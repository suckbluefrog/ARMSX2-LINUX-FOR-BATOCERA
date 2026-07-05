// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+
//
// ARM64 DMA Recompiler — Native DMA channel execution
//
// Replaces interpreter DMA transfer functions (dmaVIF0, dmaVIF1, etc.)
// with native ARM64/NEON implementations. Each channel can be individually
// toggled between interpreter fallback and native codegen.
//
// Channels:
//   D0  VIF0      D5  SIF0
//   D1  VIF1      D6  SIF1
//   D2  GIF       D7  SIF2
//   D3  fromIPU   D8  fromSPR
//   D4  toIPU     D9  toSPR

#include "Common.h"
#include "arm64/arm64Emitter.h"
#include "arm64/aDMAC.h"

#include "Dmac.h"
#include "R5900.h"
#include "Memory.h"
#include "Hw.h"
#include "Vif.h"
#include "Vif_Dma.h"
#include "Gif.h"
#include "Gif_Unit.h"
#include "Sif.h"
#include "SPR.h"
#include "VUmicro.h"
#include "MTVU.h"

#include <arm_neon.h>
#include <algorithm>

// Forward-declare IPU functions to avoid IPU.h/IPUdma.h circular include.
extern void dmaIPU0();
extern void dmaIPU1();
extern void ipu0Interrupt();
extern void ipu1Interrupt();
extern void IPU0dma();
extern void IPU1dma();
struct IPUDMAStatus { bool InProgress; bool DMAFinished; };
extern IPUDMAStatus IPU1Status;

// Forward-declare internal helpers from other DMA files.
extern void _VIF0chain();
extern void vif0SetupTransfer();
extern void _VIF1chain();
extern void vif1SetupTransfer();
extern void vif1VUFinish();
extern void vifExecQueue(int idx);
extern void vifMFIFOInterrupt();
extern void SIF0Dma();
extern void SIF1Dma();
extern void SIF2Dma();

// Globals from other TUs that we need.
extern u32 g_vif0Cycles;
extern u32 g_vif1Cycles;

// ============================================================================
//  Per-channel interpreter stub toggles (1 = interp, 0 = native)
//
//  Bisect: set individual channels to 1 to isolate native codegen bugs.
//  INTERP_DMAC overrides all to interpreter.
// ============================================================================

#ifdef INTERP_DMAC
// Group toggle: force all to interpreter
#define ISTUB_DMAC_VIF0     1
#define ISTUB_DMAC_VIF1     1
#define ISTUB_DMAC_GIF      1
#define ISTUB_DMAC_IPU0     1
#define ISTUB_DMAC_IPU1     1
#define ISTUB_DMAC_SIF0     1
#define ISTUB_DMAC_SIF1     1
#define ISTUB_DMAC_SIF2     1
#define ISTUB_DMAC_SPR0     1
#define ISTUB_DMAC_SPR1     1

#define ISTUB_DMAC_INT_VIF0 1
#define ISTUB_DMAC_INT_VIF1 1
#define ISTUB_DMAC_INT_GIF  1
#define ISTUB_DMAC_INT_IPU0 1
#define ISTUB_DMAC_INT_IPU1 1
#define ISTUB_DMAC_INT_SPR0 1
#define ISTUB_DMAC_INT_SPR1 1
#else
// Per-channel control: set to 0 to enable native ARM64 codegen
#define ISTUB_DMAC_VIF0     0
#define ISTUB_DMAC_VIF1     0
#define ISTUB_DMAC_GIF      0
#define ISTUB_DMAC_IPU0     0
#define ISTUB_DMAC_IPU1     0
#define ISTUB_DMAC_SIF0     0
#define ISTUB_DMAC_SIF1     0
#define ISTUB_DMAC_SIF2     0
#define ISTUB_DMAC_SPR0     0
#define ISTUB_DMAC_SPR1     0

#define ISTUB_DMAC_INT_VIF0 0
#define ISTUB_DMAC_INT_VIF1 0
#define ISTUB_DMAC_INT_GIF  0
#define ISTUB_DMAC_INT_IPU0 0
#define ISTUB_DMAC_INT_IPU1 0
#define ISTUB_DMAC_INT_SPR0 0
#define ISTUB_DMAC_INT_SPR1 0
#endif

// ============================================================================
//  Interpreter fallback macros
// ============================================================================

#define REC_DMAC_INTERP(name, interp_fn) \
	void recDMAC_##name() { interp_fn(); }

#define REC_DMAC_INT_INTERP(name, interp_fn) \
	void recDMACInterrupt_##name() { interp_fn(); }

// ============================================================================
//  NEON bulk memory transfer helpers
// ============================================================================

// Copy `qwc` quadwords (16 bytes each) using NEON LDP/STP.
// Both src and dst must be 16-byte aligned (guaranteed by PS2 DMA).
static __ri void neonCopyQW(void* __restrict dst, const void* __restrict src, u32 qwc)
{
	u8* __restrict d = static_cast<u8*>(dst);
	const u8* __restrict s = static_cast<const u8*>(src);

	// 4-QW (64-byte, cache-line) main loop — generates ldp/stp pairs
	while (qwc >= 4)
	{
		__builtin_prefetch(s + 256, 0, 0);
		uint8x16_t a = vld1q_u8(s);
		uint8x16_t b = vld1q_u8(s + 16);
		uint8x16_t c = vld1q_u8(s + 32);
		uint8x16_t e = vld1q_u8(s + 48);
		vst1q_u8(d, a);
		vst1q_u8(d + 16, b);
		vst1q_u8(d + 32, c);
		vst1q_u8(d + 48, e);
		s += 64;
		d += 64;
		qwc -= 4;
	}
	// Tail: 2 QW
	if (qwc & 2)
	{
		uint8x16_t a = vld1q_u8(s);
		uint8x16_t b = vld1q_u8(s + 16);
		vst1q_u8(d, a);
		vst1q_u8(d + 16, b);
		s += 32;
		d += 32;
	}
	// Tail: 1 QW
	if (qwc & 1)
	{
		vst1q_u8(d, vld1q_u8(s));
	}
}

// Copy from scratchpad to main memory, handling 16K wraparound.
static __ri void neonCopyFromSPR(u8* __restrict dst, u32 spr_addr, u32 qwc)
{
	spr_addr &= 0x3FFF;
	const u32 bytes = qwc * 16;

	if (spr_addr + bytes > _16kb)
	{
		u32 first_bytes = _16kb - spr_addr;
		u32 first_qwc = first_bytes >> 4;
		neonCopyQW(dst, &eeMem->Scratch[spr_addr], first_qwc);
		dst += first_bytes;
		neonCopyQW(dst, &eeMem->Scratch[0], qwc - first_qwc);
	}
	else
	{
		neonCopyQW(dst, &eeMem->Scratch[spr_addr], qwc);
	}
}

// Copy from main memory to scratchpad, handling 16K wraparound.
static __ri void neonCopyToSPR(u32 spr_addr, const u8* __restrict src, u32 qwc)
{
	spr_addr &= 0x3FFF;
	const u32 bytes = qwc * 16;

	if (spr_addr + bytes > _16kb)
	{
		u32 first_bytes = _16kb - spr_addr;
		u32 first_qwc = first_bytes >> 4;
		neonCopyQW(&eeMem->Scratch[spr_addr], src, first_qwc);
		src += first_bytes;
		neonCopyQW(&eeMem->Scratch[0], src, qwc - first_qwc);
	}
	else
	{
		neonCopyQW(&eeMem->Scratch[spr_addr], src, qwc);
	}
}

// VU cache clear helper — matches SPR.cpp TestClearVUs logic.
static __ri void neonTestClearVUs(u32 madr, u32 qwc, bool isWrite)
{
	if (madr >= 0x11000000 && madr < 0x11010000)
	{
		if ((madr < 0x11008000) && (VU0.VI[REG_VPU_STAT].UL & 0x1))
		{
			_vu0FinishMicro();
			CpuVU1->ExecuteBlock(0);
		}
		if ((madr >= 0x11008000) && (VU0.VI[REG_VPU_STAT].UL & 0x100) && (!THREAD_VU1 || !isWrite))
		{
			if (THREAD_VU1)
				vu1Thread.WaitVU();
			else
				CpuVU1->Execute(vu1RunCycles);
			cpuRegs.cycle = VU1.cycle;
			CpuVU0->ExecuteBlock(0);
		}

		if (madr < 0x11004000)
		{
			if (isWrite)
				CpuVU0->Clear(madr & 0xfff, qwc * 16);
		}
		else if (madr >= 0x11008000 && madr < 0x1100c000)
		{
			if (isWrite)
				CpuVU1->Clear(madr & 0x3fff, qwc * 16);
		}
	}
}

// ============================================================================
//  Native SPR helpers — NEON-accelerated transfer chains
// ============================================================================

// SPR0 (fromSPR) chain transfer — replaces _SPR0chain() with NEON copies.
static bool neonSPR0finished;

static int neonSPR0chain()
{
	tDMA_TAG* pMem;
	int partialqwc = 0;
	if (spr0ch.qwc == 0) return 0;
	pMem = SPRdmaGetAddr(spr0ch.madr, true);
	if (pMem == NULL) return -1;

	if (spr0ch.madr >= dmacRegs.rbor.ADDR &&
	    spr0ch.madr < (dmacRegs.rbor.ADDR + dmacRegs.rbsr.RMSK + 16u))
	{
		if (dmacRegs.rbsr.RMSK == 0)
		{
			spr0ch.madr += spr0ch.qwc << 4;
			spr0ch.sadr += spr0ch.qwc << 4;
			spr0ch.sadr &= 0x3FFF;
			spr0ch.qwc = 0;
		}
		else
		{
			partialqwc = std::min(spr0ch.qwc, 0x400 - ((spr0ch.sadr & 0x3fff) >> 4));
			hwMFIFOWrite(spr0ch.madr, &psSu128(spr0ch.sadr), partialqwc);
			spr0ch.madr += partialqwc << 4;
			spr0ch.madr = dmacRegs.rbor.ADDR + (spr0ch.madr & dmacRegs.rbsr.RMSK);
			spr0ch.sadr += partialqwc << 4;
			spr0ch.sadr &= 0x3FFF;
			spr0ch.qwc -= partialqwc;
		}
		neonSPR0finished = true;
	}
	else
	{
		partialqwc = std::min(spr0ch.qwc, 0x400 - ((spr0ch.sadr & 0x3fff) >> 4));

		// NEON-accelerated scratchpad → main memory copy
		neonCopyFromSPR((u8*)pMem, spr0ch.sadr, partialqwc);

		neonTestClearVUs(spr0ch.madr, partialqwc, true);

		spr0ch.madr += partialqwc << 4;
		spr0ch.sadr += partialqwc << 4;
		spr0ch.sadr &= 0x3FFF;
		spr0ch.qwc -= partialqwc;
	}

	if (spr0ch.qwc == 0 && dmacRegs.ctrl.STS == STS_fromSPR)
	{
		if (spr0ch.chcr.MOD == NORMAL_MODE || ((spr0ch.chcr.TAG >> 28) & 0x7) == TAG_CNTS)
			dmacRegs.stadr.ADDR = spr0ch.madr;
	}

	return partialqwc;
}

static void neonSPR0chainWithInt()
{
	const int cycles = neonSPR0chain() * BIAS;
	CPU_INT(DMAC_FROM_SPR, cycles);
}

static void neonSPR0interleave()
{
	int qwc = spr0ch.qwc;
	int tqwc = dmacRegs.sqwc.TQWC;
	const int sqwc = dmacRegs.sqwc.SQWC;
	tDMA_TAG* pMem;

	if (tqwc == 0) tqwc = qwc;
	CPU_INT(DMAC_FROM_SPR, qwc * BIAS);

	while (qwc > 0)
	{
		spr0ch.qwc = std::min(tqwc, qwc);
		qwc -= spr0ch.qwc;
		pMem = SPRdmaGetAddr(spr0ch.madr, true);

		switch (dmacRegs.ctrl.MFD)
		{
			case MFD_VIF1:
			case MFD_GIF:
				hwMFIFOWrite(spr0ch.madr, &psSu128(spr0ch.sadr), spr0ch.qwc);
				break;
			case NO_MFD:
			case MFD_RESERVED:
				neonTestClearVUs(spr0ch.madr, spr0ch.qwc, true);
				neonCopyFromSPR((u8*)pMem, spr0ch.sadr, spr0ch.qwc);
				break;
		}
		spr0ch.sadr += spr0ch.qwc * 16;
		spr0ch.sadr &= 0x3FFF;
		spr0ch.madr += (sqwc + spr0ch.qwc) * 16;
	}
	if (dmacRegs.ctrl.STS == STS_fromSPR)
		dmacRegs.stadr.ADDR = spr0ch.madr;
	spr0ch.qwc = 0;
}

static void neonDmaSPR0()
{
	switch (spr0ch.chcr.MOD)
	{
		case NORMAL_MODE:
			if (dmacRegs.ctrl.STS == STS_fromSPR)
				dmacRegs.stadr.ADDR = spr0ch.madr;
			neonSPR0chainWithInt();
			neonSPR0finished = true;
			return;

		case CHAIN_MODE:
		{
			tDMA_TAG* ptag;
			bool done = false;

			if (spr0ch.qwc > 0)
			{
				neonSPR0chainWithInt();
				return;
			}
			ptag = (tDMA_TAG*)&psSu32(spr0ch.sadr);
			spr0ch.sadr += 16;
			spr0ch.sadr &= 0x3FFF;
			spr0ch.unsafeTransfer(ptag);
			spr0ch.madr = ptag[1]._u32;

			switch (ptag->ID)
			{
				case TAG_CNTS:
					if (dmacRegs.ctrl.STS == STS_fromSPR)
						dmacRegs.stadr.ADDR = spr0ch.madr;
					break;
				case TAG_CNT:
					done = false;
					break;
				case TAG_END:
					done = true;
					break;
			}

			neonSPR0chainWithInt();

			if (spr0ch.chcr.TIE && ptag->IRQ)
				done = true;

			neonSPR0finished = done;
			break;
		}
		default:
			neonSPR0interleave();
			neonSPR0finished = true;
			break;
	}
}

// SPR1 (toSPR) NEON transfer helper
static bool neonSPR1finished;

static void neonSPR1transfer(const void* data, int qwc)
{
	if ((spr1ch.madr >= 0x11000000) && (spr1ch.madr < 0x11010000))
		neonTestClearVUs(spr1ch.madr, spr1ch.qwc, false);

	neonCopyToSPR(spr1ch.sadr, (const u8*)data, qwc);
	spr1ch.sadr += qwc * 16;
	spr1ch.sadr &= 0x3FFF;
}

static int neonSPR1chain()
{
	tDMA_TAG* pMem;
	if (spr1ch.qwc == 0) return 0;

	pMem = SPRdmaGetAddr(spr1ch.madr, false);
	if (pMem == NULL) return -1;

	int partialqwc = std::min(spr1ch.qwc, 0x400u);

	// NEON-accelerated main memory → scratchpad copy
	neonSPR1transfer(pMem, partialqwc);
	spr1ch.madr += partialqwc * 16;
	spr1ch.qwc -= partialqwc;

	hwDmacSrcTadrInc(spr1ch);
	return partialqwc;
}

static void neonSPR1chainWithInt()
{
	int cycles = neonSPR1chain() * BIAS;
	CPU_INT(DMAC_TO_SPR, cycles);
}

static void neonSPR1interleave()
{
	int qwc = spr1ch.qwc;
	int tqwc = dmacRegs.sqwc.TQWC;
	const int sqwc = dmacRegs.sqwc.SQWC;
	tDMA_TAG* pMem;

	if (tqwc == 0) tqwc = qwc;
	CPU_INT(DMAC_TO_SPR, qwc * BIAS);

	while (qwc > 0)
	{
		spr1ch.qwc = std::min(tqwc, qwc);
		qwc -= spr1ch.qwc;
		pMem = SPRdmaGetAddr(spr1ch.madr, false);

		neonCopyToSPR(spr1ch.sadr, (const u8*)pMem, spr1ch.qwc);
		spr1ch.sadr += spr1ch.qwc * 16;
		spr1ch.sadr &= 0x3FFF;
		spr1ch.madr += (sqwc + spr1ch.qwc) * 16;
	}
	spr1ch.qwc = 0;
}

static void neonDmaSPR1()
{
	switch (spr1ch.chcr.MOD)
	{
		case NORMAL_MODE:
			neonSPR1chainWithInt();
			neonSPR1finished = true;
			return;

		case CHAIN_MODE:
		{
			tDMA_TAG* ptag;
			bool done = false;

			if (spr1ch.qwc > 0)
			{
				neonSPR1chainWithInt();
				return;
			}

			ptag = SPRdmaGetAddr(spr1ch.tadr, false);
			if (!spr1ch.transfer("SPR1 Tag", ptag))
			{
				done = true;
				neonSPR1finished = done;
			}

			spr1ch.madr = ptag[1]._u32;

			if (spr1ch.chcr.TTE)
				neonSPR1transfer(ptag, 1);

			done = hwDmacSrcChain(spr1ch, ptag->ID);
			neonSPR1chainWithInt();

			if (spr1ch.chcr.TIE && ptag->IRQ)
				done = true;

			neonSPR1finished = done;
			break;
		}
		default:
			neonSPR1interleave();
			neonSPR1finished = true;
			break;
	}
}

// ============================================================================
//  DMA channel execution — per-channel rec functions
// ============================================================================

// --- D0: VIF0 ---
#if ISTUB_DMAC_VIF0
REC_DMAC_INTERP(VIF0, dmaVIF0)
#else
void recDMAC_VIF0()
{
	g_vif0Cycles = 0;
	CPU_SET_DMASTALL(DMAC_VIF0, false);

	if (vif0ch.qwc > 0)
	{
		if (vif0ch.chcr.MOD == CHAIN_MODE)
		{
			vif0.dmamode = VIF_CHAIN_MODE;
			vif0.done = ((vif0ch.chcr.tag().ID == TAG_REFE) ||
			             (vif0ch.chcr.tag().ID == TAG_END) ||
			             (vif0ch.chcr.tag().IRQ && vif0ch.chcr.TIE));
		}
		else
		{
			vif0.dmamode = VIF_NORMAL_FROM_MEM_MODE;
			vif0.done = true;
		}
		vif0.inprogress |= 1;
	}
	else
	{
		vif0.dmamode = VIF_CHAIN_MODE;
		vif0.done = false;
		vif0.inprogress &= ~0x1;
	}

	vif0Regs.stat.FQC = std::min((u32)0x8, vif0ch.qwc);

	if (!vif0Regs.stat.test(VIF0_STAT_VSS | VIF0_STAT_VIS | VIF0_STAT_VFS))
		CPU_INT(DMAC_VIF0, 4);
}
#endif

// --- D1: VIF1 ---
#if ISTUB_DMAC_VIF1
REC_DMAC_INTERP(VIF1, dmaVIF1)
#else
void recDMAC_VIF1()
{
	g_vif1Cycles = 0;
	vif1.inprogress = 0;
	CPU_SET_DMASTALL(DMAC_VIF1, false);

	if (vif1ch.qwc > 0)
	{
		if (vif1ch.chcr.MOD == CHAIN_MODE && vif1ch.chcr.DIR)
		{
			vif1.dmamode = VIF_CHAIN_MODE;
			vif1.done = ((vif1ch.chcr.tag().ID == TAG_REFE) ||
			             (vif1ch.chcr.tag().ID == TAG_END) ||
			             (vif1ch.chcr.tag().IRQ && vif1ch.chcr.TIE));
		}
		else
		{
			if (vif1ch.chcr.DIR)
				vif1.dmamode = VIF_NORMAL_FROM_MEM_MODE;
			else
				vif1.dmamode = VIF_NORMAL_TO_MEM_MODE;
			vif1.done = true;
		}
		vif1.inprogress |= 1;
	}
	else
	{
		vif1.inprogress &= ~0x1;
		vif1.dmamode = VIF_CHAIN_MODE;
		vif1.done = false;
	}

	if (vif1ch.chcr.DIR)
		vif1Regs.stat.FQC = std::min((u32)0x10, vif1ch.qwc);

	if (!vif1ch.chcr.DIR || !vif1Regs.stat.test(VIF1_STAT_VSS | VIF1_STAT_VIS | VIF1_STAT_VFS))
		CPU_INT(DMAC_VIF1, 4);
}
#endif

// --- D2: GIF ---
#if ISTUB_DMAC_GIF
REC_DMAC_INTERP(GIF, dmaGIF)
#else
void recDMAC_GIF()
{
	gif.gspath3done = false;
	CPU_SET_DMASTALL(DMAC_GIF, false);

	if (gifch.chcr.MOD == NORMAL_MODE)
		gif.gspath3done = true;

	if (gifch.chcr.MOD == CHAIN_MODE && gifch.qwc > 0)
	{
		if ((gifch.chcr.tag().ID == TAG_REFE) ||
		    (gifch.chcr.tag().ID == TAG_END) ||
		    (gifch.chcr.tag().IRQ && gifch.chcr.TIE))
		{
			gif.gspath3done = true;
		}
	}

	recDMACInterrupt_GIF();
}
#endif

// --- D3: fromIPU ---
#if ISTUB_DMAC_IPU0
REC_DMAC_INTERP(IPU0, dmaIPU0)
#else
void recDMAC_IPU0()
{
	if (dmacRegs.ctrl.STS == STS_fromIPU)
		dmacRegs.stadr.ADDR = ipu0ch.madr;

	CPU_SET_DMASTALL(DMAC_FROM_IPU, false);
	IPU0dma();

	if (ipu0ch.qwc == 0x10000)
	{
		ipu0ch.qwc = 0;
		ipu0ch.chcr.STR = false;
		hwDmacIrq(DMAC_FROM_IPU);
	}
}
#endif

// --- D4: toIPU ---
#if ISTUB_DMAC_IPU1
REC_DMAC_INTERP(IPU1, dmaIPU1)
#else
void recDMAC_IPU1()
{
	CPU_SET_DMASTALL(DMAC_TO_IPU, false);

	if (ipu1ch.chcr.MOD == CHAIN_MODE)
	{
		if (ipu1ch.qwc == 0)
		{
			IPU1Status.InProgress = false;
			IPU1Status.DMAFinished = false;
		}
		else
		{
			IPU1Status.InProgress = true;
			IPU1Status.DMAFinished =
				((ipu1ch.chcr.tag().ID == TAG_REFE) ||
				 (ipu1ch.chcr.tag().ID == TAG_END) ||
				 (ipu1ch.chcr.tag().IRQ && ipu1ch.chcr.TIE));
		}
	}
	else
	{
		IPU1Status.InProgress = true;
		IPU1Status.DMAFinished = true;
	}

	IPU1dma();
}
#endif

// --- D5: SIF0 ---
#if ISTUB_DMAC_SIF0
REC_DMAC_INTERP(SIF0, dmaSIF0)
#else
void recDMAC_SIF0()
{
	psHu32(SBUS_F240) |= 0x2000;
	sif0.ee.busy = true;
	sif0.ee.end = false;
	CPU_SET_DMASTALL(DMAC_SIF0, false);
	SIF0Dma();
}
#endif

// --- D6: SIF1 ---
#if ISTUB_DMAC_SIF1
REC_DMAC_INTERP(SIF1, dmaSIF1)
#else
void recDMAC_SIF1()
{
	psHu32(SBUS_F240) |= 0x4000;
	sif1.ee.busy = true;
	CPU_SET_DMASTALL(DMAC_SIF1, false);
	sif1.ee.end = false;

	if (sif1ch.chcr.MOD == CHAIN_MODE && sif1ch.qwc > 0)
	{
		if ((sif1ch.chcr.tag().ID == TAG_REFE) ||
		    (sif1ch.chcr.tag().ID == TAG_END) ||
		    (sif1ch.chcr.tag().IRQ && vif1ch.chcr.TIE))
		{
			sif1.ee.end = true;
		}
	}

	SIF1Dma();
}
#endif

// --- D7: SIF2 ---
#if ISTUB_DMAC_SIF2
REC_DMAC_INTERP(SIF2, dmaSIF2)
#else
void recDMAC_SIF2()
{
	psHu32(SBUS_F240) |= 0x8000;
	sif2.ee.busy = true;
	SIF2Dma();
}
#endif

// --- D8: fromSPR ---
#if ISTUB_DMAC_SPR0
REC_DMAC_INTERP(SPR0, dmaSPR0)
#else
void recDMAC_SPR0()
{
	neonSPR0finished = false;

	if (spr0ch.chcr.MOD == CHAIN_MODE && spr0ch.qwc > 0)
	{
		if (spr0ch.chcr.tag().ID == TAG_END)
			neonSPR0finished = true;
	}

	recDMACInterrupt_SPR0();
}
#endif

// --- D9: toSPR ---
#if ISTUB_DMAC_SPR1
REC_DMAC_INTERP(SPR1, dmaSPR1)
#else
void recDMAC_SPR1()
{
	neonSPR1finished = false;

	if (spr1ch.chcr.MOD == CHAIN_MODE && spr1ch.qwc > 0)
	{
		if ((spr1ch.chcr.tag().ID == TAG_END) ||
		    (spr1ch.chcr.tag().ID == TAG_REFE) ||
		    (spr1ch.chcr.tag().IRQ && spr1ch.chcr.TIE))
		{
			neonSPR1finished = true;
		}
	}

	recDMACInterrupt_SPR1();
}
#endif

// ============================================================================
//  DMA interrupt handlers — per-channel rec functions
// ============================================================================

// --- VIF0 interrupt ---
#if ISTUB_DMAC_INT_VIF0
REC_DMAC_INT_INTERP(VIF0, vif0Interrupt)
#else
void recDMACInterrupt_VIF0()
{
	g_vif0Cycles = 0;
	vif0Regs.stat.FQC = std::min(vif0ch.qwc, (u32)8);

	if (vif0.waitforvu)
	{
		CPU_INT(VIF_VU0_FINISH, 16);
		CPU_SET_DMASTALL(DMAC_VIF0, true);
		return;
	}

	if (vif0.irq && vif0.vifstalled.enabled && vif0.vifstalled.value == VIF_IRQ_STALL)
	{
		if (!vif0Regs.stat.ER1)
			vif0Regs.stat.INT = true;

		if (((vif0Regs.code >> 24) & 0x7f) != 0x7)
			vif0Regs.stat.VIS = true;

		hwIntcIrq(VIF0intc);
		--vif0.irq;

		if (vif0Regs.stat.test(VIF0_STAT_VSS | VIF0_STAT_VIS | VIF0_STAT_VFS))
		{
			vif0Regs.stat.FQC = std::min((u32)0x8, vif0ch.qwc);
			if (vif0ch.qwc > 0 || !vif0.done)
			{
				vif0Regs.stat.VPS = VPS_DECODING;
				CPU_SET_DMASTALL(DMAC_VIF0, true);
				return;
			}
		}
	}

	vif0.vifstalled.enabled = false;

	if (vif0.cmd)
	{
		if (vif0.done && vif0ch.qwc == 0) vif0Regs.stat.VPS = VPS_WAITING;
	}
	else
	{
		vif0Regs.stat.VPS = VPS_IDLE;
	}

	if (vif0.inprogress & 0x1)
	{
		_VIF0chain();
		vif0Regs.stat.FQC = std::min(vif0ch.qwc, (u32)8);
		CPU_INT(DMAC_VIF0, g_vif0Cycles);
		return;
	}

	if (!vif0.done)
	{
		if (!(dmacRegs.ctrl.DMAE) || vif0Regs.stat.VSS)
			return;

		if ((vif0.inprogress & 0x1) == 0) vif0SetupTransfer();
		vif0Regs.stat.FQC = std::min(vif0ch.qwc, (u32)8);
		CPU_INT(DMAC_VIF0, g_vif0Cycles);
		return;
	}

	if (vif0.vifstalled.enabled && vif0.done)
	{
		CPU_INT(DMAC_VIF0, 0);
		return;
	}

	vif0ch.chcr.STR = false;
	vif0Regs.stat.FQC = std::min((u32)0x8, vif0ch.qwc);
	vif0.vifstalled.enabled = false;
	vif0.irqoffset.enabled = false;
	if (vif0.queued_program) vifExecQueue(0);
	g_vif0Cycles = 0;
	hwDmacIrq(DMAC_VIF0);
	CPU_SET_DMASTALL(DMAC_VIF0, false);
	vif0Regs.stat.FQC = 0;
}
#endif

// --- VIF1 interrupt ---
#if ISTUB_DMAC_INT_VIF1
REC_DMAC_INT_INTERP(VIF1, vif1Interrupt)
#else
void recDMACInterrupt_VIF1()
{
	g_vif1Cycles = 0;

	if (gifRegs.stat.APATH == 2 && gifUnit.gifPath[GIF_PATH_2].isDone())
	{
		gifRegs.stat.APATH = 0;
		gifRegs.stat.OPH = 0;
		vif1Regs.stat.VGW = false;
		if (gifUnit.checkPaths(1, 0, 1))
			gifUnit.Execute(false, true);
	}

	if (dmacRegs.ctrl.MFD == MFD_VIF1)
	{
		if (vif1ch.chcr.MOD == NORMAL_MODE)
			Console.WriteLn("MFIFO mode is normal (which isn't normal here)! %x", vif1ch.chcr._u32);
		vif1Regs.stat.FQC = std::min((u32)0x10, vif1ch.qwc);
		vifMFIFOInterrupt();
		return;
	}

	if (vif1ch.chcr.DIR)
	{
		const bool isDirect = (vif1.cmd & 0x7f) == 0x50;
		const bool isDirectHL = (vif1.cmd & 0x7f) == 0x51;
		if ((isDirect && !gifUnit.CanDoPath2()) || (isDirectHL && !gifUnit.CanDoPath2HL()))
		{
			CPU_INT(DMAC_VIF1, 128);
			if (gifRegs.stat.APATH == 3)
				vif1Regs.stat.VGW = 1;
			CPU_SET_DMASTALL(DMAC_VIF1, true);
			return;
		}
		vif1Regs.stat.VGW = 0;
		vif1Regs.stat.FQC = std::min(vif1ch.qwc, (u32)16);
	}

	if (vif1.waitforvu)
	{
		CPU_INT(VIF_VU1_FINISH, std::max(16, cpuGetCycles(VU_MTVU_BUSY)));
		CPU_SET_DMASTALL(DMAC_VIF1, true);
		return;
	}

	if (vif1Regs.stat.VGW)
	{
		CPU_SET_DMASTALL(DMAC_VIF1, true);
		return;
	}

	if (!vif1ch.chcr.STR)
		return;

	if (vif1.irq && vif1.vifstalled.enabled && vif1.vifstalled.value == VIF_IRQ_STALL)
	{
		if (!vif1Regs.stat.ER1)
			vif1Regs.stat.INT = true;

		if (((vif1Regs.code >> 24) & 0x7f) != 0x7)
			vif1Regs.stat.VIS = true;

		hwIntcIrq(VIF1intc);
		--vif1.irq;

		if (vif1Regs.stat.test(VIF1_STAT_VSS | VIF1_STAT_VIS | VIF1_STAT_VFS))
		{
			vif1Regs.stat.FQC = std::min((u32)0x10, vif1ch.qwc);
			if ((vif1ch.qwc > 0 || !vif1.done) && !CHECK_VIF1STALLHACK)
			{
				vif1Regs.stat.VPS = VPS_DECODING;
				CPU_SET_DMASTALL(DMAC_VIF1, true);
				return;
			}
		}
	}

	vif1.vifstalled.enabled = false;

	if (vif1.cmd)
	{
		if (vif1.done && (vif1ch.qwc == 0))
			vif1Regs.stat.VPS = VPS_WAITING;
	}
	else
	{
		vif1Regs.stat.VPS = VPS_IDLE;
	}

	if (vif1.inprogress & 0x1)
	{
		_VIF1chain();
		if (vif1ch.chcr.DIR)
			vif1Regs.stat.FQC = std::min(vif1ch.qwc, (u32)16);

		if (!(vif1Regs.stat.VGW && gifUnit.gifPath[GIF_PATH_3].state != GIF_PATH_IDLE))
		{
			if (vif1.waitforvu)
				CPU_INT(DMAC_VIF1, std::max(static_cast<int>(g_vif1Cycles), cpuGetCycles(VU_MTVU_BUSY)));
			else
				CPU_INT(DMAC_VIF1, g_vif1Cycles);
		}
		return;
	}

	if (!vif1.done)
	{
		if (!(dmacRegs.ctrl.DMAE) || vif1Regs.stat.VSS)
			return;

		if ((vif1.inprogress & 0x1) == 0)
			vif1SetupTransfer();
		if (vif1ch.chcr.DIR)
			vif1Regs.stat.FQC = std::min(vif1ch.qwc, (u32)16);

		if (!(vif1Regs.stat.VGW && gifUnit.gifPath[GIF_PATH_3].state != GIF_PATH_IDLE))
		{
			if (vif1.waitforvu)
				CPU_INT(DMAC_VIF1, std::max(static_cast<int>(g_vif1Cycles), cpuGetCycles(VU_MTVU_BUSY)));
			else
				CPU_INT(DMAC_VIF1, g_vif1Cycles);
		}
		return;
	}

	if (vif1.vifstalled.enabled && vif1.done)
	{
		CPU_INT(DMAC_VIF1, 0);
		CPU_SET_DMASTALL(DMAC_VIF1, true);
		return;
	}

	if ((vif1ch.chcr.DIR == VIF_NORMAL_TO_MEM_MODE) && vif1.GSLastDownloadSize <= 16)
		gifRegs.stat.OPH = false;

	if (vif1ch.chcr.DIR)
		vif1Regs.stat.FQC = std::min(vif1ch.qwc, (u32)16);

	vif1ch.chcr.STR = false;
	vif1.vifstalled.enabled = false;
	vif1.irqoffset.enabled = false;
	if (vif1.queued_program) vifExecQueue(1);
	g_vif1Cycles = 0;
	hwDmacIrq(DMAC_VIF1);
	CPU_SET_DMASTALL(DMAC_VIF1, false);
}
#endif

// --- GIF interrupt ---
#if ISTUB_DMAC_INT_GIF
REC_DMAC_INT_INTERP(GIF, gifInterrupt)
#else
void recDMACInterrupt_GIF()
{
	gifCheckPathStatus(false);

	if (gifUnit.gifPath[GIF_PATH_3].state == GIF_PATH_IDLE)
	{
		if (vif1Regs.stat.VGW)
		{
			if (!(cpuRegs.interrupt & (1 << DMAC_VIF1)))
				CPU_INT(DMAC_VIF1, 1);

			if (!gifUnit.Path3Masked() || gifch.qwc == 0)
				GifDMAInt(16);

			CPU_SET_DMASTALL(DMAC_GIF, gifUnit.Path3Masked() || !gifUnit.CanDoPath3());
			return;
		}
	}

	if (dmacRegs.ctrl.MFD == MFD_GIF)
	{
		gifMFIFOInterrupt();
		return;
	}

	if (gifUnit.gsSIGNAL.queued)
	{
		GifDMAInt(128);
		CPU_SET_DMASTALL(DMAC_GIF, true);
		if (gif_fifo.fifoSize == 16)
			return;
	}

	if (gif_fifo.fifoSize > 0)
	{
		const int readSize = gif_fifo.read_fifo();
		if (readSize)
			GifDMAInt(readSize * BIAS);

		if ((!CheckPaths() && gif_fifo.fifoSize == 16) || readSize)
		{
			CPU_SET_DMASTALL(DMAC_GIF, gifUnit.Path3Masked() || !gifUnit.CanDoPath3());
			return;
		}
	}

	if (!(gifch.chcr.STR))
		return;

	if ((gifch.qwc > 0) || (!gif.gspath3done))
	{
		if (!dmacRegs.ctrl.DMAE)
		{
			GifDMAInt(64);
			CPU_SET_DMASTALL(DMAC_GIF, true);
			return;
		}
		GIFdma();
		return;
	}

	gif.gscycles = 0;
	gifch.chcr.STR = false;
	gifRegs.stat.FQC = gif_fifo.fifoSize;
	CalculateFIFOCSR();
	hwDmacIrq(DMAC_GIF);

	if (gif_fifo.fifoSize)
		GifDMAInt(8 * BIAS);
}
#endif

// --- IPU0 interrupt ---
#if ISTUB_DMAC_INT_IPU0
REC_DMAC_INT_INTERP(IPU0, ipu0Interrupt)
#else
void recDMACInterrupt_IPU0()
{
	if (ipu0ch.qwc > 0)
	{
		IPU0dma();
		return;
	}

	ipu0ch.chcr.STR = false;
	hwDmacIrq(DMAC_FROM_IPU);
	CPU_SET_DMASTALL(DMAC_FROM_IPU, false);
}
#endif

// --- IPU1 interrupt ---
#if ISTUB_DMAC_INT_IPU1
REC_DMAC_INT_INTERP(IPU1, ipu1Interrupt)
#else
void recDMACInterrupt_IPU1()
{
	if (!IPU1Status.DMAFinished || IPU1Status.InProgress)
	{
		IPU1dma();
		return;
	}

	ipu1ch.chcr.STR = false;
	hwDmacIrq(DMAC_TO_IPU);
	CPU_SET_DMASTALL(DMAC_TO_IPU, false);
}
#endif

// --- SPR0 interrupt (fromSPR) ---
#if ISTUB_DMAC_INT_SPR0
REC_DMAC_INT_INTERP(SPR0, SPRFROMinterrupt)
#else
void recDMACInterrupt_SPR0()
{
	if (!neonSPR0finished || spr0ch.qwc > 0)
	{
		neonDmaSPR0();

		if (spr0ch.qwc == 0)
		{
			switch (dmacRegs.ctrl.MFD)
			{
				case MFD_VIF1:
				case MFD_GIF:
					hwMFIFOResume();
					break;
				default:
					break;
			}
		}
		return;
	}

	spr0ch.chcr.STR = false;
	hwDmacIrq(DMAC_FROM_SPR);
}
#endif

// --- SPR1 interrupt (toSPR) ---
#if ISTUB_DMAC_INT_SPR1
REC_DMAC_INT_INTERP(SPR1, SPRTOinterrupt)
#else
void recDMACInterrupt_SPR1()
{
	if (!neonSPR1finished || spr1ch.qwc > 0)
	{
		neonDmaSPR1();
		return;
	}

	spr1ch.chcr.STR = false;
	hwDmacIrq(DMAC_TO_SPR);
}
#endif
