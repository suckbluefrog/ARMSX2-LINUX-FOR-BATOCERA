// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — Load / Store Instructions
// LB, LBU, LH, LHU, LW, LWU, LD, LQ, LWL, LWR, LDL, LDR, LWC1, LQC2
// SB, SH, SW, SD, SQ, SWL, SWR, SDL, SDR, SWC1, SQC2
//
// Aligned 8/16/32/64-bit loads and stores use VTLB fastmem when available:
// a single LDR/STR through RFASTMEMBASE (x23 = vtlbdata.fastmem_base).
// On fault the signal handler backpatches to a thunk that calls the C
// vtlb_memRead/Write functions (see recVTLB_arm64.cpp).

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "VU.h"
#include "arm64/arm64Emitter.h"
#include "arm64/AsmHelpers.h"
#include "iRecAnalysis.h"
#include "vtlb.h"

using namespace R5900;

// FPR offset from RCPUSTATE (x19) — for LWC1/SWC1. fpuRegs lives at FPUREGS_BASE
// inside cpuRegistersPack, so we address FPU state without a dedicated base reg.
static constexpr s64 FPR_OFFSET(int reg) { return FPUREGS_BASE + offsetof(fpuRegisters, fpr) + reg * sizeof(FPRreg); }

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
#if defined(INTERP_LOAD) || defined(INTERP_EE)
#define ISTUB_LB       1
#define ISTUB_LBU      1
#define ISTUB_LH       1
#define ISTUB_LHU      1
#define ISTUB_LW       1
#define ISTUB_LWU      1
#define ISTUB_LD       1
#define ISTUB_LWC1     1
#define ISTUB_LWL      1
#define ISTUB_LWR      1
#define ISTUB_LDL      1
#define ISTUB_LDR      1
#define ISTUB_LQ       1
#define ISTUB_LQC2     1
#else
#define ISTUB_LB       0
#define ISTUB_LBU      0
#define ISTUB_LH       0
#define ISTUB_LHU      0
#define ISTUB_LW       0
#define ISTUB_LWU      0
#define ISTUB_LD       0
#define ISTUB_LWC1     0
#define ISTUB_LWL      0   // unaligned — native shift+mask+merge
#define ISTUB_LWR      0
#define ISTUB_LDL      0
#define ISTUB_LDR      0
#define ISTUB_LQ       0   // 128-bit — native NEON Ldr q-reg
#define ISTUB_LQC2     0   // 128-bit VU — native, falls back on SYNC/FINISH
#endif

#if defined(INTERP_STORE) || defined(INTERP_EE)
#define ISTUB_SB       1
#define ISTUB_SH       1
#define ISTUB_SW       1
#define ISTUB_SD       1
#define ISTUB_SWC1     1
#define ISTUB_SWL      1
#define ISTUB_SWR      1
#define ISTUB_SDL      1
#define ISTUB_SDR      1
#define ISTUB_SQ       1
#define ISTUB_SQC2     1
#else
#define ISTUB_SB       0
#define ISTUB_SH       0
#define ISTUB_SW       0
#define ISTUB_SD       0
#define ISTUB_SWC1     0
#define ISTUB_SWL      0   // unaligned — native shift+mask+merge
#define ISTUB_SWR      0
#define ISTUB_SDL      0
#define ISTUB_SDR      0
#define ISTUB_SQ       0   // 128-bit — native NEON Str q-reg
#define ISTUB_SQC2     0   // 128-bit VU — native, falls back on SYNC/FINISH
#endif

// ============================================================================
//  Helpers: address computation and vtlb call wrappers
// ============================================================================

// Emit code to compute effective address (Rs + sign-extended Imm) into w0.
// Uses constant propagation when Rs is known at compile time.
static void armComputeAddress()
{
	if (GPR_IS_CONST1(_Rs_))
	{
		u32 addr = g_cpuConstRegs[_Rs_].UL[0] + _Imm_;
		armAsm->Mov(a64::w0, addr);
	}
	else
	{
		armLoadGPR32(a64::w0, _Rs_);
		if (_Imm_ != 0)
			armAsm->Add(a64::w0, a64::w0, _Imm_);
	}
}

// Flush state before a slow-path vtlb call.
static void armPreVtlbCall()
{
	armFlushConstRegs();
	armFlushPC();
}

// Record a fastmem load/store for backpatching.
// addr_reg and data_reg are ARM64 register codes.
static void armRecordFastmem(const u8* code_start, u8 addr_reg, u8 data_reg,
	u8 bits, bool is_signed, bool is_load, bool is_fpr)
{
	u32 code_size = static_cast<u32>(armGetCurrentCodePointer() - code_start);
	vtlb_AddLoadStoreInfo(reinterpret_cast<uptr>(code_start), code_size, pc,
		0 /*gpr_bitmask*/, 0 /*fpr_bitmask*/,
		addr_reg, data_reg, bits, is_signed, is_load, is_fpr);
}

// Returns true if we should use the fastmem path for the current instruction.
static bool armUseFastmem()
{
	return CHECK_FASTMEM && !vtlb_IsFaultingPC(pc);
}

// Force an event test after a load from the EE counter/timer register range
// (0x10000000..0x10001FFF). Mirrors x86 iR5900LoadStore.cpp:117-119 + 139-143
// (`iFlushCall(FLUSH_INTERPRETER); g_branch = 2;` — the "ESPN Games" fix). Without
// this, a tight polling loop on the EE counter can stay in the JIT block long
// enough to miss interrupt delivery.
//
// Flushes the in-flight cycle delta (cpuRegs.cycle = nextEventCycle + RCYCLE),
// then forces `nextEventCycle = cpuRegs.cycle` so the next iBranchTest at block
// exit dispatches events. Zeros RCYCLE so the block epilogue's cycle-flush is
// a no-op (the invariant `cycle = nec + RCYCLE` still holds with RCYCLE=0).
// Sets g_branch = 2 so the EE compile loop ends the block here.
static void armForceEventTestAfterRead()
{
	// cpuRegs.cycle = nextEventCycle + RCYCLE
	armAsm->Ldr(a64::x9, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Add(a64::x9, a64::x9, RCYCLE);
	armAsm->Str(a64::x9, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
	// nextEventCycle = cpuRegs.cycle — next iBranchTest fires event dispatch.
	armAsm->Str(a64::x9, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	// RCYCLE = cycle - nec = 0 — preserves the Phase B invariant for the
	// block epilogue's cycle writeback.
	armAsm->Mov(RCYCLE, a64::xzr);

	armFlushPC();
	armFlushCode();
	armFlushConstRegs();

	g_cpuHasConstReg = 1;
	g_cpuFlushedConstReg = 1;
	g_branch = 2;
}

// Emit the EE counter event test if this load's constant address lands in
// the 0x10000000..0x10001FFF range. Only valid when _Rs_ is const-tracked
// (matches x86 which only applies the fix in the const-address path).
static void armMaybeForceEventTestAfterRead()
{
	if (!GPR_IS_CONST1(_Rs_))
		return;
	const u32 addr = g_cpuConstRegs[_Rs_].UL[0] + _Imm_;
	if ((addr & 0xFFFFE000u) == 0x10000000u)
		armForceEventTestAfterRead();
}

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ============================================================================
//                              LOADS
// ============================================================================

// ---- LB — Load Byte (sign-extended to 64 bits) ----

#if ISTUB_LB
// ESPN-counter event-test fix — emit BEFORE armCallInterpreter so the const
// tracker still holds _Rs_ when armMaybeForceEventTestAfterRead reads it.
// armCallInterpreter would otherwise drop const tracking on return.
void recLB() { armMaybeForceEventTestAfterRead(); armCallInterpreter(R5900::Interpreter::OpcodeImpl::LB); }
#else
void recLB()
{
	// NOTE: No `if (!_Rt_) return;` — PS2 memory reads can have side effects
	// (hardware FIFO drains, interrupt-on-read registers, counter triggers),
	// and `LB $0, 0(reg)` is a legitimate way to issue one.  armStoreGPR64
	// already no-ops on gpr==0, so the read happens and the write is dropped.
	// Matches x86 recLoad (iR5900LoadStore.cpp:101-137).
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldrsb(RSCRATCHGPR, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 8, true, true, false);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem8_t>);
		armEmitReloadCycleAfterCall();
		armAsm->Sxtb(a64::x0, a64::w0);
		armStoreGPR64(a64::x0, _Rt_);
	}

	armMaybeForceEventTestAfterRead();
}
#endif

// ---- LBU — Load Byte Unsigned ----

#if ISTUB_LBU
void recLBU() { armMaybeForceEventTestAfterRead(); armCallInterpreter(R5900::Interpreter::OpcodeImpl::LBU); }
#else
void recLBU()
{
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldrb(RWSCRATCH, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 8, false, true, false);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem8_t>);
		armEmitReloadCycleAfterCall();
		armStoreGPR64(a64::x0, _Rt_);
	}

	armMaybeForceEventTestAfterRead();
}
#endif

// ---- LH — Load Halfword (sign-extended) ----

#if ISTUB_LH
void recLH() { armMaybeForceEventTestAfterRead(); armCallInterpreter(R5900::Interpreter::OpcodeImpl::LH); }
#else
void recLH()
{
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldrsh(RSCRATCHGPR, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 16, true, true, false);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem16_t>);
		armEmitReloadCycleAfterCall();
		armAsm->Sxth(a64::x0, a64::w0);
		armStoreGPR64(a64::x0, _Rt_);
	}

	armMaybeForceEventTestAfterRead();
}
#endif

// ---- LHU — Load Halfword Unsigned ----

#if ISTUB_LHU
void recLHU() { armMaybeForceEventTestAfterRead(); armCallInterpreter(R5900::Interpreter::OpcodeImpl::LHU); }
#else
void recLHU()
{
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldrh(RWSCRATCH, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 16, false, true, false);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem16_t>);
		armEmitReloadCycleAfterCall();
		armStoreGPR64(a64::x0, _Rt_);
	}

	armMaybeForceEventTestAfterRead();
}
#endif

// ---- LW — Load Word (sign-extended) ----

#if ISTUB_LW
void recLW() { armMaybeForceEventTestAfterRead(); armCallInterpreter(R5900::Interpreter::OpcodeImpl::LW); }
#else
void recLW()
{
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldrsw(RSCRATCHGPR, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 32, true, true, false);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem32_t>);
		armEmitReloadCycleAfterCall();
		armStoreGPR64SignExt32(a64::w0, _Rt_);
	}

	armMaybeForceEventTestAfterRead();
}
#endif

// ---- LWU — Load Word Unsigned ----

#if ISTUB_LWU
void recLWU() { armMaybeForceEventTestAfterRead(); armCallInterpreter(R5900::Interpreter::OpcodeImpl::LWU); }
#else
void recLWU()
{
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 32, false, true, false);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem32_t>);
		armEmitReloadCycleAfterCall();
		armStoreGPR64(a64::x0, _Rt_);
	}

	armMaybeForceEventTestAfterRead();
}
#endif

// ---- LD — Load Doubleword ----

#if ISTUB_LD
void recLD() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LD); }
#else
void recLD()
{
	// x86 recLoad doesn't apply the EE-counter event-test fix to 64-bit loads
	// (`bits <= 32 && needs_flush`), so LD skips armMaybeForceEventTestAfterRead.
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldr(RSCRATCHGPR, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 64, false, true, false);
		armStoreGPR64(RSCRATCHGPR, _Rt_);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem64_t>);
		armEmitReloadCycleAfterCall();
		armStoreGPR64(a64::x0, _Rt_);
	}
}
#endif

// ---- LWC1 — Load Word to COP1 (FPU register) ----

#if ISTUB_LWC1
void recLWC1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LWC1); }
#else
void recLWC1()
{
	armComputeAddress();

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldr(RWSCRATCH, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RSCRATCHGPR.GetCode(), 32, false, true, false);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Rt_)));
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead<mem32_t>);
		armEmitReloadCycleAfterCall();
		armAsm->Str(a64::w0, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Rt_)));
	}
}
#endif

// ============================================================================
//  Unaligned loads (LWL/LWR/LDL/LDR)
//
//  PS2 unaligned MIPS-style: address is byte-granular; the op reads the
//  ALIGNED word/dword and merges some of its bytes into the destination GPR.
//  Bit-offset within the aligned unit drives the shift/mask. We re-derive
//  bit_offset AFTER the slow-path BL since vtlb_memRead doesn't touch GPR
//  memory and the BL clobbers caller-saved scratch regs.
//
//  No fastmem path — the read is followed by complex merge logic, and the
//  fastmem backpatcher only handles single-instruction load/store fault
//  windows. Slow path only: BL vtlb_memRead<mem32_t> / <mem64_t>.
// ============================================================================

// ---- LWL — Load Word Left (high bytes of unaligned word) ----
//
// Mirrors x86 recLWL in iR5900LoadStore.cpp:273. Sign-extends merged result
// to 64 bits (low 32 of GPR + sign extension into high 32).
#if ISTUB_LWL
void recLWL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LWL); }
#else
void recLWL()
{
	armComputeAddress();                 // w0 = addr (Rs + Imm)
	armAsm->Bic(a64::w0, a64::w0, 3u);   // w0 = aligned

	armPreVtlbCall();
	armEmitFlushCycleBeforeCall();
	armEmitCall((const void*)&vtlb_memRead<mem32_t>);
	armEmitReloadCycleAfterCall();
	// w0 = loaded aligned 32-bit word

	if (!_Rt_)
		return;

	// Re-derive bit_offset (= (addr & 3) * 8) — re-read Rs since BL clobbered
	// caller-saved scratch but cpuRegs.GPR is intact.
	armLoadGPR32(a64::w1, _Rs_);
	if (_Imm_ != 0)
		armAsm->Add(a64::w1, a64::w1, _Imm_);
	armAsm->And(a64::w1, a64::w1, 3u);
	armAsm->Lsl(a64::w1, a64::w1, 3);    // w1 = bit_offset (0,8,16,24)

	// Mask: 0xFFFFFF >> bit_offset → keep low (24 - bit_offset) bits of Rt.
	armAsm->Mov(a64::w2, 0x00FFFFFFu);
	armAsm->Lsr(a64::w2, a64::w2, a64::w1);

	// w3 = Rt low 32; mask off bytes loaded.
	armLoadGPR32(a64::w3, _Rt_);
	armAsm->And(a64::w3, a64::w3, a64::w2);

	// shift = 24 - bit_offset; w0 <<= shift; w0 |= w3.
	armAsm->Mov(a64::w4, 24);
	armAsm->Sub(a64::w4, a64::w4, a64::w1);
	armAsm->Lsl(a64::w0, a64::w0, a64::w4);
	armAsm->Orr(a64::w0, a64::w0, a64::w3);

	// Sign-extend to 64-bit and store into GPR[_Rt_] low qword.
	armDelConstReg(_Rt_);
	armAsm->Sxtw(a64::x0, a64::w0);
	armAsm->Str(a64::x0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
}
#endif

// ---- LWR — Load Word Right (low bytes of unaligned word) ----
//
// Mirrors x86 recLWR in iR5900LoadStore.cpp:334. When bit_offset == 0, the
// loaded word IS the result (sign-extend to 64). Otherwise mask + merge,
// upper 32 bits cleared (matches x86's xRegister32 ops zeroing the high 32).
#if ISTUB_LWR
void recLWR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LWR); }
#else
void recLWR()
{
	armComputeAddress();
	armAsm->Bic(a64::w0, a64::w0, 3u);

	armPreVtlbCall();
	armEmitFlushCycleBeforeCall();
	armEmitCall((const void*)&vtlb_memRead<mem32_t>);
	armEmitReloadCycleAfterCall();

	if (!_Rt_)
		return;

	armDelConstReg(_Rt_);

	// Re-derive byte offset (0..3).
	armLoadGPR32(a64::w1, _Rs_);
	if (_Imm_ != 0)
		armAsm->Add(a64::w1, a64::w1, _Imm_);
	armAsm->And(a64::w1, a64::w1, 3u);

	a64::Label do_merge, end;
	armAsm->Cbnz(a64::w1, &do_merge);

	// bit_offset == 0: result = sign_extend_64(loaded_word).
	armAsm->Sxtw(a64::x0, a64::w0);
	armAsm->Str(a64::x0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
	armAsm->B(&end);

	armAsm->Bind(&do_merge);
	armAsm->Lsl(a64::w1, a64::w1, 3);    // w1 = bit_offset (8,16,24)

	// Mask: 0xFFFFFF00 << (24 - bit_offset) → keeps upper bytes of Rt.
	armAsm->Mov(a64::w2, 24);
	armAsm->Sub(a64::w2, a64::w2, a64::w1);
	armAsm->Mov(a64::w3, 0xFFFFFF00u);
	armAsm->Lsl(a64::w3, a64::w3, a64::w2);

	armLoadGPR32(a64::w4, _Rt_);
	armAsm->And(a64::w4, a64::w4, a64::w3);

	// w0 = loaded >> bit_offset; w0 |= w4. Result is 32-bit; high 32 = 0.
	armAsm->Lsr(a64::w0, a64::w0, a64::w1);
	armAsm->Orr(a64::w0, a64::w0, a64::w4);

	// 32-bit Str clears nothing on its own; explicitly write 0 to high 32 by
	// using a 64-bit Str of zero-extended w0.
	armAsm->Str(a64::x0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));

	armAsm->Bind(&end);
}
#endif

// ---- LDL — Load Doubleword Left (high bytes of unaligned dword) ----
//
// Mirrors x86 recLDL in iR5900LoadStore.cpp:587. 64-bit aligned read,
// 8-bit-granular merge. When (addr & 7) == 7, no shift needed — result
// IS the loaded dword.
#if ISTUB_LDL
void recLDL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LDL); }
#else
void recLDL()
{
	if (!_Rt_)
		return;

	armComputeAddress();
	armAsm->Bic(a64::w0, a64::w0, 7u);   // align to 8 bytes

	armPreVtlbCall();
	armEmitFlushCycleBeforeCall();
	armEmitCall((const void*)&vtlb_memRead<mem64_t>);
	armEmitReloadCycleAfterCall();
	// x0 = loaded aligned 64-bit dword

	armDelConstReg(_Rt_);

	// Re-derive byte_off (0..7).
	armLoadGPR32(a64::w1, _Rs_);
	if (_Imm_ != 0)
		armAsm->Add(a64::w1, a64::w1, _Imm_);
	armAsm->And(a64::w1, a64::w1, 7u);

	a64::Label do_merge, end;
	armAsm->Cmp(a64::w1, 7);
	armAsm->B(a64::ne, &do_merge);

	// byte_off == 7: result = loaded dword as-is (no shift).
	armAsm->Str(a64::x0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
	armAsm->B(&end);

	armAsm->Bind(&do_merge);
	// shift = (byte_off + 1) * 8;  rt_kept = Rt & (~0ULL >> shift);
	// merged = (loaded << (64 - shift)) | rt_kept;
	armAsm->Add(a64::w1, a64::w1, 1);
	armAsm->Lsl(a64::w1, a64::w1, 3);    // w1 = shift (8..56)

	// w2 = 64 - shift  (used for left-shift of loaded)
	armAsm->Mov(a64::w2, 64);
	armAsm->Sub(a64::w2, a64::w2, a64::w1);

	// x3 = (~0ULL) >> shift; x3 &= Rt; merged = (x0 << w2) | x3
	armAsm->Mov(a64::x3, ~u64{0});
	armAsm->Lsr(a64::x3, a64::x3, a64::x1);
	armAsm->Ldr(a64::x4, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
	armAsm->And(a64::x3, a64::x3, a64::x4);
	armAsm->Lsl(a64::x0, a64::x0, a64::x2);
	armAsm->Orr(a64::x0, a64::x0, a64::x3);
	armAsm->Str(a64::x0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));

	armAsm->Bind(&end);
}
#endif

// ---- LDR — Load Doubleword Right (low bytes of unaligned dword) ----
//
// Mirrors x86 recLDR in iR5900LoadStore.cpp:674. When byte_off == 0, the
// loaded dword IS the result.
#if ISTUB_LDR
void recLDR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LDR); }
#else
void recLDR()
{
	if (!_Rt_)
		return;

	armComputeAddress();
	armAsm->Bic(a64::w0, a64::w0, 7u);

	armPreVtlbCall();
	armEmitFlushCycleBeforeCall();
	armEmitCall((const void*)&vtlb_memRead<mem64_t>);
	armEmitReloadCycleAfterCall();

	armDelConstReg(_Rt_);

	armLoadGPR32(a64::w1, _Rs_);
	if (_Imm_ != 0)
		armAsm->Add(a64::w1, a64::w1, _Imm_);
	armAsm->And(a64::w1, a64::w1, 7u);

	a64::Label do_merge, end;
	armAsm->Cbnz(a64::w1, &do_merge);

	// byte_off == 0: result = loaded as-is.
	armAsm->Str(a64::x0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
	armAsm->B(&end);

	armAsm->Bind(&do_merge);
	// shift = byte_off * 8;
	// merged = (Rt & (~0ULL << (64 - shift))) | (loaded >> shift);
	armAsm->Lsl(a64::w1, a64::w1, 3);    // w1 = shift (8..56)
	armAsm->Mov(a64::w2, 64);
	armAsm->Sub(a64::w2, a64::w2, a64::w1);

	// x3 = (~0ULL) << w2 (mask preserving high (64 - shift) bits of Rt).
	armAsm->Mov(a64::x3, ~u64{0});
	armAsm->Lsl(a64::x3, a64::x3, a64::x2);
	armAsm->Ldr(a64::x4, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
	armAsm->And(a64::x3, a64::x3, a64::x4);
	armAsm->Lsr(a64::x0, a64::x0, a64::x1);
	armAsm->Orr(a64::x0, a64::x0, a64::x3);
	armAsm->Str(a64::x0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));

	armAsm->Bind(&end);
}
#endif

// ---- LQ — Load Quadword (128-bit) to GPR ----
//
// Address forced to 16-byte alignment (`addr & ~0x0F`) per PS2 spec — matches
// upstream recLoadQuad in x86/ix86-32/iR5900LoadStore.cpp:64-97. Fastmem path
// is a single 128-bit Ldr q-reg + Str q-reg into the GPR slot. Slow path BLs
// vtlb_memRead128 which returns r128 in q0 (AAPCS64 vector return).
#if ISTUB_LQ
void recLQ() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LQ); }
#else
void recLQ()
{
	armComputeAddress();
	armAsm->And(a64::w0, a64::w0, ~0xFu);

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldr(a64::q0, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), a64::q0.GetCode(), 128, false, true, false);
		if (_Rt_)
		{
			armDelConstReg(_Rt_);
			armAsm->Str(a64::q0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
		}
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead128);
		armEmitReloadCycleAfterCall();
		if (_Rt_)
		{
			armDelConstReg(_Rt_);
			armAsm->Str(a64::q0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
		}
	}
}
#endif

// ---- LQC2 — Load Quadword to COP2 (VU0 VF register) ----
//
// Mirrors x86 recLQC2 in microVU_Macro.inl:808. Falls back to interp when the
// analysis pass flagged EEINST_COP2_SYNC_VU0 / EEINST_COP2_FINISH_VU0 (so
// vu0Sync / vu0Finish runs first — same fallback shape as QMFC2/QMTC2 in
// iR5900COP2_arm64.cpp). Otherwise emits native 128-bit load to &VU0.VF[_Ft_].
#if ISTUB_LQC2
void recLQC2() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LQC2); }
#else
void recLQC2()
{
	if (g_pCurInstInfo && (g_pCurInstInfo->info & (EEINST_COP2_SYNC_VU0 | EEINST_COP2_FINISH_VU0)))
	{
		armCallInterpreter(R5900::Interpreter::OpcodeImpl::LQC2);
		return;
	}

	armComputeAddress();
	armAsm->And(a64::w0, a64::w0, ~0xFu);

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Ldr(a64::q0, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), a64::q0.GetCode(), 128, false, true, false);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memRead128);
		armEmitReloadCycleAfterCall();
	}

	if (_Rt_)
	{
		// _Rt_ is the VF index (named _Ft_ in interp). Store q0 → VU0.VF[_Rt_].
		armAsm->Mov(RSCRATCHGPR, reinterpret_cast<uintptr_t>(&VU0.VF[_Rt_]));
		armAsm->Str(a64::q0, a64::MemOperand(RSCRATCHGPR));
	}
}
#endif

// ============================================================================
//                              STORES
// ============================================================================

// ---- SB — Store Byte ----

#if ISTUB_SB
void recSB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SB); }
#else
void recSB()
{
	armComputeAddress();
	armLoadGPR32(a64::w1, _Rt_);

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Strb(a64::w1, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RXARG2.GetCode(), 8, false, false, false);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memWrite<mem8_t>);
		armEmitReloadCycleAfterCall();
	}
}
#endif

// ---- SH — Store Halfword ----

#if ISTUB_SH
void recSH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SH); }
#else
void recSH()
{
	armComputeAddress();
	armLoadGPR32(a64::w1, _Rt_);

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Strh(a64::w1, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RXARG2.GetCode(), 16, false, false, false);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memWrite<mem16_t>);
		armEmitReloadCycleAfterCall();
	}
}
#endif

// ---- SW — Store Word ----

#if ISTUB_SW
void recSW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SW); }
#else
void recSW()
{
	armComputeAddress();
	armLoadGPR32(a64::w1, _Rt_);

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Str(a64::w1, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RXARG2.GetCode(), 32, false, false, false);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memWrite<mem32_t>);
		armEmitReloadCycleAfterCall();
	}
}
#endif

// ---- SD — Store Doubleword ----

#if ISTUB_SD
void recSD() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SD); }
#else
void recSD()
{
	armComputeAddress();
	armLoadGPR64(a64::x1, _Rt_);

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Str(a64::x1, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RXARG2.GetCode(), 64, false, false, false);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memWrite<mem64_t>);
		armEmitReloadCycleAfterCall();
	}
}
#endif

// ---- SWC1 — Store Word from COP1 (FPU register) ----

#if ISTUB_SWC1
void recSWC1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SWC1); }
#else
void recSWC1()
{
	armComputeAddress();
	armAsm->Ldr(a64::w1, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Rt_)));

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Str(a64::w1, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), RXARG2.GetCode(), 32, false, false, false);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memWrite<mem32_t>);
		armEmitReloadCycleAfterCall();
	}
}
#endif

// ============================================================================
//  Unaligned stores (SWL/SWR/SDL/SDR)
//
//  Two-step: read aligned mem, mask + merge with shifted Rt, write back. When
//  bit_offset == 0 (8-aligned for word, 8-aligned-end for dword), no merge
//  needed — write Rt directly. We re-derive bit_offset after the slow-path
//  read BL since vtlb_memRead clobbers caller-saved scratch.
// ============================================================================

// ---- SWL — Store Word Left (high bytes of Rt → high bytes of unaligned mem) ----
//
// Mirrors x86 recSWL in iR5900LoadStore.cpp:402.
#if ISTUB_SWL
void recSWL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SWL); }
#else
void recSWL()
{
	// Special path for byte_off == 3: just write Rt as a 32-bit word.
	// Detect at runtime via the same merge/skip pattern as upstream.
	armComputeAddress();
	armAsm->Bic(a64::w0, a64::w0, 3u);

	// Read aligned word (always — even when byte_off==3 we'll discard).
	armPreVtlbCall();
	armEmitFlushCycleBeforeCall();
	armEmitCall((const void*)&vtlb_memRead<mem32_t>);
	armEmitReloadCycleAfterCall();
	// w0 = loaded aligned word

	// Re-derive byte_off + recompute aligned addr.
	armLoadGPR32(a64::w1, _Rs_);
	if (_Imm_ != 0)
		armAsm->Add(a64::w1, a64::w1, _Imm_);
	armAsm->And(a64::w2, a64::w1, 3u);   // w2 = byte_off
	armAsm->Bic(a64::w1, a64::w1, 3u);   // w1 = aligned addr (for write call below)

	a64::Label do_write;
	armAsm->Cmp(a64::w2, 3);
	armAsm->B(a64::eq, &do_write);       // byte_off == 3 → write Rt directly

	armAsm->Lsl(a64::w2, a64::w2, 3);    // w2 = bit_offset (0,8,16)

	// Mask read: keep upper bytes — w_merged = loaded & (0xFFFFFF00 << bit_off).
	armAsm->Mov(a64::w3, 0xFFFFFF00u);
	armAsm->Lsl(a64::w3, a64::w3, a64::w2);
	armAsm->And(a64::w0, a64::w0, a64::w3);

	if (_Rt_)
	{
		// shift = 24 - bit_off; merged |= Rt >> shift.
		armAsm->Mov(a64::w4, 24);
		armAsm->Sub(a64::w4, a64::w4, a64::w2);
		armLoadGPR32(a64::w5, _Rt_);
		armAsm->Lsr(a64::w5, a64::w5, a64::w4);
		armAsm->Orr(a64::w0, a64::w0, a64::w5);
	}

	a64::Label end;
	armAsm->B(&end);

	armAsm->Bind(&do_write);
	// byte_off == 3: write Rt's low 32 bits directly.
	armLoadGPR32(a64::w0, _Rt_);

	armAsm->Bind(&end);
	// w0 = value to write, w1 = aligned address. Swap into vtlb_memWrite ABI
	// (addr in w0, value in w1) via a scratch.
	armAsm->Mov(a64::w2, a64::w0);
	armAsm->Mov(a64::w0, a64::w1);
	armAsm->Mov(a64::w1, a64::w2);

	armEmitFlushCycleBeforeCall();
	armEmitCall((const void*)&vtlb_memWrite<mem32_t>);
	armEmitReloadCycleAfterCall();
}
#endif

// ---- SWR — Store Word Right (low bytes of Rt → low bytes of unaligned mem) ----
#if ISTUB_SWR
void recSWR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SWR); }
#else
void recSWR()
{
	armComputeAddress();
	armAsm->Bic(a64::w0, a64::w0, 3u);

	armPreVtlbCall();
	armEmitFlushCycleBeforeCall();
	armEmitCall((const void*)&vtlb_memRead<mem32_t>);
	armEmitReloadCycleAfterCall();

	armLoadGPR32(a64::w1, _Rs_);
	if (_Imm_ != 0)
		armAsm->Add(a64::w1, a64::w1, _Imm_);
	armAsm->And(a64::w2, a64::w1, 3u);   // byte_off
	armAsm->Bic(a64::w1, a64::w1, 3u);   // aligned addr

	a64::Label do_write;
	armAsm->Cbz(a64::w2, &do_write);     // byte_off == 0 → write Rt directly

	armAsm->Lsl(a64::w2, a64::w2, 3);    // bit_offset (8,16,24)

	// Keep low bytes of read: shift_amt = 24 - bit_off; mask = 0xFFFFFF >> shift_amt.
	armAsm->Mov(a64::w3, 24);
	armAsm->Sub(a64::w3, a64::w3, a64::w2);
	armAsm->Mov(a64::w4, 0x00FFFFFFu);
	armAsm->Lsr(a64::w4, a64::w4, a64::w3);
	armAsm->And(a64::w0, a64::w0, a64::w4);

	if (_Rt_)
	{
		armLoadGPR32(a64::w5, _Rt_);
		armAsm->Lsl(a64::w5, a64::w5, a64::w2);
		armAsm->Orr(a64::w0, a64::w0, a64::w5);
	}

	a64::Label end;
	armAsm->B(&end);

	armAsm->Bind(&do_write);
	armLoadGPR32(a64::w0, _Rt_);

	armAsm->Bind(&end);
	armAsm->Mov(a64::w2, a64::w0);
	armAsm->Mov(a64::w0, a64::w1);
	armAsm->Mov(a64::w1, a64::w2);

	armEmitFlushCycleBeforeCall();
	armEmitCall((const void*)&vtlb_memWrite<mem32_t>);
	armEmitReloadCycleAfterCall();
}
#endif

// ---- SDL — Store Doubleword Left ----
#if ISTUB_SDL
void recSDL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SDL); }
#else
void recSDL()
{
	armComputeAddress();
	armAsm->Bic(a64::w0, a64::w0, 7u);

	armPreVtlbCall();
	armEmitFlushCycleBeforeCall();
	armEmitCall((const void*)&vtlb_memRead<mem64_t>);
	armEmitReloadCycleAfterCall();
	// x0 = loaded dword

	armLoadGPR32(a64::w1, _Rs_);
	if (_Imm_ != 0)
		armAsm->Add(a64::w1, a64::w1, _Imm_);
	armAsm->And(a64::w2, a64::w1, 7u);   // byte_off
	armAsm->Bic(a64::w1, a64::w1, 7u);   // aligned addr

	a64::Label do_write;
	armAsm->Cmp(a64::w2, 7);
	armAsm->B(a64::eq, &do_write);       // byte_off == 7 → write Rt directly

	// shift = (byte_off + 1) * 8 (range 8..56);
	// merged = (loaded & (~0ULL << shift)) | (Rt >> (64 - shift));
	armAsm->Add(a64::w2, a64::w2, 1);
	armAsm->Lsl(a64::w2, a64::w2, 3);    // shift
	armAsm->Mov(a64::w3, 64);
	armAsm->Sub(a64::w3, a64::w3, a64::w2);

	armAsm->Mov(a64::x4, ~u64{0});
	armAsm->Lsl(a64::x4, a64::x4, a64::x2);
	armAsm->And(a64::x0, a64::x0, a64::x4);

	if (_Rt_)
	{
		armAsm->Ldr(a64::x5, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
		armAsm->Lsr(a64::x5, a64::x5, a64::x3);
		armAsm->Orr(a64::x0, a64::x0, a64::x5);
	}

	a64::Label end;
	armAsm->B(&end);

	armAsm->Bind(&do_write);
	armAsm->Ldr(a64::x0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));

	armAsm->Bind(&end);
	// vtlb_memWrite<mem64_t>(addr in w0, value in x1). Swap.
	armAsm->Mov(a64::x2, a64::x0);
	armAsm->Mov(a64::w0, a64::w1);
	armAsm->Mov(a64::x1, a64::x2);

	armEmitFlushCycleBeforeCall();
	armEmitCall((const void*)&vtlb_memWrite<mem64_t>);
	armEmitReloadCycleAfterCall();
}
#endif

// ---- SDR — Store Doubleword Right ----
#if ISTUB_SDR
void recSDR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SDR); }
#else
void recSDR()
{
	armComputeAddress();
	armAsm->Bic(a64::w0, a64::w0, 7u);

	armPreVtlbCall();
	armEmitFlushCycleBeforeCall();
	armEmitCall((const void*)&vtlb_memRead<mem64_t>);
	armEmitReloadCycleAfterCall();

	armLoadGPR32(a64::w1, _Rs_);
	if (_Imm_ != 0)
		armAsm->Add(a64::w1, a64::w1, _Imm_);
	armAsm->And(a64::w2, a64::w1, 7u);
	armAsm->Bic(a64::w1, a64::w1, 7u);

	a64::Label do_write;
	armAsm->Cbz(a64::w2, &do_write);     // byte_off == 0 → write Rt directly

	// shift = byte_off * 8 (range 8..56);
	// merged = (loaded & (~0ULL >> (64 - shift))) | (Rt << shift);
	armAsm->Lsl(a64::w2, a64::w2, 3);
	armAsm->Mov(a64::w3, 64);
	armAsm->Sub(a64::w3, a64::w3, a64::w2);

	armAsm->Mov(a64::x4, ~u64{0});
	armAsm->Lsr(a64::x4, a64::x4, a64::x3);
	armAsm->And(a64::x0, a64::x0, a64::x4);

	if (_Rt_)
	{
		armAsm->Ldr(a64::x5, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
		armAsm->Lsl(a64::x5, a64::x5, a64::x2);
		armAsm->Orr(a64::x0, a64::x0, a64::x5);
	}

	a64::Label end;
	armAsm->B(&end);

	armAsm->Bind(&do_write);
	armAsm->Ldr(a64::x0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));

	armAsm->Bind(&end);
	armAsm->Mov(a64::x2, a64::x0);
	armAsm->Mov(a64::w0, a64::w1);
	armAsm->Mov(a64::x1, a64::x2);

	armEmitFlushCycleBeforeCall();
	armEmitCall((const void*)&vtlb_memWrite<mem64_t>);
	armEmitReloadCycleAfterCall();
}
#endif

// ---- SQ — Store Quadword (128-bit) from GPR ----
#if ISTUB_SQ
void recSQ() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SQ); }
#else
void recSQ()
{
	armComputeAddress();
	armAsm->And(a64::w0, a64::w0, ~0xFu);

	// Commit any pending const-prop value before the direct 128-bit Ldr.
	armFlushConstReg(_Rt_);
	armAsm->Ldr(a64::q0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Str(a64::q0, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), a64::q0.GetCode(), 128, false, false, false);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		// vtlb_memWrite128(addr, r128 value): addr in w0 (already there),
		// value in q0 (already loaded above).
		armEmitCall((const void*)&vtlb_memWrite128);
		armEmitReloadCycleAfterCall();
	}
}
#endif

// ---- SQC2 — Store Quadword to memory from COP2 (VU0 VF register) ----
//
// Same SYNC/FINISH fallback shape as LQC2.
#if ISTUB_SQC2
void recSQC2() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SQC2); }
#else
void recSQC2()
{
	if (g_pCurInstInfo && (g_pCurInstInfo->info & (EEINST_COP2_SYNC_VU0 | EEINST_COP2_FINISH_VU0)))
	{
		armCallInterpreter(R5900::Interpreter::OpcodeImpl::SQC2);
		return;
	}

	armComputeAddress();
	armAsm->And(a64::w0, a64::w0, ~0xFu);

	// Load value from VU0.VF[_Rt_] (interp uses _Ft_; same field for COP2).
	armAsm->Mov(RSCRATCHGPR, reinterpret_cast<uintptr_t>(&VU0.VF[_Rt_]));
	armAsm->Ldr(a64::q0, a64::MemOperand(RSCRATCHGPR));

	if (armUseFastmem())
	{
		armPreVtlbCall();
		const u8* code_start = armGetCurrentCodePointer();
		armAsm->Str(a64::q0, a64::MemOperand(RFASTMEMBASE, a64::x0));
		armRecordFastmem(code_start, RXARG1.GetCode(), a64::q0.GetCode(), 128, false, false, false);
	}
	else
	{
		armPreVtlbCall();
		armEmitFlushCycleBeforeCall();
		armEmitCall((const void*)&vtlb_memWrite128);
		armEmitReloadCycleAfterCall();
	}
}
#endif

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
