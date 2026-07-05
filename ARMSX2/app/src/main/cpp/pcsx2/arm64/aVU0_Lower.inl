// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU0 Recompiler — Lower Instruction Stubs
// Integer ALU, load/store, branches, FDIV, flag ops,
// move/transfer, random, special (XTOP/XITOP; no XGKICK, no EFU on VU0)
//
// Mirrors iVU1Lower_arm64.cpp exactly; VU0_BASE_REG (x23) points to &VU0.
// XGKICK/XTOP/XITOP always use the interpreter (VU0 doesn't drive the GIF).
// EFU ops (ESADD/ERSADD/ELENG/ERLENG/EATANxy/EATANxz/ESUM/ERCPR/ESQRT/
// ERSQRT/ESIN/EATAN/EEXP) are NOPs on VU0 — VU0 hardware has no EFU, so
// writing VU0.p is undefined. Matches x86 microVU (pass1 sets isNOP=true
// when isVU0).
// ISTUB_VU0_* macros shadow ISTUB_VU_* so both units bisect independently.

// Included from aVU0.cpp. All headers (Common.h / VUops.h / VU.h / MTVU.h /
// arm64/arm64Emitter.h / arm64/AsmHelpers.h / <cmath>) and
// `using namespace vixl::aarch64` are already in effect in the parent.
// VU0_BASE_REG / viOff / vfOff also come from the parent.

// ============================================================================
//  Native codegen helpers
// ============================================================================

// Non-inline VI backup wrapper — callable from JIT-emitted code.
// Mirrors _vuBackupVI() in VUops.cpp (which is __fi and can't be linked).
static void vu0BackupVI(VURegs* VU, u32 reg)
{
	if (VU->VIBackupCycles && reg == VU->VIRegNumber)
	{
		VU->VIBackupCycles = 2;
		return;
	}
	VU->VIBackupCycles = 2;
	VU->VIRegNumber = reg;
	VU->VIOldValue = VU->VI[reg].US[0];
}

// Emit a BL to vu0BackupVI(VU, reg).
// Must be emitted before any VI register write.
static void emitBackupVI(u32 reg)
{
	armAsm->Mov(x0, VU0_BASE_REG);
	armAsm->Mov(w1, reg);
	armEmitCall(reinterpret_cast<const void*>(vu0BackupVI));
}

// ============================================================================
//  C wrapper helpers — replicate interpreter logic for JIT-called functions.
//  These use the same field-extraction macros as VUops.cpp, but take VURegs*.
// ============================================================================

// Field extraction (mirrors VUops.cpp macros, but parameter-based)
#define W_Ft(VU) (((VU)->code >> 16) & 0x1F)
#define W_Fs(VU) (((VU)->code >> 11) & 0x1F)
#define W_Fd(VU) (((VU)->code >>  6) & 0x1F)
#define W_It(VU) (W_Ft(VU) & 0xF)
#define W_Is(VU) (W_Fs(VU) & 0xF)
#define W_Id(VU) (W_Fd(VU) & 0xF)
#define W_X(VU)  (((VU)->code >> 24) & 0x1)
#define W_Y(VU)  (((VU)->code >> 23) & 0x1)
#define W_Z(VU)  (((VU)->code >> 22) & 0x1)
#define W_W(VU)  (((VU)->code >> 21) & 0x1)
#define W_XYZW(VU) (((VU)->code >> 21) & 0xF)
#define W_Fsf(VU)   (((VU)->code >> 21) & 0x03)
#define W_Ftf(VU)   (((VU)->code >> 23) & 0x03)
#define W_Imm11(VU) ((s32)((VU)->code & 0x400 ? 0xfffffc00 | ((VU)->code & 0x3ff) : (VU)->code & 0x3ff))

// Type aliases to avoid vixl namespace clashes (vixl defines s16/u16 as registers)
using vu_s16 = int16_t;
using vu_u16 = uint16_t;
using vu_s32 = int32_t;
using vu_u32 = uint32_t;

// Emit inline ARM64 for vuDouble clamping on a W register.
// Flushes denormals to ±0, clamps inf/NaN to ±max if vu0SignOverflow is set.
// INTENTIONAL DIVERGENCE FROM UPSTREAM x86 JIT: the port uses
// CHECK_VU_SIGN_OVERFLOW here (not CHECK_VU_OVERFLOW as upstream does) as
// the signal for an event-test optimization. Reverting to CHECK_VU_OVERFLOW
// breaks event testing on this port — keep SIGN_OVERFLOW.
// wreg: u32 float bits (modified in place)
// wtmp: scratch register (clobbered)
static void emitVuDouble(const Register& wreg, const Register& wtmp)
{
	a64::Label done;
	armAsm->Ubfx(wtmp, wreg, 23, 8); // extract 8-bit exponent

	if (CHECK_VU_SIGN_OVERFLOW(0))
	{
		a64::Label denormal;
		armAsm->Cbz(wtmp, &denormal);           // exp==0 -> denormal
		armAsm->Cmp(wtmp, 0xFF);
		armAsm->B(&done, a64::ne);               // normal -> done
		// Infinity/NaN: clamp to ±max
		armAsm->And(wtmp, wreg, 0x80000000u);
		armAsm->Mov(wreg, 0x7f7fffff);
		armAsm->Orr(wreg, wreg, wtmp);
		armAsm->B(&done);
		armAsm->Bind(&denormal);
		armAsm->And(wreg, wreg, 0x80000000u);    // flush to ±0
	}
	else
	{
		armAsm->Cbnz(wtmp, &done);               // exp!=0 -> done
		armAsm->And(wreg, wreg, 0x80000000u);    // flush to ±0
	}
	armAsm->Bind(&done);
}

// Float denormal/overflow clamping — mirrors the port's local VUops.cpp
// vuDouble() (gated on CHECK_VU_SIGN_OVERFLOW, not CHECK_VU_OVERFLOW as
// upstream). See emitVuDouble comment — SIGN_OVERFLOW is the event-test
// optimization signal on this port; do not flip to OVERFLOW.
static float vu0Double(vu_u32 f)
{
	switch (f & 0x7f800000)
	{
		case 0x0:
			f &= 0x80000000;
			return *(float*)&f;
		case 0x7f800000:
			if (CHECK_VU_SIGN_OVERFLOW(0))
			{
				u32 d = (f & 0x80000000) | 0x7f7fffff;
				return *(float*)&d;
			}
			break;
	}
	return *(float*)&f;
}

// Branch address calculation — mirrors _branchAddr() in VUops.cpp.
// VU0 has 4KB micro memory (mask 0x0fff); the VU1 mask 0x3fff would
// compute bpc outside VU0's range and corrupt TPC on taken branches.
static s32 vu0BranchAddr(VURegs* VU)
{
	s32 bpc = VU->VI[REG_TPC].SL + (W_Imm11(VU) * 8);
	bpc &= 0x0fff;
	return bpc;
}

// Set branch state — mirrors _setBranch() in VUops.cpp
static void vu0SetBranch(VURegs* VU, u32 bpc)
{
	if (VU->branch == 1)
	{
		VU->delaybranchpc = bpc;
		VU->takedelaybranch = true;
	}
	else
	{
		VU->branch = 2;
		VU->branchpc = bpc;
	}
}

// LFSR advance — mirrors AdvanceLFSR() in VUops.cpp
static void vu0AdvanceLFSR(VURegs* VU)
{
	int x = (VU->VI[REG_R].UL >> 4) & 1;
	int y = (VU->VI[REG_R].UL >> 22) & 1;
	VU->VI[REG_R].UL <<= 1;
	VU->VI[REG_R].UL ^= x ^ y;
	VU->VI[REG_R].UL = (VU->VI[REG_R].UL & 0x7fffff) | 0x3f800000;
}

// ============================================================================
//  C wrapper functions — called from JIT via BL.
//  Each takes VURegs* VU (passed in x0 = VU0_BASE_REG).
// ============================================================================

// Macro to generate rec function that calls a C wrapper
#define REC_VU0_LOWER_CALL(name) \
	void recVU0_##name() { \
		armAsm->Mov(x0, VU0_BASE_REG); \
		armEmitCall(reinterpret_cast<const void*>(vu0_##name)); \
	}

// --- FDIV wrappers ---
static void vu0_DIV(VURegs* VU)
{
	float ft = vu0Double(VU->VF[W_Ft(VU)].UL[W_Ftf(VU)]);
	float fs = vu0Double(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)]);
	VU->statusflag &= ~0x30;
	if (ft == 0.0)
	{
		if (fs == 0.0)
			VU->statusflag |= 0x10;
		else
			VU->statusflag |= 0x20;
		if ((VU->VF[W_Ft(VU)].UL[W_Ftf(VU)] & 0x80000000) ^
			(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)] & 0x80000000))
			VU->q.UL = 0xFF7FFFFF;
		else
			VU->q.UL = 0x7F7FFFFF;
	}
	else
	{
		VU->q.F = fs / ft;
		VU->q.F = vu0Double(VU->q.UL);
	}
}

static void vu0_SQRT(VURegs* VU)
{
	float ft = vu0Double(VU->VF[W_Ft(VU)].UL[W_Ftf(VU)]);
	VU->statusflag &= ~0x30;
	if (ft < 0.0)
		VU->statusflag |= 0x10;
	VU->q.F = sqrt(fabs(ft));
	VU->q.F = vu0Double(VU->q.UL);
}

static void vu0_RSQRT(VURegs* VU)
{
	float ft = vu0Double(VU->VF[W_Ft(VU)].UL[W_Ftf(VU)]);
	float fs = vu0Double(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)]);
	VU->statusflag &= ~0x30;
	if (ft == 0.0)
	{
		VU->statusflag |= 0x20;
		if (fs != 0)
		{
			if ((VU->VF[W_Ft(VU)].UL[W_Ftf(VU)] & 0x80000000) ^
				(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)] & 0x80000000))
				VU->q.UL = 0xFF7FFFFF;
			else
				VU->q.UL = 0x7F7FFFFF;
		}
		else
		{
			if ((VU->VF[W_Ft(VU)].UL[W_Ftf(VU)] & 0x80000000) ^
				(VU->VF[W_Fs(VU)].UL[W_Fsf(VU)] & 0x80000000))
				VU->q.UL = 0x80000000;
			else
				VU->q.UL = 0;
			VU->statusflag |= 0x10;
		}
	}
	else
	{
		if (ft < 0.0)
			VU->statusflag |= 0x10;
		float temp = sqrt(fabs(ft));
		VU->q.F = fs / temp;
		VU->q.F = vu0Double(VU->q.UL);
	}
}

// WAITQ/WAITP are NOPs in the interpreter
static void vu0_WAITQ(VURegs* VU) { (void)VU; }
static void vu0_WAITP(VURegs* VU) { (void)VU; }

// Under THREAD_VU1, when VU0 helpers touch an address with bit 0x4000 set,
// GET_VU_MEM aliases the pointer onto VU1.VF/VI (shared with the MTVU thread).
// We must drain the MTVU ring before reading/writing, otherwise we race on
// live VF/VI bytes. Mirrors microVU's mVUaddrFix → mVU.waitMTVU emission.
static inline void vu0MTVUSyncVU1Reg(u32 addr)
{
	if (THREAD_VU1 && (addr & 0x4000))
		vu1Thread.WaitVU();
}

// --- Load/Store wrappers ---
static void vu0_LQ(VURegs* VU)
{
	if (W_Ft(VU) == 0) return;
	vu_s16 imm = (VU->code & 0x400) ? (VU->code & 0x3ff) | 0xfc00 : (VU->code & 0x3ff);
	vu_u16 addr = ((imm + VU->VI[W_Is(VU)].SS[0]) * 16);
	vu0MTVUSyncVU1Reg(addr);
	u32* ptr = (u32*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) VU->VF[W_Ft(VU)].UL[0] = ptr[0];
	if (W_Y(VU)) VU->VF[W_Ft(VU)].UL[1] = ptr[1];
	if (W_Z(VU)) VU->VF[W_Ft(VU)].UL[2] = ptr[2];
	if (W_W(VU)) VU->VF[W_Ft(VU)].UL[3] = ptr[3];
}

static void vu0_LQD(VURegs* VU)
{
	vu0BackupVI(VU, W_Is(VU));
	// x86 microVU uses the decremented value for the effective address even
	// when _Is_==0 (allocGPR+xDEC regardless of register). VI[0] itself stays
	// hardwired to 0 — only the address calc sees the decrement wrap.
	const u16 is_new = static_cast<u16>(VU->VI[W_Is(VU)].US[0] - 1);
	if (W_Is(VU) != 0) VU->VI[W_Is(VU)].US[0] = is_new;
	if (W_Ft(VU) == 0) return;
	u32 addr = (is_new * 16);
	vu0MTVUSyncVU1Reg(addr);
	u32* ptr = (u32*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) VU->VF[W_Ft(VU)].UL[0] = ptr[0];
	if (W_Y(VU)) VU->VF[W_Ft(VU)].UL[1] = ptr[1];
	if (W_Z(VU)) VU->VF[W_Ft(VU)].UL[2] = ptr[2];
	if (W_W(VU)) VU->VF[W_Ft(VU)].UL[3] = ptr[3];
}

static void vu0_LQI(VURegs* VU)
{
	vu0BackupVI(VU, W_Is(VU));
	if (W_Ft(VU))
	{
		u32 addr = (VU->VI[W_Is(VU)].US[0] * 16);
		vu0MTVUSyncVU1Reg(addr);
		u32* ptr = (u32*)GET_VU_MEM(VU, addr);
		if (W_X(VU)) VU->VF[W_Ft(VU)].UL[0] = ptr[0];
		if (W_Y(VU)) VU->VF[W_Ft(VU)].UL[1] = ptr[1];
		if (W_Z(VU)) VU->VF[W_Ft(VU)].UL[2] = ptr[2];
		if (W_W(VU)) VU->VF[W_Ft(VU)].UL[3] = ptr[3];
	}
	if (W_Fs(VU) != 0) VU->VI[W_Is(VU)].US[0]++;
}

static void vu0_SQ(VURegs* VU)
{
	vu_s16 imm = (VU->code & 0x400) ? (VU->code & 0x3ff) | 0xfc00 : (VU->code & 0x3ff);
	vu_u16 addr = ((imm + VU->VI[W_It(VU)].SS[0]) * 16);
	vu0MTVUSyncVU1Reg(addr);
	u32* ptr = (u32*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) ptr[0] = VU->VF[W_Fs(VU)].UL[0];
	if (W_Y(VU)) ptr[1] = VU->VF[W_Fs(VU)].UL[1];
	if (W_Z(VU)) ptr[2] = VU->VF[W_Fs(VU)].UL[2];
	if (W_W(VU)) ptr[3] = VU->VF[W_Fs(VU)].UL[3];
}

static void vu0_SQD(VURegs* VU)
{
	vu0BackupVI(VU, W_It(VU));
	// Match x86 microVU: address uses the decremented value regardless of
	// whether _It_==0. Writeback still guards on _Ft_!=0 (existing semantic
	// that treats SQD with Ft==0 as a full NOP) AND _It_!=0 (don't corrupt
	// the VI[0]-is-zero invariant — x86's allocator caches a copy so the
	// dirty value may never reach memory).
	const u16 it_new = static_cast<u16>(VU->VI[W_It(VU)].US[0] - 1);
	if (W_Ft(VU) != 0 && W_It(VU) != 0) VU->VI[W_It(VU)].US[0] = it_new;
	u32 addr = (it_new * 16);
	vu0MTVUSyncVU1Reg(addr);
	u32* ptr = (u32*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) ptr[0] = VU->VF[W_Fs(VU)].UL[0];
	if (W_Y(VU)) ptr[1] = VU->VF[W_Fs(VU)].UL[1];
	if (W_Z(VU)) ptr[2] = VU->VF[W_Fs(VU)].UL[2];
	if (W_W(VU)) ptr[3] = VU->VF[W_Fs(VU)].UL[3];
}

static void vu0_SQI(VURegs* VU)
{
	vu0BackupVI(VU, W_It(VU));
	u32 addr = (VU->VI[W_It(VU)].US[0] * 16);
	vu0MTVUSyncVU1Reg(addr);
	u32* ptr = (u32*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) ptr[0] = VU->VF[W_Fs(VU)].UL[0];
	if (W_Y(VU)) ptr[1] = VU->VF[W_Fs(VU)].UL[1];
	if (W_Z(VU)) ptr[2] = VU->VF[W_Fs(VU)].UL[2];
	if (W_W(VU)) ptr[3] = VU->VF[W_Fs(VU)].UL[3];
	if (W_Ft(VU) != 0) VU->VI[W_It(VU)].US[0]++;
}

static void vu0_ILW(VURegs* VU)
{
	if (W_It(VU) == 0) return;
	vu_s16 imm = (VU->code & 0x400) ? (VU->code & 0x3ff) | 0xfc00 : (VU->code & 0x3ff);
	vu_u16 addr = ((imm + VU->VI[W_Is(VU)].SS[0]) * 16);
	vu0MTVUSyncVU1Reg(addr);
	vu_u16* ptr = (vu_u16*)GET_VU_MEM(VU, addr);
	// First-set-wins priority (X>Y>Z>W) — matches x86 microVU's offsetSS
	// selection. Only diverges from the sequential-if last-set-wins order
	// when multiple lane bits are set (invalid encoding).
	if      (W_X(VU)) VU->VI[W_It(VU)].US[0] = ptr[0];
	else if (W_Y(VU)) VU->VI[W_It(VU)].US[0] = ptr[2];
	else if (W_Z(VU)) VU->VI[W_It(VU)].US[0] = ptr[4];
	else if (W_W(VU)) VU->VI[W_It(VU)].US[0] = ptr[6];
}

static void vu0_ISW(VURegs* VU)
{
	vu_s16 imm = (VU->code & 0x400) ? (VU->code & 0x3ff) | 0xfc00 : (VU->code & 0x3ff);
	vu_u16 addr = ((imm + VU->VI[W_Is(VU)].SS[0]) * 16);
	vu0MTVUSyncVU1Reg(addr);
	vu_u16* ptr = (vu_u16*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) { ptr[0] = VU->VI[W_It(VU)].US[0]; ptr[1] = 0; }
	if (W_Y(VU)) { ptr[2] = VU->VI[W_It(VU)].US[0]; ptr[3] = 0; }
	if (W_Z(VU)) { ptr[4] = VU->VI[W_It(VU)].US[0]; ptr[5] = 0; }
	if (W_W(VU)) { ptr[6] = VU->VI[W_It(VU)].US[0]; ptr[7] = 0; }
}

static void vu0_ILWR(VURegs* VU)
{
	if (W_It(VU) == 0) return;
	u32 addr = (VU->VI[W_Is(VU)].US[0] * 16);
	vu0MTVUSyncVU1Reg(addr);
	vu_u16* ptr = (vu_u16*)GET_VU_MEM(VU, addr);
	// First-set-wins priority (X>Y>Z>W) — matches x86 microVU's offsetSS.
	if      (W_X(VU)) VU->VI[W_It(VU)].US[0] = ptr[0];
	else if (W_Y(VU)) VU->VI[W_It(VU)].US[0] = ptr[2];
	else if (W_Z(VU)) VU->VI[W_It(VU)].US[0] = ptr[4];
	else if (W_W(VU)) VU->VI[W_It(VU)].US[0] = ptr[6];
}

static void vu0_ISWR(VURegs* VU)
{
	u32 addr = (VU->VI[W_Is(VU)].US[0] * 16);
	vu0MTVUSyncVU1Reg(addr);
	vu_u16* ptr = (vu_u16*)GET_VU_MEM(VU, addr);
	if (W_X(VU)) { ptr[0] = VU->VI[W_It(VU)].US[0]; ptr[1] = 0; }
	if (W_Y(VU)) { ptr[2] = VU->VI[W_It(VU)].US[0]; ptr[3] = 0; }
	if (W_Z(VU)) { ptr[4] = VU->VI[W_It(VU)].US[0]; ptr[5] = 0; }
	if (W_W(VU)) { ptr[6] = VU->VI[W_It(VU)].US[0]; ptr[7] = 0; }
}

// --- Branch wrappers ---
static void vu0_B(VURegs* VU)
{
	s32 bpc = vu0BranchAddr(VU);
	vu0SetBranch(VU, bpc);
}

static void vu0_BAL(VURegs* VU)
{
	s32 bpc = vu0BranchAddr(VU);
	if (W_It(VU))
	{
		if (VU->branch == 1)
			VU->VI[W_It(VU)].US[0] = (VU->branchpc + 8) / 8;
		else
			VU->VI[W_It(VU)].US[0] = (VU->VI[REG_TPC].UL + 8) / 8;
	}
	vu0SetBranch(VU, bpc);
}

static void vu0_JR(VURegs* VU)
{
	u32 bpc = VU->VI[W_Is(VU)].US[0] * 8;
	vu0SetBranch(VU, bpc);
}

static void vu0_JALR(VURegs* VU)
{
	u32 bpc = VU->VI[W_Is(VU)].US[0] * 8;
	if (W_It(VU))
	{
		if (VU->branch == 1)
			VU->VI[W_It(VU)].US[0] = (VU->branchpc + 8) / 8;
		else
			VU->VI[W_It(VU)].US[0] = (VU->VI[REG_TPC].UL + 8) / 8;
	}
	vu0SetBranch(VU, bpc);
}

static void vu0_IBEQ(VURegs* VU)
{
	vu_s16 dest = VU->VI[W_It(VU)].US[0];
	vu_s16 src = VU->VI[W_Is(VU)].US[0];
	if (VU->VIBackupCycles > 0)
	{
		if (VU->VIRegNumber == W_It(VU)) dest = VU->VIOldValue;
		if (VU->VIRegNumber == W_Is(VU)) src = VU->VIOldValue;
	}
	if (dest == src)
	{
		s32 bpc = vu0BranchAddr(VU);
		vu0SetBranch(VU, bpc);
	}
}

static void vu0_IBNE(VURegs* VU)
{
	vu_s16 dest = VU->VI[W_It(VU)].US[0];
	vu_s16 src = VU->VI[W_Is(VU)].US[0];
	if (VU->VIBackupCycles > 0)
	{
		if (VU->VIRegNumber == W_It(VU)) dest = VU->VIOldValue;
		if (VU->VIRegNumber == W_Is(VU)) src = VU->VIOldValue;
	}
	if (dest != src)
	{
		s32 bpc = vu0BranchAddr(VU);
		vu0SetBranch(VU, bpc);
	}
}

static void vu0_IBLTZ(VURegs* VU)
{
	vu_s16 src = VU->VI[W_Is(VU)].US[0];
	if (VU->VIBackupCycles > 0)
	{
		if (VU->VIRegNumber == W_Is(VU)) src = VU->VIOldValue;
	}
	if (src < 0)
	{
		s32 bpc = vu0BranchAddr(VU);
		vu0SetBranch(VU, bpc);
	}
}

static void vu0_IBGTZ(VURegs* VU)
{
	vu_s16 src = VU->VI[W_Is(VU)].US[0];
	if (VU->VIBackupCycles > 0)
	{
		if (VU->VIRegNumber == W_Is(VU)) src = VU->VIOldValue;
	}
	if (src > 0)
	{
		s32 bpc = vu0BranchAddr(VU);
		vu0SetBranch(VU, bpc);
	}
}

static void vu0_IBLEZ(VURegs* VU)
{
	vu_s16 src = VU->VI[W_Is(VU)].US[0];
	if (VU->VIBackupCycles > 0)
	{
		if (VU->VIRegNumber == W_Is(VU)) src = VU->VIOldValue;
	}
	if (src <= 0)
	{
		s32 bpc = vu0BranchAddr(VU);
		vu0SetBranch(VU, bpc);
	}
}

static void vu0_IBGEZ(VURegs* VU)
{
	vu_s16 src = VU->VI[W_Is(VU)].US[0];
	if (VU->VIBackupCycles > 0)
	{
		if (VU->VIRegNumber == W_Is(VU)) src = VU->VIOldValue;
	}
	if (src >= 0)
	{
		s32 bpc = vu0BranchAddr(VU);
		vu0SetBranch(VU, bpc);
	}
}

// --- Flag operation wrappers ---
static void vu0_FSAND(VURegs* VU)
{
	vu_u16 imm = (((VU->code >> 21) & 0x1) << 11) | (VU->code & 0x7ff);
	if (W_It(VU) == 0) return;
	VU->VI[W_It(VU)].US[0] = (VU->VI[REG_STATUS_FLAG].US[0] & 0xFFF) & imm;
}

static void vu0_FSEQ(VURegs* VU)
{
	vu_u16 imm = (((VU->code >> 21) & 0x1) << 11) | (VU->code & 0x7ff);
	if (W_It(VU) == 0) return;
	if ((VU->VI[REG_STATUS_FLAG].US[0] & 0xFFF) == imm)
		VU->VI[W_It(VU)].US[0] = 1;
	else
		VU->VI[W_It(VU)].US[0] = 0;
}

static void vu0_FSOR(VURegs* VU)
{
	vu_u16 imm = (((VU->code >> 21) & 0x1) << 11) | (VU->code & 0x7ff);
	if (W_It(VU) == 0) return;
	VU->VI[W_It(VU)].US[0] = (VU->VI[REG_STATUS_FLAG].US[0] & 0xFFF) | imm;
}

static void vu0_FSSET(VURegs* VU)
{
	vu_u16 imm = (((VU->code >> 21) & 0x1) << 11) | (VU->code & 0x7FF);
	VU->statusflag = (imm & 0xFC0) | (VU->statusflag & 0x3F);
}

static void vu0_FMAND(VURegs* VU)
{
	if (W_It(VU) == 0) return;
	VU->VI[W_It(VU)].US[0] = VU->VI[W_Is(VU)].US[0] & (VU->VI[REG_MAC_FLAG].UL & 0xFFFF);
}

static void vu0_FMEQ(VURegs* VU)
{
	if (W_It(VU) == 0) return;
	if ((VU->VI[REG_MAC_FLAG].UL & 0xFFFF) == VU->VI[W_Is(VU)].US[0])
		VU->VI[W_It(VU)].US[0] = 1;
	else
		VU->VI[W_It(VU)].US[0] = 0;
}

static void vu0_FMOR(VURegs* VU)
{
	if (W_It(VU) == 0) return;
	VU->VI[W_It(VU)].US[0] = (VU->VI[REG_MAC_FLAG].UL & 0xFFFF) | VU->VI[W_Is(VU)].US[0];
}

static void vu0_FCAND(VURegs* VU)
{
	if ((VU->VI[REG_CLIP_FLAG].UL & 0xFFFFFF) & (VU->code & 0xFFFFFF))
		VU->VI[1].US[0] = 1;
	else
		VU->VI[1].US[0] = 0;
}

static void vu0_FCEQ(VURegs* VU)
{
	if ((VU->VI[REG_CLIP_FLAG].UL & 0xFFFFFF) == (VU->code & 0xFFFFFF))
		VU->VI[1].US[0] = 1;
	else
		VU->VI[1].US[0] = 0;
}

static void vu0_FCOR(VURegs* VU)
{
	u32 hold = (VU->VI[REG_CLIP_FLAG].UL & 0xFFFFFF) | (VU->code & 0xFFFFFF);
	if (hold == 0xFFFFFF)
		VU->VI[1].US[0] = 1;
	else
		VU->VI[1].US[0] = 0;
}

static void vu0_FCSET(VURegs* VU)
{
	VU->clipflag = (u32)(VU->code & 0xFFFFFF);
}

static void vu0_FCGET(VURegs* VU)
{
	if (W_It(VU) == 0) return;
	VU->VI[W_It(VU)].US[0] = VU->VI[REG_CLIP_FLAG].UL & 0x0FFF;
}

// --- Random number generator wrappers ---
static void vu0_RINIT(VURegs* VU)
{
	VU->VI[REG_R].UL = 0x3F800000 | (VU->VF[W_Fs(VU)].UL[W_Fsf(VU)] & 0x007FFFFF);
}

static void vu0_RGET(VURegs* VU)
{
	if (W_Ft(VU) == 0) return;
	if (W_X(VU)) VU->VF[W_Ft(VU)].UL[0] = VU->VI[REG_R].UL;
	if (W_Y(VU)) VU->VF[W_Ft(VU)].UL[1] = VU->VI[REG_R].UL;
	if (W_Z(VU)) VU->VF[W_Ft(VU)].UL[2] = VU->VI[REG_R].UL;
	if (W_W(VU)) VU->VF[W_Ft(VU)].UL[3] = VU->VI[REG_R].UL;
}

static void vu0_RNEXT(VURegs* VU)
{
	if (W_Ft(VU) == 0) return;
	vu0AdvanceLFSR(VU);
	if (W_X(VU)) VU->VF[W_Ft(VU)].UL[0] = VU->VI[REG_R].UL;
	if (W_Y(VU)) VU->VF[W_Ft(VU)].UL[1] = VU->VI[REG_R].UL;
	if (W_Z(VU)) VU->VF[W_Ft(VU)].UL[2] = VU->VI[REG_R].UL;
	if (W_W(VU)) VU->VF[W_Ft(VU)].UL[3] = VU->VI[REG_R].UL;
}

static void vu0_RXOR(VURegs* VU)
{
	VU->VI[REG_R].UL = 0x3F800000 | ((VU->VI[REG_R].UL ^ VU->VF[W_Fs(VU)].UL[W_Fsf(VU)]) & 0x007FFFFF);
}

// EFU ops (ESADD/ERSADD/ELENG/ERLENG/EATANxy/EATANxz/ESUM/ERCPR/ESQRT/
// ERSQRT/ESIN/EATAN/EEXP) have no C-wrapper path on VU0 — native emitter
// below is a NOP matching x86 microVU. If you need to bisect back to
// computed values, set the ISTUB flag for that op (routes to interp via
// VU0_LOWER_OPCODE[]).

// ============================================================================
//  Per-instruction interp stub toggles (1 = interp, 0 = native)
// ============================================================================

// ---- FDIV group ----
#ifdef INTERP_VU0_LOWER_FDIV
#define ISTUB_VU0_DIV     1
#define ISTUB_VU0_SQRT    1
#define ISTUB_VU0_RSQRT   1
#define ISTUB_VU0_WAITQ   1
#define ISTUB_VU0_WAITP   1
#else
#define ISTUB_VU0_DIV     0
#define ISTUB_VU0_SQRT    0
#define ISTUB_VU0_RSQRT   0
#define ISTUB_VU0_WAITQ   0
#define ISTUB_VU0_WAITP   0
#endif

// ---- IALU group ----
#ifdef INTERP_VU0_LOWER_IALU
#define ISTUB_VU0_IADD    1
#define ISTUB_VU0_ISUB    1
#define ISTUB_VU0_IADDI   1
#define ISTUB_VU0_IADDIU  1
#define ISTUB_VU0_ISUBIU  1
#define ISTUB_VU0_IAND    1
#define ISTUB_VU0_IOR     1
#else
#define ISTUB_VU0_IADD    0
#define ISTUB_VU0_ISUB    0
#define ISTUB_VU0_IADDI   0
#define ISTUB_VU0_IADDIU  0
#define ISTUB_VU0_ISUBIU  0
#define ISTUB_VU0_IAND    0
#define ISTUB_VU0_IOR     0
#endif

// ---- LoadStore group ----
#ifdef INTERP_VU0_LOWER_LOADSTORE
#define ISTUB_VU0_LQ      1
#define ISTUB_VU0_LQD     1
#define ISTUB_VU0_LQI     1
#define ISTUB_VU0_SQ      1
#define ISTUB_VU0_SQD     1
#define ISTUB_VU0_SQI     1
#define ISTUB_VU0_ILW     1
#define ISTUB_VU0_ISW     1
#define ISTUB_VU0_ILWR    1
#define ISTUB_VU0_ISWR    1
#else
#define ISTUB_VU0_LQ      0
#define ISTUB_VU0_LQD     0
#define ISTUB_VU0_LQI     0
#define ISTUB_VU0_SQ      0
#define ISTUB_VU0_SQD     0
#define ISTUB_VU0_SQI     0
#define ISTUB_VU0_ILW     0
#define ISTUB_VU0_ISW     0
#define ISTUB_VU0_ILWR    0
#define ISTUB_VU0_ISWR    0
#endif

// ---- Branch group ----
#ifdef INTERP_VU0_LOWER_BRANCH
#define ISTUB_VU0_B       1
#define ISTUB_VU0_BAL     1
#define ISTUB_VU0_JR      1
#define ISTUB_VU0_JALR    1
#define ISTUB_VU0_IBEQ    1
#define ISTUB_VU0_IBNE    1
#define ISTUB_VU0_IBLTZ   1
#define ISTUB_VU0_IBGTZ   1
#define ISTUB_VU0_IBLEZ   1
#define ISTUB_VU0_IBGEZ   1
#else
#define ISTUB_VU0_B       0
#define ISTUB_VU0_BAL     0
#define ISTUB_VU0_JR      0
#define ISTUB_VU0_JALR    0
#define ISTUB_VU0_IBEQ    0
#define ISTUB_VU0_IBNE    0
#define ISTUB_VU0_IBLTZ   0
#define ISTUB_VU0_IBGTZ   0
#define ISTUB_VU0_IBLEZ   0
#define ISTUB_VU0_IBGEZ   0
#endif

// ---- Misc group (move, flag, random, EFU, special) ----
// XGKICK/XTOP/XITOP always use the interpreter on VU0 — VU0 has no GIF interface.
#ifdef INTERP_VU0_LOWER_MISC
#define ISTUB_VU0_MOVE    1
#define ISTUB_VU0_MR32    1
#define ISTUB_VU0_MFIR    1
#define ISTUB_VU0_MTIR    1
#define ISTUB_VU0_MFP     1
#define ISTUB_VU0_FSAND   1
#define ISTUB_VU0_FSEQ    1
#define ISTUB_VU0_FSOR    1
#define ISTUB_VU0_FSSET   1
#define ISTUB_VU0_FMAND   1
#define ISTUB_VU0_FMEQ    1
#define ISTUB_VU0_FMOR    1
#define ISTUB_VU0_FCAND   1
#define ISTUB_VU0_FCEQ    1
#define ISTUB_VU0_FCOR    1
#define ISTUB_VU0_FCSET   1
#define ISTUB_VU0_FCGET   1
#define ISTUB_VU0_RINIT   1
#define ISTUB_VU0_RGET    1
#define ISTUB_VU0_RNEXT   1
#define ISTUB_VU0_RXOR    1
#define ISTUB_VU0_ESADD   1
#define ISTUB_VU0_ERSADD  1
#define ISTUB_VU0_ELENG   1
#define ISTUB_VU0_ERLENG  1
#define ISTUB_VU0_EATANxy 1
#define ISTUB_VU0_EATANxz 1
#define ISTUB_VU0_ESUM    1
#define ISTUB_VU0_ERCPR   1
#define ISTUB_VU0_ESQRT_EFU 1
#define ISTUB_VU0_ERSQRT  1
#define ISTUB_VU0_ESIN    1
#define ISTUB_VU0_EATAN   1
#define ISTUB_VU0_EEXP    1
#else
#define ISTUB_VU0_MOVE    0
#define ISTUB_VU0_MR32    0
#define ISTUB_VU0_MFIR    0
#define ISTUB_VU0_MTIR    0
#define ISTUB_VU0_MFP     0
#define ISTUB_VU0_FSAND   0
#define ISTUB_VU0_FSEQ    0
#define ISTUB_VU0_FSOR    0
#define ISTUB_VU0_FSSET   0
#define ISTUB_VU0_FMAND   0
#define ISTUB_VU0_FMEQ    0
#define ISTUB_VU0_FMOR    0
#define ISTUB_VU0_FCAND   0
#define ISTUB_VU0_FCEQ    0
#define ISTUB_VU0_FCOR    0
#define ISTUB_VU0_FCSET   0
#define ISTUB_VU0_FCGET   0
#define ISTUB_VU0_RINIT   0
#define ISTUB_VU0_RGET    0
#define ISTUB_VU0_RNEXT   0
#define ISTUB_VU0_RXOR    0
#define ISTUB_VU0_ESADD   0
#define ISTUB_VU0_ERSADD  0
#define ISTUB_VU0_ELENG   0
#define ISTUB_VU0_ERLENG  0
#define ISTUB_VU0_EATANxy 0
#define ISTUB_VU0_EATANxz 0
#define ISTUB_VU0_ESUM    0
#define ISTUB_VU0_ERCPR   0
#define ISTUB_VU0_ESQRT_EFU 0
#define ISTUB_VU0_ERSQRT  0
#define ISTUB_VU0_ESIN    0
#define ISTUB_VU0_EATAN   0
#define ISTUB_VU0_EEXP    0
#endif

// ============================================================================
//  Interpreter stub: dispatch through VU0 lower opcode table.
//  VU0.code must be set before the rec function is called.
// ============================================================================

// Code-emitter macro: called at block-compile time to emit an ARM64 BL to
// the specific interpreter function for this lower opcode.
#define REC_VU0_LOWER_INTERP(name) \
	void recVU0_##name() { \
		armEmitCall(reinterpret_cast<const void*>(VU0_LOWER_OPCODE[VU0.code >> 25])); \
	}

// Non-INTERP path (ISTUB=0, no native codegen yet): same emitter as ISTUB=1.
#define REC_VU0_LOWER_EMIT(name) \
	void recVU0_##name() { \
		armEmitCall(reinterpret_cast<const void*>(VU0_LOWER_OPCODE[VU0.code >> 25])); \
	}

// ============================================================================
//  FDIV — Division pipeline
// ============================================================================

#if ISTUB_VU0_DIV
REC_VU0_LOWER_INTERP(DIV)
#else
void recVU0_DIV() {
	const u32 ft = (VU0.code >> 16) & 0x1F;
	const u32 ftf = (VU0.code >> 23) & 0x3;
	const u32 fs = (VU0.code >> 11) & 0x1F;
	const u32 fsf = (VU0.code >> 21) & 0x3;
	const int64_t sf_off = static_cast<int64_t>(offsetof(VURegs, statusflag));
	const int64_t q_off  = static_cast<int64_t>(offsetof(VURegs, q));

	// Load raw u32 float bits
	armAsm->Ldr(w0, MemOperand(VU0_BASE_REG, vfOff(ft) + ftf * 4));
	armAsm->Ldr(w1, MemOperand(VU0_BASE_REG, vfOff(fs) + fsf * 4));
	// Save raw copies for sign check
	armAsm->Mov(w4, w0);
	armAsm->Mov(w5, w1);
	// vuDouble clamp both operands
	emitVuDouble(w0, w2);
	emitVuDouble(w1, w2);
	// Clear statusflag bits [5:4]
	armAsm->Ldr(w2, MemOperand(VU0_BASE_REG, sf_off));
	armAsm->Bic(w2, w2, 0x30);
	armAsm->Str(w2, MemOperand(VU0_BASE_REG, sf_off));
	// Convert to float
	armAsm->Fmov(s0, w0); // ft
	armAsm->Fmov(s1, w1); // fs

	a64::Label ftNotZero, done;
	armAsm->Fcmp(s0, 0.0);
	armAsm->B(&ftNotZero, a64::ne);

	// --- ft == 0 path ---
	a64::Label fsNotZero, signCheck;
	armAsm->Fcmp(s1, 0.0);
	armAsm->B(&fsNotZero, a64::ne);
	// fs==0: statusflag |= 0x10
	armAsm->Ldr(w2, MemOperand(VU0_BASE_REG, sf_off));
	armAsm->Orr(w2, w2, 0x10);
	armAsm->Str(w2, MemOperand(VU0_BASE_REG, sf_off));
	armAsm->B(&signCheck);
	armAsm->Bind(&fsNotZero);
	// fs!=0: statusflag |= 0x20
	armAsm->Ldr(w2, MemOperand(VU0_BASE_REG, sf_off));
	armAsm->Orr(w2, w2, 0x20);
	armAsm->Str(w2, MemOperand(VU0_BASE_REG, sf_off));
	armAsm->Bind(&signCheck);
	// Sign check: (raw_ft ^ raw_fs) bit 31
	armAsm->Eor(w3, w4, w5);
	a64::Label diffSign;
	armAsm->Tbnz(w3, 31, &diffSign);
	// Same sign: q = 0x7F7FFFFF
	armAsm->Mov(w0, 0x7F7FFFFFu);
	armAsm->Str(w0, MemOperand(VU0_BASE_REG, q_off));
	armAsm->B(&done);
	armAsm->Bind(&diffSign);
	// Different sign: q = 0xFF7FFFFF
	armAsm->Mov(w0, 0xFF7FFFFFu);
	armAsm->Str(w0, MemOperand(VU0_BASE_REG, q_off));
	armAsm->B(&done);

	// --- ft != 0 path ---
	armAsm->Bind(&ftNotZero);
	armAsm->Fdiv(s2, s1, s0); // fs / ft
	armAsm->Fmov(w0, s2);
	emitVuDouble(w0, w2);      // clamp result
	armAsm->Str(w0, MemOperand(VU0_BASE_REG, q_off));

	armAsm->Bind(&done);
}
#endif

#if ISTUB_VU0_SQRT
REC_VU0_LOWER_INTERP(SQRT)
#else
void recVU0_SQRT() {
	const u32 ft = (VU0.code >> 16) & 0x1F;
	const u32 ftf = (VU0.code >> 23) & 0x3;
	const int64_t sf_off = static_cast<int64_t>(offsetof(VURegs, statusflag));
	const int64_t q_off  = static_cast<int64_t>(offsetof(VURegs, q));

	// Load and clamp ft
	armAsm->Ldr(w0, MemOperand(VU0_BASE_REG, vfOff(ft) + ftf * 4));
	emitVuDouble(w0, w2);
	// Clear statusflag bits [5:4]
	armAsm->Ldr(w2, MemOperand(VU0_BASE_REG, sf_off));
	armAsm->Bic(w2, w2, 0x30);
	armAsm->Str(w2, MemOperand(VU0_BASE_REG, sf_off));
	// Convert to float
	armAsm->Fmov(s0, w0);
	// If negative: statusflag |= 0x10, use fabs
	a64::Label notNeg, store;
	armAsm->Fcmp(s0, 0.0);
	armAsm->B(&notNeg, a64::ge);
	armAsm->Ldr(w2, MemOperand(VU0_BASE_REG, sf_off));
	armAsm->Orr(w2, w2, 0x10);
	armAsm->Str(w2, MemOperand(VU0_BASE_REG, sf_off));
	armAsm->Bind(&notNeg);
	// sqrt(fabs(ft))
	armAsm->Fabs(s0, s0);
	armAsm->Fsqrt(s0, s0);
	// Clamp result
	armAsm->Fmov(w0, s0);
	emitVuDouble(w0, w2);
	armAsm->Str(w0, MemOperand(VU0_BASE_REG, q_off));
}
#endif

#if ISTUB_VU0_RSQRT
REC_VU0_LOWER_INTERP(RSQRT)
#else
void recVU0_RSQRT() {
	const u32 ft = (VU0.code >> 16) & 0x1F;
	const u32 ftf = (VU0.code >> 23) & 0x3;
	const u32 fs = (VU0.code >> 11) & 0x1F;
	const u32 fsf = (VU0.code >> 21) & 0x3;
	const int64_t sf_off = static_cast<int64_t>(offsetof(VURegs, statusflag));
	const int64_t q_off  = static_cast<int64_t>(offsetof(VURegs, q));

	// Load raw bits
	armAsm->Ldr(w0, MemOperand(VU0_BASE_REG, vfOff(ft) + ftf * 4));
	armAsm->Ldr(w1, MemOperand(VU0_BASE_REG, vfOff(fs) + fsf * 4));
	armAsm->Mov(w4, w0); // save raw ft
	armAsm->Mov(w5, w1); // save raw fs
	emitVuDouble(w0, w2);
	emitVuDouble(w1, w2);
	// Clear statusflag [5:4]
	armAsm->Ldr(w2, MemOperand(VU0_BASE_REG, sf_off));
	armAsm->Bic(w2, w2, 0x30);
	armAsm->Str(w2, MemOperand(VU0_BASE_REG, sf_off));
	armAsm->Fmov(s0, w0); // ft
	armAsm->Fmov(s1, w1); // fs

	a64::Label ftNotZero, done;
	armAsm->Fcmp(s0, 0.0);
	armAsm->B(&ftNotZero, a64::ne);

	// --- ft == 0 ---
	// statusflag |= 0x20
	armAsm->Ldr(w2, MemOperand(VU0_BASE_REG, sf_off));
	armAsm->Orr(w2, w2, 0x20);
	armAsm->Str(w2, MemOperand(VU0_BASE_REG, sf_off));

	a64::Label fsIsZero;
	armAsm->Fcmp(s1, 0.0);
	armAsm->B(&fsIsZero, a64::eq);

	// fs != 0, ft == 0: q = ±max based on sign XOR
	a64::Label diffSign1;
	armAsm->Eor(w3, w4, w5);
	armAsm->Tbnz(w3, 31, &diffSign1);
	armAsm->Mov(w0, 0x7F7FFFFFu);
	armAsm->Str(w0, MemOperand(VU0_BASE_REG, q_off));
	armAsm->B(&done);
	armAsm->Bind(&diffSign1);
	armAsm->Mov(w0, 0xFF7FFFFFu);
	armAsm->Str(w0, MemOperand(VU0_BASE_REG, q_off));
	armAsm->B(&done);

	armAsm->Bind(&fsIsZero);
	// fs == 0, ft == 0: q = ±0, statusflag |= 0x10
	armAsm->Ldr(w2, MemOperand(VU0_BASE_REG, sf_off));
	armAsm->Orr(w2, w2, 0x10);
	armAsm->Str(w2, MemOperand(VU0_BASE_REG, sf_off));
	a64::Label diffSign2;
	armAsm->Eor(w3, w4, w5);
	armAsm->Tbnz(w3, 31, &diffSign2);
	armAsm->Str(wzr, MemOperand(VU0_BASE_REG, q_off)); // +0
	armAsm->B(&done);
	armAsm->Bind(&diffSign2);
	armAsm->Mov(w0, 0x80000000u);
	armAsm->Str(w0, MemOperand(VU0_BASE_REG, q_off)); // -0
	armAsm->B(&done);

	// --- ft != 0 ---
	armAsm->Bind(&ftNotZero);
	// If ft < 0: statusflag |= 0x10
	a64::Label ftPos;
	armAsm->Fcmp(s0, 0.0);
	armAsm->B(&ftPos, a64::ge);
	armAsm->Ldr(w2, MemOperand(VU0_BASE_REG, sf_off));
	armAsm->Orr(w2, w2, 0x10);
	armAsm->Str(w2, MemOperand(VU0_BASE_REG, sf_off));
	armAsm->Bind(&ftPos);
	// temp = sqrt(fabs(ft)); q = fs / temp
	armAsm->Fabs(s0, s0);
	armAsm->Fsqrt(s0, s0);
	armAsm->Fdiv(s2, s1, s0); // fs / sqrt(|ft|)
	armAsm->Fmov(w0, s2);
	emitVuDouble(w0, w2);
	armAsm->Str(w0, MemOperand(VU0_BASE_REG, q_off));

	armAsm->Bind(&done);
}
#endif

#if ISTUB_VU0_WAITQ
REC_VU0_LOWER_INTERP(WAITQ)
#else
void recVU0_WAITQ() { /* NOP — no instructions to emit */ }
#endif

#if ISTUB_VU0_WAITP
REC_VU0_LOWER_INTERP(WAITP)
#else
void recVU0_WAITP() { /* NOP — no instructions to emit */ }
#endif

// ============================================================================
//  Integer ALU
// ============================================================================

#if ISTUB_VU0_IADD
REC_VU0_LOWER_INTERP(IADD)
#else
void recVU0_IADD() {
	const u32 id = (VU0.code >> 6) & 0xF;
	if (id == 0) return;
	const u32 is = (VU0.code >> 11) & 0xF;
	const u32 it = (VU0.code >> 16) & 0xF;
	emitBackupVI(id);
	armAsm->Ldrh(w0, MemOperand(VU0_BASE_REG, viOff(is)));
	armAsm->Ldrh(w1, MemOperand(VU0_BASE_REG, viOff(it)));
	armAsm->Add(w0, w0, w1);
	armAsm->Strh(w0, MemOperand(VU0_BASE_REG, viOff(id)));
}
#endif

#if ISTUB_VU0_ISUB
REC_VU0_LOWER_INTERP(ISUB)
#else
void recVU0_ISUB() {
	const u32 id = (VU0.code >> 6) & 0xF;
	if (id == 0) return;
	const u32 is = (VU0.code >> 11) & 0xF;
	const u32 it = (VU0.code >> 16) & 0xF;
	emitBackupVI(id);
	armAsm->Ldrh(w0, MemOperand(VU0_BASE_REG, viOff(is)));
	armAsm->Ldrh(w1, MemOperand(VU0_BASE_REG, viOff(it)));
	armAsm->Sub(w0, w0, w1);
	armAsm->Strh(w0, MemOperand(VU0_BASE_REG, viOff(id)));
}
#endif

#if ISTUB_VU0_IADDI
REC_VU0_LOWER_INTERP(IADDI)
#else
void recVU0_IADDI() {
	const u32 it = (VU0.code >> 16) & 0xF;
	if (it == 0) return;
	const u32 is = (VU0.code >> 11) & 0xF;
	emitBackupVI(it);
	armAsm->Ldrh(w0, MemOperand(VU0_BASE_REG, viOff(is)));
	if (EmuConfig.Gamefixes.IbitHack)
	{
		// Live 5-bit signed immediate from bits [10:6] of VU->code. Matches
		// x86 microVU_Lower.inl IADDI (xSHL 21; xSAR 27 = sign-extract [10:6]).
		armAsm->Ldr(w1, MemOperand(VU0_BASE_REG, static_cast<int64_t>(offsetof(VURegs, code))));
		armAsm->Sbfx(w1, w1, 6, 5);
		armAsm->Add(w0, w0, w1);
	}
	else
	{
		// 5-bit signed immediate at bits [10:6], sign-extended at JIT time.
		s32 imm = (VU0.code >> 6) & 0x1f;
		imm = ((imm & 0x10) ? static_cast<s32>(0xfffffff0) : 0) | (imm & 0xf);
		if (imm > 0)
			armAsm->Add(w0, w0, imm);
		else if (imm < 0)
			armAsm->Sub(w0, w0, static_cast<u32>(-imm));
	}
	armAsm->Strh(w0, MemOperand(VU0_BASE_REG, viOff(it)));
}
#endif

#if ISTUB_VU0_IADDIU
REC_VU0_LOWER_INTERP(IADDIU)
#else
void recVU0_IADDIU() {
	const u32 it = (VU0.code >> 16) & 0xF;
	if (it == 0) return;
	const u32 is = (VU0.code >> 11) & 0xF;
	emitBackupVI(it);
	armAsm->Ldrh(w0, MemOperand(VU0_BASE_REG, viOff(is)));
	if (EmuConfig.Gamefixes.IbitHack)
	{
		// Live 15-bit unsigned immediate from VU->code: ((code >> 10) & 0x7800) | (code & 0x7FF).
		// Mirrors x86 microVU_Lower.inl IADDIU IbitHack path.
		armAsm->Ldr(w1, MemOperand(VU0_BASE_REG, static_cast<int64_t>(offsetof(VURegs, code))));
		armAsm->Lsr(w2, w1, 10);
		armAsm->And(w2, w2, 0x7800);
		armAsm->And(w1, w1, 0x7FF);
		armAsm->Orr(w1, w1, w2);
		armAsm->Add(w0, w0, w1);
	}
	else
	{
		// 15-bit unsigned immediate: bits [24:21] → [14:11], bits [10:0] → [10:0].
		const u32 imm = ((VU0.code >> 10) & 0x7800) | (VU0.code & 0x7ff);
		if (imm != 0)
		{
			armAsm->Mov(w1, imm);
			armAsm->Add(w0, w0, w1);
		}
	}
	armAsm->Strh(w0, MemOperand(VU0_BASE_REG, viOff(it)));
}
#endif

#if ISTUB_VU0_ISUBIU
REC_VU0_LOWER_INTERP(ISUBIU)
#else
void recVU0_ISUBIU() {
	const u32 it = (VU0.code >> 16) & 0xF;
	if (it == 0) return;
	const u32 is = (VU0.code >> 11) & 0xF;
	emitBackupVI(it);
	armAsm->Ldrh(w0, MemOperand(VU0_BASE_REG, viOff(is)));
	if (EmuConfig.Gamefixes.IbitHack)
	{
		armAsm->Ldr(w1, MemOperand(VU0_BASE_REG, static_cast<int64_t>(offsetof(VURegs, code))));
		armAsm->Lsr(w2, w1, 10);
		armAsm->And(w2, w2, 0x7800);
		armAsm->And(w1, w1, 0x7FF);
		armAsm->Orr(w1, w1, w2);
		armAsm->Sub(w0, w0, w1);
	}
	else
	{
		const u32 imm = ((VU0.code >> 10) & 0x7800) | (VU0.code & 0x7ff);
		if (imm != 0)
		{
			armAsm->Mov(w1, imm);
			armAsm->Sub(w0, w0, w1);
		}
	}
	armAsm->Strh(w0, MemOperand(VU0_BASE_REG, viOff(it)));
}
#endif

#if ISTUB_VU0_IAND
REC_VU0_LOWER_INTERP(IAND)
#else
void recVU0_IAND() {
	const u32 id = (VU0.code >> 6) & 0xF;
	if (id == 0) return;
	const u32 is = (VU0.code >> 11) & 0xF;
	const u32 it = (VU0.code >> 16) & 0xF;
	emitBackupVI(id);
	armAsm->Ldrh(w0, MemOperand(VU0_BASE_REG, viOff(is)));
	armAsm->Ldrh(w1, MemOperand(VU0_BASE_REG, viOff(it)));
	armAsm->And(w0, w0, w1);
	armAsm->Strh(w0, MemOperand(VU0_BASE_REG, viOff(id)));
}
#endif

#if ISTUB_VU0_IOR
REC_VU0_LOWER_INTERP(IOR)
#else
void recVU0_IOR() {
	const u32 id = (VU0.code >> 6) & 0xF;
	if (id == 0) return;
	const u32 is = (VU0.code >> 11) & 0xF;
	const u32 it = (VU0.code >> 16) & 0xF;
	emitBackupVI(id);
	armAsm->Ldrh(w0, MemOperand(VU0_BASE_REG, viOff(is)));
	armAsm->Ldrh(w1, MemOperand(VU0_BASE_REG, viOff(it)));
	armAsm->Orr(w0, w0, w1);
	armAsm->Strh(w0, MemOperand(VU0_BASE_REG, viOff(id)));
}
#endif

// ============================================================================
//  Load / Store (VU data memory)
// ============================================================================

#if ISTUB_VU0_LQ
REC_VU0_LOWER_INTERP(LQ)
#else
REC_VU0_LOWER_CALL(LQ)
#endif

#if ISTUB_VU0_LQD
REC_VU0_LOWER_INTERP(LQD)
#else
REC_VU0_LOWER_CALL(LQD)
#endif

#if ISTUB_VU0_LQI
REC_VU0_LOWER_INTERP(LQI)
#else
REC_VU0_LOWER_CALL(LQI)
#endif

#if ISTUB_VU0_SQ
REC_VU0_LOWER_INTERP(SQ)
#else
REC_VU0_LOWER_CALL(SQ)
#endif

#if ISTUB_VU0_SQD
REC_VU0_LOWER_INTERP(SQD)
#else
REC_VU0_LOWER_CALL(SQD)
#endif

#if ISTUB_VU0_SQI
REC_VU0_LOWER_INTERP(SQI)
#else
REC_VU0_LOWER_CALL(SQI)
#endif

#if ISTUB_VU0_ILW
REC_VU0_LOWER_INTERP(ILW)
#else
REC_VU0_LOWER_CALL(ILW)
#endif

#if ISTUB_VU0_ISW
REC_VU0_LOWER_INTERP(ISW)
#else
REC_VU0_LOWER_CALL(ISW)
#endif

#if ISTUB_VU0_ILWR
REC_VU0_LOWER_INTERP(ILWR)
#else
REC_VU0_LOWER_CALL(ILWR)
#endif

#if ISTUB_VU0_ISWR
REC_VU0_LOWER_INTERP(ISWR)
#else
REC_VU0_LOWER_CALL(ISWR)
#endif

// ============================================================================
//  Branches
// ============================================================================

#if ISTUB_VU0_B
REC_VU0_LOWER_INTERP(B)
#else
REC_VU0_LOWER_CALL(B)
#endif

#if ISTUB_VU0_BAL
REC_VU0_LOWER_INTERP(BAL)
#else
REC_VU0_LOWER_CALL(BAL)
#endif

#if ISTUB_VU0_JR
REC_VU0_LOWER_INTERP(JR)
#else
REC_VU0_LOWER_CALL(JR)
#endif

#if ISTUB_VU0_JALR
REC_VU0_LOWER_INTERP(JALR)
#else
REC_VU0_LOWER_CALL(JALR)
#endif

#if ISTUB_VU0_IBEQ
REC_VU0_LOWER_INTERP(IBEQ)
#else
REC_VU0_LOWER_CALL(IBEQ)
#endif

#if ISTUB_VU0_IBNE
REC_VU0_LOWER_INTERP(IBNE)
#else
REC_VU0_LOWER_CALL(IBNE)
#endif

#if ISTUB_VU0_IBLTZ
REC_VU0_LOWER_INTERP(IBLTZ)
#else
REC_VU0_LOWER_CALL(IBLTZ)
#endif

#if ISTUB_VU0_IBGTZ
REC_VU0_LOWER_INTERP(IBGTZ)
#else
REC_VU0_LOWER_CALL(IBGTZ)
#endif

#if ISTUB_VU0_IBLEZ
REC_VU0_LOWER_INTERP(IBLEZ)
#else
REC_VU0_LOWER_CALL(IBLEZ)
#endif

#if ISTUB_VU0_IBGEZ
REC_VU0_LOWER_INTERP(IBGEZ)
#else
REC_VU0_LOWER_CALL(IBGEZ)
#endif

// ============================================================================
//  Move / Transfer
// ============================================================================

#if ISTUB_VU0_MOVE
REC_VU0_LOWER_INTERP(MOVE)
#else
void recVU0_MOVE() {
	const u32 ft = (VU0.code >> 16) & 0x1F;
	if (ft == 0) return;
	const u32 fs = (VU0.code >> 11) & 0x1F;
	const u32 xyzw = (VU0.code >> 21) & 0xF;
	if (xyzw & 8) { armAsm->Ldr(w0, MemOperand(VU0_BASE_REG, vfOff(fs) + 0));  armAsm->Str(w0, MemOperand(VU0_BASE_REG, vfOff(ft) + 0));  }
	if (xyzw & 4) { armAsm->Ldr(w0, MemOperand(VU0_BASE_REG, vfOff(fs) + 4));  armAsm->Str(w0, MemOperand(VU0_BASE_REG, vfOff(ft) + 4));  }
	if (xyzw & 2) { armAsm->Ldr(w0, MemOperand(VU0_BASE_REG, vfOff(fs) + 8));  armAsm->Str(w0, MemOperand(VU0_BASE_REG, vfOff(ft) + 8));  }
	if (xyzw & 1) { armAsm->Ldr(w0, MemOperand(VU0_BASE_REG, vfOff(fs) + 12)); armAsm->Str(w0, MemOperand(VU0_BASE_REG, vfOff(ft) + 12)); }
}
#endif

#if ISTUB_VU0_MR32
REC_VU0_LOWER_INTERP(MR32)
#else
void recVU0_MR32() {
	const u32 ft = (VU0.code >> 16) & 0x1F;
	if (ft == 0) return;
	const u32 fs = (VU0.code >> 11) & 0x1F;
	const u32 xyzw = (VU0.code >> 21) & 0xF;
	// Save VF[fs].x — needed for W component (W receives the original X).
	armAsm->Ldr(w1, MemOperand(VU0_BASE_REG, vfOff(fs) + 0));
	if (xyzw & 8) { // X = fs.y
		armAsm->Ldr(w0, MemOperand(VU0_BASE_REG, vfOff(fs) + 4));
		armAsm->Str(w0, MemOperand(VU0_BASE_REG, vfOff(ft) + 0));
	}
	if (xyzw & 4) { // Y = fs.z
		armAsm->Ldr(w0, MemOperand(VU0_BASE_REG, vfOff(fs) + 8));
		armAsm->Str(w0, MemOperand(VU0_BASE_REG, vfOff(ft) + 4));
	}
	if (xyzw & 2) { // Z = fs.w
		armAsm->Ldr(w0, MemOperand(VU0_BASE_REG, vfOff(fs) + 12));
		armAsm->Str(w0, MemOperand(VU0_BASE_REG, vfOff(ft) + 8));
	}
	if (xyzw & 1) { // W = saved fs.x
		armAsm->Str(w1, MemOperand(VU0_BASE_REG, vfOff(ft) + 12));
	}
}
#endif

#if ISTUB_VU0_MFIR
REC_VU0_LOWER_INTERP(MFIR)
#else
void recVU0_MFIR() {
	const u32 ft = (VU0.code >> 16) & 0x1F;
	if (ft == 0) return;
	const u32 is = (VU0.code >> 11) & 0xF;
	const u32 xyzw = (VU0.code >> 21) & 0xF;
	// Sign-extend VI[is].SS[0] to 32 bits, broadcast to selected VF components.
	armAsm->Ldrsh(w0, MemOperand(VU0_BASE_REG, viOff(is)));
	if (xyzw & 8) armAsm->Str(w0, MemOperand(VU0_BASE_REG, vfOff(ft) + 0));
	if (xyzw & 4) armAsm->Str(w0, MemOperand(VU0_BASE_REG, vfOff(ft) + 4));
	if (xyzw & 2) armAsm->Str(w0, MemOperand(VU0_BASE_REG, vfOff(ft) + 8));
	if (xyzw & 1) armAsm->Str(w0, MemOperand(VU0_BASE_REG, vfOff(ft) + 12));
}
#endif

#if ISTUB_VU0_MTIR
REC_VU0_LOWER_INTERP(MTIR)
#else
void recVU0_MTIR() {
	const u32 it = (VU0.code >> 16) & 0xF;
	if (it == 0) return;
	const u32 fs = (VU0.code >> 11) & 0x1F;
	const u32 fsf = (VU0.code >> 21) & 0x3; // 0=x, 1=y, 2=z, 3=w
	emitBackupVI(it);
	// Load lower 16 bits of VF[fs].F[fsf] into VI[it].
	armAsm->Ldrh(w0, MemOperand(VU0_BASE_REG, vfOff(fs) + fsf * 4));
	armAsm->Strh(w0, MemOperand(VU0_BASE_REG, viOff(it)));
}
#endif

#if ISTUB_VU0_MFP
REC_VU0_LOWER_INTERP(MFP)
#else
void recVU0_MFP() {
	// P register only exists on VU1 (it's the EFU result register).
	// MFP on VU0 is a no-op — matches interpreter _vuMFP in VUops.cpp.
}
#endif

// ============================================================================
//  Flag read/write
// ============================================================================

#if ISTUB_VU0_FSAND
REC_VU0_LOWER_INTERP(FSAND)
#else
REC_VU0_LOWER_CALL(FSAND)
#endif

#if ISTUB_VU0_FSEQ
REC_VU0_LOWER_INTERP(FSEQ)
#else
REC_VU0_LOWER_CALL(FSEQ)
#endif

#if ISTUB_VU0_FSOR
REC_VU0_LOWER_INTERP(FSOR)
#else
REC_VU0_LOWER_CALL(FSOR)
#endif

#if ISTUB_VU0_FSSET
REC_VU0_LOWER_INTERP(FSSET)
#else
REC_VU0_LOWER_CALL(FSSET)
#endif

#if ISTUB_VU0_FMAND
REC_VU0_LOWER_INTERP(FMAND)
#else
REC_VU0_LOWER_CALL(FMAND)
#endif

#if ISTUB_VU0_FMEQ
REC_VU0_LOWER_INTERP(FMEQ)
#else
REC_VU0_LOWER_CALL(FMEQ)
#endif

#if ISTUB_VU0_FMOR
REC_VU0_LOWER_INTERP(FMOR)
#else
REC_VU0_LOWER_CALL(FMOR)
#endif

#if ISTUB_VU0_FCAND
REC_VU0_LOWER_INTERP(FCAND)
#else
REC_VU0_LOWER_CALL(FCAND)
#endif

#if ISTUB_VU0_FCEQ
REC_VU0_LOWER_INTERP(FCEQ)
#else
REC_VU0_LOWER_CALL(FCEQ)
#endif

#if ISTUB_VU0_FCOR
REC_VU0_LOWER_INTERP(FCOR)
#else
REC_VU0_LOWER_CALL(FCOR)
#endif

#if ISTUB_VU0_FCSET
REC_VU0_LOWER_INTERP(FCSET)
#else
REC_VU0_LOWER_CALL(FCSET)
#endif

#if ISTUB_VU0_FCGET
REC_VU0_LOWER_INTERP(FCGET)
#else
REC_VU0_LOWER_CALL(FCGET)
#endif

// ============================================================================
//  Random number generator
// ============================================================================

#if ISTUB_VU0_RINIT
REC_VU0_LOWER_INTERP(RINIT)
#else
REC_VU0_LOWER_CALL(RINIT)
#endif

#if ISTUB_VU0_RGET
REC_VU0_LOWER_INTERP(RGET)
#else
REC_VU0_LOWER_CALL(RGET)
#endif

#if ISTUB_VU0_RNEXT
REC_VU0_LOWER_INTERP(RNEXT)
#else
REC_VU0_LOWER_CALL(RNEXT)
#endif

#if ISTUB_VU0_RXOR
REC_VU0_LOWER_INTERP(RXOR)
#else
REC_VU0_LOWER_CALL(RXOR)
#endif

// ============================================================================
//  EFU — Elementary Function Unit
//
//  VU0 has no EFU hardware. Native path emits no instructions (matches x86
//  microVU's isNOP=true on isVU0). ISTUB=1 routes to the interpreter, which
//  will compute the polynomial and update VU0.p — divergent from x86, kept
//  for bisection only.
// ============================================================================

#if ISTUB_VU0_ESADD
REC_VU0_LOWER_INTERP(ESADD)
#else
void recVU0_ESADD() { /* NOP — no EFU on VU0 */ }
#endif

#if ISTUB_VU0_ERSADD
REC_VU0_LOWER_INTERP(ERSADD)
#else
void recVU0_ERSADD() { /* NOP — no EFU on VU0 */ }
#endif

#if ISTUB_VU0_ELENG
REC_VU0_LOWER_INTERP(ELENG)
#else
void recVU0_ELENG() { /* NOP — no EFU on VU0 */ }
#endif

#if ISTUB_VU0_ERLENG
REC_VU0_LOWER_INTERP(ERLENG)
#else
void recVU0_ERLENG() { /* NOP — no EFU on VU0 */ }
#endif

#if ISTUB_VU0_EATANxy
REC_VU0_LOWER_INTERP(EATANxy)
#else
void recVU0_EATANxy() { /* NOP — no EFU on VU0 */ }
#endif

#if ISTUB_VU0_EATANxz
REC_VU0_LOWER_INTERP(EATANxz)
#else
void recVU0_EATANxz() { /* NOP — no EFU on VU0 */ }
#endif

#if ISTUB_VU0_ESUM
REC_VU0_LOWER_INTERP(ESUM)
#else
void recVU0_ESUM() { /* NOP — no EFU on VU0 */ }
#endif

#if ISTUB_VU0_ERCPR
REC_VU0_LOWER_INTERP(ERCPR)
#else
void recVU0_ERCPR() { /* NOP — no EFU on VU0 */ }
#endif

#if ISTUB_VU0_ESQRT_EFU
REC_VU0_LOWER_INTERP(ESQRT)
#else
void recVU0_ESQRT() { /* NOP — no EFU on VU0 */ }
#endif

#if ISTUB_VU0_ERSQRT
REC_VU0_LOWER_INTERP(ERSQRT)
#else
void recVU0_ERSQRT() { /* NOP — no EFU on VU0 */ }
#endif

#if ISTUB_VU0_ESIN
REC_VU0_LOWER_INTERP(ESIN)
#else
void recVU0_ESIN() { /* NOP — no EFU on VU0 */ }
#endif

#if ISTUB_VU0_EATAN
REC_VU0_LOWER_INTERP(EATAN)
#else
void recVU0_EATAN() { /* NOP — no EFU on VU0 */ }
#endif

#if ISTUB_VU0_EEXP
REC_VU0_LOWER_INTERP(EEXP)
#else
void recVU0_EEXP() { /* NOP — no EFU on VU0 */ }
#endif

// ============================================================================
//  Special — VU/GIF interface.
//
//  XGKICK and XTOP are NOPs on VU0 — matches x86 microVU's isNOP=true on
//  isVU0 (microVU_Lower.inl:1653, 1826) and the VU0 interp stubs which are
//  empty bodies (VU0MI_XGKICK/VU0MI_XTOP in VUops.cpp).
//
//  XITOP does real work on VU0 — reads VifRegs.itop masked by 0xff (VU1
//  masks by 0x3ff). x86 microVU emits inline code; arm64 routes to interp.
// ============================================================================

void recVU0_XGKICK() { /* NOP on VU0 — no GIF interface */ }
void recVU0_XTOP()   { /* NOP on VU0 — VIF top register not exposed */ }
void recVU0_XITOP()  { armEmitCall(reinterpret_cast<const void*>(VU0_LOWER_OPCODE[VU0.code >> 25])); }

// ============================================================================
//  Generic fallback emitter for unknown / reserved lower opcode slots.
//  VU0.code is already set; the interpreter will handle the unknown case
//  (logging / NOP).
//
//  Exported (non-static) so iVU0micro_arm64.cpp can pointer-compare dispatch
//  table entries for bad-op detection (C-7 from the VU0 audit).
// ============================================================================
void recVU0_Lower_Unknown()
{
	armEmitCall(reinterpret_cast<const void*>(VU0_LOWER_OPCODE[VU0.code >> 25]));
}

// ============================================================================
//  LowerOP sub-table dispatch (compile-time, mirrors interpreter tables)
//
//  When recVU0_LowerTable[0x40] is called, VU0.code is set to the lower
//  instruction word.  The sub-table dispatch reads bits from VU0.code at
//  compile time to route to the correct rec function.  Non-native entries
//  fall back to recVU0_Lower_Unknown (interpreter BL chain).
// ============================================================================
using VU0RecFn = void (*)();

// T3_00 sub-table: indexed by (VU0.code >> 6) & 0x1f
static VU0RecFn recVU0_LowerOP_T3_00_Table[32] = {
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x00
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x04
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x08
	recVU0_MOVE,          recVU0_LQI,           recVU0_DIV,           recVU0_MTIR,           // 0x0C
	recVU0_RNEXT,         recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x10
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x14
	recVU0_Lower_Unknown, recVU0_MFP,           recVU0_XTOP,          recVU0_XGKICK,         // 0x18
	recVU0_ESADD,         recVU0_EATANxy,       recVU0_ESQRT,         recVU0_ESIN,           // 0x1C
};

// T3_01 sub-table: indexed by (VU0.code >> 6) & 0x1f
static VU0RecFn recVU0_LowerOP_T3_01_Table[32] = {
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x00
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x04
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x08
	recVU0_MR32,          recVU0_SQI,           recVU0_SQRT,          recVU0_MFIR,           // 0x0C
	recVU0_RGET,          recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x10
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x14
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_XITOP,         recVU0_Lower_Unknown, // 0x18
	recVU0_ERSADD,        recVU0_EATANxz,       recVU0_ERSQRT,        recVU0_EATAN,          // 0x1C
};

// T3_10 sub-table: indexed by (VU0.code >> 6) & 0x1f
static VU0RecFn recVU0_LowerOP_T3_10_Table[32] = {
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x00
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x04
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x08
	recVU0_Lower_Unknown, recVU0_LQD,           recVU0_RSQRT,         recVU0_ILWR,           // 0x0C
	recVU0_RINIT,         recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x10
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x14
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x18
	recVU0_ELENG,         recVU0_ESUM,          recVU0_ERCPR,         recVU0_EEXP,           // 0x1C
};

// T3_11 sub-table: indexed by (VU0.code >> 6) & 0x1f
static VU0RecFn recVU0_LowerOP_T3_11_Table[32] = {
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x00
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x04
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x08
	recVU0_Lower_Unknown, recVU0_SQD,           recVU0_WAITQ,         recVU0_ISWR,           // 0x0C
	recVU0_RXOR,          recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x10
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x14
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x18
	recVU0_ERLENG,        recVU0_Lower_Unknown, recVU0_WAITP,         recVU0_Lower_Unknown, // 0x1C
};

// T3 dispatch functions (compile-time)
static void recVU0_LowerOP_T3_00() { recVU0_LowerOP_T3_00_Table[(VU0.code >> 6) & 0x1f](); }
static void recVU0_LowerOP_T3_01() { recVU0_LowerOP_T3_01_Table[(VU0.code >> 6) & 0x1f](); }
static void recVU0_LowerOP_T3_10() { recVU0_LowerOP_T3_10_Table[(VU0.code >> 6) & 0x1f](); }
static void recVU0_LowerOP_T3_11() { recVU0_LowerOP_T3_11_Table[(VU0.code >> 6) & 0x1f](); }

// LowerOP main sub-table: indexed by VU0.code & 0x3f
static VU0RecFn recVU0_LowerOP_Table[64] = {
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x00
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x04
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x08
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x0C
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x10
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x14
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x18
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x1C
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x20
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x24
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x28
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x2C
	recVU0_IADD,          recVU0_ISUB,          recVU0_IADDI,         recVU0_Lower_Unknown, // 0x30
	recVU0_IAND,          recVU0_IOR,           recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x34
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, // 0x38
	recVU0_LowerOP_T3_00, recVU0_LowerOP_T3_01, recVU0_LowerOP_T3_10, recVU0_LowerOP_T3_11, // 0x3C
};

// LowerOP dispatch entry point (compile-time)
static void recVU0_LowerOP() { recVU0_LowerOP_Table[VU0.code & 0x3f](); }

// ============================================================================
//  recVU0_LowerTable[128]
//
//  Maps lower opcode index (lower_word >> 25) to a code-emitter function.
//  Layout mirrors VU0_LOWER_OPCODE in VUops.cpp.
//  Index 0x40 dispatches through recVU0_LowerOP sub-table chain.
// ============================================================================

// clang-format off
// 128 entries, 4 per line = 32 lines
VU0RecFn recVU0_LowerTable[128] = {
	// 0x00-0x03: LQ, SQ, unk, unk
	recVU0_LQ,            recVU0_SQ,            recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x04-0x07: ILW, ISW, unk, unk
	recVU0_ILW,           recVU0_ISW,           recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x08-0x0B: IADDIU, ISUBIU, unk, unk
	recVU0_IADDIU,        recVU0_ISUBIU,        recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x0C-0x0F: unk x4
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x10-0x13: FCEQ, FCSET, FCAND, FCOR
	recVU0_FCEQ,          recVU0_FCSET,         recVU0_FCAND,         recVU0_FCOR,
	// 0x14-0x17: FSEQ, FSSET, FSAND, FSOR
	recVU0_FSEQ,          recVU0_FSSET,         recVU0_FSAND,         recVU0_FSOR,
	// 0x18-0x1B: FMEQ, unk, FMAND, FMOR
	recVU0_FMEQ,          recVU0_Lower_Unknown, recVU0_FMAND,         recVU0_FMOR,
	// 0x1C-0x1F: FCGET, unk, unk, unk
	recVU0_FCGET,         recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x20-0x23: B, BAL, unk, unk
	recVU0_B,             recVU0_BAL,           recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x24-0x27: JR, JALR, unk, unk
	recVU0_JR,            recVU0_JALR,          recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x28-0x2B: IBEQ, IBNE, unk, unk
	recVU0_IBEQ,          recVU0_IBNE,          recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x2C-0x2F: IBLTZ, IBGTZ, IBLEZ, IBGEZ
	recVU0_IBLTZ,         recVU0_IBGTZ,         recVU0_IBLEZ,         recVU0_IBGEZ,
	// 0x30-0x3F: unknown x16
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x40: LowerOP (sub-table dispatch via recVU0_LowerOP)
	// 0x41-0x43: unknown
	recVU0_LowerOP,       recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x44-0x47: unknown x4
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x48-0x4B: unknown x4
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x4C-0x4F: unknown x4
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x50-0x5F: unknown x16
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x60-0x6F: unknown x16
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	// 0x70-0x7F: unknown x16
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
	recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown, recVU0_Lower_Unknown,
};
// clang-format on
