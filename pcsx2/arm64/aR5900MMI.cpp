// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — MMI (MultiMedia Instructions)
// Packed SIMD ops: PADD*/PSUB*, PCGT*, PMAX/MIN*, PCEQ*, PABS*,
// PSxx shifts, PEXTL*/PPAC*/PEXTU*, PINTH/PINTEH, PADSBH,
// PAND/POR/PXOR/PNOR, PMFHI/LO, PMTHI/LO, PCPYLD/UD,
// PREVH, PCPYH, PLZCW, PEXEH/PEXEW/PEXCH/PEXCW, PROT3W,
// PSLLVW/PSRLVW/PSRAVW, and more.
//
// Native implementations use ARM64 NEON 128-bit SIMD instructions.
// Complex or rarely-used ops remain as interpreter stubs (ISTUB=1).

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"

using namespace R5900;

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {
namespace MMI {

// ============================================================================
//  INTERP_MMI master switch: ALL ops become unconditional interpreter stubs.
//  No templates, no _Rd_ guards, no NEON code — just plain interpreter calls.
// ============================================================================

#if defined(INTERP_MMI) || defined(INTERP_EE)

#define REC_MMI_STUB(name) \
	void rec##name() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::name); }

// MMI main
REC_MMI_STUB(PLZCW)
REC_MMI_STUB(PMFHL)
REC_MMI_STUB(PMTHL)
REC_MMI_STUB(PSLLH)
REC_MMI_STUB(PSRLH)
REC_MMI_STUB(PSRAH)
REC_MMI_STUB(PSLLW)
REC_MMI_STUB(PSRLW)
REC_MMI_STUB(PSRAW)

// MMI0
REC_MMI_STUB(PADDW)
REC_MMI_STUB(PSUBW)
REC_MMI_STUB(PCGTW)
REC_MMI_STUB(PMAXW)
REC_MMI_STUB(PADDH)
REC_MMI_STUB(PSUBH)
REC_MMI_STUB(PCGTH)
REC_MMI_STUB(PMAXH)
REC_MMI_STUB(PADDB)
REC_MMI_STUB(PSUBB)
REC_MMI_STUB(PCGTB)
REC_MMI_STUB(PADDSW)
REC_MMI_STUB(PSUBSW)
REC_MMI_STUB(PEXTLW)
REC_MMI_STUB(PPACW)
REC_MMI_STUB(PADDSH)
REC_MMI_STUB(PSUBSH)
REC_MMI_STUB(PEXTLH)
REC_MMI_STUB(PPACH)
REC_MMI_STUB(PADDSB)
REC_MMI_STUB(PSUBSB)
REC_MMI_STUB(PEXTLB)
REC_MMI_STUB(PPACB)
REC_MMI_STUB(PEXT5)
REC_MMI_STUB(PPAC5)

// MMI1
REC_MMI_STUB(PABSW)
REC_MMI_STUB(PCEQW)
REC_MMI_STUB(PMINW)
REC_MMI_STUB(PADSBH)
REC_MMI_STUB(PABSH)
REC_MMI_STUB(PCEQH)
REC_MMI_STUB(PMINH)
REC_MMI_STUB(PCEQB)
REC_MMI_STUB(PADDUW)
REC_MMI_STUB(PSUBUW)
REC_MMI_STUB(PEXTUW)
REC_MMI_STUB(PADDUH)
REC_MMI_STUB(PSUBUH)
REC_MMI_STUB(PEXTUH)
REC_MMI_STUB(PADDUB)
REC_MMI_STUB(PSUBUB)
REC_MMI_STUB(PEXTUB)
REC_MMI_STUB(QFSRV)

// MMI2
REC_MMI_STUB(PMADDW)
REC_MMI_STUB(PSLLVW)
REC_MMI_STUB(PMSUBW)
REC_MMI_STUB(PMFHI)
REC_MMI_STUB(PMFLO)
REC_MMI_STUB(PINTH)
REC_MMI_STUB(PMULTW)
REC_MMI_STUB(PDIVW)
REC_MMI_STUB(PCPYLD)
REC_MMI_STUB(PMADDH)
REC_MMI_STUB(PHMADH)
REC_MMI_STUB(PAND)
REC_MMI_STUB(PXOR)
REC_MMI_STUB(PMSUBH)
REC_MMI_STUB(PHMSBH)
REC_MMI_STUB(PEXEH)
REC_MMI_STUB(PREVH)
REC_MMI_STUB(PMULTH)
REC_MMI_STUB(PDIVBW)
REC_MMI_STUB(PEXEW)
REC_MMI_STUB(PROT3W)

// MMI3
REC_MMI_STUB(PMADDUW)
REC_MMI_STUB(PSRLVW)
REC_MMI_STUB(PSRAVW)
REC_MMI_STUB(PMTHI)
REC_MMI_STUB(PMTLO)
REC_MMI_STUB(PINTEH)
REC_MMI_STUB(PCPYUD)
REC_MMI_STUB(POR)
REC_MMI_STUB(PNOR)
REC_MMI_STUB(PMULTUW)
REC_MMI_STUB(PDIVUW)
REC_MMI_STUB(PEXCH)
REC_MMI_STUB(PCPYH)
REC_MMI_STUB(PEXCW)

#undef REC_MMI_STUB

#else // !INTERP_MMI — native NEON implementations with per-op ISTUB toggles

// ============================================================================
//  NEON scratch register aliases (128-bit Q registers, caller-saved)
// ============================================================================

#define RVMMI0  a64::q0
#define RVMMI1  a64::q1
#define RVMMI2  a64::q2

// ============================================================================
//  Per-instruction interp stub toggles
//  Individual ISTUB_* = 1 for interpreter fallback, 0 for native NEON
// ============================================================================

// Simple packed arithmetic — native NEON (direct equivalents)
#define ISTUB_PADDW    0
#define ISTUB_PADDH    0
#define ISTUB_PADDB    0
#define ISTUB_PADDSW   0
#define ISTUB_PADDSH   0
#define ISTUB_PADDSB   0
#define ISTUB_PADDUW   0
#define ISTUB_PADDUH   0
#define ISTUB_PADDUB   0
#define ISTUB_PSUBW    0
#define ISTUB_PSUBH    0
#define ISTUB_PSUBB    0
#define ISTUB_PSUBSW   0
#define ISTUB_PSUBSH   0
#define ISTUB_PSUBSB   0
#define ISTUB_PSUBUW   0
#define ISTUB_PSUBUH   0
#define ISTUB_PSUBUB   0

// Compare / min / max — native NEON
#define ISTUB_PCGTW    0
#define ISTUB_PCGTH    0
#define ISTUB_PCGTB    0
#define ISTUB_PMAXW    0
#define ISTUB_PMAXH    0
#define ISTUB_PMINW    0
#define ISTUB_PMINH    0
#define ISTUB_PCEQW    0
#define ISTUB_PCEQH    0
#define ISTUB_PCEQB    0
#define ISTUB_PABSW    0
#define ISTUB_PABSH    0

// Logic — native NEON
#define ISTUB_PAND     0
#define ISTUB_POR      0
#define ISTUB_PXOR     0
#define ISTUB_PNOR     0

// HI/LO transfers — native (simple 128b load/store)
#define ISTUB_PMFHI    0
#define ISTUB_PMFLO    0
#define ISTUB_PMTHI    0
#define ISTUB_PMTLO    0

// 128-bit copy — native
#define ISTUB_PCPYLD   0
#define ISTUB_PCPYUD   0

// Packed shifts (immediate _Sa_) — native NEON
#define ISTUB_PSLLH    0
#define ISTUB_PSRLH    0
#define ISTUB_PSRAH    0
#define ISTUB_PSLLW    0
#define ISTUB_PSRLW    0
#define ISTUB_PSRAW    0

// Variable-shift — native NEON (Sshl/Ushl with negation)
#define ISTUB_PSLLVW   0
#define ISTUB_PSRLVW   0
#define ISTUB_PSRAVW   0

// Interleave / pack / extract — native (zip1/zip2/uzp1/ext)
#define ISTUB_PEXTLW   0
#define ISTUB_PEXTLH   0
#define ISTUB_PEXTLB   0
#define ISTUB_PPACW    0
#define ISTUB_PPACH    0
#define ISTUB_PPACB    0
#define ISTUB_PEXTUW   0
#define ISTUB_PEXTUH   0
#define ISTUB_PEXTUB   0
#define ISTUB_PINTH    0
#define ISTUB_PINTEH   0

// Misc packed — native NEON
#define ISTUB_PADSBH   0
#define ISTUB_PLZCW    0
#define ISTUB_PREVH    0
#define ISTUB_PCPYH    0

// Shuffle ops — native (Ins sequences)
#define ISTUB_PEXEH    0
#define ISTUB_PEXEW    0
#define ISTUB_PEXCW    0
#define ISTUB_PEXCH    0
#define ISTUB_PROT3W   0

// Complex ops — ported natively to match x86 coverage.
#define ISTUB_PMFHL    0   // 4 native modes (LW/UW/LH/SH) + SLW/invalid → interp
#define ISTUB_PMTHL    0   // mode 0 only (other modes are no-ops per x86)
#define ISTUB_PEXT5    0   // RGB5→RGBA8 bit shuffle, mirrors x86 mask+shift chain
#define ISTUB_PPAC5    0   // RGBA8→RGB5 bit shuffle, mirrors x86 mask+shift chain
#define ISTUB_QFSRV    0   // 32-byte stage + unaligned 128b load from [tempqw+sa]
#define ISTUB_PMADDH   0   // 8×16b MAC to HI/LO
#define ISTUB_PHMADH   0   // horizontal MAC
#define ISTUB_PMSUBH   0   // 8×16b MSUB to HI/LO
#define ISTUB_PHMSBH   0   // horizontal MSUB
#define ISTUB_PMULTH   0   // 8×16b multiply to HI/LO
#define ISTUB_PMADDW   0   // 2×32b MAC to HI/LO
#define ISTUB_PMSUBW   0   // 2×32b MSUB to HI/LO
#define ISTUB_PMULTW   0   // 2×32b multiply to HI/LO
#define ISTUB_PMADDUW  0   // 2×32b unsigned MAC to HI/LO
#define ISTUB_PMULTUW  0   // 2×32b unsigned multiply to HI/LO
#define ISTUB_PDIVW    1   // 2×32b signed divide to HI/LO
#define ISTUB_PDIVUW   1   // 2×32b unsigned divide to HI/LO
#define ISTUB_PDIVBW   1   // 4×32/16b divide to HI/LO

// ============================================================================
//  Codegen helpers
// ============================================================================

// Load PS2 128-bit GPR into a NEON Q register.
// Must commit any pending const-prop value AND any cached lower-64-bit GPR
// value first — both only track the lower half, so the upper half in memory
// is authoritative but the lower half may be stale.
static __fi void armLoadGPR128(const a64::VRegister& dst, int gpr)
{
	armFlushConstReg(gpr);
	armGprFlush(gpr);
	armAsm->Ldr(dst, a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
}

// Store NEON Q register into PS2 128-bit GPR. Any cached lower-64-bit value
// must be retired BEFORE the NEON store: armGprInvalidate flushes a dirty
// cache slot to memory, and if we did that AFTER the NEON store it would
// overwrite the new bytes with the stale lower-64. The slot's old value
// would have been wasted anyway (the NEON store replaces it), but keeping
// the order correct is what matters.
static __fi void armStoreGPR128(int gpr, const a64::VRegister& src)
{
	armGprInvalidate(gpr);
	armAsm->Str(src, a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
}

// Binary op template: loads rs→q0, rt→q1, runs opFunc(), stores q0→rd.
template<typename OpFunc>
static void armMMIBinOp(OpFunc opFunc)
{
	if (!_Rd_) return;
	armDelConstReg(_Rd_);
	armLoadGPR128(RVMMI0, _Rs_);
	armLoadGPR128(RVMMI1, _Rt_);
	opFunc();
	armStoreGPR128(_Rd_, RVMMI0);
}

// Unary op template: loads rt→q0, runs opFunc(), stores q0→rd.
template<typename OpFunc>
static void armMMIUnaryOp(OpFunc opFunc)
{
	if (!_Rd_) return;
	armDelConstReg(_Rd_);
	armLoadGPR128(RVMMI0, _Rt_);
	opFunc();
	armStoreGPR128(_Rd_, RVMMI0);
}

// ============================================================================
//  MMI — PLZCW, PMFHL, PMTHL, PSxx shifts (main MMI sub-table)
// ============================================================================

#if ISTUB_PLZCW
void recPLZCW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PLZCW); }
#else
void recPLZCW()
{
	if (!_Rd_) return;
	armDelConstReg(_Rd_);
	// Direct upper-half read bypasses armLoadGPR* — commit const AND any
	// cached lower-64 first. The _Rd_ store overwrites the cached lower-64,
	// so drop the cache slot (before the store, to avoid clobbering it).
	armFlushConstReg(_Rs_);
	armGprFlush(_Rs_);
	armGprInvalidate(_Rd_);
	armAsm->Ldr(RWSCRATCH,  a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rs_) + 0));
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rs_) + 4));
	armAsm->Cls(RWSCRATCH,  RWSCRATCH);
	armAsm->Cls(RWSCRATCH2, RWSCRATCH2);
	armAsm->Str(RWSCRATCH,  a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_) + 0));
	armAsm->Str(RWSCRATCH2, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_) + 4));
}
#endif

#if ISTUB_PMFHL
void recPMFHL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMFHL); }
#else
// _Sa_ selects one of 5 extraction modes from HI/LO into Rd:
//   0x00 LW  — Rd = {LO.L[0], HI.L[0], LO.L[2], HI.L[2]}
//   0x01 UW  — Rd = {LO.L[1], HI.L[1], LO.L[3], HI.L[3]}
//   0x02 SLW — saturate HI/LO doubleword to 32b signed; x86 falls to interp
//   0x03 LH  — Rd.H = {LO[0], LO[2], HI[0], HI[2], LO[4], LO[6], HI[4], HI[6]}
//   0x04 SH  — Rd.H = {sat(LO[0]), sat(LO[1]), sat(HI[0]), sat(HI[1]),
//                      sat(LO[2]), sat(LO[3]), sat(HI[2]), sat(HI[3])}
//              (signed 32→16 saturation per lane, then interleave L/H halves)
// Mirrors x86 recPMFHL in iMMI.cpp:162-232.
void recPMFHL()
{
	if (!_Rd_)
		return;
	armDelConstReg(_Rd_);

	switch (_Sa_)
	{
		case 0x00: // LW
			armAsm->Ldr(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
			armAsm->Ldr(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));
			// Even-indexed 32b lanes of each, packed to low 64b, then interleave.
			armAsm->Uzp1(a64::v3.V4S(), a64::v3.V4S(), a64::v3.V4S());
			armAsm->Uzp1(a64::v4.V4S(), a64::v4.V4S(), a64::v4.V4S());
			armAsm->Zip1(a64::v0.V4S(), a64::v3.V4S(), a64::v4.V4S());
			armStoreGPR128(_Rd_, a64::q0);
			return;

		case 0x01: // UW
			armAsm->Ldr(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
			armAsm->Ldr(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));
			// Odd-indexed 32b lanes, then interleave.
			armAsm->Uzp2(a64::v3.V4S(), a64::v3.V4S(), a64::v3.V4S());
			armAsm->Uzp2(a64::v4.V4S(), a64::v4.V4S(), a64::v4.V4S());
			armAsm->Zip1(a64::v0.V4S(), a64::v3.V4S(), a64::v4.V4S());
			armStoreGPR128(_Rd_, a64::q0);
			return;

		case 0x02: // SLW — x86 falls through to interp (iMMI.cpp:193-198)
			armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMFHL);
			return;

		case 0x03: // LH
			armAsm->Ldr(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
			armAsm->Ldr(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));
			// Even-indexed 16b lanes of each. After Uzp1.V8H the low 4H of each
			// holds {lane 0, 2, 4, 6}; interleaving as 4S packs them as halfword
			// pairs per PS2 spec (same shape as x86's PSHUF.LW/HW 0x88 + PSRL.DQ
			// + PUNPCK.LDQ sequence).
			armAsm->Uzp1(a64::v3.V8H(), a64::v3.V8H(), a64::v3.V8H());
			armAsm->Uzp1(a64::v4.V8H(), a64::v4.V8H(), a64::v4.V8H());
			armAsm->Zip1(a64::v0.V4S(), a64::v3.V4S(), a64::v4.V4S());
			armStoreGPR128(_Rd_, a64::q0);
			return;

		case 0x04: // SH — 32→16 signed saturating pack + swap middle 2 words
			armAsm->Ldr(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
			armAsm->Ldr(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));
			// Pack LO 4×s32 → low 4×s16, then HI 4×s32 → upper 4×s16.
			armAsm->Sqxtn(a64::v0.V4H(), a64::v3.V4S());
			armAsm->Sqxtn2(a64::v0.V8H(), a64::v4.V4S());
			// Swap lanes 1 and 2 (PSHUF.D 0xd8 equivalent).
			armAsm->Mov(a64::v1.V16B(), a64::v0.V16B());
			armAsm->Ins(a64::v1.S(), 1, a64::v0.S(), 2);
			armAsm->Ins(a64::v1.S(), 2, a64::v0.S(), 1);
			armStoreGPR128(_Rd_, a64::q1);
			return;

		default:
			// Invalid mode — x86 pxFails; we fall to interp for parity.
			armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMFHL);
			return;
	}
}
#endif

#if ISTUB_PMTHL
void recPMTHL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMTHL); }
#else
// _Sa_ == 0: LO.L[0,2] = Rs.L[0,2], HI.L[0,2] = Rs.L[1,3]; other lanes preserved.
// Other _Sa_ values are silently no-op (matches x86 recPMTHL in iMMI.cpp:234-248,
// which early-returns on _Sa_ != 0).
void recPMTHL()
{
	if (_Sa_ != 0)
		return;

	armLoadGPR128(a64::q0, _Rs_);
	armAsm->Ldr(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Ldr(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));

	// LO.L[0] ← Rs.L[0], LO.L[2] ← Rs.L[2] (lanes 1 and 3 of LO preserved).
	armAsm->Ins(a64::v3.S(), 0, a64::v0.S(), 0);
	armAsm->Ins(a64::v3.S(), 2, a64::v0.S(), 2);
	// HI.L[0] ← Rs.L[1], HI.L[2] ← Rs.L[3] (lanes 1 and 3 of HI preserved).
	armAsm->Ins(a64::v4.S(), 0, a64::v0.S(), 1);
	armAsm->Ins(a64::v4.S(), 2, a64::v0.S(), 3);

	armAsm->Str(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Str(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));
}
#endif

// ---- Packed shifts (immediate sa) ----

#if ISTUB_PSLLH
void recPSLLH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSLLH); }
#else
void recPSLLH()
{
	const int sa = _Sa_ & 0xF;
	armMMIUnaryOp([sa]() {
		armAsm->Shl(a64::v0.V8H(), a64::v0.V8H(), sa);
	});
}
#endif

#if ISTUB_PSRLH
void recPSRLH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSRLH); }
#else
void recPSRLH()
{
	const int sa = _Sa_ & 0xF;
	armMMIUnaryOp([sa]() {
		if (sa != 0)
			armAsm->Ushr(a64::v0.V8H(), a64::v0.V8H(), sa);
	});
}
#endif

#if ISTUB_PSRAH
void recPSRAH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSRAH); }
#else
void recPSRAH()
{
	const int sa = _Sa_ & 0xF;
	armMMIUnaryOp([sa]() {
		if (sa != 0)
			armAsm->Sshr(a64::v0.V8H(), a64::v0.V8H(), sa);
	});
}
#endif

#if ISTUB_PSLLW
void recPSLLW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSLLW); }
#else
void recPSLLW()
{
	const int sa = _Sa_;
	armMMIUnaryOp([sa]() {
		armAsm->Shl(a64::v0.V4S(), a64::v0.V4S(), sa);
	});
}
#endif

#if ISTUB_PSRLW
void recPSRLW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSRLW); }
#else
void recPSRLW()
{
	const int sa = _Sa_;
	armMMIUnaryOp([sa]() {
		if (sa != 0)
			armAsm->Ushr(a64::v0.V4S(), a64::v0.V4S(), sa);
	});
}
#endif

#if ISTUB_PSRAW
void recPSRAW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSRAW); }
#else
void recPSRAW()
{
	const int sa = _Sa_;
	armMMIUnaryOp([sa]() {
		if (sa != 0)
			armAsm->Sshr(a64::v0.V4S(), a64::v0.V4S(), sa);
	});
}
#endif

// ============================================================================
//  MMI0 — PADDW/H/B, PSUBW/H/B, saturating, unsigned-sat, compare, pack/interleave
// ============================================================================

#if ISTUB_PADDW
void recPADDW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDW); }
#else
void recPADDW() { armMMIBinOp([]() { armAsm->Add(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PSUBW
void recPSUBW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBW); }
#else
void recPSUBW() { armMMIBinOp([]() { armAsm->Sub(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PCGTW
void recPCGTW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCGTW); }
#else
void recPCGTW() { armMMIBinOp([]() { armAsm->Cmgt(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PMAXW
void recPMAXW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMAXW); }
#else
void recPMAXW() { armMMIBinOp([]() { armAsm->Smax(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PADDH
void recPADDH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDH); }
#else
void recPADDH() { armMMIBinOp([]() { armAsm->Add(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PSUBH
void recPSUBH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBH); }
#else
void recPSUBH() { armMMIBinOp([]() { armAsm->Sub(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PCGTH
void recPCGTH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCGTH); }
#else
void recPCGTH() { armMMIBinOp([]() { armAsm->Cmgt(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PMAXH
void recPMAXH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMAXH); }
#else
void recPMAXH() { armMMIBinOp([]() { armAsm->Smax(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PADDB
void recPADDB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDB); }
#else
void recPADDB() { armMMIBinOp([]() { armAsm->Add(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PSUBB
void recPSUBB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBB); }
#else
void recPSUBB() { armMMIBinOp([]() { armAsm->Sub(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PCGTB
void recPCGTB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCGTB); }
#else
void recPCGTB() { armMMIBinOp([]() { armAsm->Cmgt(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PADDSW
void recPADDSW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDSW); }
#else
void recPADDSW() { armMMIBinOp([]() { armAsm->Sqadd(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PSUBSW
void recPSUBSW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBSW); }
#else
void recPSUBSW() { armMMIBinOp([]() { armAsm->Sqsub(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PEXTLW
void recPEXTLW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXTLW); }
#else
void recPEXTLW() { armMMIBinOp([]() { armAsm->Zip1(a64::v0.V4S(), a64::v1.V4S(), a64::v0.V4S()); }); }
#endif

#if ISTUB_PPACW
void recPPACW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PPACW); }
#else
void recPPACW() { armMMIBinOp([]() { armAsm->Uzp1(a64::v0.V4S(), a64::v1.V4S(), a64::v0.V4S()); }); }
#endif

#if ISTUB_PADDSH
void recPADDSH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDSH); }
#else
void recPADDSH() { armMMIBinOp([]() { armAsm->Sqadd(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PSUBSH
void recPSUBSH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBSH); }
#else
void recPSUBSH() { armMMIBinOp([]() { armAsm->Sqsub(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PEXTLH
void recPEXTLH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXTLH); }
#else
void recPEXTLH() { armMMIBinOp([]() { armAsm->Zip1(a64::v0.V8H(), a64::v1.V8H(), a64::v0.V8H()); }); }
#endif

#if ISTUB_PPACH
void recPPACH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PPACH); }
#else
void recPPACH() { armMMIBinOp([]() { armAsm->Uzp1(a64::v0.V8H(), a64::v1.V8H(), a64::v0.V8H()); }); }
#endif

#if ISTUB_PADDSB
void recPADDSB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDSB); }
#else
void recPADDSB() { armMMIBinOp([]() { armAsm->Sqadd(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PSUBSB
void recPSUBSB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBSB); }
#else
void recPSUBSB() { armMMIBinOp([]() { armAsm->Sqsub(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PEXTLB
void recPEXTLB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXTLB); }
#else
void recPEXTLB() { armMMIBinOp([]() { armAsm->Zip1(a64::v0.V16B(), a64::v1.V16B(), a64::v0.V16B()); }); }
#endif

#if ISTUB_PPACB
void recPPACB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PPACB); }
#else
void recPPACB() { armMMIBinOp([]() { armAsm->Uzp1(a64::v0.V16B(), a64::v1.V16B(), a64::v0.V16B()); }); }
#endif

#if ISTUB_PEXT5
void recPEXT5() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXT5); }
#else
// RGB5A1 (5+5+5+1 packed per 32-bit word) → RGBA8888 expansion.  Direct port
// of x86 recPEXT5 shift/mask chain (iMMI.cpp:548-578) — per 32-bit lane:
//   out.bit[  0.. 4] = in.bit[ 0.. 4] << 3   (R5 → R8 high-5)
//   out.bit[  8..12] = in.bit[ 5.. 9] << 3   (G5 → G8 high-5)
//   out.bit[ 16..20] = in.bit[10..14] << 3   (B5 → B8 high-5)
//   out.bit[     31] = in.bit[    15]        (A1 → A8 MSB)
// Low 3 bits of each channel and bits 5..7 of alpha are zero (PS2 behaviour).
void recPEXT5()
{
	if (!_Rd_)
		return;
	armDelConstReg(_Rd_);
	armLoadGPR128(a64::q0, _Rt_);

	// v1 = t0reg (B5 bits + A1 bit pre-position), v2 = t1reg, v3 = D (R5 + G5).

	// Extract G5 (bits 5..9) → bits 0..4 of v1.
	armAsm->Shl(a64::v1.V4S(), a64::v0.V4S(), 22);
	// Extract A1 (bit 15 of each halfword) → bit 0 of each halfword in v2.
	armAsm->Ushr(a64::v2.V8H(), a64::v0.V8H(), 15);
	armAsm->Ushr(a64::v1.V4S(), a64::v1.V4S(), 27);
	// A1 → bit 20 of each 32b lane (so final shift<<11 puts it at bit 31).
	armAsm->Shl(a64::v2.V4S(), a64::v2.V4S(), 20);
	armAsm->Orr(a64::v1.V16B(), a64::v1.V16B(), a64::v2.V16B());

	// Extract B5 (bits 10..14) → bits 0..4 of high halfword of each 32b lane
	// (via dword shift 17 then word shift 11).
	armAsm->Shl(a64::v2.V4S(), a64::v0.V4S(), 17);
	// Extract R5 (bits 0..4) → bits 0..4 of v3.
	armAsm->Shl(a64::v3.V4S(), a64::v0.V4S(), 27);
	armAsm->Ushr(a64::v3.V4S(), a64::v3.V4S(), 27);
	armAsm->Ushr(a64::v2.V8H(), a64::v2.V8H(), 11);
	armAsm->Orr(a64::v3.V16B(), a64::v3.V16B(), a64::v2.V16B());

	// Final assembly: low halfword of each 32b contains R5 (bits 0..4) + B5
	// (bits 16..20 after the halfword shift).  << 3 places them at RGBA8
	// positions.  t0reg holds G5 at bits 0..4 and A1 at bit 20; << 11 puts
	// G5 at bit 11 (wait, matches PS2 RGB5→RGBA8 per x86 shift chain).
	armAsm->Shl(a64::v3.V8H(), a64::v3.V8H(), 3);
	armAsm->Shl(a64::v1.V8H(), a64::v1.V8H(), 11);
	armAsm->Orr(a64::v3.V16B(), a64::v3.V16B(), a64::v1.V16B());

	armStoreGPR128(_Rd_, a64::q3);
}
#endif

#if ISTUB_PPAC5
void recPPAC5() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PPAC5); }
#else
// RGBA8888 → RGB5A1 compression.  Direct port of x86 recPPAC5 (iMMI.cpp:581-613):
//   out.bit[ 0.. 4] = in.bit[ 3.. 7]   (R high 5)
//   out.bit[ 5.. 9] = in.bit[11..15]   (G high 5)
//   out.bit[10..14] = in.bit[19..23]   (B high 5)
//   out.bit[    15] = in.bit[    31]   (A MSB)
//   out.bit[16..31] = 0
// The x86 sequence builds R+G into D, B+A into t0, then uses a 0x3FF mask
// to merge (keep R+G from D, B+A from t0).  We mirror via NEON Bic.
void recPPAC5()
{
	if (!_Rd_)
		return;
	armDelConstReg(_Rd_);
	armLoadGPR128(a64::q0, _Rt_);

	// v1 = t0reg (B5 + A1 bit), v2 = t1reg (scratch + final mask), v3 = D (R5 + G5).

	// t0 = (in << 8) >> 17  → bits 10..14 of T at bits 0..4 of v1.
	armAsm->Shl(a64::v1.V4S(), a64::v0.V4S(), 8);
	// t1 = (in >> 31) << 15 → bit 31 of T at bit 15 of v2.
	armAsm->Ushr(a64::v2.V4S(), a64::v0.V4S(), 31);
	armAsm->Ushr(a64::v1.V4S(), a64::v1.V4S(), 17);
	armAsm->Shl(a64::v2.V4S(), a64::v2.V4S(), 15);
	armAsm->Orr(a64::v1.V16B(), a64::v1.V16B(), a64::v2.V16B());

	// D = (in << 24) >> 27 | (in >> 11) << 5  → R5 at bits 0..4, G5 at bits 5..9.
	armAsm->Ushr(a64::v2.V4S(), a64::v0.V4S(), 11);
	armAsm->Shl(a64::v3.V4S(), a64::v0.V4S(), 24);
	armAsm->Ushr(a64::v3.V4S(), a64::v3.V4S(), 27);
	armAsm->Shl(a64::v2.V4S(), a64::v2.V4S(), 5);
	armAsm->Orr(a64::v3.V16B(), a64::v3.V16B(), a64::v2.V16B());

	// Build 0x3FF mask per 32b lane (all-ones >> 22) to keep only R+G bits of D,
	// then OR in B+A from t0 (via Bic for PANDN semantics).
	armAsm->Movi(a64::v2.V16B(), 0xFF);
	armAsm->Ushr(a64::v2.V4S(), a64::v2.V4S(), 22);
	armAsm->And(a64::v3.V16B(), a64::v3.V16B(), a64::v2.V16B());
	// PANDN(t1, t0) → t1 = (~t1) & t0.  NEON Bic(Vd, Vn, Vm) = Vn AND NOT Vm,
	// so Bic(v2, v1, v2) = v1 AND NOT v2.
	armAsm->Bic(a64::v2.V16B(), a64::v1.V16B(), a64::v2.V16B());
	armAsm->Orr(a64::v3.V16B(), a64::v3.V16B(), a64::v2.V16B());

	armStoreGPR128(_Rd_, a64::q3);
}
#endif

#if ISTUB_PADDUW
void recPADDUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDUW); }
#else
void recPADDUW() { armMMIBinOp([]() { armAsm->Uqadd(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PSUBUW
void recPSUBUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBUW); }
#else
void recPSUBUW() { armMMIBinOp([]() { armAsm->Uqsub(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PEXTUW
void recPEXTUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXTUW); }
#else
void recPEXTUW() { armMMIBinOp([]() { armAsm->Zip2(a64::v0.V4S(), a64::v1.V4S(), a64::v0.V4S()); }); }
#endif

#if ISTUB_PADDUH
void recPADDUH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDUH); }
#else
void recPADDUH() { armMMIBinOp([]() { armAsm->Uqadd(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PSUBUH
void recPSUBUH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBUH); }
#else
void recPSUBUH() { armMMIBinOp([]() { armAsm->Uqsub(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PEXTUH
void recPEXTUH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXTUH); }
#else
void recPEXTUH() { armMMIBinOp([]() { armAsm->Zip2(a64::v0.V8H(), a64::v1.V8H(), a64::v0.V8H()); }); }
#endif

#if ISTUB_PADDUB
void recPADDUB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADDUB); }
#else
void recPADDUB() { armMMIBinOp([]() { armAsm->Uqadd(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PSUBUB
void recPSUBUB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSUBUB); }
#else
void recPSUBUB() { armMMIBinOp([]() { armAsm->Uqsub(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PEXTUB
void recPEXTUB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXTUB); }
#else
void recPEXTUB() { armMMIBinOp([]() { armAsm->Zip2(a64::v0.V16B(), a64::v1.V16B(), a64::v0.V16B()); }); }
#endif

// ============================================================================
//  MMI1 — PABSW/H, PCEQW/H/B, PMIN/MAX, PADSBH, QFSRV
// ============================================================================

#if ISTUB_PABSW
void recPABSW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PABSW); }
#else
void recPABSW() { armMMIUnaryOp([]() { armAsm->Sqabs(a64::v0.V4S(), a64::v0.V4S()); }); }
#endif

#if ISTUB_PCEQW
void recPCEQW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCEQW); }
#else
void recPCEQW() { armMMIBinOp([]() { armAsm->Cmeq(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PMINW
void recPMINW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMINW); }
#else
void recPMINW() { armMMIBinOp([]() { armAsm->Smin(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); }); }
#endif

#if ISTUB_PADSBH
void recPADSBH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PADSBH); }
#else
void recPADSBH()
{
	if (!_Rd_) return;
	armDelConstReg(_Rd_);
	armLoadGPR128(RVMMI0, _Rs_);
	armLoadGPR128(RVMMI1, _Rt_);
	armAsm->Sub(a64::v2.V8H(), a64::v0.V8H(), a64::v1.V8H());
	armAsm->Add(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H());
	armAsm->Ins(a64::v2.D(), 1, a64::v0.D(), 1);
	armStoreGPR128(_Rd_, RVMMI2);
}
#endif

#if ISTUB_PABSH
void recPABSH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PABSH); }
#else
void recPABSH() { armMMIUnaryOp([]() { armAsm->Sqabs(a64::v0.V8H(), a64::v0.V8H()); }); }
#endif

#if ISTUB_PCEQH
void recPCEQH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCEQH); }
#else
void recPCEQH() { armMMIBinOp([]() { armAsm->Cmeq(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PMINH
void recPMINH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMINH); }
#else
void recPMINH() { armMMIBinOp([]() { armAsm->Smin(a64::v0.V8H(), a64::v0.V8H(), a64::v1.V8H()); }); }
#endif

#if ISTUB_PCEQB
void recPCEQB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCEQB); }
#else
void recPCEQB() { armMMIBinOp([]() { armAsm->Cmeq(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PSLLVW
void recPSLLVW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSLLVW); }
#else
void recPSLLVW()
{
	if (!_Rd_) return;
	armDelConstReg(_Rd_);
	armLoadGPR128(RVMMI0, _Rt_);
	armLoadGPR128(RVMMI1, _Rs_);
	armAsm->Movi(a64::v2.V4S(), 31);
	armAsm->And(a64::v1.V16B(), a64::v1.V16B(), a64::v2.V16B());
	armAsm->Ushl(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S());
	armAsm->Ins(a64::v0.S(), 1, a64::v0.S(), 2);
	armAsm->Sxtl(a64::v0.V2D(), a64::v0.V2S());
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_PSRLVW
void recPSRLVW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSRLVW); }
#else
void recPSRLVW()
{
	if (!_Rd_) return;
	armDelConstReg(_Rd_);
	armLoadGPR128(RVMMI0, _Rt_);
	armLoadGPR128(RVMMI1, _Rs_);
	armAsm->Movi(a64::v2.V4S(), 31);
	armAsm->And(a64::v1.V16B(), a64::v1.V16B(), a64::v2.V16B());
	armAsm->Neg(a64::v1.V4S(), a64::v1.V4S());
	armAsm->Ushl(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S());
	armAsm->Ins(a64::v0.S(), 1, a64::v0.S(), 2);
	armAsm->Sxtl(a64::v0.V2D(), a64::v0.V2S());
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_QFSRV
void recQFSRV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::QFSRV); }
#else
// 32-byte staging buffer for QFSRV's "concatenate Rt||Rs, extract 16B at +sa".
// File-local static: EE JIT dispatches single-threaded so no synchronisation
// is needed (same assumption as the XGKICK scratch in iVU1Lower_arm64.cpp).
alignas(16) static u32 qfsrv_tempqw[8];

// QFSRV: Rd = bytes[sa..sa+15] of the 32-byte concatenation {Rt, Rs},
// where `sa` is a runtime value in cpuRegs.sa (0..15 per PS2 spec).
// Mirrors x86 recQFSRV (iMMI.cpp:1268-1297) including the "_Rs_ == _Rt_ + 1"
// fast path that avoids the staging buffer when the two GPRs are already
// contiguous in cpuRegs.GPR.r[].
void recQFSRV()
{
	if (!_Rd_)
		return;
	armDelConstReg(_Rd_);

	if (_Rs_ == _Rt_ + 1)
	{
		// Fast path: adjacent GPRs are contiguous in memory.  Flush any
		// cached/const state on Rt and Rs (matches x86 _flushEEreg), then
		// load directly from &cpuRegs.GPR.r[_Rt_] + sa.
		armFlushConstReg(_Rt_);
		armGprFlush(_Rt_);
		armFlushConstReg(_Rs_);
		armGprFlush(_Rs_);

		armAsm->Add(RSCRATCHGPR, RCPUSTATE, GPR_OFFSET(_Rt_));
		armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, SA_OFFSET));
		armAsm->Add(RSCRATCHGPR, RSCRATCHGPR, RSCRATCHGPR2);
		armAsm->Ldr(a64::q0, a64::MemOperand(RSCRATCHGPR));
		armStoreGPR128(_Rd_, a64::q0);
		return;
	}

	// General path: stage Rt and Rs contiguously in qfsrv_tempqw, then do
	// unaligned 128b load from [tempqw + sa].  ARM64 LDR Q supports unaligned
	// access by default.
	armLoadGPR128(a64::q0, _Rt_);
	armLoadGPR128(a64::q1, _Rs_);

	armAsm->Mov(RSCRATCHGPR, reinterpret_cast<uintptr_t>(qfsrv_tempqw));
	armAsm->Str(a64::q0, a64::MemOperand(RSCRATCHGPR));
	armAsm->Str(a64::q1, a64::MemOperand(RSCRATCHGPR, 16));

	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, SA_OFFSET));
	armAsm->Add(RSCRATCHGPR, RSCRATCHGPR, RSCRATCHGPR2);
	armAsm->Ldr(a64::q0, a64::MemOperand(RSCRATCHGPR));

	armStoreGPR128(_Rd_, a64::q0);
}
#endif

// ============================================================================
//  MMI2 — PMADDW, PSLLVW, PMFHI/LO, PINTH, PMULTW, PDIVW, PCPYLD, PAND, PXOR,
//          PMADDH, PHMADH, PMSUBH, PHMSBH, PMULTH, PDIVBW, PEXEW, PROT3W
// ============================================================================

#if ISTUB_PMADDW
void recPMADDW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMADDW); }
#else
void recPMADDW()
{
	// LO/HI are 128-bit accumulators, treated as two 64-bit slots (UD[0] and UD[1]).
	// acc = { (s64)HI.SL[0]<<32 | (u32)LO.SL[0],  (s64)HI.SL[2]<<32 | (u32)LO.SL[2] }
	// prod = { (s64)Rs.SL[0] * (s64)Rt.SL[0],  (s64)Rs.SL[2] * (s64)Rt.SL[2] }
	// result = acc + prod
	// new LO.SD[i] = sext32(result[i] & 0xFFFFFFFF)
	// new HI.SD[i] = sext32(result[i] >> 32)
	// Rd.SD[i]     = result[i]   (if Rd)
	armDelConstReg(_Rd_);
	armLoadGPR128(a64::q0, _Rs_);
	armLoadGPR128(a64::q1, _Rt_);
	// Extract words 0 and 2 into the lower 2×32 lanes
	armAsm->Uzp1(a64::v0.V4S(), a64::v0.V4S(), a64::v0.V4S()); // {Rs[0],Rs[2],Rs[0],Rs[2]}
	armAsm->Uzp1(a64::v1.V4S(), a64::v1.V4S(), a64::v1.V4S()); // {Rt[0],Rt[2],Rt[0],Rt[2]}
	armAsm->Smull(a64::v2.V2D(), a64::v0.V2S(), a64::v1.V2S()); // prod = {Rs[0]*Rt[0], Rs[2]*Rt[2]}
	// Build 64-bit accumulator from LO and HI
	armAsm->Ldr(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Ldr(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	armAsm->Uzp1(a64::v3.V4S(), a64::v3.V4S(), a64::v3.V4S()); // {LO[0],LO[2],...}
	armAsm->Uzp1(a64::v4.V4S(), a64::v4.V4S(), a64::v4.V4S()); // {HI[0],HI[2],...}
	armAsm->Uxtl(a64::v3.V2D(), a64::v3.V2S());                 // zero-extend LO elements
	armAsm->Sxtl(a64::v4.V2D(), a64::v4.V2S());                 // sign-extend HI elements
	armAsm->Shl(a64::v4.V2D(), a64::v4.V2D(), 32);              // shift HI to upper 32 bits
	armAsm->Orr(a64::v3.V16B(), a64::v3.V16B(), a64::v4.V16B()); // acc = HI<<32|LO
	armAsm->Add(a64::v2.V2D(), a64::v2.V2D(), a64::v3.V2D());   // result = prod + acc
	// Split result: LO = sext32(low32), HI = sext32(high32)
	armAsm->Xtn(a64::v3.V2S(), a64::v2.V2D());
	armAsm->Sxtl(a64::v3.V2D(), a64::v3.V2S());
	armAsm->Str(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Sshr(a64::v4.V2D(), a64::v2.V2D(), 32);
	armAsm->Str(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	if (_Rd_) armStoreGPR128(_Rd_, a64::q2);
}
#endif

#if ISTUB_PMSUBW
void recPMSUBW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMSUBW); }
#else
void recPMSUBW()
{
	// Same as PMADDW but result = acc - prod
	armDelConstReg(_Rd_);
	armLoadGPR128(a64::q0, _Rs_);
	armLoadGPR128(a64::q1, _Rt_);
	armAsm->Uzp1(a64::v0.V4S(), a64::v0.V4S(), a64::v0.V4S());
	armAsm->Uzp1(a64::v1.V4S(), a64::v1.V4S(), a64::v1.V4S());
	armAsm->Smull(a64::v2.V2D(), a64::v0.V2S(), a64::v1.V2S());
	armAsm->Ldr(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Ldr(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	armAsm->Uzp1(a64::v3.V4S(), a64::v3.V4S(), a64::v3.V4S());
	armAsm->Uzp1(a64::v4.V4S(), a64::v4.V4S(), a64::v4.V4S());
	armAsm->Uxtl(a64::v3.V2D(), a64::v3.V2S());
	armAsm->Sxtl(a64::v4.V2D(), a64::v4.V2S());
	armAsm->Shl(a64::v4.V2D(), a64::v4.V2D(), 32);
	armAsm->Orr(a64::v3.V16B(), a64::v3.V16B(), a64::v4.V16B());
	armAsm->Sub(a64::v2.V2D(), a64::v3.V2D(), a64::v2.V2D());   // result = acc - prod
	armAsm->Xtn(a64::v3.V2S(), a64::v2.V2D());
	armAsm->Sxtl(a64::v3.V2D(), a64::v3.V2S());
	armAsm->Str(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Sshr(a64::v4.V2D(), a64::v2.V2D(), 32);
	armAsm->Str(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	if (_Rd_) armStoreGPR128(_Rd_, a64::q2);
}
#endif

#if ISTUB_PMFHI
void recPMFHI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMFHI); }
#else
void recPMFHI()
{
	if (!_Rd_) return;
	armDelConstReg(_Rd_);
	armAsm->Ldr(RVMMI0, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_PMFLO
void recPMFLO() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMFLO); }
#else
void recPMFLO()
{
	if (!_Rd_) return;
	armDelConstReg(_Rd_);
	armAsm->Ldr(RVMMI0, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_PINTH
void recPINTH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PINTH); }
#else
void recPINTH()
{
	if (!_Rd_) return;
	armDelConstReg(_Rd_);
	armLoadGPR128(RVMMI0, _Rt_);
	armLoadGPR128(RVMMI1, _Rs_);
	armAsm->Ext(a64::v2.V16B(), a64::v1.V16B(), a64::v1.V16B(), 8);
	armAsm->Zip1(a64::v0.V8H(), a64::v0.V8H(), a64::v2.V8H());
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_PMULTW
void recPMULTW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMULTW); }
#else
void recPMULTW()
{
	// prod = { (s64)Rs.SL[0]*(s64)Rt.SL[0], (s64)Rs.SL[2]*(s64)Rt.SL[2] }
	// LO.SD[i] = sext32(prod[i] & 0xFFFFFFFF)
	// HI.SD[i] = sext32(prod[i] >> 32)
	// Rd.SD[i] = prod[i]  (if Rd)
	armDelConstReg(_Rd_);
	armLoadGPR128(a64::q0, _Rs_);
	armLoadGPR128(a64::q1, _Rt_);
	armAsm->Uzp1(a64::v0.V4S(), a64::v0.V4S(), a64::v0.V4S()); // {Rs[0],Rs[2],Rs[0],Rs[2]}
	armAsm->Uzp1(a64::v1.V4S(), a64::v1.V4S(), a64::v1.V4S()); // {Rt[0],Rt[2],Rt[0],Rt[2]}
	armAsm->Smull(a64::v2.V2D(), a64::v0.V2S(), a64::v1.V2S()); // {prod0, prod1}
	armAsm->Xtn(a64::v3.V2S(), a64::v2.V2D());
	armAsm->Sxtl(a64::v3.V2D(), a64::v3.V2S());
	armAsm->Str(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Sshr(a64::v4.V2D(), a64::v2.V2D(), 32);
	armAsm->Str(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	if (_Rd_) armStoreGPR128(_Rd_, a64::q2);
}
#endif

#if ISTUB_PDIVW
void recPDIVW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PDIVW); }
#else
void recPDIVW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PDIVW); }
#endif

#if ISTUB_PCPYLD
void recPCPYLD() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCPYLD); }
#else
void recPCPYLD()
{
	if (!_Rd_) return;
	armDelConstReg(_Rd_);
	armLoadGPR128(RVMMI0, _Rt_);
	armLoadGPR128(RVMMI1, _Rs_);
	armAsm->Ins(a64::v0.D(), 1, a64::v1.D(), 0);
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_PMADDH
void recPMADDH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMADDH); }
#else
void recPMADDH()
{
	// p[i] = Rs.SS[i] * Rt.SS[i]  (signed 16×16→32, 8 products)
	// LO += {p0,p1,p4,p5}  (32-bit wrapping add per element)
	// HI += {p2,p3,p6,p7}
	// Rd = {new_LO[0], new_HI[0], new_LO[2], new_HI[2]}  (if Rd)
	armDelConstReg(_Rd_);
	armLoadGPR128(a64::q0, _Rs_);
	armLoadGPR128(a64::q1, _Rt_);
	armAsm->Smull(a64::v2.V4S(), a64::v0.V4H(), a64::v1.V4H());  // {p0,p1,p2,p3}
	armAsm->Smull2(a64::v3.V4S(), a64::v0.V8H(), a64::v1.V8H()); // {p4,p5,p6,p7}
	armAsm->Ldr(a64::q4, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Ldr(a64::q5, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	// lo_add={p0,p1,p4,p5} = {v2.D[0],v3.D[0]}, hi_add={p2,p3,p6,p7} = {v2.D[1],v3.D[1]}
	armAsm->Uzp1(a64::v6.V2D(), a64::v2.V2D(), a64::v3.V2D()); // lo_add
	armAsm->Uzp2(a64::v7.V2D(), a64::v2.V2D(), a64::v3.V2D()); // hi_add
	armAsm->Add(a64::v4.V4S(), a64::v4.V4S(), a64::v6.V4S());  // new LO
	armAsm->Add(a64::v5.V4S(), a64::v5.V4S(), a64::v7.V4S());  // new HI
	armAsm->Str(a64::q4, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Str(a64::q5, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	if (_Rd_) {
		// Rd = {LO[0],HI[0],LO[2],HI[2]}: interleave even elements of LO and HI
		armAsm->Uzp1(a64::v0.V4S(), a64::v4.V4S(), a64::v4.V4S()); // {LO[0],LO[2],LO[0],LO[2]}
		armAsm->Uzp1(a64::v1.V4S(), a64::v5.V4S(), a64::v5.V4S()); // {HI[0],HI[2],HI[0],HI[2]}
		armAsm->Zip1(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S()); // {LO[0],HI[0],LO[2],HI[2]}
		armStoreGPR128(_Rd_, a64::q0);
	}
}
#endif

#if ISTUB_PHMADH
void recPHMADH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PHMADH); }
#else
void recPHMADH()
{
	// p[i] = Rs.SS[i] * Rt.SS[i];  pairs (0,1), (2,3), (4,5), (6,7)
	// LO = {p0+p1, p1, p4+p5, p5}
	// HI = {p2+p3, p3, p6+p7, p7}
	// Rd = {p0+p1, p2+p3, p4+p5, p6+p7}  (if Rd)
	armDelConstReg(_Rd_);
	armLoadGPR128(a64::q0, _Rs_);
	armLoadGPR128(a64::q1, _Rt_);
	armAsm->Smull(a64::v2.V4S(), a64::v0.V4H(), a64::v1.V4H());  // {p0,p1,p2,p3}
	armAsm->Smull2(a64::v3.V4S(), a64::v0.V8H(), a64::v1.V8H()); // {p4,p5,p6,p7}
	// even={p0,p2,p4,p6}, odd={p1,p3,p5,p7}
	armAsm->Uzp1(a64::v4.V4S(), a64::v2.V4S(), a64::v3.V4S());
	armAsm->Uzp2(a64::v5.V4S(), a64::v2.V4S(), a64::v3.V4S());
	armAsm->Add(a64::v2.V4S(), a64::v4.V4S(), a64::v5.V4S());    // sums (= Rd)
	// Interleave sums and odds to build LO/HI halves
	armAsm->Zip1(a64::v6.V4S(), a64::v2.V4S(), a64::v5.V4S()); // {p01,p1,p23,p3}
	armAsm->Zip2(a64::v7.V4S(), a64::v2.V4S(), a64::v5.V4S()); // {p45,p5,p67,p7}
	armAsm->Uzp1(a64::v3.V2D(), a64::v6.V2D(), a64::v7.V2D()); // LO = {p01,p1, p45,p5}
	armAsm->Str(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Uzp2(a64::v4.V2D(), a64::v6.V2D(), a64::v7.V2D()); // HI = {p23,p3, p67,p7}
	armAsm->Str(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	if (_Rd_) armStoreGPR128(_Rd_, a64::q2); // sums = {p01,p23,p45,p67}
}
#endif

#if ISTUB_PAND
void recPAND() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PAND); }
#else
void recPAND() { armMMIBinOp([]() { armAsm->And(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PXOR
void recPXOR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PXOR); }
#else
void recPXOR()
{
	if (!_Rd_) return;
	armDelConstReg(_Rd_);
	if (_Rs_ == _Rt_)
	{
		// Direct 128-bit zero store bypasses armStoreGPR128 — drop any
		// cached lower-64 for _Rd_ first (it'd otherwise be flushed over
		// our zero on the next invalidate).
		armGprInvalidate(_Rd_);
		armAsm->Stp(a64::xzr, a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	armMMIBinOp([]() { armAsm->Eor(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); });
}
#endif

#if ISTUB_PMSUBH
void recPMSUBH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMSUBH); }
#else
void recPMSUBH()
{
	// Same as PMADDH but subtracts: LO -= {p0,p1,p4,p5}, HI -= {p2,p3,p6,p7}
	armDelConstReg(_Rd_);
	armLoadGPR128(a64::q0, _Rs_);
	armLoadGPR128(a64::q1, _Rt_);
	armAsm->Smull(a64::v2.V4S(), a64::v0.V4H(), a64::v1.V4H());
	armAsm->Smull2(a64::v3.V4S(), a64::v0.V8H(), a64::v1.V8H());
	armAsm->Ldr(a64::q4, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Ldr(a64::q5, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	armAsm->Uzp1(a64::v6.V2D(), a64::v2.V2D(), a64::v3.V2D()); // lo_sub={p0,p1,p4,p5}
	armAsm->Uzp2(a64::v7.V2D(), a64::v2.V2D(), a64::v3.V2D()); // hi_sub={p2,p3,p6,p7}
	armAsm->Sub(a64::v4.V4S(), a64::v4.V4S(), a64::v6.V4S());  // new LO
	armAsm->Sub(a64::v5.V4S(), a64::v5.V4S(), a64::v7.V4S());  // new HI
	armAsm->Str(a64::q4, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Str(a64::q5, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	if (_Rd_) {
		armAsm->Uzp1(a64::v0.V4S(), a64::v4.V4S(), a64::v4.V4S());
		armAsm->Uzp1(a64::v1.V4S(), a64::v5.V4S(), a64::v5.V4S());
		armAsm->Zip1(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S());
		armStoreGPR128(_Rd_, a64::q0);
	}
}
#endif

#if ISTUB_PHMSBH
void recPHMSBH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PHMSBH); }
#else
void recPHMSBH()
{
	// p[i] = Rs.SS[i] * Rt.SS[i];  odd products minus even products per pair
	// LO = {p1-p0, ~p1, p5-p4, ~p5}
	// HI = {p3-p2, ~p3, p7-p6, ~p7}
	// Rd = {p1-p0, p3-p2, p5-p4, p7-p6}  (if Rd)
	armDelConstReg(_Rd_);
	armLoadGPR128(a64::q0, _Rs_);
	armLoadGPR128(a64::q1, _Rt_);
	armAsm->Smull(a64::v2.V4S(), a64::v0.V4H(), a64::v1.V4H());  // {p0,p1,p2,p3}
	armAsm->Smull2(a64::v3.V4S(), a64::v0.V8H(), a64::v1.V8H()); // {p4,p5,p6,p7}
	armAsm->Uzp1(a64::v4.V4S(), a64::v2.V4S(), a64::v3.V4S());   // even: {p0,p2,p4,p6}
	armAsm->Uzp2(a64::v5.V4S(), a64::v2.V4S(), a64::v3.V4S());   // odd:  {p1,p3,p5,p7}
	armAsm->Sub(a64::v2.V4S(), a64::v5.V4S(), a64::v4.V4S());    // diff: {p1-p0,p3-p2,p5-p4,p7-p6} = Rd
	armAsm->Mvn(a64::v3.V16B(), a64::v5.V16B());                  // ~odd: {~p1,~p3,~p5,~p7}
	// Interleave diff and ~odd to build LO/HI halves
	armAsm->Zip1(a64::v6.V4S(), a64::v2.V4S(), a64::v3.V4S()); // {p1-p0,~p1, p3-p2,~p3}
	armAsm->Zip2(a64::v7.V4S(), a64::v2.V4S(), a64::v3.V4S()); // {p5-p4,~p5, p7-p6,~p7}
	armAsm->Uzp1(a64::v4.V2D(), a64::v6.V2D(), a64::v7.V2D()); // LO = {p1-p0,~p1, p5-p4,~p5}
	armAsm->Str(a64::q4, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Uzp2(a64::v5.V2D(), a64::v6.V2D(), a64::v7.V2D()); // HI = {p3-p2,~p3, p7-p6,~p7}
	armAsm->Str(a64::q5, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	if (_Rd_) armStoreGPR128(_Rd_, a64::q2);
}
#endif

#if ISTUB_PEXEH
void recPEXEH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXEH); }
#else
void recPEXEH()
{
	armMMIUnaryOp([]() {
		armAsm->Mov(a64::v1.V16B(), a64::v0.V16B());
		armAsm->Ins(a64::v1.H(), 0, a64::v0.H(), 2);
		armAsm->Ins(a64::v1.H(), 2, a64::v0.H(), 0);
		armAsm->Ins(a64::v1.H(), 4, a64::v0.H(), 6);
		armAsm->Ins(a64::v1.H(), 6, a64::v0.H(), 4);
		armAsm->Mov(a64::v0.V16B(), a64::v1.V16B());
	});
}
#endif

#if ISTUB_PREVH
void recPREVH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PREVH); }
#else
void recPREVH() { armMMIUnaryOp([]() { armAsm->Rev64(a64::v0.V8H(), a64::v0.V8H()); }); }
#endif

#if ISTUB_PMULTH
void recPMULTH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMULTH); }
#else
void recPMULTH()
{
	// p[i] = Rs.SS[i] * Rt.SS[i]  (signed 16×16→32, 8 products)
	// LO = {p0,p1,p4,p5},  HI = {p2,p3,p6,p7}
	// Rd = {p0,p2,p4,p6}  (if Rd)
	armDelConstReg(_Rd_);
	armLoadGPR128(a64::q0, _Rs_);
	armLoadGPR128(a64::q1, _Rt_);
	armAsm->Smull(a64::v2.V4S(), a64::v0.V4H(), a64::v1.V4H());  // {p0,p1,p2,p3}
	armAsm->Smull2(a64::v3.V4S(), a64::v0.V8H(), a64::v1.V8H()); // {p4,p5,p6,p7}
	// LO = {v2.D[0], v3.D[0]} = {p0,p1,p4,p5}
	armAsm->Uzp1(a64::v4.V2D(), a64::v2.V2D(), a64::v3.V2D());
	armAsm->Str(a64::q4, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	// HI = {v2.D[1], v3.D[1]} = {p2,p3,p6,p7}
	armAsm->Uzp2(a64::v5.V2D(), a64::v2.V2D(), a64::v3.V2D());
	armAsm->Str(a64::q5, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	if (_Rd_) {
		// Rd = {p0,p2,p4,p6} = even elements from v2 and v3
		armAsm->Uzp1(a64::v2.V4S(), a64::v2.V4S(), a64::v3.V4S());
		armStoreGPR128(_Rd_, a64::q2);
	}
}
#endif

#if ISTUB_PDIVBW
void recPDIVBW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PDIVBW); }
#else
void recPDIVBW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PDIVBW); }
#endif

#if ISTUB_PEXEW
void recPEXEW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXEW); }
#else
void recPEXEW()
{
	armMMIUnaryOp([]() {
		armAsm->Mov(a64::v1.V16B(), a64::v0.V16B());
		armAsm->Ins(a64::v1.S(), 0, a64::v0.S(), 2);
		armAsm->Ins(a64::v1.S(), 2, a64::v0.S(), 0);
		armAsm->Mov(a64::v0.V16B(), a64::v1.V16B());
	});
}
#endif

#if ISTUB_PROT3W
void recPROT3W() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PROT3W); }
#else
void recPROT3W()
{
	armMMIUnaryOp([]() {
		armAsm->Mov(a64::v1.V16B(), a64::v0.V16B());
		armAsm->Ins(a64::v1.S(), 0, a64::v0.S(), 1);
		armAsm->Ins(a64::v1.S(), 1, a64::v0.S(), 2);
		armAsm->Ins(a64::v1.S(), 2, a64::v0.S(), 0);
		armAsm->Mov(a64::v0.V16B(), a64::v1.V16B());
	});
}
#endif

// ============================================================================
//  MMI3 — PMADDUW, PSRAVW, PMTHI/LO, PINTEH, PCPYUD, POR, PNOR,
//          PMULTUW, PDIVUW, PEXCH, PCPYH, PEXCW
// ============================================================================

#if ISTUB_PMADDUW
void recPMADDUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMADDUW); }
#else
void recPMADDUW()
{
	// Unsigned: acc = {(u64)HI.UL[0]<<32|(u64)LO.UL[0], (u64)HI.UL[2]<<32|(u64)LO.UL[2]}
	// prod     = {(u64)Rs.UL[0]*(u64)Rt.UL[0], (u64)Rs.UL[2]*(u64)Rt.UL[2]}
	// result   = acc + prod
	// new LO.SD[i] = sext32(result[i] & 0xFFFFFFFF)
	// new HI.SD[i] = sext32(result[i] >> 32)
	// Rd.UD[i] = result[i]  (if Rd)
	armDelConstReg(_Rd_);
	armLoadGPR128(a64::q0, _Rs_);
	armLoadGPR128(a64::q1, _Rt_);
	armAsm->Uzp1(a64::v0.V4S(), a64::v0.V4S(), a64::v0.V4S());
	armAsm->Uzp1(a64::v1.V4S(), a64::v1.V4S(), a64::v1.V4S());
	armAsm->Umull(a64::v2.V2D(), a64::v0.V2S(), a64::v1.V2S()); // unsigned prod
	armAsm->Ldr(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Ldr(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	armAsm->Uzp1(a64::v3.V4S(), a64::v3.V4S(), a64::v3.V4S());
	armAsm->Uzp1(a64::v4.V4S(), a64::v4.V4S(), a64::v4.V4S());
	armAsm->Uxtl(a64::v3.V2D(), a64::v3.V2S()); // zero-extend LO elements
	armAsm->Uxtl(a64::v4.V2D(), a64::v4.V2S()); // zero-extend HI elements (unsigned acc)
	armAsm->Shl(a64::v4.V2D(), a64::v4.V2D(), 32);
	armAsm->Orr(a64::v3.V16B(), a64::v3.V16B(), a64::v4.V16B());
	armAsm->Add(a64::v2.V2D(), a64::v2.V2D(), a64::v3.V2D());
	armAsm->Xtn(a64::v3.V2S(), a64::v2.V2D());
	armAsm->Sxtl(a64::v3.V2D(), a64::v3.V2S());
	armAsm->Str(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Sshr(a64::v4.V2D(), a64::v2.V2D(), 32);
	armAsm->Str(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	if (_Rd_) armStoreGPR128(_Rd_, a64::q2);
}
#endif

#if ISTUB_PSRAVW
void recPSRAVW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PSRAVW); }
#else
void recPSRAVW()
{
	if (!_Rd_) return;
	armDelConstReg(_Rd_);
	armLoadGPR128(RVMMI0, _Rt_);
	armLoadGPR128(RVMMI1, _Rs_);
	armAsm->Movi(a64::v2.V4S(), 31);
	armAsm->And(a64::v1.V16B(), a64::v1.V16B(), a64::v2.V16B());
	armAsm->Neg(a64::v1.V4S(), a64::v1.V4S());
	armAsm->Sshl(a64::v0.V4S(), a64::v0.V4S(), a64::v1.V4S());
	armAsm->Ins(a64::v0.S(), 1, a64::v0.S(), 2);
	armAsm->Sxtl(a64::v0.V2D(), a64::v0.V2S());
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_PMTHI
void recPMTHI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMTHI); }
#else
void recPMTHI()
{
	armLoadGPR128(RVMMI0, _Rs_);
	armAsm->Str(RVMMI0, a64::MemOperand(RCPUSTATE, HI_OFFSET));
}
#endif

#if ISTUB_PMTLO
void recPMTLO() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMTLO); }
#else
void recPMTLO()
{
	armLoadGPR128(RVMMI0, _Rs_);
	armAsm->Str(RVMMI0, a64::MemOperand(RCPUSTATE, LO_OFFSET));
}
#endif

#if ISTUB_PINTEH
void recPINTEH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PINTEH); }
#else
void recPINTEH()
{
	if (!_Rd_) return;
	armDelConstReg(_Rd_);
	armLoadGPR128(RVMMI0, _Rt_);
	armLoadGPR128(RVMMI1, _Rs_);
	armAsm->Uzp1(a64::v2.V8H(), a64::v0.V8H(), a64::v1.V8H());
	armAsm->Ext(a64::v1.V16B(), a64::v2.V16B(), a64::v2.V16B(), 8);
	armAsm->Zip1(a64::v0.V8H(), a64::v2.V8H(), a64::v1.V8H());
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_PCPYUD
void recPCPYUD() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCPYUD); }
#else
void recPCPYUD()
{
	if (!_Rd_) return;
	armDelConstReg(_Rd_);
	armLoadGPR128(RVMMI0, _Rs_);
	armLoadGPR128(RVMMI1, _Rt_);
	armAsm->Ins(a64::v0.D(), 0, a64::v0.D(), 1);
	armAsm->Ins(a64::v0.D(), 1, a64::v1.D(), 1);
	armStoreGPR128(_Rd_, RVMMI0);
}
#endif

#if ISTUB_POR
void recPOR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::POR); }
#else
void recPOR() { armMMIBinOp([]() { armAsm->Orr(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B()); }); }
#endif

#if ISTUB_PNOR
void recPNOR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PNOR); }
#else
void recPNOR()
{
	armMMIBinOp([]() {
		armAsm->Orr(a64::v0.V16B(), a64::v0.V16B(), a64::v1.V16B());
		armAsm->Mvn(a64::v0.V16B(), a64::v0.V16B());
	});
}
#endif

#if ISTUB_PMULTUW
void recPMULTUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PMULTUW); }
#else
void recPMULTUW()
{
	// Unsigned: prod = {(u64)Rs.UL[0]*(u64)Rt.UL[0], (u64)Rs.UL[2]*(u64)Rt.UL[2]}
	// LO.SD[i] = sext32(prod[i] & 0xFFFFFFFF)
	// HI.SD[i] = sext32(prod[i] >> 32)
	// Rd.UD[i] = prod[i]  (if Rd)
	armDelConstReg(_Rd_);
	armLoadGPR128(a64::q0, _Rs_);
	armLoadGPR128(a64::q1, _Rt_);
	armAsm->Uzp1(a64::v0.V4S(), a64::v0.V4S(), a64::v0.V4S());
	armAsm->Uzp1(a64::v1.V4S(), a64::v1.V4S(), a64::v1.V4S());
	armAsm->Umull(a64::v2.V2D(), a64::v0.V2S(), a64::v1.V2S());
	armAsm->Xtn(a64::v3.V2S(), a64::v2.V2D());
	armAsm->Sxtl(a64::v3.V2D(), a64::v3.V2S());
	armAsm->Str(a64::q3, a64::MemOperand(RCPUSTATE, LO_OFFSET));
	armAsm->Sshr(a64::v4.V2D(), a64::v2.V2D(), 32);
	armAsm->Str(a64::q4, a64::MemOperand(RCPUSTATE, HI_OFFSET));
	if (_Rd_) armStoreGPR128(_Rd_, a64::q2);
}
#endif

#if ISTUB_PDIVUW
void recPDIVUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PDIVUW); }
#else
void recPDIVUW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PDIVUW); }
#endif

#if ISTUB_PEXCH
void recPEXCH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXCH); }
#else
void recPEXCH()
{
	armMMIUnaryOp([]() {
		armAsm->Mov(a64::v1.V16B(), a64::v0.V16B());
		armAsm->Ins(a64::v1.H(), 1, a64::v0.H(), 2);
		armAsm->Ins(a64::v1.H(), 2, a64::v0.H(), 1);
		armAsm->Ins(a64::v1.H(), 5, a64::v0.H(), 6);
		armAsm->Ins(a64::v1.H(), 6, a64::v0.H(), 5);
		armAsm->Mov(a64::v0.V16B(), a64::v1.V16B());
	});
}
#endif

#if ISTUB_PCPYH
void recPCPYH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PCPYH); }
#else
void recPCPYH()
{
	armMMIUnaryOp([]() {
		armAsm->Dup(a64::v1.V8H(), a64::v0.H(), 0);
		armAsm->Dup(a64::v2.V8H(), a64::v0.H(), 4);
		armAsm->Ins(a64::v1.D(), 1, a64::v2.D(), 0);
		armAsm->Mov(a64::v0.V16B(), a64::v1.V16B());
	});
}
#endif

#if ISTUB_PEXCW
void recPEXCW() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MMI::PEXCW); }
#else
void recPEXCW()
{
	armMMIUnaryOp([]() {
		armAsm->Mov(a64::v1.V16B(), a64::v0.V16B());
		armAsm->Ins(a64::v1.S(), 1, a64::v0.S(), 2);
		armAsm->Ins(a64::v1.S(), 2, a64::v0.S(), 1);
		armAsm->Mov(a64::v0.V16B(), a64::v1.V16B());
	});
}
#endif

#endif // !INTERP_MMI

} // namespace MMI
} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
