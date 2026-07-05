// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — Multiply / Divide Instructions
// MULT/MULTU, MULT1/MULTU1, MADD/MADDU, MADD1/MADDU1, DIV/DIVU, DIV1/DIVU1
//
// All ops update HI:LO. The MULT/MADD family also conditionally update Rd
// with the (sign-extended) low 32 bits of the new LO. The /1 variants
// operate on the upper 64-bit half of HI/LO (HI.SD[1] / LO.SD[1]).

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"

using namespace R5900;

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
#if defined(INTERP_MULTDIV) || defined(INTERP_EE)
#define ISTUB_MULT     1
#define ISTUB_MULTU    1
#define ISTUB_MULT1    1
#define ISTUB_MULTU1   1
#define ISTUB_MADD     1
#define ISTUB_MADDU    1
#define ISTUB_MADD1    1
#define ISTUB_MADDU1   1
#define ISTUB_DIV      1
#define ISTUB_DIVU     1
#define ISTUB_DIV1     1
#define ISTUB_DIVU1    1
#else
#define ISTUB_MULT     0
#define ISTUB_MULTU    0
#define ISTUB_MULT1    0
#define ISTUB_MULTU1   0
#define ISTUB_MADD     0
#define ISTUB_MADDU    0
#define ISTUB_MADD1    0
#define ISTUB_MADDU1   0
#define ISTUB_DIV      0
#define ISTUB_DIVU     0
#define ISTUB_DIV1     0
#define ISTUB_DIVU1    0
#endif

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ============================================================================
//  Helpers
// ============================================================================
//
// After a 32×32 multiply (signed or unsigned) the 64-bit result lives in an
// X register. The PS2 wants:
//   LO[hilo_off]   = sign_extend((s32)(result & 0xFFFFFFFF))
//   HI[hilo_off]   = sign_extend((s32)(result >> 32))
//   if (rd != 0)   GPR[rd].UD[0] = LO[hilo_off]   (low 64 bits of GPR)
//
// hilo_off is 0 for the normal pipe and 8 for the /1 (pipeline 1) variants.

static void emitMultWritebackHILO(const a64::Register& xres, s64 hilo_off, int rd_for_writeback)
{
	// xres holds the full 64-bit multiply (or accumulator) result.
	// We need both halves sign-extended into separate X registers.
	// xlo gets LO (sign-extended low 32). xres is then reused to hold HI
	// (sign-extended high 32 via ASR).
	const a64::Register xlo = RSCRATCHGPR2;
	const a64::Register wres = a64::WRegister(xres.GetCode());

	armAsm->Sxtw(xlo, wres);            // xlo = (s64)(s32)(xres & 0xFFFFFFFF)
	armAsm->Asr(xres, xres, 32);        // xres = (s64)(s32)(xres >> 32)
	armAsm->Str(xlo, a64::MemOperand(RCPUSTATE, LO_OFFSET + hilo_off));
	armAsm->Str(xres, a64::MemOperand(RCPUSTATE, HI_OFFSET + hilo_off));

	if (rd_for_writeback != 0)
		armStoreGPR64(xlo, rd_for_writeback);
}

// Build the 64-bit MADD accumulator from the LOW 32 bits of HI/LO at hilo_off
// into the destination X register. Reads only LO_low32 and HI_low32 — matches
// interpreter semantics: temp = ((u64)HI.UL[off] << 32) | LO.UL[off].
static void emitLoadMaddAccumulator(const a64::Register& xacc, s64 hilo_off)
{
	const a64::Register wacc = a64::WRegister(xacc.GetCode());
	const a64::Register xtmp = RSCRATCHGPR2;
	const a64::Register wtmp = a64::WRegister(xtmp.GetCode());

	armAsm->Ldr(wacc, a64::MemOperand(RCPUSTATE, LO_OFFSET + hilo_off));   // xacc = zero-ext LO_low32
	armAsm->Ldr(wtmp, a64::MemOperand(RCPUSTATE, HI_OFFSET + hilo_off));   // xtmp = zero-ext HI_low32
	armAsm->Bfi(xacc, xtmp, 32, 32);                                       // xacc[63:32] = xtmp[31:0]
}

// ============================================================================
//  MULT — HI:LO = (s64)(s32)Rs * (s32)Rt;  if (rd) Rd = LO
// ============================================================================

#if ISTUB_MULT
void recMULT() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MULT); }
#else
void recMULT()
{
	armDelConstReg(_Rd_);

	armLoadGPR32(RWSCRATCH, _Rs_);
	armLoadGPR32(RWSCRATCH2, _Rt_);
	armAsm->Smull(RSCRATCHGPR, RWSCRATCH, RWSCRATCH2);
	emitMultWritebackHILO(RSCRATCHGPR, 0, _Rd_);
}
#endif

// ============================================================================
//  MULTU — HI:LO = (u64)(u32)Rs * (u32)Rt; HI/LO sign-extended; if (rd) Rd = LO
// ============================================================================

#if ISTUB_MULTU
void recMULTU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MULTU); }
#else
void recMULTU()
{
	armDelConstReg(_Rd_);

	armLoadGPR32(RWSCRATCH, _Rs_);
	armLoadGPR32(RWSCRATCH2, _Rt_);
	armAsm->Umull(RSCRATCHGPR, RWSCRATCH, RWSCRATCH2);
	emitMultWritebackHILO(RSCRATCHGPR, 0, _Rd_);
}
#endif

// ============================================================================
//  MULT1 — same as MULT but writes HI.SD[1] / LO.SD[1] (pipeline 1)
// ============================================================================

#if ISTUB_MULT1
void recMULT1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MULT1); }
#else
void recMULT1()
{
	armDelConstReg(_Rd_);

	armLoadGPR32(RWSCRATCH, _Rs_);
	armLoadGPR32(RWSCRATCH2, _Rt_);
	armAsm->Smull(RSCRATCHGPR, RWSCRATCH, RWSCRATCH2);
	emitMultWritebackHILO(RSCRATCHGPR, 8, _Rd_);
}
#endif

// ============================================================================
//  MULTU1 — same as MULTU but writes HI.SD[1] / LO.SD[1] (pipeline 1)
// ============================================================================

#if ISTUB_MULTU1
void recMULTU1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MULTU1); }
#else
void recMULTU1()
{
	armDelConstReg(_Rd_);

	armLoadGPR32(RWSCRATCH, _Rs_);
	armLoadGPR32(RWSCRATCH2, _Rt_);
	armAsm->Umull(RSCRATCHGPR, RWSCRATCH, RWSCRATCH2);
	emitMultWritebackHILO(RSCRATCHGPR, 8, _Rd_);
}
#endif

// ============================================================================
//  MADD — temp = ((u64)HI.UL[0]<<32 | LO.UL[0]) + (s64)(s32)Rs * (s32)Rt
//         HI:LO = sign-extended halves of temp; if (rd) Rd = LO
//  Note: only the LOW 32 bits of HI/LO feed the accumulator (matches interp).
// ============================================================================

#if ISTUB_MADD
void recMADD() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MADD); }
#else
void recMADD()
{
	armDelConstReg(_Rd_);

	emitLoadMaddAccumulator(RSCRATCHGPR, 0);                       // x4 = acc (x5 scratch dead after this)
	armLoadGPR32(RWSCRATCH2, _Rs_);                                // w5 = rs
	armLoadGPR32(RWSCRATCH3, _Rt_);                                // w6 = rt
	armAsm->Smaddl(RSCRATCHGPR, RWSCRATCH2, RWSCRATCH3, RSCRATCHGPR); // x4 = w5*w6 + x4
	emitMultWritebackHILO(RSCRATCHGPR, 0, _Rd_);
}
#endif

// ============================================================================
//  MADDU — same as MADD but unsigned multiply
// ============================================================================

#if ISTUB_MADDU
void recMADDU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MADDU); }
#else
void recMADDU()
{
	armDelConstReg(_Rd_);

	emitLoadMaddAccumulator(RSCRATCHGPR, 0);
	armLoadGPR32(RWSCRATCH2, _Rs_);
	armLoadGPR32(RWSCRATCH3, _Rt_);
	armAsm->Umaddl(RSCRATCHGPR, RWSCRATCH2, RWSCRATCH3, RSCRATCHGPR);
	emitMultWritebackHILO(RSCRATCHGPR, 0, _Rd_);
}
#endif

// ============================================================================
//  MADD1 — same as MADD but writes pipeline 1 (HI.SD[1] / LO.SD[1])
//  Reads accumulator from HI.UL[2] / LO.UL[2] (the LOW 32 bits of the upper
//  64-bit half), exactly mirroring the interpreter.
// ============================================================================

#if ISTUB_MADD1
void recMADD1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MADD1); }
#else
void recMADD1()
{
	armDelConstReg(_Rd_);

	emitLoadMaddAccumulator(RSCRATCHGPR, 8);
	armLoadGPR32(RWSCRATCH2, _Rs_);
	armLoadGPR32(RWSCRATCH3, _Rt_);
	armAsm->Smaddl(RSCRATCHGPR, RWSCRATCH2, RWSCRATCH3, RSCRATCHGPR);
	emitMultWritebackHILO(RSCRATCHGPR, 8, _Rd_);
}
#endif

// ============================================================================
//  MADDU1 — same as MADDU but writes pipeline 1 (HI.SD[1] / LO.SD[1])
// ============================================================================

#if ISTUB_MADDU1
void recMADDU1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MADDU1); }
#else
void recMADDU1()
{
	armDelConstReg(_Rd_);

	emitLoadMaddAccumulator(RSCRATCHGPR, 8);
	armLoadGPR32(RWSCRATCH2, _Rs_);
	armLoadGPR32(RWSCRATCH3, _Rt_);
	armAsm->Umaddl(RSCRATCHGPR, RWSCRATCH2, RWSCRATCH3, RSCRATCHGPR);
	emitMultWritebackHILO(RSCRATCHGPR, 8, _Rd_);
}
#endif

// ============================================================================
//  Divide helpers
// ============================================================================
//
// For DIV/DIVU we need to handle divide-by-zero (ARM64 SDIV/UDIV produce 0
// silently — but the PS2 wants specific values for HI/LO). The INT_MIN/-1
// case for SDIV is also defined by ARM (returns INT_MIN), and the resulting
// MSUB remainder works out to 0 — exactly what the interpreter wants — so
// no special branch is needed for it.

// emitSignedDivBody: w_rs and w_rt hold the operands (must be RWSCRATCH /
// RWSCRATCH2). On entry both registers must already be loaded. Stores the
// final HI/LO at the requested offset.
static void emitSignedDivBody(s64 hilo_off)
{
	a64::Label divzero;
	a64::Label done;

	armAsm->Cbz(RWSCRATCH2, &divzero);

	// Normal path: quotient = rs / rt, remainder = rs - quot * rt
	armAsm->Sdiv(RWSCRATCH3, RWSCRATCH, RWSCRATCH2);                    // w6 = rs / rt
	armAsm->Msub(RWSCRATCH, RWSCRATCH3, RWSCRATCH2, RWSCRATCH);         // w4 = rs - w6*rt (remainder)
	armAsm->Sxtw(RSCRATCHGPR3, RWSCRATCH3);                             // x6 = sign-ext quotient
	armAsm->Sxtw(RSCRATCHGPR, RWSCRATCH);                               // x4 = sign-ext remainder
	armAsm->Str(RSCRATCHGPR3, a64::MemOperand(RCPUSTATE, LO_OFFSET + hilo_off));
	armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, HI_OFFSET + hilo_off));
	armAsm->B(&done);

	armAsm->Bind(&divzero);
	// LO = (rs < 0) ? 1 : -1   (sign-extended to 64)
	// HI = sign_extend(rs)
	armAsm->Mov(RWSCRATCH3, 1);                                         // w6 = 1
	armAsm->Mov(RWARG1, -1);                                            // w0 = -1
	armAsm->Cmp(RWSCRATCH, 0);                                          // compare rs with 0
	armAsm->Csel(RWSCRATCH3, RWSCRATCH3, RWARG1, a64::lt);              // w6 = lt ? 1 : -1
	armAsm->Sxtw(RSCRATCHGPR3, RWSCRATCH3);
	armAsm->Sxtw(RSCRATCHGPR, RWSCRATCH);                               // sign-ext rs
	armAsm->Str(RSCRATCHGPR3, a64::MemOperand(RCPUSTATE, LO_OFFSET + hilo_off));
	armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, HI_OFFSET + hilo_off));

	armAsm->Bind(&done);
}

// emitUnsignedDivBody: same convention, unsigned variant.
static void emitUnsignedDivBody(s64 hilo_off)
{
	a64::Label divzero;
	a64::Label done;

	armAsm->Cbz(RWSCRATCH2, &divzero);

	armAsm->Udiv(RWSCRATCH3, RWSCRATCH, RWSCRATCH2);                    // w6 = rs / rt
	armAsm->Msub(RWSCRATCH, RWSCRATCH3, RWSCRATCH2, RWSCRATCH);         // w4 = rs - w6*rt
	armAsm->Sxtw(RSCRATCHGPR3, RWSCRATCH3);                             // sign-ext (s32) cast → s64
	armAsm->Sxtw(RSCRATCHGPR, RWSCRATCH);
	armAsm->Str(RSCRATCHGPR3, a64::MemOperand(RCPUSTATE, LO_OFFSET + hilo_off));
	armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, HI_OFFSET + hilo_off));
	armAsm->B(&done);

	armAsm->Bind(&divzero);
	// LO = -1 (sign-extended)
	// HI = sign_extend((s32)rs)
	armAsm->Mov(RSCRATCHGPR3, static_cast<u64>(-1));
	armAsm->Sxtw(RSCRATCHGPR, RWSCRATCH);
	armAsm->Str(RSCRATCHGPR3, a64::MemOperand(RCPUSTATE, LO_OFFSET + hilo_off));
	armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, HI_OFFSET + hilo_off));

	armAsm->Bind(&done);
}

// ============================================================================
//  DIV — signed 32-bit divide. HI:LO updated; Rd unused.
// ============================================================================

#if ISTUB_DIV
void recDIV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DIV); }
#else
void recDIV()
{
	armLoadGPR32(RWSCRATCH, _Rs_);
	armLoadGPR32(RWSCRATCH2, _Rt_);
	emitSignedDivBody(0);
}
#endif

// ============================================================================
//  DIVU — unsigned 32-bit divide.
// ============================================================================

#if ISTUB_DIVU
void recDIVU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DIVU); }
#else
void recDIVU()
{
	armLoadGPR32(RWSCRATCH, _Rs_);
	armLoadGPR32(RWSCRATCH2, _Rt_);
	emitUnsignedDivBody(0);
}
#endif

// ============================================================================
//  DIV1 — signed divide writing pipeline 1 (HI.SD[1] / LO.SD[1])
// ============================================================================

#if ISTUB_DIV1
void recDIV1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DIV1); }
#else
void recDIV1()
{
	armLoadGPR32(RWSCRATCH, _Rs_);
	armLoadGPR32(RWSCRATCH2, _Rt_);
	emitSignedDivBody(8);
}
#endif

// ============================================================================
//  DIVU1 — unsigned divide writing pipeline 1 (HI.SD[1] / LO.SD[1])
// ============================================================================

#if ISTUB_DIVU1
void recDIVU1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DIVU1); }
#else
void recDIVU1()
{
	armLoadGPR32(RWSCRATCH, _Rs_);
	armLoadGPR32(RWSCRATCH2, _Rt_);
	emitUnsignedDivBody(8);
}
#endif

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
