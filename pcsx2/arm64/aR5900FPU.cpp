// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — COP1 (FPU) Instructions
// MFC1, MTC1, CFC1, CTC1, BC1x, arithmetic, compare, convert

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"
#include "common/FPControl.h"

using namespace R5900;

// fpuRegisters field offsets from RCPUSTATE (x19) — fpuRegs lives at FPUREGS_BASE
// inside cpuRegistersPack, so we can address FPU state without a dedicated base reg.
static constexpr s64 FPR_OFFSET(int reg) { return FPUREGS_BASE + offsetof(fpuRegisters, fpr) + reg * sizeof(FPRreg); }
static constexpr s64 FPRC_OFFSET(int reg) { return FPUREGS_BASE + offsetof(fpuRegisters, fprc) + reg * sizeof(u32); }

// COP1 field aliases: _Fs_=bits 15:11, _Ft_=bits 20:16, _Fd_=bits 10:6
#define _Fs_cop1_ _Rd_
#define _Ft_cop1_ _Rt_
#define _Fd_cop1_ _Sa_

// FPU accumulator offset from RCPUSTATE
static constexpr s64 ACC_OFFSET = FPUREGS_BASE + offsetof(fpuRegisters, ACC);

// PS2 FPU constants
static constexpr u32 PS2_POS_FMAX  = 0x7F7FFFFF;
static constexpr u32 PS2_FPU_FLAG_C  = 0x00800000;
static constexpr u32 PS2_FPU_FLAG_I  = 0x00020000;
static constexpr u32 PS2_FPU_FLAG_D  = 0x00010000;
static constexpr u32 PS2_FPU_FLAG_O  = 0x00008000;
static constexpr u32 PS2_FPU_FLAG_U  = 0x00004000;
static constexpr u32 PS2_FPU_FLAG_SI = 0x00000040;
static constexpr u32 PS2_FPU_FLAG_SD = 0x00000020;
static constexpr u32 PS2_FPU_FLAG_SO = 0x00000010;
static constexpr u32 PS2_FPU_FLAG_SU = 0x00000008;

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
#if defined(INTERP_COP1) || defined(INTERP_EE)
#define ISTUB_MFC1     1
#define ISTUB_MTC1     1
#define ISTUB_CFC1     1
#define ISTUB_CTC1     1
#define ISTUB_BC1F     1
#define ISTUB_BC1T     1
#define ISTUB_BC1FL    1
#define ISTUB_BC1TL    1
#define ISTUB_ADD_S    1
#define ISTUB_ADDA_S   1
#define ISTUB_SUB_S    1
#define ISTUB_SUBA_S   1
#define ISTUB_ABS_S    1
#define ISTUB_MOV_S    1
#define ISTUB_NEG_S    1
#define ISTUB_MAX_S    1
#define ISTUB_MIN_S    1
#define ISTUB_MUL_S    1
#define ISTUB_DIV_S    1
#define ISTUB_SQRT_S   1
#define ISTUB_RSQRT_S  1
#define ISTUB_MULA_S   1
#define ISTUB_MADD_S   1
#define ISTUB_MSUB_S   1
#define ISTUB_MADDA_S  1
#define ISTUB_MSUBA_S  1
#define ISTUB_C_F      1
#define ISTUB_C_EQ     1
#define ISTUB_C_LT     1
#define ISTUB_C_LE     1
#define ISTUB_CVT_S    1
#define ISTUB_CVT_W    1
#else
#define ISTUB_MFC1     0
#define ISTUB_MTC1     0
#define ISTUB_CFC1     0
#define ISTUB_CTC1     0
#define ISTUB_BC1F     0
#define ISTUB_BC1T     0
#define ISTUB_BC1FL    0
#define ISTUB_BC1TL    0
#define ISTUB_ADD_S    0
#define ISTUB_ADDA_S   0
#define ISTUB_SUB_S    0
#define ISTUB_SUBA_S   0
#define ISTUB_ABS_S    0
#define ISTUB_MOV_S    0
#define ISTUB_NEG_S    0
#define ISTUB_MAX_S    0
#define ISTUB_MIN_S    0
#define ISTUB_MUL_S    0
#define ISTUB_DIV_S    0
#define ISTUB_SQRT_S   0
#define ISTUB_RSQRT_S  0
#define ISTUB_MULA_S   0
#define ISTUB_MADD_S   0
#define ISTUB_MSUB_S   0
#define ISTUB_MADDA_S  0
#define ISTUB_MSUBA_S  0
#define ISTUB_C_F      0
#define ISTUB_C_EQ     0
#define ISTUB_C_LT     0
#define ISTUB_C_LE     0
#define ISTUB_CVT_S    0
#define ISTUB_CVT_W    0
#endif

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {
namespace COP1 {

// ============================================================================
//  MFC1 — Move from FPU register
//  Interpreter: if (!_Rt_) return; GPR[rt].SD[0] = fpuRegs.fpr[fs].SL
//  (sign-extend 32-bit FPR to 64-bit GPR)
// ============================================================================

#if ISTUB_MFC1
void recMFC1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MFC1); }
#else
void recMFC1()
{
	if (!_Rt_)
		return;

	armDelConstReg(_Rt_);
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
}
#endif

// ============================================================================
//  MTC1 — Move to FPU register
//  Interpreter: fpuRegs.fpr[fs].UL = GPR[rt].UL[0]
//  (32-bit copy from GPR to FPR)
// ============================================================================

#if ISTUB_MTC1
void recMTC1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MTC1); }
#else
void recMTC1()
{
	// Load 32-bit from GPR[rt], store to fpr[fs]
	armLoadGPR32(RWSCRATCH, _Rt_);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fs_cop1_)));
}
#endif

// ============================================================================
//  CFC1 — Move from FPU control register
//  Interpreter:
//    if (!_Rt_) return;
//    if (fs == 31) GPR[rt].SD[0] = (s32)fprc[31]
//    else if (fs == 0) GPR[rt].SD[0] = 0x2E00
//    else GPR[rt].SD[0] = 0
// ============================================================================

#if ISTUB_CFC1
void recCFC1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::CFC1); }
#else
void recCFC1()
{
	if (!_Rt_)
		return;

	armDelConstReg(_Rt_);
	// PS2 EE FPU control register read behavior:
	// FCR0-15: all return FCR0 value (0x2E30 = implementation/revision)
	// FCR16-31: all mirror FCR31, masked and with forced bits
	if (_Fs_cop1_ >= 16)
	{
		// Load FCR31, apply read mask, set forced bits
		constexpr u32 FCR31_READ_MASK = 0x0083C078;
		constexpr u32 FCR31_FORCED_BITS = 0x01000001;
		armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
		armAsm->And(RWSCRATCH, RWSCRATCH, FCR31_READ_MASK);
		armAsm->Orr(RWSCRATCH, RWSCRATCH, FCR31_FORCED_BITS);
		armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
	}
	else
	{
		// FCR0-15 all return fprc[0] (implementation/revision register)
		armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(0)));
		armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
	}
}
#endif

// ============================================================================
//  CTC1 — Move to FPU control register
//  Interpreter: if (fs != 31) return; fprc[fs] = GPR[rt].UL[0]
// ============================================================================

#if ISTUB_CTC1
void recCTC1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::CTC1); }
#else
void recCTC1()
{
	if (_Fs_cop1_ != 31)
		return;

	// Store raw value; masking happens on read (CFC1)
	armLoadGPR32(RWSCRATCH, _Rt_);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
}
#endif

// ============================================================================
//  BC1F / BC1T / BC1FL / BC1TL — COP1 branches
//  Test FPU condition flag (bit 23 of fprc[31])
// ============================================================================

} // namespace COP1
} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900

extern void recompileNextInstruction(bool delayslot, bool swapped_delay_slot);

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {
namespace COP1 {

// Helper: non-likely BC1 branch.
// branchIfSet=false → BC1F (branch when flag clear),
// branchIfSet=true  → BC1T (branch when flag set).
static void recBC1_helper(bool branchIfSet)
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	// Test FPU condition flag
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
	armAsm->Tst(RWSCRATCH, PS2_FPU_FLAG_C);
	// Cset: 1 = taken. BC1F taken when flag clear (eq), BC1T when set (ne).
	armAsm->Cset(RDELAYSLOTGPR, branchIfSet ? a64::ne : a64::eq);

	// Pre-DS flush gated on DS safety (see armFlushConstRegsBeforeDS).
	armFlushConstRegsBeforeDS();
	recompileNextInstruction(true, false);
	armFlushConstRegs();

	armAsm->Mov(RWSCRATCH, branchTarget);
	armAsm->Mov(RWSCRATCH2, fallthrough);
	armAsm->Cmp(RDELAYSLOTGPR, 0);
	armAsm->Csel(RWSCRATCH, RWSCRATCH, RWSCRATCH2, a64::ne);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));

	g_branch = 1;
	g_cpuFlushedPC = true;
}

// Helper: likely BC1 branch (skip delay slot if not taken).
static void recBC1_Likely_helper(bool branchIfSet)
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	// Flush unconditionally up front — both paths exit the block, so any
	// unflushed const-tracked GPRs must hit memory regardless of which side
	// of the branch we take. Const tracking is preserved (only the flushed
	// bit is set), so subsequent armLoadGPR* in the delay slot still uses
	// Mov-imm for any const operands. Flushing before the Ldr also protects
	// RWSCRATCH (the W view of RSCRATCHGPR, which armFlushConstRegs uses
	// internally as scratch) so the Tbz/Tbnz below sees the FPU C flag.
	//
	// NB: armFlushConstRegsBeforeDS is NOT applicable here — the skip-DS
	// path exits the block without ever recompiling the DS, so the DS
	// safety gate doesn't help. The dirty-mask fast-path in
	// armFlushConstRegs makes the no-dirty case cheap anyway.
	armFlushConstRegs();

	// Test FPU condition flag (bit 23)
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));

	// If NOT taken, skip delay slot.
	// BC1FL not-taken when C SET → Tbnz; BC1TL not-taken when C CLEAR → Tbz.
	// Tbz/Tbnz on bit 23 replaces Tst + B(cond) — saves 1 instruction.
	a64::Label skipDelaySlot, done;
	if (branchIfSet)
		armAsm->Tbz(RWSCRATCH, 23, &skipDelaySlot);  // BC1TL: skip when C clear
	else
		armAsm->Tbnz(RWSCRATCH, 23, &skipDelaySlot); // BC1FL: skip when C set

	// Taken: execute delay slot, branch to target
	recompileNextInstruction(true, false);
	armFlushConstRegs();
	armAsm->Mov(RWSCRATCH, branchTarget);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	armAsm->B(&done);

	// Not taken: skip delay slot, PC = fallthrough
	armAsm->Bind(&skipDelaySlot);
	armAsm->Mov(RWSCRATCH, fallthrough);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));

	armAsm->Bind(&done);
	g_branch = 1;
	g_cpuFlushedPC = true;
}

#if ISTUB_BC1F
void recBC1F()  { armBranchInterpWithDSCycles(R5900::Interpreter::OpcodeImpl::COP1::BC1F); }
#else
void recBC1F()  { recBC1_helper(false); }
#endif

#if ISTUB_BC1T
void recBC1T()  { armBranchInterpWithDSCycles(R5900::Interpreter::OpcodeImpl::COP1::BC1T); }
#else
void recBC1T()  { recBC1_helper(true); }
#endif

#if ISTUB_BC1FL
void recBC1FL() { armBranchInterpWithDSCycles(R5900::Interpreter::OpcodeImpl::COP1::BC1FL); }
#else
void recBC1FL() { recBC1_Likely_helper(false); }
#endif

#if ISTUB_BC1TL
void recBC1TL() { armBranchInterpWithDSCycles(R5900::Interpreter::OpcodeImpl::COP1::BC1TL); }
#else
void recBC1TL() { recBC1_Likely_helper(true); }
#endif

// ============================================================================
//  FPU codegen helpers
// ============================================================================

// NEON/FP scratch registers (caller-saved)
#define RFSCRATCH0 a64::s0
#define RFSCRATCH1 a64::s1
#define RFSCRATCH2 a64::s2

// Emit PS2 FPU input clamping (fpuDouble equivalent):
//   denormal (exp==0) → ±0, inf/NaN (exp==0xFF) → ±Fmax
// Input: wSrc = raw FPR bits. Output: sDst = clamped float.
// Clobbers: wSrc, RWSCRATCH3
static void armFpuClampInput(const a64::Register& wSrc, const a64::VRegister& sDst)
{
	// Extract exponent field (bits 30:23)
	armAsm->Ubfx(RWSCRATCH3, wSrc, 23, 8);
	// If exponent == 0: flush to ±0 (keep sign, zero rest)
	a64::Label notDenorm, done;
	armAsm->Cbnz(RWSCRATCH3, &notDenorm);
	armAsm->And(wSrc, wSrc, 0x80000000);
	armAsm->B(&done);
	armAsm->Bind(&notDenorm);
	// If exponent == 0xFF: clamp to ±Fmax
	armAsm->Cmp(RWSCRATCH3, 0xFF);
	a64::Label notInf;
	armAsm->B(&notInf, a64::ne);
	armAsm->And(wSrc, wSrc, 0x80000000);
	armAsm->Mov(RWSCRATCH3, PS2_POS_FMAX);
	armAsm->Orr(wSrc, wSrc, RWSCRATCH3);
	armAsm->Bind(&notInf);
	armAsm->Bind(&done);
	armAsm->Fmov(sDst, wSrc);
}

// Emit PS2 FPU output clamping (checkOverflow + checkUnderflow):
//   infinity → ±Fmax, denormal → ±0
// Input: sDst has the result. Output: wDst = clamped result bits.
// Clobbers: RWSCRATCH3
static void armFpuClampOutput(const a64::VRegister& sSrc, const a64::Register& wDst)
{
	armAsm->Fmov(wDst, sSrc);
	armAsm->Ubfx(RWSCRATCH3, wDst, 23, 8);
	// Overflow: exp == 0xFF → ±Fmax
	armAsm->Cmp(RWSCRATCH3, 0xFF);
	a64::Label notOvf, checkUdf, storeDone;
	armAsm->B(&notOvf, a64::ne);
	armAsm->And(wDst, wDst, 0x80000000);
	armAsm->Mov(RWSCRATCH3, PS2_POS_FMAX);
	armAsm->Orr(wDst, wDst, RWSCRATCH3);
	armAsm->B(&storeDone);
	armAsm->Bind(&notOvf);
	// Underflow: exp == 0 && mantissa != 0 → ±0
	armAsm->Cbnz(RWSCRATCH3, &storeDone);
	armAsm->Tst(wDst, 0x007FFFFF);
	armAsm->B(&storeDone, a64::eq);
	armAsm->And(wDst, wDst, 0x80000000);
	armAsm->Bind(&storeDone);
}

// Emit: Msr FPCR, #bitmask — write FPCR to a compile-time-known value.
// Clobbers RSCRATCHGPR. Used to bracket SQRT / DIV / RSQRT whose PS2 semantics
// require a rounding mode different from the ambient FPUFPCR.
static void armEmitFpcrSet(u64 bitmask)
{
	armAsm->Mov(RSCRATCHGPR, bitmask);
	armAsm->Msr(a64::FPCR, RSCRATCHGPR);
}

// Emit a two-operand FPU arithmetic op: fd = clamp(op(clamp(fs), clamp(ft)))
// opFunc emits the actual instruction given (sDst, sSrc0, sSrc1)
template<typename OpFunc>
static void armFpuBinOp(int fd, int fs, int ft, OpFunc opFunc)
{
	// Load and clamp fs
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(fs)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	// Load and clamp ft
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, FPR_OFFSET(ft)));
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);
	// Operation
	opFunc(RFSCRATCH0, RFSCRATCH0, RFSCRATCH1);
	// Clamp output and store
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(fd)));
}

// Same but stores to ACC instead of fd
template<typename OpFunc>
static void armFpuBinOpAcc(int fs, int ft, OpFunc opFunc)
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(fs)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, FPR_OFFSET(ft)));
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);
	opFunc(RFSCRATCH0, RFSCRATCH0, RFSCRATCH1);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, ACC_OFFSET));
}

// ============================================================================
//  FPU arithmetic / compare / convert
// ============================================================================

// ---- ABS_S ----
#if ISTUB_ABS_S
void recABS_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::ABS_S); }
#else
void recABS_S()
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armAsm->And(RWSCRATCH, RWSCRATCH, 0x7FFFFFFF);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fd_cop1_)));
}
#endif

// ---- NEG_S ----
#if ISTUB_NEG_S
void recNEG_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::NEG_S); }
#else
void recNEG_S()
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armAsm->Eor(RWSCRATCH, RWSCRATCH, 0x80000000);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fd_cop1_)));
}
#endif

// ---- MOV_S ----
#if ISTUB_MOV_S
void recMOV_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MOV_S); }
#else
void recMOV_S()
{
	if (_Fs_cop1_ == _Fd_cop1_)
		return;
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fd_cop1_)));
}
#endif

// ---- MAX_S / MIN_S ----
// x86 iFPU.cpp:1390-1406 uses SSE MAXSS/MINSS via recCommutativeOp(op >= 2 path),
// which applies fpuFloat2 (PS2 NaN/Inf → ±Fmax clamp) to both operands before the
// SSE op. Our armFpuBinOp does equivalent input clamping + op + output clamping
// (the output clamp is redundant for MAX/MIN since the result is bounded by the
// already-clamped inputs, but it's harmless and keeps the code path uniform).
#if ISTUB_MAX_S
void recMAX_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MAX_S); }
#else
void recMAX_S()
{
	armFpuBinOp(_Fd_cop1_, _Fs_cop1_, _Ft_cop1_,
		[](auto d, auto a, auto b) { armAsm->Fmax(d, a, b); });
}
#endif

#if ISTUB_MIN_S
void recMIN_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MIN_S); }
#else
void recMIN_S()
{
	armFpuBinOp(_Fd_cop1_, _Fs_cop1_, _Ft_cop1_,
		[](auto d, auto a, auto b) { armAsm->Fmin(d, a, b); });
}
#endif

// ---- ADD_S ----
#if ISTUB_ADD_S
void recADD_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::ADD_S); }
#else
void recADD_S()
{
	armFpuBinOp(_Fd_cop1_, _Fs_cop1_, _Ft_cop1_,
		[](auto d, auto a, auto b) { armAsm->Fadd(d, a, b); });
}
#endif

// ---- SUB_S ----
#if ISTUB_SUB_S
void recSUB_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::SUB_S); }
#else
void recSUB_S()
{
	armFpuBinOp(_Fd_cop1_, _Fs_cop1_, _Ft_cop1_,
		[](auto d, auto a, auto b) { armAsm->Fsub(d, a, b); });
}
#endif

// ---- MUL_S ----
#if ISTUB_MUL_S
void recMUL_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MUL_S); }
#else
void recMUL_S()
{
	armFpuBinOp(_Fd_cop1_, _Fs_cop1_, _Ft_cop1_,
		[](auto d, auto a, auto b) { armAsm->Fmul(d, a, b); });
}
#endif

// ---- ADDA_S ----
#if ISTUB_ADDA_S
void recADDA_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::ADDA_S); }
#else
void recADDA_S()
{
	armFpuBinOpAcc(_Fs_cop1_, _Ft_cop1_,
		[](auto d, auto a, auto b) { armAsm->Fadd(d, a, b); });
}
#endif

// ---- SUBA_S ----
#if ISTUB_SUBA_S
void recSUBA_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::SUBA_S); }
#else
void recSUBA_S()
{
	armFpuBinOpAcc(_Fs_cop1_, _Ft_cop1_,
		[](auto d, auto a, auto b) { armAsm->Fsub(d, a, b); });
}
#endif

// ---- MULA_S ----
#if ISTUB_MULA_S
void recMULA_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MULA_S); }
#else
void recMULA_S()
{
	armFpuBinOpAcc(_Fs_cop1_, _Ft_cop1_,
		[](auto d, auto a, auto b) { armAsm->Fmul(d, a, b); });
}
#endif

// ---- MADD_S ----
#if ISTUB_MADD_S
void recMADD_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MADD_S); }
#else
void recMADD_S()
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Ft_cop1_)));
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);
	armAsm->Fmul(RFSCRATCH0, RFSCRATCH0, RFSCRATCH1);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, ACC_OFFSET));
	armFpuClampInput(RWSCRATCH, RFSCRATCH1);
	armAsm->Fadd(RFSCRATCH0, RFSCRATCH1, RFSCRATCH0);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fd_cop1_)));
}
#endif

// ---- MSUB_S ----
#if ISTUB_MSUB_S
void recMSUB_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MSUB_S); }
#else
void recMSUB_S()
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Ft_cop1_)));
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);
	armAsm->Fmul(RFSCRATCH0, RFSCRATCH0, RFSCRATCH1);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, ACC_OFFSET));
	armFpuClampInput(RWSCRATCH, RFSCRATCH1);
	armAsm->Fsub(RFSCRATCH0, RFSCRATCH1, RFSCRATCH0);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fd_cop1_)));
}
#endif

// ---- MADDA_S ----
// ACC = clamp(clamp(fs * ft) + clamp(ACC)). Mirrors recMADD_S's clamp discipline;
// earlier revision skipped the intermediate-product clamp and the ACC-clamp,
// making MADDA's rounding inconsistent with MADD. x86 recMADDtemp clamps both
// the product and ACC (via fpuFloat2/fpuFloat) before the FPU_ADD regardless of
// whether the result goes to Fd (MADD) or ACC (MADDA), so we match.
#if ISTUB_MADDA_S
void recMADDA_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MADDA_S); }
#else
void recMADDA_S()
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Ft_cop1_)));
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);
	armAsm->Fmul(RFSCRATCH0, RFSCRATCH0, RFSCRATCH1);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, ACC_OFFSET));
	armFpuClampInput(RWSCRATCH, RFSCRATCH1);
	armAsm->Fadd(RFSCRATCH0, RFSCRATCH1, RFSCRATCH0);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, ACC_OFFSET));
}
#endif

// ---- MSUBA_S ----
// ACC = clamp(clamp(ACC) - clamp(fs * ft)). Same clamp-discipline fix as MADDA.
#if ISTUB_MSUBA_S
void recMSUBA_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::MSUBA_S); }
#else
void recMSUBA_S()
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Ft_cop1_)));
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);
	armAsm->Fmul(RFSCRATCH0, RFSCRATCH0, RFSCRATCH1);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, ACC_OFFSET));
	armFpuClampInput(RWSCRATCH, RFSCRATCH1);
	armAsm->Fsub(RFSCRATCH0, RFSCRATCH1, RFSCRATCH0);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, ACC_OFFSET));
}
#endif

// ---- DIV_S ----
// Matches x86 recDIV_S_xmm + recDIVhelper1 (iFPU.cpp:1037-1169). Flow:
//   if (FPUFPCR != FPUDivFPCR): switch FPCR to FPUDivFPCR
//   clear I|D flags
//   clamp fs, ft
//   if (ft == 0):
//     if (fs == 0): set I|SI (0/0 → indeterminate)
//     else:         set D|SD (x/0 → divide by zero)
//     fd = sign(fs XOR ft) | POS_FMAX
//   else:
//     fd = clamp(fs / ft)
//   if (FPUFPCR != FPUDivFPCR): restore FPCR to FPUFPCR
//
// FPCR bracketing: PS2 DIV uses the FPUDivFPCR rounding mode (default Nearest),
// which by default differs from FPUFPCR (ChopZero). The x86 JIT brackets with
// xLDMXCSR; we mirror with Msr FPCR. Check at emit time so we don't pay the
// cost when the user's config has identical FPUFPCR/FPUDivFPCR.
//
// SI/SD are sticky — x86 clears only I|D up front, preserving previous sticky
// bits. We do the same (Bic against I|D only).
#if ISTUB_DIV_S
void recDIV_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::DIV_S); }
#else
void recDIV_S()
{
	const bool fpcr_switch = (EmuConfig.Cpu.FPUFPCR.bitmask != EmuConfig.Cpu.FPUDivFPCR.bitmask);

	// Switch FPCR to FPUDivFPCR before the Fdiv. Bracket is placed around the
	// whole op rather than just the Fdiv because flag-path stores and integer
	// ops are FPCR-independent anyway, and emitting it once avoids duplication
	// between the normal and div-by-zero paths.
	if (fpcr_switch)
		armEmitFpcrSet(EmuConfig.Cpu.FPUDivFPCR.bitmask);

	// Clear I and D flags
	armAsm->Ldr(RWSCRATCH3, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
	armAsm->Bic(RWSCRATCH3, RWSCRATCH3, PS2_FPU_FLAG_I | PS2_FPU_FLAG_D);
	armAsm->Str(RWSCRATCH3, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));

	// Load and clamp fs
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	// RWSCRATCH now holds clamped-fs bits; RFSCRATCH0 holds clamped-fs float.

	// Load and clamp ft
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Ft_cop1_)));
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);
	// RWSCRATCH2 now holds clamped-ft bits; RFSCRATCH1 holds clamped-ft float.

	// Test ft == 0 (compare bits to zero — after clamp, ±0 both have mantissa 0
	// and sign the only non-zero bit, so a non-zero bit-pattern means non-zero).
	// Use Tst to mask off the sign bit, then check if the remaining bits are 0.
	armAsm->Tst(RWSCRATCH2, 0x7FFFFFFFu);
	a64::Label divByZero, done;
	armAsm->B(&divByZero, a64::eq);

	// Normal divide: fd = clamp(fs / ft)
	armAsm->Fdiv(RFSCRATCH0, RFSCRATCH0, RFSCRATCH1);
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fd_cop1_)));
	armAsm->B(&done);

	// Divide-by-zero
	armAsm->Bind(&divByZero);
	{
		// Determine flags: fs == 0 → I|SI (0/0), else D|SD (x/0)
		armAsm->Tst(RWSCRATCH, 0x7FFFFFFFu);
		a64::Label zeroZero, flagsDone;
		armAsm->B(&zeroZero, a64::eq);

		// x/0 → D|SD
		armAsm->Ldr(RWSCRATCH3, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
		armAsm->Orr(RWSCRATCH3, RWSCRATCH3, PS2_FPU_FLAG_D | PS2_FPU_FLAG_SD);
		armAsm->Str(RWSCRATCH3, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
		armAsm->B(&flagsDone);

		// 0/0 → I|SI
		armAsm->Bind(&zeroZero);
		armAsm->Ldr(RWSCRATCH3, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
		armAsm->Orr(RWSCRATCH3, RWSCRATCH3, PS2_FPU_FLAG_I | PS2_FPU_FLAG_SI);
		armAsm->Str(RWSCRATCH3, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));

		armAsm->Bind(&flagsDone);

		// Result = sign(fs XOR ft) | POS_FMAX
		armAsm->Eor(RWSCRATCH, RWSCRATCH, RWSCRATCH2);
		armAsm->And(RWSCRATCH, RWSCRATCH, 0x80000000u);
		armAsm->Mov(RWSCRATCH3, PS2_POS_FMAX);
		armAsm->Orr(RWSCRATCH, RWSCRATCH, RWSCRATCH3);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fd_cop1_)));
	}

	armAsm->Bind(&done);

	// Restore FPCR if we switched.
	if (fpcr_switch)
		armEmitFpcrSet(EmuConfig.Cpu.FPUFPCR.bitmask);
}
#endif

// ---- SQRT_S ----
// Matches x86 recSQRT_S_xmm (iFPU.cpp:1739-1783). Flow:
//   if (FPUFPCR round != Nearest): switch FPCR to FPUFPCR-with-Nearest
//   clear I|D
//   if (ft < 0): set I|SI; ft = abs(ft)
//   clamp ft (positive side only needed since sign is cleared)
//   fd = sqrt(ft)
//   if (FPUFPCR round != Nearest): restore FPCR to FPUFPCR
//
// PS2 SQRT is always round-to-nearest regardless of FCR31. x86 brackets with
// xLDMXCSR around the SQRTSS; we mirror with Msr FPCR. The bracket is gated
// at emit time — if the user's FPUFPCR already rounds to nearest (rare —
// default is ChopZero), skip the switch.
#if ISTUB_SQRT_S
void recSQRT_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::SQRT_S); }
#else
void recSQRT_S()
{
	const bool fpcr_switch = (EmuConfig.Cpu.FPUFPCR.GetRoundMode() != FPRoundMode::Nearest);
	FPControlRegister nearest_fpcr = EmuConfig.Cpu.FPUFPCR;
	nearest_fpcr.SetRoundMode(FPRoundMode::Nearest);

	if (fpcr_switch)
		armEmitFpcrSet(nearest_fpcr.bitmask);

	// Clear I and D flags
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
	armAsm->Bic(RWSCRATCH2, RWSCRATCH2, PS2_FPU_FLAG_I | PS2_FPU_FLAG_D);
	armAsm->Str(RWSCRATCH2, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));

	// Load ft bits
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Ft_cop1_)));

	// If negative, set I|SI flags and take abs
	a64::Label notNeg;
	armAsm->Tbz(RWSCRATCH, 31, &notNeg);
	{
		armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
		armAsm->Orr(RWSCRATCH2, RWSCRATCH2, PS2_FPU_FLAG_I | PS2_FPU_FLAG_SI);
		armAsm->Str(RWSCRATCH2, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
		armAsm->And(RWSCRATCH, RWSCRATCH, 0x7FFFFFFFu);
	}
	armAsm->Bind(&notNeg);

	// Clamp (inf/NaN → Fmax, denormal → 0) and sqrt
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Fsqrt(RFSCRATCH0, RFSCRATCH0);

	// No output clamp needed: sqrt(Fmax) ≈ 1.8e19 is finite and normal.
	armAsm->Str(RFSCRATCH0, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fd_cop1_)));

	if (fpcr_switch)
		armEmitFpcrSet(EmuConfig.Cpu.FPUFPCR.bitmask);
}
#endif

// ---- RSQRT_S ----
// Matches x86 DOUBLE::recRSQRT_S_xmm's bracketing (iFPUd.cpp:1005-1027).
// The fast-path iFPU.cpp version skips FPCR bracketing because it uses RSQRTSS
// (an approximation that ignores MXCSR rounding). We don't have an equivalent
// "always-round-to-nearest" SQRT-and-divide on arm64 — Fsqrt/Fdiv respect FPCR
// — so we follow the DOUBLE variant's lead and force FPCR to Nearest around
// the whole op.
//
// Flow:
//   if (FPUFPCR round != Nearest): switch FPCR to FPUFPCR-with-Nearest
//   clear I|D
//   if (ft < 0): set I|SI; ft = abs(ft)
//   if (ft == 0):
//     if (fs == 0): set I|SI (0/0)
//     else:         set D|SD (x/0)
//     fd = sign(fs) | POS_FMAX
//   else:
//     fd = clamp(fs / sqrt(clamp(ft)))
//   if (FPUFPCR round != Nearest): restore FPCR
#if ISTUB_RSQRT_S
void recRSQRT_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::RSQRT_S); }
#else
void recRSQRT_S()
{
	const bool fpcr_switch = (EmuConfig.Cpu.FPUFPCR.GetRoundMode() != FPRoundMode::Nearest);
	FPControlRegister nearest_fpcr = EmuConfig.Cpu.FPUFPCR;
	nearest_fpcr.SetRoundMode(FPRoundMode::Nearest);

	if (fpcr_switch)
		armEmitFpcrSet(nearest_fpcr.bitmask);

	// Clear I and D flags
	armAsm->Ldr(RWSCRATCH3, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
	armAsm->Bic(RWSCRATCH3, RWSCRATCH3, PS2_FPU_FLAG_I | PS2_FPU_FLAG_D);
	armAsm->Str(RWSCRATCH3, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));

	// Load fs bits (into RWSCRATCH) and ft bits (into RWSCRATCH2)
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Ft_cop1_)));

	// If ft negative: set I|SI, clear sign (abs)
	a64::Label ftNotNeg;
	armAsm->Tbz(RWSCRATCH2, 31, &ftNotNeg);
	{
		armAsm->Ldr(RWSCRATCH3, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
		armAsm->Orr(RWSCRATCH3, RWSCRATCH3, PS2_FPU_FLAG_I | PS2_FPU_FLAG_SI);
		armAsm->Str(RWSCRATCH3, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
		armAsm->And(RWSCRATCH2, RWSCRATCH2, 0x7FFFFFFFu);
	}
	armAsm->Bind(&ftNotNeg);

	// Check ft == 0 (magnitude bits all zero after sign clear)
	armAsm->Tst(RWSCRATCH2, 0x7FFFFFFFu);
	a64::Label divByZero, done;
	armAsm->B(&divByZero, a64::eq);

	// Normal RSQRT: fd = clamp(fs / sqrt(clamp(ft)))
	armFpuClampInput(RWSCRATCH2, RFSCRATCH1);         // clamp(ft) → s1
	armAsm->Fsqrt(RFSCRATCH1, RFSCRATCH1);            // s1 = sqrt(clamp(ft))
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);          // clamp(fs) → s0
	armAsm->Fdiv(RFSCRATCH0, RFSCRATCH0, RFSCRATCH1); // s0 = fs / sqrt(ft)
	armFpuClampOutput(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fd_cop1_)));
	armAsm->B(&done);

	// Divide-by-zero (ft == 0)
	armAsm->Bind(&divByZero);
	{
		// Determine flags: fs == 0 → I|SI (0/0), else D|SD (x/0).
		// Note: x86 checks fs raw bits here — we match. If ft was negative
		// (I|SI already set above), the 0/0 branch re-ORs I|SI which is a no-op.
		armAsm->Tst(RWSCRATCH, 0x7FFFFFFFu);
		a64::Label zeroZero, flagsDone;
		armAsm->B(&zeroZero, a64::eq);

		// x/0 → D|SD
		armAsm->Ldr(RWSCRATCH3, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
		armAsm->Orr(RWSCRATCH3, RWSCRATCH3, PS2_FPU_FLAG_D | PS2_FPU_FLAG_SD);
		armAsm->Str(RWSCRATCH3, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
		armAsm->B(&flagsDone);

		// 0/0 → I|SI
		armAsm->Bind(&zeroZero);
		armAsm->Ldr(RWSCRATCH3, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
		armAsm->Orr(RWSCRATCH3, RWSCRATCH3, PS2_FPU_FLAG_I | PS2_FPU_FLAG_SI);
		armAsm->Str(RWSCRATCH3, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));

		armAsm->Bind(&flagsDone);

		// Result = sign(fs) | POS_FMAX (x86 doesn't XOR with ft here since ft
		// was already made non-negative by the abs above).
		armAsm->And(RWSCRATCH, RWSCRATCH, 0x80000000u);
		armAsm->Mov(RWSCRATCH3, PS2_POS_FMAX);
		armAsm->Orr(RWSCRATCH, RWSCRATCH, RWSCRATCH3);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fd_cop1_)));
	}

	armAsm->Bind(&done);

	if (fpcr_switch)
		armEmitFpcrSet(EmuConfig.Cpu.FPUFPCR.bitmask);
}
#endif

// ---- C_F ----
#if ISTUB_C_F
void recC_F() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::C_F); }
#else
void recC_F()
{
	// Bic clears specific bits: 0x00800000 is a valid ARM64 logical immediate,
	// ~0x00800000 is not — Bic saves the scratch+And pair.
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
	armAsm->Bic(RWSCRATCH, RWSCRATCH, PS2_FPU_FLAG_C);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
}
#endif

// ---- C_EQ / C_LT / C_LE ----
// Matches x86 iFPU.cpp:721-796 (C_EQ_xmm), 808-885 (C_LE_xmm), 888-965 (C_LT_xmm):
// clamp both operands (Inf/NaN → ±Fmax, denormals → ±0) then compare, set the
// FPU C flag (bit 23 of fprc[31]) based on the result.
//
// Previous arm64 revision used a sign-magnitude→two's-complement integer-key
// compare. It was clever but had edge cases: Inf/NaN bit patterns have
// exponent=0xFF which compared as extremely-large integers — correct for +Inf
// but wrong if a NaN slipped through (PS2 rarely produces NaN but VU interop
// can). Clamping up front normalizes to ±Fmax so Fcmp's ordered-compare is
// semantically equivalent to x86's clamp+UCOMISS without the edge cases.
//
// After armFpuClampInput, operands are guaranteed finite ±0..±Fmax, so the
// unordered (NaN) outcome of Fcmp can't occur — ordered eq/lt/le set NZCV
// unambiguously and Cset reads them directly.
static void armFpuCompare(a64::Condition cond)
{
	// Load and clamp fs → s0 (uses RWSCRATCH as intermediate)
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);

	// Load and clamp ft → s1
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Ft_cop1_)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH1);

	// Ordered float compare (no NaN possible after clamp).
	armAsm->Fcmp(RFSCRATCH0, RFSCRATCH1);

	// Load fprc[31], clear C, OR in Cset(cond) shifted to bit 23.
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
	armAsm->Bic(RWSCRATCH, RWSCRATCH, PS2_FPU_FLAG_C);
	armAsm->Cset(RWSCRATCH2, cond);
	armAsm->Lsl(RWSCRATCH2, RWSCRATCH2, 23);
	armAsm->Orr(RWSCRATCH, RWSCRATCH, RWSCRATCH2);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPRC_OFFSET(31)));
}

#if ISTUB_C_EQ
void recC_EQ() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::C_EQ); }
#else
void recC_EQ() { armFpuCompare(a64::eq); }
#endif

#if ISTUB_C_LT
void recC_LT() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::C_LT); }
#else
void recC_LT() { armFpuCompare(a64::lt); }
#endif

#if ISTUB_C_LE
void recC_LE() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::C_LE); }
#else
void recC_LE() { armFpuCompare(a64::le); }
#endif

// ---- CVT_S ----
#if ISTUB_CVT_S
void recCVT_S() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::CVT_S); }
#else
void recCVT_S()
{
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armAsm->Scvtf(RFSCRATCH0, RWSCRATCH);
	armAsm->Str(RFSCRATCH0, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fd_cop1_)));
}
#endif

// ---- CVT_W ----
#if ISTUB_CVT_W
void recCVT_W() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP1::CVT_W); }
#else
void recCVT_W()
{
	// Clamp input (denormal → ±0, inf/NaN → ±Fmax), then convert to s32.
	// ARM64 FCVTZS saturates: >MAX→0x7FFFFFFF, <MIN→0x80000000
	// which matches PS2 CVT_W overflow behavior for clamped inputs.
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fs_cop1_)));
	armFpuClampInput(RWSCRATCH, RFSCRATCH0);
	armAsm->Fcvtzs(RWSCRATCH, RFSCRATCH0);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, FPR_OFFSET(_Fd_cop1_)));
}
#endif

} // namespace COP1
} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
