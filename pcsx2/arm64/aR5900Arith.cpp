// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — Arithmetic family.
// Combines the original iR5900{ALU,Move,Shift}_arm64.cpp into a single
// translation unit that mirrors the macOS-port arm64/aR5900Arith.cpp file
// boundary. The three op classes were always disjoint static-helper-wise
// (no name collisions), so this is a pure concatenation — same code, one
// file. Per-op ISTUB_* / INTERP_ALU / INTERP_MOVE / INTERP_SHIFT bisect
// toggles still live where they were.
//
// Coverage (matches mac aR5900Arith.cpp + Android-port extensions):
//   ALU:   ADD/ADDU/SUB/SUBU, ADDI/ADDIU, DADD/DADDU/DSUB/DSUBU,
//          DADDI/DADDIU, AND/OR/XOR/NOR, ANDI/ORI/XORI, SLT/SLTU,
//          SLTI/SLTIU
//   Move:  LUI, MFHI/MFLO, MTHI/MTLO, MFHI1/MFLO1, MTHI1/MTLO1,
//          MOVZ, MOVN, MFSA, MTSA, MTSAB, MTSAH
//   Shift: SLL/SRL/SRA, SLLV/SRLV/SRAV, DSLL/DSRL/DSRA, DSLL32/DSRL32/
//          DSRA32, DSLLV/DSRLV/DSRAV

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"

using namespace R5900;

// ============================================================================
// === Begin iR5900ALU_arm64.cpp body =========================================
// ============================================================================


using namespace R5900;

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
//
// The overflow-trapping variants (ADD/ADDI/SUB/DADD/DADDI/DSUB) alias directly
// to their non-trapping -U counterparts at the bottom of this file, matching
// x86 JIT behavior — upstream silently wraps on overflow and never raises
// cpuException from JIT-compiled code. So there are no ISTUB_* toggles for the
// signed variants; they follow whatever ISTUB_*U says.
#if defined(INTERP_ALU) || defined(INTERP_EE)
#define ISTUB_ADDU     1
#define ISTUB_SUBU     1
#define ISTUB_ADDIU    1
#define ISTUB_DADDU    1
#define ISTUB_DSUBU    1
#define ISTUB_DADDIU   1
#define ISTUB_AND      1
#define ISTUB_OR       1
#define ISTUB_XOR      1
#define ISTUB_NOR      1
#define ISTUB_ANDI     1
#define ISTUB_ORI      1
#define ISTUB_XORI     1
#define ISTUB_SLT      1
#define ISTUB_SLTU     1
#define ISTUB_SLTI     1
#define ISTUB_SLTIU    1
#else
#define ISTUB_ADDU     0
#define ISTUB_SUBU     0
#define ISTUB_ADDIU    0
#define ISTUB_DADDU    0
#define ISTUB_DSUBU    0
#define ISTUB_DADDIU   0
#define ISTUB_AND      0
#define ISTUB_OR       0
#define ISTUB_XOR      0
#define ISTUB_NOR      0
#define ISTUB_ANDI     0
#define ISTUB_ORI      0
#define ISTUB_XORI     0
#define ISTUB_SLT      0
#define ISTUB_SLTU     0
#define ISTUB_SLTI     0
#define ISTUB_SLTIU    0
#endif

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ============================================================================
//  ADDU — rd = sign_extend_32(rs + rt)  [no overflow trap]
//  Interpreter: GPR[rd].UD[0] = u64(s64(s32(GPR[rs].UL[0] + GPR[rt].UL[0])))
// ============================================================================

#if ISTUB_ADDU
void recADDU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::ADDU); }
#else
void recADDU()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] =
			(s64)(s32)(g_cpuConstRegs[_Rs_].UL[0] + g_cpuConstRegs[_Rt_].UL[0]);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rs_ && !_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rt_)
	{
		auto rs = armGprAlloc(_Rs_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Sxtw(rd, rs.W());
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Sxtw(rd, rt.W());
		return;
	}
	if (_Rs_ == _Rt_)
	{
		auto rs = armGprAlloc(_Rs_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Add(rd.W(), rs.W(), rs.W());
		armAsm->Sxtw(rd, rd.W());
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Add(rd.W(), rs.W(), rt.W());
	armAsm->Sxtw(rd, rd.W());
}
#endif

// ============================================================================
//  SUBU — rd = sign_extend_32(rs - rt)  [no overflow trap]
//  Interpreter: GPR[rd].UD[0] = u64(s64(s32(GPR[rs].UL[0] - GPR[rt].UL[0])))
// ============================================================================

#if ISTUB_SUBU
void recSUBU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SUBU); }
#else
void recSUBU()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] =
			(s64)(s32)(g_cpuConstRegs[_Rs_].UL[0] - g_cpuConstRegs[_Rt_].UL[0]);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (_Rs_ == _Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rt_)
	{
		auto rs = armGprAlloc(_Rs_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Sxtw(rd, rs.W());
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Neg(rd.W(), rt.W());
		armAsm->Sxtw(rd, rd.W());
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Sub(rd.W(), rs.W(), rt.W());
	armAsm->Sxtw(rd, rd.W());
}
#endif

// ============================================================================
//  ADDIU — rt = sign_extend_32(rs + sign_extend(imm16))  [no overflow trap]
//  Interpreter: GPR[rt].UD[0] = u64(s64(s32(GPR[rs].UL[0] + u32(s32(_Imm_)))))
// ============================================================================

#if ISTUB_ADDIU
void recADDIU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::ADDIU); }
#else
void recADDIU()
{
	if (!_Rt_)
		return;

	if (GPR_IS_CONST1(_Rs_))
	{
		g_cpuConstRegs[_Rt_].SD[0] =
			(s64)(s32)(g_cpuConstRegs[_Rs_].UL[0] + (u32)(s32)_Imm_);
		GPR_SET_CONST(_Rt_);
		return;
	}

	armDelConstReg(_Rt_);

	const s32 imm = _Imm_;
	if (imm == 0)
	{
		auto rs = armGprAlloc(_Rs_, false);
		auto rt = armGprAlloc(_Rt_, true);
		armAsm->Sxtw(rt, rs.W());
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, true);
		armAsm->Mov(rt, static_cast<u64>(static_cast<s64>(imm)));
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, true);
	if (imm > 0)
		armAsm->Add(rt.W(), rs.W(), imm);
	else
		armAsm->Sub(rt.W(), rs.W(), -imm);
	armAsm->Sxtw(rt, rt.W());
}
#endif

// ============================================================================
//  DADDU — rd = rs + rt  (64-bit, no overflow trap)
//  Interpreter: GPR[rd].UD[0] = GPR[rs].UD[0] + GPR[rt].UD[0]
// ============================================================================

#if ISTUB_DADDU
void recDADDU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DADDU); }
#else
void recDADDU()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] =
			g_cpuConstRegs[_Rs_].UD[0] + g_cpuConstRegs[_Rt_].UD[0];
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rs_ && !_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rt_)
	{
		auto rs = armGprAlloc(_Rs_, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (rd.GetCode() != rs.GetCode())
			armAsm->Mov(rd, rs);
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (rd.GetCode() != rt.GetCode())
			armAsm->Mov(rd, rt);
		return;
	}
	if (_Rs_ == _Rt_)
	{
		auto rs = armGprAlloc(_Rs_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Add(rd, rs, rs);
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Add(rd, rs, rt);
}
#endif

// ============================================================================
//  DSUBU — rd = rs - rt  (64-bit, no overflow trap)
//  Interpreter: GPR[rd].UD[0] = GPR[rs].UD[0] - GPR[rt].UD[0]
// ============================================================================

#if ISTUB_DSUBU
void recDSUBU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSUBU); }
#else
void recDSUBU()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] =
			g_cpuConstRegs[_Rs_].UD[0] - g_cpuConstRegs[_Rt_].UD[0];
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (_Rs_ == _Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rt_)
	{
		auto rs = armGprAlloc(_Rs_, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (rd.GetCode() != rs.GetCode())
			armAsm->Mov(rd, rs);
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Neg(rd, rt);
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Sub(rd, rs, rt);
}
#endif

// ============================================================================
//  DADDIU — rt = rs + sign_extend(imm16)  (64-bit, no overflow trap)
//  Interpreter: GPR[rt].UD[0] = GPR[rs].UD[0] + u64(s64(_Imm_))
// ============================================================================

#if ISTUB_DADDIU
void recDADDIU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DADDIU); }
#else
void recDADDIU()
{
	if (!_Rt_)
		return;

	if (GPR_IS_CONST1(_Rs_))
	{
		g_cpuConstRegs[_Rt_].UD[0] =
			g_cpuConstRegs[_Rs_].UD[0] + (u64)(s64)(s32)_Imm_;
		GPR_SET_CONST(_Rt_);
		return;
	}

	armDelConstReg(_Rt_);

	const s32 imm = _Imm_;
	if (imm == 0)
	{
		auto rs = armGprAlloc(_Rs_, false);
		auto rt = armGprAlloc(_Rt_, true);
		if (rt.GetCode() != rs.GetCode())
			armAsm->Mov(rt, rs);
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, true);
		armAsm->Mov(rt, static_cast<u64>(static_cast<s64>(imm)));
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, true);
	if (imm > 0)
		armAsm->Add(rt, rs, imm);
	else
		armAsm->Sub(rt, rs, -imm);
}
#endif

// ============================================================================
//  AND / OR / XOR / NOR — 64-bit logical, rd = rs OP rt
// ============================================================================

#if ISTUB_AND
void recAND() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::AND); }
#else
void recAND()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] =
			g_cpuConstRegs[_Rs_].UD[0] & g_cpuConstRegs[_Rt_].UD[0];
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rs_ || !_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	// Single-operand const fold (mirrors x86 recLogicalOp_constv for AND:
	// absorbing=0, identity=-1).
	if (GPR_IS_CONST1(_Rs_) || GPR_IS_CONST1(_Rt_))
	{
		const u32 creg = GPR_IS_CONST1(_Rs_) ? _Rs_ : _Rt_;
		const u32 vreg = GPR_IS_CONST1(_Rs_) ? _Rt_ : _Rs_;
		const u64 cval = g_cpuConstRegs[creg].UD[0];

		if (cval == 0)
		{
			armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
			return;
		}
		auto rv = armGprAlloc(vreg, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (cval == ~static_cast<u64>(0))
		{
			if (rd.GetCode() != rv.GetCode())
				armAsm->Mov(rd, rv);
			return;
		}
		armAsm->And(rd, rv, cval);
		return;
	}

	if (_Rs_ == _Rt_)
	{
		auto rs = armGprAlloc(_Rs_, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (rd.GetCode() != rs.GetCode())
			armAsm->Mov(rd, rs);
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->And(rd, rs, rt);
}
#endif

#if ISTUB_OR
void recOR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::OR); }
#else
void recOR()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] =
			g_cpuConstRegs[_Rs_].UD[0] | g_cpuConstRegs[_Rt_].UD[0];
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rs_ && !_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_ || _Rs_ == _Rt_)
	{
		// 0 | rt = rt, or rs | rs = rs (MIPS "move" idiom)
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (rd.GetCode() != rt.GetCode())
			armAsm->Mov(rd, rt);
		return;
	}
	if (!_Rt_)
	{
		auto rs = armGprAlloc(_Rs_, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (rd.GetCode() != rs.GetCode())
			armAsm->Mov(rd, rs);
		return;
	}

	// Single-operand const fold (mirrors x86 recLogicalOp_constv for OR:
	// absorbing=-1, identity=0).
	if (GPR_IS_CONST1(_Rs_) || GPR_IS_CONST1(_Rt_))
	{
		const u32 creg = GPR_IS_CONST1(_Rs_) ? _Rs_ : _Rt_;
		const u32 vreg = GPR_IS_CONST1(_Rs_) ? _Rt_ : _Rs_;
		const u64 cval = g_cpuConstRegs[creg].UD[0];

		if (cval == ~static_cast<u64>(0))
		{
			auto rd = armGprAlloc(_Rd_, true);
			armAsm->Mov(rd, ~static_cast<u64>(0));
			return;
		}
		auto rv = armGprAlloc(vreg, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (cval == 0)
		{
			if (rd.GetCode() != rv.GetCode())
				armAsm->Mov(rd, rv);
			return;
		}
		armAsm->Orr(rd, rv, cval);
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Orr(rd, rs, rt);
}
#endif

#if ISTUB_XOR
void recXOR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::XOR); }
#else
void recXOR()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] =
			g_cpuConstRegs[_Rs_].UD[0] ^ g_cpuConstRegs[_Rt_].UD[0];
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (_Rs_ == _Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (rd.GetCode() != rt.GetCode())
			armAsm->Mov(rd, rt);
		return;
	}
	if (!_Rt_)
	{
		auto rs = armGprAlloc(_Rs_, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (rd.GetCode() != rs.GetCode())
			armAsm->Mov(rd, rs);
		return;
	}

	// Single-operand const fold (mirrors x86 recLogicalOp_constv for XOR:
	// no absorbing value, identity=0).
	if (GPR_IS_CONST1(_Rs_) || GPR_IS_CONST1(_Rt_))
	{
		const u32 creg = GPR_IS_CONST1(_Rs_) ? _Rs_ : _Rt_;
		const u32 vreg = GPR_IS_CONST1(_Rs_) ? _Rt_ : _Rs_;
		const u64 cval = g_cpuConstRegs[creg].UD[0];

		auto rv = armGprAlloc(vreg, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (cval == 0)
		{
			if (rd.GetCode() != rv.GetCode())
				armAsm->Mov(rd, rv);
			return;
		}
		armAsm->Eor(rd, rv, cval);
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Eor(rd, rs, rt);
}
#endif

#if ISTUB_NOR
void recNOR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::NOR); }
#else
void recNOR()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] =
			~(g_cpuConstRegs[_Rs_].UD[0] | g_cpuConstRegs[_Rt_].UD[0]);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rs_ && !_Rt_)
	{
		// ~(0 | 0) = all ones
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Mov(rd, ~static_cast<u64>(0));
		return;
	}
	if (_Rs_ == _Rt_ || !_Rt_)
	{
		// ~(rs | rs) = ~rs, or ~(rs | 0) = ~rs
		auto rs = armGprAlloc(_Rs_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Mvn(rd, rs);
		return;
	}
	if (!_Rs_)
	{
		// ~(0 | rt) = ~rt
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Mvn(rd, rt);
		return;
	}

	// Single-operand const fold (mirrors x86 recLogicalOp_constv for NOR:
	// absorbing=-1 → result 0, identity=0 → result ~vreg).
	if (GPR_IS_CONST1(_Rs_) || GPR_IS_CONST1(_Rt_))
	{
		const u32 creg = GPR_IS_CONST1(_Rs_) ? _Rs_ : _Rt_;
		const u32 vreg = GPR_IS_CONST1(_Rs_) ? _Rt_ : _Rs_;
		const u64 cval = g_cpuConstRegs[creg].UD[0];

		if (cval == ~static_cast<u64>(0))
		{
			// x NOR -1 = ~(-1) = 0
			armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
			return;
		}
		auto rv = armGprAlloc(vreg, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (cval == 0)
		{
			// x NOR 0 = ~x
			armAsm->Mvn(rd, rv);
			return;
		}
		armAsm->Orr(rd, rv, cval);
		armAsm->Mvn(rd, rd);
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Orr(rd, rs, rt);
	armAsm->Mvn(rd, rd);
}
#endif

// ============================================================================
//  ANDI / ORI / XORI — 64-bit logical with zero-extended immediate
//  Interpreter: GPR[rt].UD[0] = GPR[rs].UD[0] OP (u64)_ImmU_
// ============================================================================

#if ISTUB_ANDI
void recANDI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::ANDI); }
#else
void recANDI()
{
	if (!_Rt_)
		return;

	if (GPR_IS_CONST1(_Rs_))
	{
		g_cpuConstRegs[_Rt_].UD[0] = g_cpuConstRegs[_Rs_].UD[0] & (u64)_ImmU_;
		GPR_SET_CONST(_Rt_);
		return;
	}

	armDelConstReg(_Rt_);

	if (_ImmU_ == 0)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, true);
	armAsm->And(rt, rs, (u64)_ImmU_);
}
#endif

#if ISTUB_ORI
void recORI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::ORI); }
#else
void recORI()
{
	if (!_Rt_)
		return;

	if (GPR_IS_CONST1(_Rs_))
	{
		g_cpuConstRegs[_Rt_].UD[0] = g_cpuConstRegs[_Rs_].UD[0] | (u64)_ImmU_;
		GPR_SET_CONST(_Rt_);
		return;
	}

	armDelConstReg(_Rt_);
	if (_ImmU_ == 0)
	{
		// ORI with 0 is a move
		auto rs = armGprAlloc(_Rs_, false);
		auto rt = armGprAlloc(_Rt_, true);
		if (rt.GetCode() != rs.GetCode())
			armAsm->Mov(rt, rs);
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, true);
	armAsm->Orr(rt, rs, (u64)_ImmU_);
}
#endif

#if ISTUB_XORI
void recXORI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::XORI); }
#else
void recXORI()
{
	if (!_Rt_)
		return;

	if (GPR_IS_CONST1(_Rs_))
	{
		g_cpuConstRegs[_Rt_].UD[0] = g_cpuConstRegs[_Rs_].UD[0] ^ (u64)_ImmU_;
		GPR_SET_CONST(_Rt_);
		return;
	}

	armDelConstReg(_Rt_);
	if (_ImmU_ == 0)
	{
		auto rs = armGprAlloc(_Rs_, false);
		auto rt = armGprAlloc(_Rt_, true);
		if (rt.GetCode() != rs.GetCode())
			armAsm->Mov(rt, rs);
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, true);
	armAsm->Eor(rt, rs, (u64)_ImmU_);
}
#endif

// ============================================================================
//  SLT / SLTU — Set on Less Than (register)
//  SLT:  GPR[rd] = (GPR[rs].SD[0] < GPR[rt].SD[0]) ? 1 : 0   (signed)
//  SLTU: GPR[rd] = (GPR[rs].UD[0] < GPR[rt].UD[0]) ? 1 : 0   (unsigned)
// ============================================================================

#if ISTUB_SLT
void recSLT() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SLT); }
#else
void recSLT()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] =
			(g_cpuConstRegs[_Rs_].SD[0] < g_cpuConstRegs[_Rt_].SD[0]) ? 1 : 0;
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (_Rs_ == _Rt_)
	{
		// x < x is always false
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	// _Rs_ or _Rt_ may be 0 — use xzr in that case (no slot needed).
	a64::Register rs = _Rs_ ? a64::Register(armGprAlloc(_Rs_, false)) : a64::xzr;
	a64::Register rt = _Rt_ ? a64::Register(armGprAlloc(_Rt_, false)) : a64::xzr;
	armAsm->Cmp(rs, rt);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Cset(rd, a64::lt);
}
#endif

#if ISTUB_SLTU
void recSLTU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SLTU); }
#else
void recSLTU()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] =
			(g_cpuConstRegs[_Rs_].UD[0] < g_cpuConstRegs[_Rt_].UD[0]) ? 1 : 0;
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (_Rs_ == _Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	a64::Register rs = _Rs_ ? a64::Register(armGprAlloc(_Rs_, false)) : a64::xzr;
	a64::Register rt = _Rt_ ? a64::Register(armGprAlloc(_Rt_, false)) : a64::xzr;
	armAsm->Cmp(rs, rt);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Cset(rd, a64::lo);
}
#endif

// ============================================================================
//  SLTI / SLTIU — Set on Less Than Immediate
//  SLTI:  GPR[rt] = (GPR[rs].SD[0] < sign_extend(imm16)) ? 1 : 0  (signed)
//  SLTIU: GPR[rt] = (GPR[rs].UD[0] < sign_extend(imm16)) ? 1 : 0  (unsigned)
// ============================================================================

#if ISTUB_SLTI
void recSLTI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SLTI); }
#else
void recSLTI()
{
	if (!_Rt_)
		return;

	if (GPR_IS_CONST1(_Rs_))
	{
		g_cpuConstRegs[_Rt_].UD[0] =
			(g_cpuConstRegs[_Rs_].SD[0] < (s64)(s32)_Imm_) ? 1 : 0;
		GPR_SET_CONST(_Rt_);
		return;
	}

	armDelConstReg(_Rt_);
	a64::Register rs = _Rs_ ? a64::Register(armGprAlloc(_Rs_, false)) : a64::xzr;
	// VIXL MacroAssembler handles Cmp with immediate optimally:
	// small positive → cmp, small negative → cmn, else → mov tmp + cmp
	armAsm->Cmp(rs, static_cast<s64>(_Imm_));
	auto rt = armGprAlloc(_Rt_, true);
	armAsm->Cset(rt, a64::lt);
}
#endif

#if ISTUB_SLTIU
void recSLTIU() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SLTIU); }
#else
void recSLTIU()
{
	if (!_Rt_)
		return;

	if (GPR_IS_CONST1(_Rs_))
	{
		g_cpuConstRegs[_Rt_].UD[0] =
			(g_cpuConstRegs[_Rs_].UD[0] < (u64)(s64)(s32)_Imm_) ? 1 : 0;
		GPR_SET_CONST(_Rt_);
		return;
	}

	armDelConstReg(_Rt_);
	a64::Register rs = _Rs_ ? a64::Register(armGprAlloc(_Rs_, false)) : a64::xzr;
	armAsm->Cmp(rs, static_cast<s64>(_Imm_));
	auto rt = armGprAlloc(_Rt_, true);
	armAsm->Cset(rt, a64::lo);
}
#endif

// ============================================================================
//  Overflow-trapping variants — alias to their non-trapping counterparts.
//
//  The x86 JIT (iR5900Arit.cpp:147, 221, 316, 397, iR5900AritImm.cpp:67-70,
//  88-91) does exactly this: recADDU literally calls recADD, recDADDIU calls
//  recDADDI, etc. — no overflow check is ever emitted. The interpreter traps
//  per the MIPS spec, but no x86-JIT-compiled game ever sees that trap. We
//  mirror the x86 JIT to keep semantics consistent across backends.
//
//  INTERP_ALU / INTERP_EE bisect escape hatches still work because they
//  force ISTUB_*U = 1, routing the -U variants (and therefore these aliases)
//  to the interpreter.
// ============================================================================

void recADD()   { recADDU();   }
void recADDI()  { recADDIU();  }
void recSUB()   { recSUBU();   }
void recDADD()  { recDADDU();  }
void recDADDI() { recDADDIU(); }
void recDSUB()  { recDSUBU();  }

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900

// ============================================================================
// === Begin iR5900Move_arm64.cpp body ========================================
// ============================================================================


using namespace R5900;

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
#if defined(INTERP_MOVE) || defined(INTERP_EE)
#define ISTUB_LUI      1
#define ISTUB_MFHI     1
#define ISTUB_MFLO     1
#define ISTUB_MTHI     1
#define ISTUB_MTLO     1
#define ISTUB_MFHI1    1
#define ISTUB_MFLO1    1
#define ISTUB_MTHI1    1
#define ISTUB_MTLO1    1
#define ISTUB_MOVZ     1
#define ISTUB_MOVN     1
#define ISTUB_MFSA     1
#define ISTUB_MTSA     1
#define ISTUB_MTSAB    1
#define ISTUB_MTSAH    1
#else
#define ISTUB_LUI      0
#define ISTUB_MFHI     0
#define ISTUB_MFLO     0
#define ISTUB_MTHI     0
#define ISTUB_MTLO     0
#define ISTUB_MFHI1    0
#define ISTUB_MFLO1    0
#define ISTUB_MTHI1    0
#define ISTUB_MTLO1    0
#define ISTUB_MOVZ     0
#define ISTUB_MOVN     0
#define ISTUB_MFSA     0
#define ISTUB_MTSA     0
#define ISTUB_MTSAB    0
#define ISTUB_MTSAH    0
#endif

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ============================================================================
//  LUI — Load Upper Immediate
//  Interpreter: GPR[rt].UD[0] = (s32)(code << 16)
// ============================================================================

#if ISTUB_LUI
void recLUI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::LUI); }
#else
void recLUI()
{
	if (!_Rt_)
		return;

	// LUI is always a compile-time constant. Track it via const-prop instead
	// of materializing now — downstream LUI;ORI / LUI;ADDIU folds at compile
	// time and the whole address-load idiom becomes a Mov-imm later.
	const s64 val = (s64)(s32)(cpuRegs.code << 16);
	g_cpuConstRegs[_Rt_].SD[0] = val;
	GPR_SET_CONST(_Rt_);
}
#endif

// ============================================================================
//  MFHI / MFLO — Move from HI/LO register
//  Interpreter: GPR[rd].UD[0] = HI.UD[0]  /  GPR[rd].UD[0] = LO.UD[0]
// ============================================================================

#if ISTUB_MFHI
void recMFHI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MFHI); }
#else
void recMFHI()
{
	if (!_Rd_)
		return;

	armDelConstReg(_Rd_);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Ldr(rd, a64::MemOperand(RCPUSTATE, HI_OFFSET));
}
#endif

#if ISTUB_MFLO
void recMFLO() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MFLO); }
#else
void recMFLO()
{
	if (!_Rd_)
		return;

	armDelConstReg(_Rd_);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Ldr(rd, a64::MemOperand(RCPUSTATE, LO_OFFSET));
}
#endif

// ============================================================================
//  MTHI / MTLO — Move to HI/LO register
//  Interpreter: HI.UD[0] = GPR[rs].UD[0]  /  LO.UD[0] = GPR[rs].UD[0]
// ============================================================================

#if ISTUB_MTHI
void recMTHI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MTHI); }
#else
void recMTHI()
{
	if (!_Rs_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, HI_OFFSET));
		return;
	}
	auto rs = armGprAlloc(_Rs_, false);
	armAsm->Str(rs, a64::MemOperand(RCPUSTATE, HI_OFFSET));
}
#endif

#if ISTUB_MTLO
void recMTLO() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MTLO); }
#else
void recMTLO()
{
	if (!_Rs_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, LO_OFFSET));
		return;
	}
	auto rs = armGprAlloc(_Rs_, false);
	armAsm->Str(rs, a64::MemOperand(RCPUSTATE, LO_OFFSET));
}
#endif

// ============================================================================
//  MFHI1 / MFLO1 — Move from HI/LO pipeline 1 (upper 64 bits)
//  Interpreter: GPR[rd].UD[0] = HI.UD[1]  /  GPR[rd].UD[0] = LO.UD[1]
// ============================================================================

#if ISTUB_MFHI1
void recMFHI1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MFHI1); }
#else
void recMFHI1()
{
	if (!_Rd_)
		return;

	armDelConstReg(_Rd_);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Ldr(rd, a64::MemOperand(RCPUSTATE, HI_OFFSET + 8));
}
#endif

#if ISTUB_MFLO1
void recMFLO1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MFLO1); }
#else
void recMFLO1()
{
	if (!_Rd_)
		return;

	armDelConstReg(_Rd_);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Ldr(rd, a64::MemOperand(RCPUSTATE, LO_OFFSET + 8));
}
#endif

// ============================================================================
//  MTHI1 / MTLO1 — Move to HI/LO pipeline 1 (upper 64 bits)
//  Interpreter: HI.UD[1] = GPR[rs].UD[0]  /  LO.UD[1] = GPR[rs].UD[0]
// ============================================================================

#if ISTUB_MTHI1
void recMTHI1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MTHI1); }
#else
void recMTHI1()
{
	if (!_Rs_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, HI_OFFSET + 8));
		return;
	}
	auto rs = armGprAlloc(_Rs_, false);
	armAsm->Str(rs, a64::MemOperand(RCPUSTATE, HI_OFFSET + 8));
}
#endif

#if ISTUB_MTLO1
void recMTLO1() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MTLO1); }
#else
void recMTLO1()
{
	if (!_Rs_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, LO_OFFSET + 8));
		return;
	}
	auto rs = armGprAlloc(_Rs_, false);
	armAsm->Str(rs, a64::MemOperand(RCPUSTATE, LO_OFFSET + 8));
}
#endif

// ============================================================================
//  MOVZ — Conditional Move if Zero
//  Interpreter: if (!_Rd_) return; if (GPR[rt].UD[0] == 0) GPR[rd].UD[0] = GPR[rs].UD[0]
// ============================================================================

#if ISTUB_MOVZ
void recMOVZ() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MOVZ); }
#else
void recMOVZ()
{
	if (!_Rd_)
		return;

	// Conditional move: if rt==0, rd=rs. The store is conditional, so we
	// always go through memory for _Rd_ rather than allocating it as a
	// cache slot (which would have undefined runtime contents on the skip
	// path).
	armDelConstReg(_Rd_);

	// rt const nonzero: move never happens
	if (GPR_IS_CONST1(_Rt_) && g_cpuConstRegs[_Rt_].UD[0] != 0)
		return;

	// rt const zero (or both const): unconditional move
	if (GPR_IS_CONST1(_Rt_))
	{
		if (!_Rs_)
		{
			armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		}
		else
		{
			auto rs = armGprAlloc(_Rs_, false);
			armAsm->Str(rs, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		}
		return;
	}

	// General case: conditional store. Allocate both rs and rt unconditionally
	// before the branch so the cache slot table stays consistent across paths.
	auto rt = armGprAlloc(_Rt_, false);
	a64::Register rs =
		(_Rs_ == 0) ? a64::Register(a64::xzr) : a64::Register(armGprAlloc(_Rs_, false));
	a64::Label skip;
	armAsm->Cbnz(rt, &skip);
	armAsm->Str(rs, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
	armAsm->Bind(&skip);
}
#endif

// ============================================================================
//  MOVN — Conditional Move if Not Zero
//  Interpreter: if (!_Rd_) return; if (GPR[rt].UD[0] != 0) GPR[rd].UD[0] = GPR[rs].UD[0]
// ============================================================================

#if ISTUB_MOVN
void recMOVN() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MOVN); }
#else
void recMOVN()
{
	if (!_Rd_)
		return;

	// Conditional move: if rt!=0, rd=rs. The store is conditional, so we
	// always go through memory for _Rd_ rather than allocating it as a
	// cache slot.
	armDelConstReg(_Rd_);

	// rt const zero: move never happens
	if (GPR_IS_CONST1(_Rt_) && g_cpuConstRegs[_Rt_].UD[0] == 0)
		return;

	// rt const nonzero (or both const): unconditional move
	if (GPR_IS_CONST1(_Rt_))
	{
		if (!_Rs_)
		{
			armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		}
		else
		{
			auto rs = armGprAlloc(_Rs_, false);
			armAsm->Str(rs, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		}
		return;
	}

	// General case: conditional store. Allocate both rs and rt unconditionally
	// before the branch so the cache slot table stays consistent across paths.
	auto rt = armGprAlloc(_Rt_, false);
	a64::Register rs =
		(_Rs_ == 0) ? a64::Register(a64::xzr) : a64::Register(armGprAlloc(_Rs_, false));
	a64::Label skip;
	armAsm->Cbz(rt, &skip);
	armAsm->Str(rs, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
	armAsm->Bind(&skip);
}
#endif

// ============================================================================
//  MFSA — Move from Shift Amount register
//  Interpreter: if (!_Rd_) return; GPR[rd].UD[0] = (u64)cpuRegs.sa
// ============================================================================

#if ISTUB_MFSA
void recMFSA() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MFSA); }
#else
void recMFSA()
{
	if (!_Rd_)
		return;

	armDelConstReg(_Rd_);
	auto rd = armGprAlloc(_Rd_, true);
	// sa is u32; LDR W zero-extends to the full X register.
	armAsm->Ldr(rd.W(), a64::MemOperand(RCPUSTATE, SA_OFFSET));
}
#endif

// ============================================================================
//  MTSA — Move to Shift Amount register
//  Interpreter: cpuRegs.sa = (u32)GPR[rs].UD[0]
// ============================================================================

#if ISTUB_MTSA
void recMTSA() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MTSA); }
#else
void recMTSA()
{
	if (!_Rs_)
	{
		armAsm->Str(a64::wzr, a64::MemOperand(RCPUSTATE, SA_OFFSET));
		return;
	}
	auto rs = armGprAlloc(_Rs_, false);
	armAsm->And(RWSCRATCH, rs.W(), 0xF);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, SA_OFFSET));
}
#endif

// ============================================================================
//  MTSAB — Move to Shift Amount Byte
//  Interpreter: cpuRegs.sa = ((GPR[rs].UL[0] & 0xF) ^ (_Imm_ & 0xF))
// ============================================================================

#if ISTUB_MTSAB
void recMTSAB() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MTSAB); }
#else
void recMTSAB()
{
	if (GPR_IS_CONST1(_Rs_))
	{
		u32 result = (g_cpuConstRegs[_Rs_].UL[0] & 0xF) ^ (_ImmU_ & 0xF);
		armAsm->Mov(RWSCRATCH, result);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, SA_OFFSET));
		return;
	}

	// (_Rs_ == 0 is always const 0 — handled by the const path above.)
	auto rs = armGprAlloc(_Rs_, false);
	// w_scratch = rs & 0xF
	armAsm->And(RWSCRATCH, rs.W(), 0xF);
	// w_scratch ^= (imm & 0xF)
	u32 immMask = _ImmU_ & 0xF;
	if (immMask)
		armAsm->Eor(RWSCRATCH, RWSCRATCH, immMask);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, SA_OFFSET));
}
#endif

// ============================================================================
//  MTSAH — Move to Shift Amount Halfword
//  Interpreter: cpuRegs.sa = ((GPR[rs].UL[0] & 0x7) ^ (_Imm_ & 0x7)) << 1
// ============================================================================

#if ISTUB_MTSAH
void recMTSAH() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::MTSAH); }
#else
void recMTSAH()
{
	if (GPR_IS_CONST1(_Rs_))
	{
		u32 result = ((g_cpuConstRegs[_Rs_].UL[0] & 0x7) ^ (_ImmU_ & 0x7)) << 1;
		armAsm->Mov(RWSCRATCH, result);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, SA_OFFSET));
		return;
	}

	// (_Rs_ == 0 is always const 0 — handled by the const path above.)
	auto rs = armGprAlloc(_Rs_, false);
	// w_scratch = rs & 0x7
	armAsm->And(RWSCRATCH, rs.W(), 0x7);
	// w_scratch ^= (imm & 0x7)
	u32 immMask = _ImmU_ & 0x7;
	if (immMask)
		armAsm->Eor(RWSCRATCH, RWSCRATCH, immMask);
	// w_scratch <<= 1
	armAsm->Lsl(RWSCRATCH, RWSCRATCH, 1);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, SA_OFFSET));
}
#endif

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900

// ============================================================================
// === Begin iR5900Shift_arm64.cpp body =======================================
// ============================================================================


using namespace R5900;

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
#if defined(INTERP_SHIFT) || defined(INTERP_EE)
#define ISTUB_SLL      1
#define ISTUB_SRL      1
#define ISTUB_SRA      1
#define ISTUB_SLLV     1
#define ISTUB_SRLV     1
#define ISTUB_SRAV     1
#define ISTUB_DSLL     1
#define ISTUB_DSRL     1
#define ISTUB_DSRA     1
#define ISTUB_DSLL32   1
#define ISTUB_DSRL32   1
#define ISTUB_DSRA32   1
#define ISTUB_DSLLV    1
#define ISTUB_DSRLV    1
#define ISTUB_DSRAV    1
#else
#define ISTUB_SLL      0
#define ISTUB_SRL      0
#define ISTUB_SRA      0
#define ISTUB_SLLV     0
#define ISTUB_SRLV     0
#define ISTUB_SRAV     0
#define ISTUB_DSLL     0
#define ISTUB_DSRL     0
#define ISTUB_DSRA     0
#define ISTUB_DSLL32   0
#define ISTUB_DSRL32   0
#define ISTUB_DSRA32   0
#define ISTUB_DSLLV    0
#define ISTUB_DSRLV    0
#define ISTUB_DSRAV    0
#endif

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ============================================================================
//  SLL — Shift Left Logical (32-bit, sign-extend result)
//  Interpreter: GPR[rd].SD[0] = (s32)(GPR[rt].UL[0] << sa)
// ============================================================================

#if ISTUB_SLL
void recSLL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SLL); }
#else
void recSLL()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] = (s64)(s32)(g_cpuConstRegs[_Rt_].UL[0] << _Sa_);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	if (_Sa_)
		armAsm->Lsl(rd.W(), rt.W(), _Sa_);
	armAsm->Sxtw(rd, _Sa_ ? rd.W() : rt.W());
}
#endif

// ============================================================================
//  SRL — Shift Right Logical (32-bit, sign-extend result)
//  Interpreter: GPR[rd].SD[0] = (s32)(GPR[rt].UL[0] >> sa)
// ============================================================================

#if ISTUB_SRL
void recSRL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SRL); }
#else
void recSRL()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] = (s64)(s32)(g_cpuConstRegs[_Rt_].UL[0] >> _Sa_);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	if (_Sa_)
		armAsm->Lsr(rd.W(), rt.W(), _Sa_);
	armAsm->Sxtw(rd, _Sa_ ? rd.W() : rt.W());
}
#endif

// ============================================================================
//  SRA — Shift Right Arithmetic (32-bit, sign-extend result)
//  Interpreter: GPR[rd].SD[0] = (s32)(GPR[rt].SL[0] >> sa)
// ============================================================================

#if ISTUB_SRA
void recSRA() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SRA); }
#else
void recSRA()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] = (s64)((s32)g_cpuConstRegs[_Rt_].UL[0] >> _Sa_);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	if (_Sa_)
		armAsm->Asr(rd.W(), rt.W(), _Sa_);
	armAsm->Sxtw(rd, _Sa_ ? rd.W() : rt.W());
}
#endif

// ============================================================================
//  SLLV — Shift Left Logical Variable (32-bit, sign-extend result)
//  Interpreter: GPR[rd].SD[0] = (s32)(GPR[rt].UL[0] << (GPR[rs].UL[0] & 0x1f))
// ============================================================================

#if ISTUB_SLLV
void recSLLV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SLLV); }
#else
void recSLLV()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] =
			(s64)(s32)(g_cpuConstRegs[_Rt_].UL[0] << (g_cpuConstRegs[_Rs_].UL[0] & 0x1F));
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Sxtw(rd, rt.W());
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Lsl(rd.W(), rt.W(), rs.W());
	armAsm->Sxtw(rd, rd.W());
}
#endif

// ============================================================================
//  SRLV — Shift Right Logical Variable (32-bit, sign-extend result)
//  Interpreter: GPR[rd].SD[0] = (s32)(GPR[rt].UL[0] >> (GPR[rs].UL[0] & 0x1f))
// ============================================================================

#if ISTUB_SRLV
void recSRLV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SRLV); }
#else
void recSRLV()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] =
			(s64)(s32)(g_cpuConstRegs[_Rt_].UL[0] >> (g_cpuConstRegs[_Rs_].UL[0] & 0x1F));
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Sxtw(rd, rt.W());
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Lsr(rd.W(), rt.W(), rs.W());
	armAsm->Sxtw(rd, rd.W());
}
#endif

// ============================================================================
//  SRAV — Shift Right Arithmetic Variable (32-bit, sign-extend result)
//  Interpreter: GPR[rd].SD[0] = (s32)(GPR[rt].SL[0] >> (GPR[rs].UL[0] & 0x1f))
// ============================================================================

#if ISTUB_SRAV
void recSRAV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::SRAV); }
#else
void recSRAV()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] =
			(s64)((s32)g_cpuConstRegs[_Rt_].UL[0] >> (g_cpuConstRegs[_Rs_].UL[0] & 0x1F));
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		armAsm->Sxtw(rd, rt.W());
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Asr(rd.W(), rt.W(), rs.W());
	armAsm->Sxtw(rd, rd.W());
}
#endif

// ============================================================================
//  DSLL — Doubleword Shift Left Logical
//  Interpreter: GPR[rd].UD[0] = GPR[rt].UD[0] << sa
// ============================================================================

#if ISTUB_DSLL
void recDSLL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSLL); }
#else
void recDSLL()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] = g_cpuConstRegs[_Rt_].UD[0] << _Sa_;
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	if (_Sa_)
		armAsm->Lsl(rd, rt, _Sa_);
	else if (!rd.Is(rt))
		armAsm->Mov(rd, rt);
}
#endif

// ============================================================================
//  DSRL — Doubleword Shift Right Logical
//  Interpreter: GPR[rd].UD[0] = GPR[rt].UD[0] >> sa
// ============================================================================

#if ISTUB_DSRL
void recDSRL() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSRL); }
#else
void recDSRL()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] = g_cpuConstRegs[_Rt_].UD[0] >> _Sa_;
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	if (_Sa_)
		armAsm->Lsr(rd, rt, _Sa_);
	else if (!rd.Is(rt))
		armAsm->Mov(rd, rt);
}
#endif

// ============================================================================
//  DSRA — Doubleword Shift Right Arithmetic
//  Interpreter: GPR[rd].SD[0] = GPR[rt].SD[0] >> sa
// ============================================================================

#if ISTUB_DSRA
void recDSRA() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSRA); }
#else
void recDSRA()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] = g_cpuConstRegs[_Rt_].SD[0] >> _Sa_;
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	if (_Sa_)
		armAsm->Asr(rd, rt, _Sa_);
	else if (!rd.Is(rt))
		armAsm->Mov(rd, rt);
}
#endif

// ============================================================================
//  DSLL32 — Doubleword Shift Left Logical +32
//  Interpreter: GPR[rd].UD[0] = GPR[rt].UD[0] << (sa + 32)
// ============================================================================

#if ISTUB_DSLL32
void recDSLL32() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSLL32); }
#else
void recDSLL32()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] = g_cpuConstRegs[_Rt_].UD[0] << (_Sa_ + 32);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Lsl(rd, rt, _Sa_ + 32);
}
#endif

// ============================================================================
//  DSRL32 — Doubleword Shift Right Logical +32
//  Interpreter: GPR[rd].UD[0] = GPR[rt].UD[0] >> (sa + 32)
// ============================================================================

#if ISTUB_DSRL32
void recDSRL32() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSRL32); }
#else
void recDSRL32()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] = g_cpuConstRegs[_Rt_].UD[0] >> (_Sa_ + 32);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Lsr(rd, rt, _Sa_ + 32);
}
#endif

// ============================================================================
//  DSRA32 — Doubleword Shift Right Arithmetic +32
//  Interpreter: GPR[rd].SD[0] = GPR[rt].SD[0] >> (sa + 32)
// ============================================================================

#if ISTUB_DSRA32
void recDSRA32() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSRA32); }
#else
void recDSRA32()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST1(_Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] = g_cpuConstRegs[_Rt_].SD[0] >> (_Sa_ + 32);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}

	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Asr(rd, rt, _Sa_ + 32);
}
#endif

// ============================================================================
//  DSLLV — Doubleword Shift Left Logical Variable
//  Interpreter: GPR[rd].UD[0] = GPR[rt].UD[0] << (GPR[rs].UL[0] & 0x3f)
// ============================================================================

#if ISTUB_DSLLV
void recDSLLV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSLLV); }
#else
void recDSLLV()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] =
			g_cpuConstRegs[_Rt_].UD[0] << (g_cpuConstRegs[_Rs_].UL[0] & 0x3F);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (!rd.Is(rt))
			armAsm->Mov(rd, rt);
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Lsl(rd, rt, rs);
}
#endif

// ============================================================================
//  DSRLV — Doubleword Shift Right Logical Variable
//  Interpreter: GPR[rd].UD[0] = GPR[rt].UD[0] >> (GPR[rs].UL[0] & 0x3f)
// ============================================================================

#if ISTUB_DSRLV
void recDSRLV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSRLV); }
#else
void recDSRLV()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].UD[0] =
			g_cpuConstRegs[_Rt_].UD[0] >> (g_cpuConstRegs[_Rs_].UL[0] & 0x3F);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (!rd.Is(rt))
			armAsm->Mov(rd, rt);
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Lsr(rd, rt, rs);
}
#endif

// ============================================================================
//  DSRAV — Doubleword Shift Right Arithmetic Variable
//  Interpreter: GPR[rd].SD[0] = GPR[rt].SD[0] >> (GPR[rs].UL[0] & 0x3f)
// ============================================================================

#if ISTUB_DSRAV
void recDSRAV() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::DSRAV); }
#else
void recDSRAV()
{
	if (!_Rd_)
		return;

	if (GPR_IS_CONST2(_Rs_, _Rt_))
	{
		g_cpuConstRegs[_Rd_].SD[0] =
			g_cpuConstRegs[_Rt_].SD[0] >> (g_cpuConstRegs[_Rs_].UL[0] & 0x3F);
		GPR_SET_CONST(_Rd_);
		return;
	}

	armDelConstReg(_Rd_);

	if (!_Rt_)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rd_)));
		return;
	}
	if (!_Rs_)
	{
		auto rt = armGprAlloc(_Rt_, false);
		auto rd = armGprAlloc(_Rd_, true);
		if (!rd.Is(rt))
			armAsm->Mov(rd, rt);
		return;
	}

	auto rs = armGprAlloc(_Rs_, false);
	auto rt = armGprAlloc(_Rt_, false);
	auto rd = armGprAlloc(_Rd_, true);
	armAsm->Asr(rd, rt, rs);
}
#endif

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
