// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 IOP (R3000A) Recompiler — Instruction Tables & Implementations
//
// All instructions start as interpreter stubs. Native ARM64 implementations
// are progressively added and controlled by ISTUB_IOP_* toggles.
//
// Liveness propagation (rpsxpropBSC etc.) is pure C++ analysis copied from
// the x86 version (x86/iR3000Atables.cpp) since that file isn't compiled
// for ARM64 targets.

#include "arm64/aR3000A.h"
#include "arm64/AsmHelpers.h"
#include "arm64/arm64Emitter.h"
#include "R3000A.h"
#include "IopMem.h"
#include "IopGte.h"
#include "IopDma.h"
#include "IopBios.h"
#include "iRecAnalysis.h" // EEINST, _recClearInst, _recFillRegister, g_pCurInstInfo

#include "common/Console.h"

using namespace vixl::aarch64;

extern int g_psxWriteOk;
extern u32 g_psxMaxRecMem;

// These are defined in x86/iCore.cpp which isn't compiled for ARM64.
// Provide local definitions (pure C++, no arch-specific code).
void _recClearInst(EEINST* pinst)
{
	std::memset(pinst, 0, sizeof(EEINST));
	std::memset(pinst->regs, EEINST_LIVE, sizeof(pinst->regs));
	std::memset(pinst->fpuregs, EEINST_LIVE, sizeof(pinst->fpuregs));
	std::memset(pinst->vfregs, EEINST_LIVE, sizeof(pinst->vfregs));
	std::memset(pinst->viregs, EEINST_LIVE, sizeof(pinst->viregs));
}

void _recFillRegister(EEINST& pinst, int type, int reg, int write)
{
	if (write)
	{
		for (size_t i = 0; i < std::size(pinst.writeType); ++i)
		{
			if (pinst.writeType[i] == XMMTYPE_TEMP)
			{
				pinst.writeType[i] = type;
				pinst.writeReg[i] = reg;
				return;
			}
		}
	}
	else
	{
		for (size_t i = 0; i < std::size(pinst.readType); ++i)
		{
			if (pinst.readType[i] == XMMTYPE_TEMP)
			{
				pinst.readType[i] = type;
				pinst.readReg[i] = reg;
				return;
			}
		}
	}
}


// ============================================================================
//  ISTUB toggles — per-instruction interpreter/native selection
// ============================================================================
//
// Each INTERP_IOP_* group forces all instructions in that group to use
// interpreter stubs. When the group define is NOT set, individual ISTUB_IOP_*
// flags control each instruction (0 = native, 1 = interp stub).
//
// For initial bringup, all individual stubs are set to 1 (interpreter)
// even when the group is enabled for native. This lets us flip them
// one at a time to isolate bugs.

// --- ALU ---
#ifdef INTERP_IOP_ALU
#define ISTUB_IOP_ADDI   1
#define ISTUB_IOP_ADDIU  1
#define ISTUB_IOP_SLTI   1
#define ISTUB_IOP_SLTIU  1
#define ISTUB_IOP_ANDI   1
#define ISTUB_IOP_ORI    1
#define ISTUB_IOP_XORI   1
#define ISTUB_IOP_LUI    1
#define ISTUB_IOP_ADD    1
#define ISTUB_IOP_ADDU   1
#define ISTUB_IOP_SUB    1
#define ISTUB_IOP_SUBU   1
#define ISTUB_IOP_AND    1
#define ISTUB_IOP_OR     1
#define ISTUB_IOP_XOR    1
#define ISTUB_IOP_NOR    1
#define ISTUB_IOP_SLT    1
#define ISTUB_IOP_SLTU   1
#else
// --- ALU bisect: flip individual 1→0 to find buggy instruction ---
// I-type (immediate) ALU — bisect round 2:
#define ISTUB_IOP_ADDI   0
#define ISTUB_IOP_ADDIU  0
#define ISTUB_IOP_SLTI   0
#define ISTUB_IOP_SLTIU  0
#define ISTUB_IOP_ANDI   0
#define ISTUB_IOP_ORI    0
#define ISTUB_IOP_XORI   0
#define ISTUB_IOP_LUI    0
// R-type (register-register) ALU:
#define ISTUB_IOP_ADD    0
#define ISTUB_IOP_ADDU   0
#define ISTUB_IOP_SUB    0
#define ISTUB_IOP_SUBU   0
#define ISTUB_IOP_AND    0
#define ISTUB_IOP_OR     0
#define ISTUB_IOP_XOR    0
#define ISTUB_IOP_NOR    0
#define ISTUB_IOP_SLT    0
#define ISTUB_IOP_SLTU   0
#endif

// --- SHIFT ---
#ifdef INTERP_IOP_SHIFT
#define ISTUB_IOP_SLL    1
#define ISTUB_IOP_SRL    1
#define ISTUB_IOP_SRA    1
#define ISTUB_IOP_SLLV   1
#define ISTUB_IOP_SRLV   1
#define ISTUB_IOP_SRAV   1
#else
#define ISTUB_IOP_SLL    0
#define ISTUB_IOP_SRL    0
#define ISTUB_IOP_SRA    0
#define ISTUB_IOP_SLLV   0
#define ISTUB_IOP_SRLV   0
#define ISTUB_IOP_SRAV   0
#endif

// --- MULTDIV ---
#ifdef INTERP_IOP_MULTDIV
#define ISTUB_IOP_MULT   1
#define ISTUB_IOP_MULTU  1
#define ISTUB_IOP_DIV    1
#define ISTUB_IOP_DIVU   1
#else
#define ISTUB_IOP_MULT   0
#define ISTUB_IOP_MULTU  0
#define ISTUB_IOP_DIV    0
#define ISTUB_IOP_DIVU   0
#endif

// --- MOVE ---
#ifdef INTERP_IOP_MOVE
#define ISTUB_IOP_MFHI   1
#define ISTUB_IOP_MTHI   1
#define ISTUB_IOP_MFLO   1
#define ISTUB_IOP_MTLO   1
#else
#define ISTUB_IOP_MFHI   0
#define ISTUB_IOP_MTHI   0
#define ISTUB_IOP_MFLO   0
#define ISTUB_IOP_MTLO   0
#endif

// --- BRANCH ---
#ifdef INTERP_IOP_BRANCH
#define ISTUB_IOP_J      1
#define ISTUB_IOP_JAL    1
#define ISTUB_IOP_JR     1
#define ISTUB_IOP_JALR   1
#define ISTUB_IOP_BEQ    1
#define ISTUB_IOP_BNE    1
#define ISTUB_IOP_BLEZ   1
#define ISTUB_IOP_BGTZ   1
#define ISTUB_IOP_BLTZ   1
#define ISTUB_IOP_BGEZ   1
#define ISTUB_IOP_BLTZAL 1
#define ISTUB_IOP_BGEZAL 1
#else
// --- BRANCH bisect: flip individual 1→0 to find buggy branch op ---
// Start with all interp (1), enable native one at a time.
#define ISTUB_IOP_J      0
#define ISTUB_IOP_JAL    0
#define ISTUB_IOP_JR     0
#define ISTUB_IOP_JALR   0
#define ISTUB_IOP_BEQ    0
#define ISTUB_IOP_BNE    0
#define ISTUB_IOP_BLEZ   0
#define ISTUB_IOP_BGTZ   0
#define ISTUB_IOP_BLTZ   0
#define ISTUB_IOP_BGEZ   0
#define ISTUB_IOP_BLTZAL 0
#define ISTUB_IOP_BGEZAL 0
#endif

// --- LOADSTORE ---
#ifdef INTERP_IOP_LOADSTORE
#define ISTUB_IOP_LB     1
#define ISTUB_IOP_LBU    1
#define ISTUB_IOP_LH     1
#define ISTUB_IOP_LHU    1
#define ISTUB_IOP_LW     1
#define ISTUB_IOP_LWL    1
#define ISTUB_IOP_LWR    1
#define ISTUB_IOP_SB     1
#define ISTUB_IOP_SH     1
#define ISTUB_IOP_SW     1
#define ISTUB_IOP_SWL    1
#define ISTUB_IOP_SWR    1
#else
#define ISTUB_IOP_LB     0
#define ISTUB_IOP_LBU    0
#define ISTUB_IOP_LH     0
#define ISTUB_IOP_LHU    0
#define ISTUB_IOP_LW     0
#define ISTUB_IOP_LWL    1  // unaligned — keep as interp for now
#define ISTUB_IOP_LWR    1
#define ISTUB_IOP_SB     0
#define ISTUB_IOP_SH     0
#define ISTUB_IOP_SW     0
#define ISTUB_IOP_SWL    1  // unaligned — keep as interp for now
#define ISTUB_IOP_SWR    1
#endif

// --- COP0 ---
#ifdef INTERP_IOP_COP0
#define ISTUB_IOP_MFC0   1
#define ISTUB_IOP_CFC0   1
#define ISTUB_IOP_MTC0   1
#define ISTUB_IOP_CTC0   1
#define ISTUB_IOP_RFE    1
#else
#define ISTUB_IOP_MFC0   0
#define ISTUB_IOP_CFC0   0
#define ISTUB_IOP_MTC0   0
#define ISTUB_IOP_CTC0   0
#define ISTUB_IOP_RFE    1  // keep as interp (exception handling)
#endif

// COP2 (GTE) and SYSTEM always use interpreter stubs
// No ISTUB toggles needed — they're always stubbed

// ============================================================================
//  Interpreter stub macro
// ============================================================================

// Emit a call to the interpreter function with full state flush.
// This is the universal fallback for any instruction.
#define REC_FUNC(f) \
	static void rpsx##f() \
	{ \
		iopArmCallInterpreter(psx##f); \
	}

// GTE variant — different naming convention
#define REC_GTE_FUNC(f) \
	static void rgte##f() \
	{ \
		iopArmCallInterpreter(gte##f); \
	}

// ============================================================================
//  Interpreter function forward declarations
// ============================================================================

// These are defined in R3000AOpcodeTables.cpp and R3000AInterpreter.cpp
extern void psxADDI();
extern void psxADDIU();
extern void psxSLTI();
extern void psxSLTIU();
extern void psxANDI();
extern void psxORI();
extern void psxXORI();
extern void psxLUI();
extern void psxADD();
extern void psxADDU();
extern void psxSUB();
extern void psxSUBU();
extern void psxAND();
extern void psxOR();
extern void psxXOR();
extern void psxNOR();
extern void psxSLT();
extern void psxSLTU();
extern void psxSLL();
extern void psxSRL();
extern void psxSRA();
extern void psxSLLV();
extern void psxSRLV();
extern void psxSRAV();
extern void psxMULT();
extern void psxMULTU();
extern void psxDIV();
extern void psxDIVU();
extern void psxMFHI();
extern void psxMTHI();
extern void psxMFLO();
extern void psxMTLO();
extern void psxJ();
extern void psxJAL();
extern void psxJR();
extern void psxJALR();
extern void psxBEQ();
extern void psxBNE();
extern void psxBLEZ();
extern void psxBGTZ();
extern void psxBLTZ();
extern void psxBGEZ();
extern void psxBLTZAL();
extern void psxBGEZAL();
extern void psxLB();
extern void psxLBU();
extern void psxLH();
extern void psxLHU();
extern void psxLW();
extern void psxLWL();
extern void psxLWR();
extern void psxSB();
extern void psxSH();
extern void psxSW();
extern void psxSWL();
extern void psxSWR();
extern void psxMFC0();
extern void psxCFC0();
extern void psxMTC0();
extern void psxCTC0();
extern void psxRFE();
extern void psxSYSCALL();
extern void psxBREAK();

// ============================================================================
//  ALU Instructions
// ============================================================================

#if ISTUB_IOP_ADDI
REC_FUNC(ADDI)
#else
static void rpsxADDI()
{
	if (!_psxRt_) return;
	if (PSX_IS_CONST1(_psxRs_))
	{
		g_psxConstRegs[_psxRt_] = g_psxConstRegs[_psxRs_] + _psxImm_;
		PSX_SET_CONST(_psxRt_);
		return;
	}
	PSX_DEL_CONST(_psxRt_);
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	if (_psxImm_ != 0)
	{
		if (Assembler::IsImmAddSub(std::abs(_psxImm_)))
		{
			if (_psxImm_ > 0)
				armAsm->Add(RWPSXSCRATCH, RWPSXSCRATCH, _psxImm_);
			else
				armAsm->Sub(RWPSXSCRATCH, RWPSXSCRATCH, -_psxImm_);
		}
		else
		{
			armAsm->Mov(RWPSXSCRATCH2, (u32)(s32)_psxImm_);
			armAsm->Add(RWPSXSCRATCH, RWPSXSCRATCH, RWPSXSCRATCH2);
		}
	}
	iopArmStoreGPR(RWPSXSCRATCH, _psxRt_);
}
#endif

// Handle IOP module import table magic (HLE calls)
// When delay slot is `addiu $zero, $zero, index` (0x2400xxxx), this may be
// an IOP BIOS import stub. Check at compile time and emit HLE call if needed.
static void psxRecompileIrxImport()
{
	u32 import_table = R3000A::irxImportTableAddr(psxpc - 4);
	u16 index = psxRegs.code & 0xffff;
	if (!import_table)
		return;

	const std::string libname = iopMemReadString(import_table + 12, 8);
	irxHLE hle = R3000A::irxImportHLE(libname, index);

	if (!hle)
		return;

	// Flush state and emit call to HLE function
	iopArmFlushCode();
	iopArmFlushPC();
	iopArmFlushConstRegs();
	armEmitCall((const void*)hle);

	// If HLE returns non-zero, it handled the call - jump to dispatcher
	extern const void* iopDispatcherReg;
	armEmitCbnz(RWRET, iopDispatcherReg);

	// Conservative: invalidate const tracking after HLE call
	g_psxHasConstReg = g_psxFlushedConstReg = 1;
}

#if ISTUB_IOP_ADDIU
REC_FUNC(ADDIU)
#else
static void rpsxADDIU()
{
	if (!_psxRt_)
	{
		// Check for IOP module import table magic
		if (psxRegs.code >> 16 == 0x2400)
			psxRecompileIrxImport();
		return;
	}
	if (PSX_IS_CONST1(_psxRs_))
	{
		g_psxConstRegs[_psxRt_] = g_psxConstRegs[_psxRs_] + _psxImm_;
		PSX_SET_CONST(_psxRt_);
		return;
	}
	PSX_DEL_CONST(_psxRt_);
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	if (_psxImm_ != 0)
	{
		if (Assembler::IsImmAddSub(std::abs(_psxImm_)))
		{
			if (_psxImm_ > 0)
				armAsm->Add(RWPSXSCRATCH, RWPSXSCRATCH, _psxImm_);
			else
				armAsm->Sub(RWPSXSCRATCH, RWPSXSCRATCH, -_psxImm_);
		}
		else
		{
			armAsm->Mov(RWPSXSCRATCH2, (u32)(s32)_psxImm_);
			armAsm->Add(RWPSXSCRATCH, RWPSXSCRATCH, RWPSXSCRATCH2);
		}
	}
	iopArmStoreGPR(RWPSXSCRATCH, _psxRt_);
}
#endif

#if ISTUB_IOP_SLTI
REC_FUNC(SLTI)
#else
static void rpsxSLTI()
{
	if (!_psxRt_) return;
	if (PSX_IS_CONST1(_psxRs_))
	{
		g_psxConstRegs[_psxRt_] = (s32)g_psxConstRegs[_psxRs_] < _psxImm_ ? 1 : 0;
		PSX_SET_CONST(_psxRt_);
		return;
	}
	PSX_DEL_CONST(_psxRt_);
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	armAsm->Mov(RWPSXSCRATCH2, (u32)(s32)_psxImm_);
	armAsm->Cmp(RWPSXSCRATCH, RWPSXSCRATCH2);
	armAsm->Cset(RWPSXSCRATCH, lt);
	iopArmStoreGPR(RWPSXSCRATCH, _psxRt_);
}
#endif

#if ISTUB_IOP_SLTIU
REC_FUNC(SLTIU)
#else
static void rpsxSLTIU()
{
	if (!_psxRt_) return;
	if (PSX_IS_CONST1(_psxRs_))
	{
		g_psxConstRegs[_psxRt_] = g_psxConstRegs[_psxRs_] < (u32)(s32)_psxImm_ ? 1 : 0;
		PSX_SET_CONST(_psxRt_);
		return;
	}
	PSX_DEL_CONST(_psxRt_);
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	armAsm->Mov(RWPSXSCRATCH2, (u32)(s32)_psxImm_);
	armAsm->Cmp(RWPSXSCRATCH, RWPSXSCRATCH2);
	armAsm->Cset(RWPSXSCRATCH, lo);
	iopArmStoreGPR(RWPSXSCRATCH, _psxRt_);
}
#endif

#if ISTUB_IOP_ANDI
REC_FUNC(ANDI)
#else
static void rpsxANDI()
{
	if (!_psxRt_) return;
	if (PSX_IS_CONST1(_psxRs_))
	{
		g_psxConstRegs[_psxRt_] = g_psxConstRegs[_psxRs_] & _psxImmU_;
		PSX_SET_CONST(_psxRt_);
		return;
	}
	PSX_DEL_CONST(_psxRt_);
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	if (_psxImmU_ != 0 && Assembler::IsImmLogical(_psxImmU_, 32))
		armAsm->And(RWPSXSCRATCH, RWPSXSCRATCH, _psxImmU_);
	else
	{
		armAsm->Mov(RWPSXSCRATCH2, (u32)_psxImmU_);
		armAsm->And(RWPSXSCRATCH, RWPSXSCRATCH, RWPSXSCRATCH2);
	}
	iopArmStoreGPR(RWPSXSCRATCH, _psxRt_);
}
#endif

#if ISTUB_IOP_ORI
REC_FUNC(ORI)
#else
static void rpsxORI()
{
	if (!_psxRt_) return;
	if (PSX_IS_CONST1(_psxRs_))
	{
		g_psxConstRegs[_psxRt_] = g_psxConstRegs[_psxRs_] | _psxImmU_;
		PSX_SET_CONST(_psxRt_);
		return;
	}
	PSX_DEL_CONST(_psxRt_);
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	if (_psxImmU_ != 0)
	{
		if (Assembler::IsImmLogical(_psxImmU_, 32))
			armAsm->Orr(RWPSXSCRATCH, RWPSXSCRATCH, _psxImmU_);
		else
		{
			armAsm->Mov(RWPSXSCRATCH2, (u32)_psxImmU_);
			armAsm->Orr(RWPSXSCRATCH, RWPSXSCRATCH, RWPSXSCRATCH2);
		}
	}
	iopArmStoreGPR(RWPSXSCRATCH, _psxRt_);
}
#endif

#if ISTUB_IOP_XORI
REC_FUNC(XORI)
#else
static void rpsxXORI()
{
	if (!_psxRt_) return;
	if (PSX_IS_CONST1(_psxRs_))
	{
		g_psxConstRegs[_psxRt_] = g_psxConstRegs[_psxRs_] ^ _psxImmU_;
		PSX_SET_CONST(_psxRt_);
		return;
	}
	PSX_DEL_CONST(_psxRt_);
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	if (_psxImmU_ != 0)
	{
		if (Assembler::IsImmLogical(_psxImmU_, 32))
			armAsm->Eor(RWPSXSCRATCH, RWPSXSCRATCH, _psxImmU_);
		else
		{
			armAsm->Mov(RWPSXSCRATCH2, (u32)_psxImmU_);
			armAsm->Eor(RWPSXSCRATCH, RWPSXSCRATCH, RWPSXSCRATCH2);
		}
	}
	iopArmStoreGPR(RWPSXSCRATCH, _psxRt_);
}
#endif

#if ISTUB_IOP_LUI
REC_FUNC(LUI)
#else
static void rpsxLUI()
{
	if (!_psxRt_) return;
	g_psxConstRegs[_psxRt_] = psxRegs.code << 16;
	PSX_SET_CONST(_psxRt_);
}
#endif

// Rd = Rs op Rt
#define REC_ALU_RD_RS_RT(name, op, interp_fn) \
	static void rpsx##name() \
	{ \
		if (!_psxRd_) return; \
		if (PSX_IS_CONST2(_psxRs_, _psxRt_)) \
		{ \
			g_psxConstRegs[_psxRd_] = g_psxConstRegs[_psxRs_] op g_psxConstRegs[_psxRt_]; \
			PSX_SET_CONST(_psxRd_); \
			return; \
		} \
		/* Load Rs/Rt BEFORE deleting Rd's const, in case Rd aliases Rs or Rt */ \
		/* with a dirty const (Has=1, Flushed=0). PSX_DEL_CONST clears the Has bit */ \
		/* but never flushes; iopArmLoadGPR would then read stale memory. */ \
		iopArmLoadGPR(RWPSXSCRATCH, _psxRs_); \
		iopArmLoadGPR(RWPSXSCRATCH2, _psxRt_); \
		PSX_DEL_CONST(_psxRd_); \
		armAsm->interp_fn(RWPSXSCRATCH, RWPSXSCRATCH, RWPSXSCRATCH2); \
		iopArmStoreGPR(RWPSXSCRATCH, _psxRd_); \
	}

#if ISTUB_IOP_ADD
REC_FUNC(ADD)
#else
REC_ALU_RD_RS_RT(ADD, +, Add)
#endif

#if ISTUB_IOP_ADDU
REC_FUNC(ADDU)
#else
REC_ALU_RD_RS_RT(ADDU, +, Add)
#endif

#if ISTUB_IOP_SUB
REC_FUNC(SUB)
#else
REC_ALU_RD_RS_RT(SUB, -, Sub)
#endif

#if ISTUB_IOP_SUBU
REC_FUNC(SUBU)
#else
REC_ALU_RD_RS_RT(SUBU, -, Sub)
#endif

#if ISTUB_IOP_AND
REC_FUNC(AND)
#else
REC_ALU_RD_RS_RT(AND, &, And)
#endif

#if ISTUB_IOP_OR
REC_FUNC(OR)
#else
REC_ALU_RD_RS_RT(OR, |, Orr)
#endif

#if ISTUB_IOP_XOR
REC_FUNC(XOR)
#else
REC_ALU_RD_RS_RT(XOR, ^, Eor)
#endif

#if ISTUB_IOP_NOR
REC_FUNC(NOR)
#else
static void rpsxNOR()
{
	if (!_psxRd_) return;
	if (PSX_IS_CONST2(_psxRs_, _psxRt_))
	{
		g_psxConstRegs[_psxRd_] = ~(g_psxConstRegs[_psxRs_] | g_psxConstRegs[_psxRt_]);
		PSX_SET_CONST(_psxRd_);
		return;
	}
	// Load Rs/Rt BEFORE deleting Rd's const, in case Rd aliases an input.
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	iopArmLoadGPR(RWPSXSCRATCH2, _psxRt_);
	PSX_DEL_CONST(_psxRd_);
	armAsm->Orr(RWPSXSCRATCH, RWPSXSCRATCH, RWPSXSCRATCH2);
	armAsm->Mvn(RWPSXSCRATCH, RWPSXSCRATCH);
	iopArmStoreGPR(RWPSXSCRATCH, _psxRd_);
}
#endif

#if ISTUB_IOP_SLT
REC_FUNC(SLT)
#else
static void rpsxSLT()
{
	if (!_psxRd_) return;
	if (PSX_IS_CONST2(_psxRs_, _psxRt_))
	{
		g_psxConstRegs[_psxRd_] = (s32)g_psxConstRegs[_psxRs_] < (s32)g_psxConstRegs[_psxRt_] ? 1 : 0;
		PSX_SET_CONST(_psxRd_);
		return;
	}
	// Load Rs/Rt BEFORE deleting Rd's const, in case Rd aliases an input.
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	iopArmLoadGPR(RWPSXSCRATCH2, _psxRt_);
	PSX_DEL_CONST(_psxRd_);
	armAsm->Cmp(RWPSXSCRATCH, RWPSXSCRATCH2);
	armAsm->Cset(RWPSXSCRATCH, lt);
	iopArmStoreGPR(RWPSXSCRATCH, _psxRd_);
}
#endif

#if ISTUB_IOP_SLTU
REC_FUNC(SLTU)
#else
static void rpsxSLTU()
{
	if (!_psxRd_) return;
	if (PSX_IS_CONST2(_psxRs_, _psxRt_))
	{
		g_psxConstRegs[_psxRd_] = g_psxConstRegs[_psxRs_] < g_psxConstRegs[_psxRt_] ? 1 : 0;
		PSX_SET_CONST(_psxRd_);
		return;
	}
	// Load Rs/Rt BEFORE deleting Rd's const, in case Rd aliases an input.
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	iopArmLoadGPR(RWPSXSCRATCH2, _psxRt_);
	PSX_DEL_CONST(_psxRd_);
	armAsm->Cmp(RWPSXSCRATCH, RWPSXSCRATCH2);
	armAsm->Cset(RWPSXSCRATCH, lo);
	iopArmStoreGPR(RWPSXSCRATCH, _psxRd_);
}
#endif

// ============================================================================
//  Shift Instructions
// ============================================================================

#if ISTUB_IOP_SLL
REC_FUNC(SLL)
#else
static void rpsxSLL()
{
	if (!_psxRd_) return;
	if (PSX_IS_CONST1(_psxRt_))
	{
		g_psxConstRegs[_psxRd_] = g_psxConstRegs[_psxRt_] << _psxSa_;
		PSX_SET_CONST(_psxRd_);
		return;
	}
	PSX_DEL_CONST(_psxRd_);
	iopArmLoadGPR(RWPSXSCRATCH, _psxRt_);
	if (_psxSa_ != 0)
		armAsm->Lsl(RWPSXSCRATCH, RWPSXSCRATCH, _psxSa_);
	iopArmStoreGPR(RWPSXSCRATCH, _psxRd_);
}
#endif

#if ISTUB_IOP_SRL
REC_FUNC(SRL)
#else
static void rpsxSRL()
{
	if (!_psxRd_) return;
	if (PSX_IS_CONST1(_psxRt_))
	{
		g_psxConstRegs[_psxRd_] = g_psxConstRegs[_psxRt_] >> _psxSa_;
		PSX_SET_CONST(_psxRd_);
		return;
	}
	PSX_DEL_CONST(_psxRd_);
	iopArmLoadGPR(RWPSXSCRATCH, _psxRt_);
	if (_psxSa_ != 0)
		armAsm->Lsr(RWPSXSCRATCH, RWPSXSCRATCH, _psxSa_);
	iopArmStoreGPR(RWPSXSCRATCH, _psxRd_);
}
#endif

#if ISTUB_IOP_SRA
REC_FUNC(SRA)
#else
static void rpsxSRA()
{
	if (!_psxRd_) return;
	if (PSX_IS_CONST1(_psxRt_))
	{
		g_psxConstRegs[_psxRd_] = (s32)g_psxConstRegs[_psxRt_] >> _psxSa_;
		PSX_SET_CONST(_psxRd_);
		return;
	}
	PSX_DEL_CONST(_psxRd_);
	iopArmLoadGPR(RWPSXSCRATCH, _psxRt_);
	if (_psxSa_ != 0)
		armAsm->Asr(RWPSXSCRATCH, RWPSXSCRATCH, _psxSa_);
	iopArmStoreGPR(RWPSXSCRATCH, _psxRd_);
}
#endif

#if ISTUB_IOP_SLLV
REC_FUNC(SLLV)
#else
static void rpsxSLLV()
{
	if (!_psxRd_) return;
	// Load inputs BEFORE deleting Rd's const, in case Rd aliases an input.
	iopArmLoadGPR(RWPSXSCRATCH, _psxRt_);
	iopArmLoadGPR(RWPSXSCRATCH2, _psxRs_);
	PSX_DEL_CONST(_psxRd_);
	armAsm->Lsl(RWPSXSCRATCH, RWPSXSCRATCH, RWPSXSCRATCH2);
	iopArmStoreGPR(RWPSXSCRATCH, _psxRd_);
}
#endif

#if ISTUB_IOP_SRLV
REC_FUNC(SRLV)
#else
static void rpsxSRLV()
{
	if (!_psxRd_) return;
	// Load inputs BEFORE deleting Rd's const, in case Rd aliases an input.
	iopArmLoadGPR(RWPSXSCRATCH, _psxRt_);
	iopArmLoadGPR(RWPSXSCRATCH2, _psxRs_);
	PSX_DEL_CONST(_psxRd_);
	armAsm->Lsr(RWPSXSCRATCH, RWPSXSCRATCH, RWPSXSCRATCH2);
	iopArmStoreGPR(RWPSXSCRATCH, _psxRd_);
}
#endif

#if ISTUB_IOP_SRAV
REC_FUNC(SRAV)
#else
static void rpsxSRAV()
{
	if (!_psxRd_) return;
	// Load inputs BEFORE deleting Rd's const, in case Rd aliases an input.
	iopArmLoadGPR(RWPSXSCRATCH, _psxRt_);
	iopArmLoadGPR(RWPSXSCRATCH2, _psxRs_);
	PSX_DEL_CONST(_psxRd_);
	armAsm->Asr(RWPSXSCRATCH, RWPSXSCRATCH, RWPSXSCRATCH2);
	iopArmStoreGPR(RWPSXSCRATCH, _psxRd_);
}
#endif

// ============================================================================
//  Multiply / Divide
// ============================================================================

#if ISTUB_IOP_MULT
REC_FUNC(MULT)
#else
static void rpsxMULT()
{
	// [HI, LO] = (s32)Rs * (s32)Rt
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	iopArmLoadGPR(RWPSXSCRATCH2, _psxRt_);
	// SMULL gives 64-bit signed result in x-register
	armAsm->Smull(RPSXSCRATCH, RWPSXSCRATCH, RWPSXSCRATCH2);
	// LO = lower 32 bits
	armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_LO_OFFSET));
	// HI = upper 32 bits
	armAsm->Lsr(RPSXSCRATCH, RPSXSCRATCH, 32);
	armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_HI_OFFSET));
	PSX_DEL_CONST(33); // LO
	PSX_DEL_CONST(32); // HI
	g_iopCyclePenalty = psxInstCycles_Mult;
}
#endif

#if ISTUB_IOP_MULTU
REC_FUNC(MULTU)
#else
static void rpsxMULTU()
{
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	iopArmLoadGPR(RWPSXSCRATCH2, _psxRt_);
	armAsm->Umull(RPSXSCRATCH, RWPSXSCRATCH, RWPSXSCRATCH2);
	armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_LO_OFFSET));
	armAsm->Lsr(RPSXSCRATCH, RPSXSCRATCH, 32);
	armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_HI_OFFSET));
	PSX_DEL_CONST(33);
	PSX_DEL_CONST(32);
	g_iopCyclePenalty = psxInstCycles_Mult;
}
#endif

#if ISTUB_IOP_DIV
REC_FUNC(DIV)
#else
static void rpsxDIV()
{
	// Use interpreter for divide — handles edge cases (div by zero, overflow)
	iopArmCallInterpreter(psxDIV);
	g_iopCyclePenalty = psxInstCycles_Div;
}
#endif

#if ISTUB_IOP_DIVU
REC_FUNC(DIVU)
#else
static void rpsxDIVU()
{
	iopArmCallInterpreter(psxDIVU);
	g_iopCyclePenalty = psxInstCycles_Div;
}
#endif

// ============================================================================
//  Move HI/LO
// ============================================================================

#if ISTUB_IOP_MFHI
REC_FUNC(MFHI)
#else
static void rpsxMFHI()
{
	if (!_psxRd_) return;
	PSX_DEL_CONST(_psxRd_);
	armAsm->Ldr(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_HI_OFFSET));
	iopArmStoreGPR(RWPSXSCRATCH, _psxRd_);
}
#endif

#if ISTUB_IOP_MTHI
REC_FUNC(MTHI)
#else
static void rpsxMTHI()
{
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_HI_OFFSET));
}
#endif

#if ISTUB_IOP_MFLO
REC_FUNC(MFLO)
#else
static void rpsxMFLO()
{
	if (!_psxRd_) return;
	PSX_DEL_CONST(_psxRd_);
	armAsm->Ldr(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_LO_OFFSET));
	iopArmStoreGPR(RWPSXSCRATCH, _psxRd_);
}
#endif

#if ISTUB_IOP_MTLO
REC_FUNC(MTLO)
#else
static void rpsxMTLO()
{
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_LO_OFFSET));
}
#endif

// ============================================================================
//  Branch Instructions — Native ARM64 implementations
// ============================================================================
//
// All branches follow the same pattern:
//   1. Evaluate condition (const-prop or emitted compare)
//   2. Compile delay slot via psxRecompileNextInstruction(true, ...)
//   3. Emit psxSetBranchImm/psxSetBranchReg for taken/not-taken paths
//
// For conditional branches with non-const operands, we emit both paths:
//   - Not-taken path: psxSetBranchImm(psxpc)  [fall-through]
//   - Taken path:     psxSetBranchImm(branchTo)
// The delay slot is compiled twice (once per path) using save/restore,
// unless psxTrySwapDelaySlot moved it before the compare.

// Helper: emit a conditional branch comparing two GPRs (or one vs zero).
// Each branch function uses a local Label for the not-taken path.

// Emit compare for BEQ/BNE: Rs vs Rt
static void rpsxEmitBranchEQ()
{
	iopArmFlushConstRegs();
	// Load both operands into scratch registers and compare.
	// iopArmLoadGPR handles const-prop (emits MOV imm if const).
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	iopArmLoadGPR(RWPSXSCRATCH2, _psxRt_);
	armAsm->Cmp(RWPSXSCRATCH, RWPSXSCRATCH2);
}

// Emit compare for BLTZ/BGEZ/BLEZ/BGTZ: Rs vs 0
static void rpsxEmitBranchRsZero()
{
	iopArmFlushConstRegs();
	iopArmLoadGPR(RWPSXSCRATCH, _psxRs_);
	armAsm->Cmp(RWPSXSCRATCH, 0);
}

// ---------- J / JAL ----------

#if ISTUB_IOP_J
static void rpsxJ() { iopArmBranchCallInterpreter(psxJ); }
#else
static void rpsxJ()
{
	u32 newpc = _psxTarget_ * 4 + (psxpc & 0xf0000000);
	psxRecompileNextInstruction(true, false);
	psxSetBranchImm(newpc);
}
#endif

#if ISTUB_IOP_JAL
static void rpsxJAL() { iopArmBranchCallInterpreter(psxJAL); }
#else
static void rpsxJAL()
{
	u32 newpc = (_psxTarget_ << 2) + (psxpc & 0xf0000000);
	PSX_SET_CONST(31);
	g_psxConstRegs[31] = psxpc + 4;
	psxRecompileNextInstruction(true, false);
	psxSetBranchImm(newpc);
}
#endif

// ---------- JR / JALR ----------

#if ISTUB_IOP_JR
static void rpsxJR() { iopArmBranchCallInterpreter(psxJR); }
#else
static void rpsxJR()
{
	const bool swap = psxTrySwapDelaySlot(_psxRs_, 0, 0);

	// Load Rs into callee-saved scratch so it survives delay slot compilation
	if (PSX_IS_CONST1(_psxRs_))
		armAsm->Mov(RWPSXDELAYSLOT, g_psxConstRegs[_psxRs_]);
	else
		armAsm->Ldr(RWPSXDELAYSLOT, MemOperand(RPSXSTATE, PSX_GPR_OFFSET(_psxRs_)));

	if (!swap)
		psxRecompileNextInstruction(true, false);

	// Store jump target to psxRegs.pc
	armAsm->Str(RWPSXDELAYSLOT, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
	psxSetBranchReg();
}
#endif

#if ISTUB_IOP_JALR
static void rpsxJALR() { iopArmBranchCallInterpreter(psxJALR); }
#else
static void rpsxJALR()
{
	// Capture link target BEFORE psxTrySwapDelaySlot — the swap path calls
	// psxRecompileNextInstruction which bumps psxpc by 4 and does NOT restore it
	// (swapped_delayslot only restores psxRegs.code and g_pCurInstInfo). Mirror
	// upstream rpsxJALR at pcsx2/x86/iR3000Atables.cpp:1457.
	const u32 newpc = psxpc + 4;
	const bool swap = (_psxRd_ == _psxRs_) ? false : psxTrySwapDelaySlot(_psxRs_, 0, _psxRd_);

	// Load Rs into callee-saved scratch so it survives delay slot compilation
	if (PSX_IS_CONST1(_psxRs_))
		armAsm->Mov(RWPSXDELAYSLOT, g_psxConstRegs[_psxRs_]);
	else
		armAsm->Ldr(RWPSXDELAYSLOT, MemOperand(RPSXSTATE, PSX_GPR_OFFSET(_psxRs_)));

	// Link register
	if (_psxRd_)
	{
		PSX_SET_CONST(_psxRd_);
		g_psxConstRegs[_psxRd_] = newpc;
	}

	if (!swap)
		psxRecompileNextInstruction(true, false);

	// Store jump target to psxRegs.pc
	armAsm->Str(RWPSXDELAYSLOT, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
	psxSetBranchReg();
}
#endif

// ---------- BEQ ----------

#if ISTUB_IOP_BEQ
static void rpsxBEQ() { iopArmBranchCallInterpreter(psxBEQ); }
#else
static void rpsxBEQ()
{
	u32 branchTo = ((s32)_psxImm_ * 4) + psxpc;

	// Both const: resolve at compile time
	if (PSX_IS_CONST2(_psxRs_, _psxRt_))
	{
		if (g_psxConstRegs[_psxRs_] != g_psxConstRegs[_psxRt_])
			branchTo = psxpc + 4;
		psxRecompileNextInstruction(true, false);
		psxSetBranchImm(branchTo);
		return;
	}

	// Rs == Rt: always taken
	if (_psxRs_ == _psxRt_)
	{
		psxRecompileNextInstruction(true, false);
		psxSetBranchImm(branchTo);
		return;
	}

	// General case: emit compare and two paths
	const bool swap = psxTrySwapDelaySlot(_psxRs_, _psxRt_, 0);
	rpsxEmitBranchEQ();
	Label notTaken;
	armAsm->B(&notTaken, ne); // skip to not-taken if Rs != Rt

	// --- Taken path ---
	if (!swap)
	{
		psxSaveBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(branchTo);

	// --- Not-taken path ---
	armAsm->Bind(&notTaken);
	if (!swap)
	{
		psxpc -= 4;
		psxLoadBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(psxpc);
}
#endif

// ---------- BNE ----------

#if ISTUB_IOP_BNE
static void rpsxBNE() { iopArmBranchCallInterpreter(psxBNE); }
#else
static void rpsxBNE()
{
	u32 branchTo = ((s32)_psxImm_ * 4) + psxpc;

	if (PSX_IS_CONST2(_psxRs_, _psxRt_))
	{
		if (g_psxConstRegs[_psxRs_] == g_psxConstRegs[_psxRt_])
			branchTo = psxpc + 4;
		psxRecompileNextInstruction(true, false);
		psxSetBranchImm(branchTo);
		return;
	}

	// Rs == Rt: never taken
	if (_psxRs_ == _psxRt_)
	{
		psxRecompileNextInstruction(true, false);
		psxSetBranchImm(psxpc);
		return;
	}

	const bool swap = psxTrySwapDelaySlot(_psxRs_, _psxRt_, 0);
	rpsxEmitBranchEQ();
	Label notTaken;
	armAsm->B(&notTaken, eq); // skip to not-taken if Rs == Rt

	// --- Taken path (Rs != Rt) ---
	if (!swap)
	{
		psxSaveBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(branchTo);

	// --- Not-taken path ---
	armAsm->Bind(&notTaken);
	if (!swap)
	{
		psxpc -= 4;
		psxLoadBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(psxpc);
}
#endif

// ---------- BLEZ ----------

#if ISTUB_IOP_BLEZ
static void rpsxBLEZ() { iopArmBranchCallInterpreter(psxBLEZ); }
#else
static void rpsxBLEZ()
{
	u32 branchTo = ((s32)_psxImm_ * 4) + psxpc;

	if (PSX_IS_CONST1(_psxRs_))
	{
		if ((s32)g_psxConstRegs[_psxRs_] > 0)
			branchTo = psxpc + 4;
		psxRecompileNextInstruction(true, false);
		psxSetBranchImm(branchTo);
		return;
	}

	const bool swap = psxTrySwapDelaySlot(_psxRs_, 0, 0);
	rpsxEmitBranchRsZero();
	Label notTaken;
	armAsm->B(&notTaken, gt); // skip to not-taken if Rs > 0

	if (!swap)
	{
		psxSaveBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(branchTo);

	armAsm->Bind(&notTaken);
	if (!swap)
	{
		psxpc -= 4;
		psxLoadBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(psxpc);
}
#endif

// ---------- BGTZ ----------

#if ISTUB_IOP_BGTZ
static void rpsxBGTZ() { iopArmBranchCallInterpreter(psxBGTZ); }
#else
static void rpsxBGTZ()
{
	u32 branchTo = ((s32)_psxImm_ * 4) + psxpc;

	if (PSX_IS_CONST1(_psxRs_))
	{
		if ((s32)g_psxConstRegs[_psxRs_] <= 0)
			branchTo = psxpc + 4;
		psxRecompileNextInstruction(true, false);
		psxSetBranchImm(branchTo);
		return;
	}

	const bool swap = psxTrySwapDelaySlot(_psxRs_, 0, 0);
	rpsxEmitBranchRsZero();
	Label notTaken;
	armAsm->B(&notTaken, le); // skip to not-taken if Rs <= 0

	if (!swap)
	{
		psxSaveBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(branchTo);

	armAsm->Bind(&notTaken);
	if (!swap)
	{
		psxpc -= 4;
		psxLoadBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(psxpc);
}
#endif

// ---------- BLTZ ----------

#if ISTUB_IOP_BLTZ
static void rpsxBLTZ() { iopArmBranchCallInterpreter(psxBLTZ); }
#else
static void rpsxBLTZ()
{
	u32 branchTo = ((s32)_psxImm_ * 4) + psxpc;

	if (PSX_IS_CONST1(_psxRs_))
	{
		if ((s32)g_psxConstRegs[_psxRs_] >= 0)
			branchTo = psxpc + 4;
		psxRecompileNextInstruction(true, false);
		psxSetBranchImm(branchTo);
		return;
	}

	const bool swap = psxTrySwapDelaySlot(_psxRs_, 0, 0);
	rpsxEmitBranchRsZero();
	Label notTaken;
	armAsm->B(&notTaken, ge); // skip to not-taken if Rs >= 0

	if (!swap)
	{
		psxSaveBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(branchTo);

	armAsm->Bind(&notTaken);
	if (!swap)
	{
		psxpc -= 4;
		psxLoadBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(psxpc);
}
#endif

// ---------- BGEZ ----------

#if ISTUB_IOP_BGEZ
static void rpsxBGEZ() { iopArmBranchCallInterpreter(psxBGEZ); }
#else
static void rpsxBGEZ()
{
	u32 branchTo = ((s32)_psxImm_ * 4) + psxpc;

	if (PSX_IS_CONST1(_psxRs_))
	{
		if ((s32)g_psxConstRegs[_psxRs_] < 0)
			branchTo = psxpc + 4;
		psxRecompileNextInstruction(true, false);
		psxSetBranchImm(branchTo);
		return;
	}

	const bool swap = psxTrySwapDelaySlot(_psxRs_, 0, 0);
	rpsxEmitBranchRsZero();
	Label notTaken;
	armAsm->B(&notTaken, lt); // skip to not-taken if Rs < 0

	if (!swap)
	{
		psxSaveBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(branchTo);

	armAsm->Bind(&notTaken);
	if (!swap)
	{
		psxpc -= 4;
		psxLoadBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(psxpc);
}
#endif

// ---------- BLTZAL ----------

#if ISTUB_IOP_BLTZAL
static void rpsxBLTZAL() { iopArmBranchCallInterpreter(psxBLTZAL); }
#else
static void rpsxBLTZAL()
{
	u32 branchTo = ((s32)_psxImm_ * 4) + psxpc;

	PSX_SET_CONST(31);
	g_psxConstRegs[31] = psxpc + 4;

	if (PSX_IS_CONST1(_psxRs_))
	{
		if ((s32)g_psxConstRegs[_psxRs_] >= 0)
			branchTo = psxpc + 4;
		psxRecompileNextInstruction(true, false);
		psxSetBranchImm(branchTo);
		return;
	}

	const bool swap = psxTrySwapDelaySlot(_psxRs_, 0, 0);
	rpsxEmitBranchRsZero();
	Label notTaken;
	armAsm->B(&notTaken, ge);

	if (!swap)
	{
		psxSaveBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(branchTo);

	armAsm->Bind(&notTaken);
	if (!swap)
	{
		psxpc -= 4;
		psxLoadBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(psxpc);
}
#endif

// ---------- BGEZAL ----------

#if ISTUB_IOP_BGEZAL
static void rpsxBGEZAL() { iopArmBranchCallInterpreter(psxBGEZAL); }
#else
static void rpsxBGEZAL()
{
	u32 branchTo = ((s32)_psxImm_ * 4) + psxpc;

	PSX_SET_CONST(31);
	g_psxConstRegs[31] = psxpc + 4;

	if (PSX_IS_CONST1(_psxRs_))
	{
		if ((s32)g_psxConstRegs[_psxRs_] < 0)
			branchTo = psxpc + 4;
		psxRecompileNextInstruction(true, false);
		psxSetBranchImm(branchTo);
		return;
	}

	const bool swap = psxTrySwapDelaySlot(_psxRs_, 0, 0);
	rpsxEmitBranchRsZero();
	Label notTaken;
	armAsm->B(&notTaken, lt);

	if (!swap)
	{
		psxSaveBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(branchTo);

	armAsm->Bind(&notTaken);
	if (!swap)
	{
		psxpc -= 4;
		psxLoadBranchState();
		psxRecompileNextInstruction(true, false);
	}
	psxSetBranchImm(psxpc);
}
#endif

// ============================================================================
//  Load/Store Instructions
// ============================================================================

#if ISTUB_IOP_LB
REC_FUNC(LB)
#else
static void rpsxLB()
{
	// NOTE: do NOT early-return on Rt == 0. The interp psxLB always calls
	// iopMemRead8 (with the result discarded if Rt is r0) because HW
	// register reads have side effects — auto-clearing INT status,
	// popping FIFOs, etc. PS1 BIOS uses `LB r0, X(rs)` to consume HW
	// state; skipping the read here causes native-only hangs that don't
	// repro in pure-interp mode.
	// Load Rs BEFORE deleting Rt's const, in case Rs == Rt with a dirty const.
	iopArmLoadGPR(RWARG1, _psxRs_);
	if (_psxImm_ != 0)
	{
		if (Assembler::IsImmAddSub(std::abs(_psxImm_)))
		{
			if (_psxImm_ > 0)
				armAsm->Add(RWARG1, RWARG1, _psxImm_);
			else
				armAsm->Sub(RWARG1, RWARG1, -_psxImm_);
		}
		else
		{
			armAsm->Mov(RWPSXSCRATCH, (u32)(s32)_psxImm_);
			armAsm->Add(RWARG1, RWARG1, RWPSXSCRATCH);
		}
	}
	if (_psxRt_)
		PSX_DEL_CONST(_psxRt_);
	iopArmFlushConstRegs();
	armEmitCall((const void*)iopMemRead8);
	if (_psxRt_)
	{
		armAsm->Sxtb(RWRET, RWRET);
		iopArmStoreGPR(RWRET, _psxRt_);
	}
}
#endif

#if ISTUB_IOP_LBU
REC_FUNC(LBU)
#else
static void rpsxLBU()
{
	// NOTE: do NOT early-return on Rt == 0. The interp psxLBU always calls
	// iopMemRead8 (discarding the result if Rt is r0) because HW reads have
	// side effects (HW_ICTRL auto-clear, FIFO pop, etc.). PS1 BIOS uses
	// `LBU r0, X(rs)` to consume HW state; the previous early-return here
	// caused native-only hangs that didn't repro in pure-interp mode.
	// Load Rs BEFORE deleting Rt's const, in case Rs == Rt with a dirty const.
	iopArmLoadGPR(RWARG1, _psxRs_);
	if (_psxImm_ != 0)
	{
		if (Assembler::IsImmAddSub(std::abs(_psxImm_)))
		{
			if (_psxImm_ > 0)
				armAsm->Add(RWARG1, RWARG1, _psxImm_);
			else
				armAsm->Sub(RWARG1, RWARG1, -_psxImm_);
		}
		else
		{
			armAsm->Mov(RWPSXSCRATCH, (u32)(s32)_psxImm_);
			armAsm->Add(RWARG1, RWARG1, RWPSXSCRATCH);
		}
	}
	if (_psxRt_)
		PSX_DEL_CONST(_psxRt_);
	iopArmFlushConstRegs();
	armEmitCall((const void*)iopMemRead8);
	if (_psxRt_)
	{
		// Explicit Uxtb because AAPCS64's "producer zero-extends u8" isn't
		// reliably emitted by Clang at iopMemRead8's return path.
		armAsm->Uxtb(RWRET, RWRET);
		iopArmStoreGPR(RWRET, _psxRt_);
	}
}
#endif

#if ISTUB_IOP_LH
REC_FUNC(LH)
#else
static void rpsxLH()
{
	// Always emit the read for HW side effects — see rpsxLB note.
	iopArmLoadGPR(RWARG1, _psxRs_);
	if (_psxImm_ != 0)
	{
		if (Assembler::IsImmAddSub(std::abs(_psxImm_)))
		{
			if (_psxImm_ > 0)
				armAsm->Add(RWARG1, RWARG1, _psxImm_);
			else
				armAsm->Sub(RWARG1, RWARG1, -_psxImm_);
		}
		else
		{
			armAsm->Mov(RWPSXSCRATCH, (u32)(s32)_psxImm_);
			armAsm->Add(RWARG1, RWARG1, RWPSXSCRATCH);
		}
	}
	if (_psxRt_)
		PSX_DEL_CONST(_psxRt_);
	iopArmFlushConstRegs();
	armEmitCall((const void*)iopMemRead16);
	if (_psxRt_)
	{
		armAsm->Sxth(RWRET, RWRET);
		iopArmStoreGPR(RWRET, _psxRt_);
	}
}
#endif

#if ISTUB_IOP_LHU
REC_FUNC(LHU)
#else
static void rpsxLHU()
{
	// Always emit the read for HW side effects — see rpsxLB note.
	iopArmLoadGPR(RWARG1, _psxRs_);
	if (_psxImm_ != 0)
	{
		if (Assembler::IsImmAddSub(std::abs(_psxImm_)))
		{
			if (_psxImm_ > 0)
				armAsm->Add(RWARG1, RWARG1, _psxImm_);
			else
				armAsm->Sub(RWARG1, RWARG1, -_psxImm_);
		}
		else
		{
			armAsm->Mov(RWPSXSCRATCH, (u32)(s32)_psxImm_);
			armAsm->Add(RWARG1, RWARG1, RWPSXSCRATCH);
		}
	}
	if (_psxRt_)
		PSX_DEL_CONST(_psxRt_);
	iopArmFlushConstRegs();
	armEmitCall((const void*)iopMemRead16);
	if (_psxRt_)
	{
		// Explicit Uxth — same AAPCS64 caveat as LBU.
		armAsm->Uxth(RWRET, RWRET);
		iopArmStoreGPR(RWRET, _psxRt_);
	}
}
#endif

#if ISTUB_IOP_LW
REC_FUNC(LW)
#else
static void rpsxLW()
{
	// Always emit the read for HW side effects — see rpsxLB note.
	iopArmLoadGPR(RWARG1, _psxRs_);
	if (_psxImm_ != 0)
	{
		if (Assembler::IsImmAddSub(std::abs(_psxImm_)))
		{
			if (_psxImm_ > 0)
				armAsm->Add(RWARG1, RWARG1, _psxImm_);
			else
				armAsm->Sub(RWARG1, RWARG1, -_psxImm_);
		}
		else
		{
			armAsm->Mov(RWPSXSCRATCH, (u32)(s32)_psxImm_);
			armAsm->Add(RWARG1, RWARG1, RWPSXSCRATCH);
		}
	}
	if (_psxRt_)
		PSX_DEL_CONST(_psxRt_);
	iopArmFlushConstRegs();
	armEmitCall((const void*)iopMemRead32);
	if (_psxRt_)
		iopArmStoreGPR(RWRET, _psxRt_);
}
#endif

// Unaligned loads — always interpreter stubs
REC_FUNC(LWL)
REC_FUNC(LWR)

#if ISTUB_IOP_SB
REC_FUNC(SB)
#else
static void rpsxSB()
{
	iopArmLoadGPR(RWARG1, _psxRs_);
	if (_psxImm_ != 0)
	{
		if (Assembler::IsImmAddSub(std::abs(_psxImm_)))
		{
			if (_psxImm_ > 0)
				armAsm->Add(RWARG1, RWARG1, _psxImm_);
			else
				armAsm->Sub(RWARG1, RWARG1, -_psxImm_);
		}
		else
		{
			armAsm->Mov(RWPSXSCRATCH, (u32)(s32)_psxImm_);
			armAsm->Add(RWARG1, RWARG1, RWPSXSCRATCH);
		}
	}
	iopArmLoadGPR(RWARG2, _psxRt_);
	armAsm->And(RWARG2, RWARG2, 0xFF);
	iopArmFlushConstRegs();
	armEmitCall((const void*)iopMemWrite8);
}
#endif

#if ISTUB_IOP_SH
REC_FUNC(SH)
#else
static void rpsxSH()
{
	iopArmLoadGPR(RWARG1, _psxRs_);
	if (_psxImm_ != 0)
	{
		if (Assembler::IsImmAddSub(std::abs(_psxImm_)))
		{
			if (_psxImm_ > 0)
				armAsm->Add(RWARG1, RWARG1, _psxImm_);
			else
				armAsm->Sub(RWARG1, RWARG1, -_psxImm_);
		}
		else
		{
			armAsm->Mov(RWPSXSCRATCH, (u32)(s32)_psxImm_);
			armAsm->Add(RWARG1, RWARG1, RWPSXSCRATCH);
		}
	}
	iopArmLoadGPR(RWARG2, _psxRt_);
	if (Assembler::IsImmLogical(0xFFFF, 32))
		armAsm->And(RWARG2, RWARG2, 0xFFFF);
	iopArmFlushConstRegs();
	armEmitCall((const void*)iopMemWrite16);
}
#endif

#if ISTUB_IOP_SW
REC_FUNC(SW)
#else
static void rpsxSW()
{
	iopArmLoadGPR(RWARG1, _psxRs_);
	if (_psxImm_ != 0)
	{
		if (Assembler::IsImmAddSub(std::abs(_psxImm_)))
		{
			if (_psxImm_ > 0)
				armAsm->Add(RWARG1, RWARG1, _psxImm_);
			else
				armAsm->Sub(RWARG1, RWARG1, -_psxImm_);
		}
		else
		{
			armAsm->Mov(RWPSXSCRATCH, (u32)(s32)_psxImm_);
			armAsm->Add(RWARG1, RWARG1, RWPSXSCRATCH);
		}
	}
	iopArmLoadGPR(RWARG2, _psxRt_);
	iopArmFlushConstRegs();
	armEmitCall((const void*)iopMemWrite32);
}
#endif

// Unaligned stores — always interpreter stubs
REC_FUNC(SWL)
REC_FUNC(SWR)

// ============================================================================
//  COP0 Instructions
// ============================================================================

#if ISTUB_IOP_MFC0
REC_FUNC(MFC0)
#else
static void rpsxMFC0()
{
	if (!_psxRt_) return;
	PSX_DEL_CONST(_psxRt_);
	armAsm->Ldr(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_CP0_OFFSET(_psxRd_)));
	iopArmStoreGPR(RWPSXSCRATCH, _psxRt_);
}
#endif

#if ISTUB_IOP_CFC0
REC_FUNC(CFC0)
#else
static void rpsxCFC0()
{
	if (!_psxRt_) return;
	PSX_DEL_CONST(_psxRt_);
	armAsm->Ldr(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_CP0_OFFSET(_psxRd_)));
	iopArmStoreGPR(RWPSXSCRATCH, _psxRt_);
}
#endif

#if ISTUB_IOP_MTC0
REC_FUNC(MTC0)
#else
static void rpsxMTC0()
{
	iopArmLoadGPR(RWPSXSCRATCH, _psxRt_);
	armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_CP0_OFFSET(_psxRd_)));
}
#endif

#if ISTUB_IOP_CTC0
REC_FUNC(CTC0)
#else
static void rpsxCTC0()
{
	iopArmLoadGPR(RWPSXSCRATCH, _psxRt_);
	armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_CP0_OFFSET(_psxRd_)));
}
#endif

// RFE always uses interpreter (exception handling)
REC_FUNC(RFE)

// ============================================================================
//  COP2 (GTE) — always interpreter stubs
// ============================================================================

REC_GTE_FUNC(MFC2)
REC_GTE_FUNC(CFC2)
REC_GTE_FUNC(MTC2)
REC_GTE_FUNC(CTC2)
REC_GTE_FUNC(LWC2)
REC_GTE_FUNC(SWC2)
REC_GTE_FUNC(RTPS)
REC_GTE_FUNC(OP)
REC_GTE_FUNC(NCLIP)
REC_GTE_FUNC(DPCS)
REC_GTE_FUNC(INTPL)
REC_GTE_FUNC(MVMVA)
REC_GTE_FUNC(NCDS)
REC_GTE_FUNC(NCDT)
REC_GTE_FUNC(CDP)
REC_GTE_FUNC(NCCS)
REC_GTE_FUNC(CC)
REC_GTE_FUNC(NCS)
REC_GTE_FUNC(NCT)
REC_GTE_FUNC(SQR)
REC_GTE_FUNC(DCPL)
REC_GTE_FUNC(DPCT)
REC_GTE_FUNC(AVSZ3)
REC_GTE_FUNC(AVSZ4)
REC_GTE_FUNC(RTPT)
REC_GTE_FUNC(GPF)
REC_GTE_FUNC(GPL)
REC_GTE_FUNC(NCCT)

// ============================================================================
//  System Instructions — always interpreter stubs
// ============================================================================

static void rpsxSYSCALL()
{
	iopArmFlushCode();
	// Store PC-4 (SYSCALL address, not next instruction)
	armAsm->Mov(RWPSXSCRATCH, psxpc - 4);
	armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
	iopArmFlushConstRegs();

	// psxException(0x20, psxbranch == 1)
	armAsm->Mov(RWARG1, 0x20u);
	armAsm->Mov(RWARG2, (psxbranch == 1) ? 1u : 0u);
	armEmitCall((const void*)psxException);

	// After SYSCALL, branch is taken
	psxbranch = 2;
}

static void rpsxBREAK()
{
	iopArmFlushCode();
	armAsm->Mov(RWPSXSCRATCH, psxpc - 4);
	armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
	iopArmFlushConstRegs();

	armAsm->Mov(RWARG1, 0x24u);
	armAsm->Mov(RWARG2, (psxbranch == 1) ? 1u : 0u);
	armEmitCall((const void*)psxException);

	psxbranch = 2;
}

// ============================================================================
//  NULL / dispatch helpers
// ============================================================================

static void rpsxNULL()
{
	Console.WriteLn("[IOP ARM64 Rec] psxUNK: %8.8x", psxRegs.code);
}

static void rpsxSPECIAL() { rpsxSPC[_psxFunct_](); }
static void rpsxREGIMM()  { rpsxREG[_psxRt_](); }
static void rpsxCOP0()    { rpsxCP0[_psxRs_](); }
static void rpsxCOP2()    { rpsxCP2[_psxFunct_](); }
static void rpsxBASIC()   { rpsxCP2BSC[_psxRs_](); }

// ============================================================================
//  Opcode Dispatch Tables
// ============================================================================

// clang-format off
void (*rpsxBSC[64])() = {
	rpsxSPECIAL, rpsxREGIMM, rpsxJ   , rpsxJAL  , rpsxBEQ , rpsxBNE , rpsxBLEZ, rpsxBGTZ,
	rpsxADDI   , rpsxADDIU , rpsxSLTI, rpsxSLTIU, rpsxANDI, rpsxORI , rpsxXORI, rpsxLUI ,
	rpsxCOP0   , rpsxNULL  , rpsxCOP2, rpsxNULL , rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL,
	rpsxNULL   , rpsxNULL  , rpsxNULL, rpsxNULL , rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL,
	rpsxLB     , rpsxLH    , rpsxLWL , rpsxLW   , rpsxLBU , rpsxLHU , rpsxLWR , rpsxNULL,
	rpsxSB     , rpsxSH    , rpsxSWL , rpsxSW   , rpsxNULL, rpsxNULL, rpsxSWR , rpsxNULL,
	rpsxNULL   , rpsxNULL  , rgteLWC2, rpsxNULL , rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL,
	rpsxNULL   , rpsxNULL  , rgteSWC2, rpsxNULL , rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL,
};

void (*rpsxSPC[64])() = {
	rpsxSLL , rpsxNULL, rpsxSRL , rpsxSRA , rpsxSLLV   , rpsxNULL , rpsxSRLV, rpsxSRAV,
	rpsxJR  , rpsxJALR, rpsxNULL, rpsxNULL, rpsxSYSCALL, rpsxBREAK, rpsxNULL, rpsxNULL,
	rpsxMFHI, rpsxMTHI, rpsxMFLO, rpsxMTLO, rpsxNULL   , rpsxNULL , rpsxNULL, rpsxNULL,
	rpsxMULT, rpsxMULTU,rpsxDIV , rpsxDIVU, rpsxNULL   , rpsxNULL , rpsxNULL, rpsxNULL,
	rpsxADD , rpsxADDU, rpsxSUB , rpsxSUBU, rpsxAND    , rpsxOR   , rpsxXOR , rpsxNOR ,
	rpsxNULL, rpsxNULL, rpsxSLT , rpsxSLTU, rpsxNULL   , rpsxNULL , rpsxNULL, rpsxNULL,
	rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL   , rpsxNULL , rpsxNULL, rpsxNULL,
	rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL   , rpsxNULL , rpsxNULL, rpsxNULL,
};

void (*rpsxREG[32])() = {
	rpsxBLTZ  , rpsxBGEZ  , rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL,
	rpsxNULL  , rpsxNULL  , rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL,
	rpsxBLTZAL, rpsxBGEZAL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL,
	rpsxNULL  , rpsxNULL  , rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL,
};

void (*rpsxCP0[32])() = {
	rpsxMFC0, rpsxNULL, rpsxCFC0, rpsxNULL, rpsxMTC0, rpsxNULL, rpsxCTC0, rpsxNULL,
	rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL,
	rpsxRFE , rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL,
	rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL,
};

void (*rpsxCP2[64])() = {
	rpsxBASIC, rgteRTPS , rpsxNULL , rpsxNULL, rpsxNULL, rpsxNULL , rgteNCLIP, rpsxNULL,
	rpsxNULL , rpsxNULL , rpsxNULL , rpsxNULL, rgteOP  , rpsxNULL , rpsxNULL , rpsxNULL,
	rgteDPCS , rgteINTPL, rgteMVMVA, rgteNCDS, rgteCDP , rpsxNULL , rgteNCDT , rpsxNULL,
	rpsxNULL , rpsxNULL , rpsxNULL , rgteNCCS, rgteCC  , rpsxNULL , rgteNCS  , rpsxNULL,
	rgteNCT  , rpsxNULL , rpsxNULL , rpsxNULL, rpsxNULL, rpsxNULL , rpsxNULL , rpsxNULL,
	rgteSQR  , rgteDCPL , rgteDPCT , rpsxNULL, rpsxNULL, rgteAVSZ3, rgteAVSZ4, rpsxNULL,
	rgteRTPT , rpsxNULL , rpsxNULL , rpsxNULL, rpsxNULL, rpsxNULL , rpsxNULL , rpsxNULL,
	rpsxNULL , rpsxNULL , rpsxNULL , rpsxNULL, rpsxNULL, rgteGPF  , rgteGPL  , rgteNCCT,
};

void (*rpsxCP2BSC[32])() = {
	rgteMFC2, rpsxNULL, rgteCFC2, rpsxNULL, rgteMTC2, rpsxNULL, rgteCTC2, rpsxNULL,
	rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL,
	rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL,
	rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL, rpsxNULL,
};
// clang-format on

// ============================================================================
//  Liveness Propagation Tables
//  (Pure C++ analysis — copied from x86/iR3000Atables.cpp since that file
//   isn't compiled for ARM64 targets)
// ============================================================================

#define rpsxpropSetRead(reg) \
	{ \
		if (!(pinst->regs[reg] & EEINST_USED)) \
			pinst->regs[reg] |= EEINST_LASTUSE; \
		prev->regs[reg] |= EEINST_LIVE | EEINST_USED; \
		pinst->regs[reg] |= EEINST_USED; \
		_recFillRegister(*pinst, XMMTYPE_GPRREG, reg, 0); \
	}

#define rpsxpropSetWrite(reg) \
	{ \
		prev->regs[reg] &= ~(EEINST_LIVE | EEINST_USED); \
		if (!(pinst->regs[reg] & EEINST_USED)) \
			pinst->regs[reg] |= EEINST_LASTUSE; \
		pinst->regs[reg] |= EEINST_USED; \
		_recFillRegister(*pinst, XMMTYPE_GPRREG, reg, 1); \
	}

void rpsxpropSPECIAL(EEINST* prev, EEINST* pinst);
void rpsxpropREGIMM(EEINST* prev, EEINST* pinst);
void rpsxpropCP0(EEINST* prev, EEINST* pinst);
void rpsxpropCP2(EEINST* prev, EEINST* pinst);

void rpsxpropBSC(EEINST* prev, EEINST* pinst)
{
	switch (psxRegs.code >> 26)
	{
		case 0: rpsxpropSPECIAL(prev, pinst); break;
		case 1: rpsxpropREGIMM(prev, pinst); break;
		case 2: break; // J
		case 3: rpsxpropSetWrite(31); break; // JAL
		case 4: case 5: rpsxpropSetRead(_psxRs_); rpsxpropSetRead(_psxRt_); break;
		case 6: case 7: rpsxpropSetRead(_psxRs_); break;
		case 15: rpsxpropSetWrite(_psxRt_); break; // LUI
		case 16: rpsxpropCP0(prev, pinst); break;
		case 18: rpsxpropCP2(prev, pinst); break;
		case 40: case 41: case 42: case 43: case 46: // stores
			rpsxpropSetRead(_psxRt_); rpsxpropSetRead(_psxRs_); break;
		case 50: case 58: break; // LWC2, SWC2
		default:
			rpsxpropSetWrite(_psxRt_); rpsxpropSetRead(_psxRs_); break;
	}
}

void rpsxpropSPECIAL(EEINST* prev, EEINST* pinst)
{
	switch (_psxFunct_)
	{
		case 0: case 2: case 3: // SLL, SRL, SRA
			rpsxpropSetWrite(_psxRd_); rpsxpropSetRead(_psxRt_); break;
		case 8: rpsxpropSetRead(_psxRs_); break; // JR
		case 9: rpsxpropSetWrite(_psxRd_); rpsxpropSetRead(_psxRs_); break; // JALR
		case 12: case 13: _recClearInst(prev); prev->info = 0; break; // SYSCALL, BREAK
		case 15: break; // SYNC
		case 16: rpsxpropSetWrite(_psxRd_); rpsxpropSetRead(PSX_HI); break; // MFHI
		case 17: rpsxpropSetWrite(PSX_HI); rpsxpropSetRead(_psxRs_); break; // MTHI
		case 18: rpsxpropSetWrite(_psxRd_); rpsxpropSetRead(PSX_LO); break; // MFLO
		case 19: rpsxpropSetWrite(PSX_LO); rpsxpropSetRead(_psxRs_); break; // MTLO
		case 24: case 25: case 26: case 27: // MULT, MULTU, DIV, DIVU
			rpsxpropSetWrite(PSX_LO); rpsxpropSetWrite(PSX_HI);
			rpsxpropSetRead(_psxRs_); rpsxpropSetRead(_psxRt_); break;
		case 32: case 33: case 34: case 35: // ADD, ADDU, SUB, SUBU
			rpsxpropSetWrite(_psxRd_);
			if (_psxRs_) rpsxpropSetRead(_psxRs_);
			if (_psxRt_) rpsxpropSetRead(_psxRt_);
			break;
		default:
			rpsxpropSetWrite(_psxRd_);
			rpsxpropSetRead(_psxRs_); rpsxpropSetRead(_psxRt_); break;
	}
}

void rpsxpropREGIMM(EEINST* prev, EEINST* pinst)
{
	switch (_psxRt_)
	{
		case 0: case 1: rpsxpropSetRead(_psxRs_); break; // BLTZ, BGEZ
		case 16: case 17: rpsxpropSetRead(_psxRs_); break; // BLTZAL, BGEZAL
		default: break;
	}
}

void rpsxpropCP0(EEINST* prev, EEINST* pinst)
{
	switch (_psxRs_)
	{
		case 0: case 2: rpsxpropSetWrite(_psxRt_); break; // MFC0, CFC0
		case 4: case 6: rpsxpropSetRead(_psxRt_); break; // MTC0, CTC0
		case 16: break; // RFE
		default: break;
	}
}

void rpsxpropCP2_basic(EEINST* prev, EEINST* pinst)
{
	switch (_psxRs_)
	{
		case 0: case 2: rpsxpropSetWrite(_psxRt_); break; // MFC2, CFC2
		case 4: case 6: rpsxpropSetRead(_psxRt_); break; // MTC2, CTC2
		default: break;
	}
}

void rpsxpropCP2(EEINST* prev, EEINST* pinst)
{
	switch (_psxFunct_)
	{
		case 0: rpsxpropCP2_basic(prev, pinst); break;
		default: break; // COP2 ops don't affect GPRs
	}
}
