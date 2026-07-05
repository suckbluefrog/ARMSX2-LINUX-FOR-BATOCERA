// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 IOP (R3000A) Recompiler — Header

#pragma once

#include "arm64/AsmHelpers.h"
#include "R3000A.h"

#include <cstddef>

namespace a64 = vixl::aarch64;

// ============================================================================
//  ARM64 IOP Recompiler — Register Conventions
// ============================================================================
//
// The IOP JIT runs in its own stack frame (created by iopEnterRecompiledCode).
// Callee-saved registers x19-x28 are saved/restored by armBeginStackFrame,
// so we can freely use them without conflicting with the EE JIT.
//
// Pinned state registers (callee-saved):
//   x28 = &psxRegs          (IOP CPU state base)
//   x27 = psxRecLUT         (IOP block dispatch lookup table)
//
// Scratch registers (caller-saved):
//   x0-x3   = arguments / return value
//   x4-x15  = temporaries within a block
//   x16-x17 = vixl scratch (IP0/IP1)
//
// Callee-saved scratch (survives across function calls):
//   x25 = delay slot scratch (value survives interp calls in delay slot)

#define RPSXSTATE   a64::x28
#define RPSXRECLUT  a64::x27

// Scratch GPRs (same as EE — caller-saved, safe within single instruction codegen)
#define RPSXSCRATCH   a64::x4
#define RPSXSCRATCH2  a64::x5
#define RPSXSCRATCH3  a64::x6
#define RWPSXSCRATCH  a64::w4
#define RWPSXSCRATCH2 a64::w5
#define RWPSXSCRATCH3 a64::w6

// Callee-saved scratch for delay slot values
#define RPSXDELAYSLOT  a64::x25
#define RWPSXDELAYSLOT a64::w25

// ============================================================================
//  psxRegs field offsets
// ============================================================================

// Register indices for HI/LO in psxRegs.GPR array (GPR[32]=HI, GPR[33]=LO)
#define PSX_HI 32
#define PSX_LO 33

// IOP GPR[n] is at offset n * 4 from psxRegs base (32-bit registers)
static constexpr s64 PSX_GPR_OFFSET(int reg) { return offsetof(psxRegisters, GPR) + reg * 4; }
static constexpr s64 PSX_HI_OFFSET = offsetof(psxRegisters, GPR) + 32 * 4;
static constexpr s64 PSX_LO_OFFSET = offsetof(psxRegisters, GPR) + 33 * 4;
static constexpr s64 PSX_CP0_OFFSET(int reg) { return offsetof(psxRegisters, CP0) + reg * 4; }
static constexpr s64 PSX_PC_OFFSET = offsetof(psxRegisters, pc);
static constexpr s64 PSX_CODE_OFFSET = offsetof(psxRegisters, code);
static constexpr s64 PSX_CYCLE_OFFSET = offsetof(psxRegisters, cycle);
static constexpr s64 PSX_INTERRUPT_OFFSET = offsetof(psxRegisters, interrupt);
static constexpr s64 PSX_PCWRITEBACK_OFFSET = offsetof(psxRegisters, pcWriteback);
static constexpr s64 PSX_IOPNEXTEVENTCYCLE_OFFSET = offsetof(psxRegisters, iopNextEventCycle);
static constexpr s64 PSX_IOPBREAK_OFFSET = offsetof(psxRegisters, iopBreak);
static constexpr s64 PSX_IOPCYCLEEE_OFFSET = offsetof(psxRegisters, iopCycleEE);
static constexpr s64 PSX_IOPCYCLEEECARRY_OFFSET = offsetof(psxRegisters, iopCycleEECarry);

// ============================================================================
//  Instruction field extraction macros (use psxRegs.code)
// ============================================================================

// These are the same as in R3000A.h but we redeclare them here for clarity
// in recompiler context (psxRegs.code is set before recompiling each instruction)
#ifndef PSX_OPCODE_MACROS
#define PSX_OPCODE_MACROS
#define _psxFunct_  ((psxRegs.code      ) & 0x3F)
#define _psxRd_     ((psxRegs.code >> 11) & 0x1F)
#define _psxRt_     ((psxRegs.code >> 16) & 0x1F)
#define _psxRs_     ((psxRegs.code >> 21) & 0x1F)
#define _psxSa_     ((psxRegs.code >>  6) & 0x1F)
#define _psxImm_    ((int16_t)(psxRegs.code & 0xFFFF))
#define _psxImmU_   ((uint16_t)(psxRegs.code & 0xFFFF))
#define _psxTarget_ (psxRegs.code & 0x03FFFFFF)
#endif

// ============================================================================
//  IOP Constant Propagation
// ============================================================================

extern u32 g_psxConstRegs[32];
extern u32 g_psxHasConstReg, g_psxFlushedConstReg;

#define PSX_IS_CONST1(reg) ((reg) < 32 && (g_psxHasConstReg & (1u << (reg))))
#define PSX_IS_CONST2(reg1, reg2) ((g_psxHasConstReg & (1u << (reg1))) && (g_psxHasConstReg & (1u << (reg2))))
#define PSX_IS_DIRTY_CONST(reg) ((reg) < 32 && (g_psxHasConstReg & (1u << (reg))) && (!(g_psxFlushedConstReg & (1u << (reg)))))
#define PSX_SET_CONST(reg) \
	do { \
		if ((reg) < 32) { \
			g_psxHasConstReg |= (1u << (reg)); \
			g_psxFlushedConstReg &= ~(1u << (reg)); \
		} \
	} while (0)
#define PSX_DEL_CONST(reg) \
	do { \
		if ((reg) < 32) \
			g_psxHasConstReg &= ~(1u << (reg)); \
	} while (0)

// ============================================================================
//  Block state
// ============================================================================

extern u32 psxpc;           // recompiler PC
extern int psxbranch;       // set for branch (0=no, 1=static, 2=dynamic, 3=link)
extern u32 g_iopCyclePenalty;
extern u32 s_psxBlockCycles;
extern bool s_recompilingDelaySlot;

// ============================================================================
//  Cycle penalty constants
// ============================================================================

static const int psxInstCycles_Mult = 7;
static const int psxInstCycles_Div = 40;

// ============================================================================
//  ARM64 IOP codegen helpers
// ============================================================================

// Flush all constant registers to psxRegs memory
void iopArmFlushConstRegs();

// Flush a single constant register
void iopArmFlushConstReg(int reg);

// Load IOP GPR (32-bit) into ARM64 register. Uses MOV imm if const.
void iopArmLoadGPR(const a64::Register& dst, int gpr);

// Store ARM64 w-register into IOP GPR and clear const flag.
void iopArmStoreGPR(const a64::Register& src, int gpr);

// Flush psxpc to psxRegs.pc if not already flushed
void iopArmFlushPC();

// Flush psxRegs.code if not already flushed
void iopArmFlushCode();

// Emit a call to an IOP interpreter function with full state flush
void iopArmCallInterpreter(void (*func)());

// Branch-call variant: flushes state, calls interpreter branch function,
// sets psxbranch = 2
void iopArmBranchCallInterpreter(void (*func)());

// Save/restore branch state for delay slot compilation
void psxSaveBranchState();
void psxLoadBranchState();

// Set branch targets
void psxSetBranchReg();
void psxSetBranchImm(u32 imm);

// Compile next instruction
void psxRecompileNextInstruction(bool delayslot, bool swapped_delayslot);

// Try to swap delay slot instruction before branch
bool psxTrySwapDelaySlot(u32 rs, u32 rt, u32 rd);

// ============================================================================
//  Dispatch tables (defined in iR3000Atables_arm64.cpp)
// ============================================================================

extern void (*rpsxBSC[64])();
extern void (*rpsxSPC[64])();
extern void (*rpsxREG[32])();
extern void (*rpsxCP0[32])();
extern void (*rpsxCP2[64])();
extern void (*rpsxCP2BSC[32])();

// Liveness propagation
struct EEINST;
void rpsxpropBSC(EEINST* prev, EEINST* pinst);

// ============================================================================
//  Recompiler LUT
// ============================================================================

extern uptr psxRecLUT[];
