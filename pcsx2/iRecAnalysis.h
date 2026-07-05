// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+
//
// Instruction liveness analysis structures shared between x86 and ARM64
// recompilers. Extracted from x86/iCore.h to avoid pulling in x86emitter.h.

#pragma once

#include "common/Pcsx2Defs.h"

// Register type constants used by liveness analysis (writeType/readType fields).
// These values must match the x86 iCore.h definitions.
#define XMMTYPE_TEMP   0
#define XMMTYPE_GPRREG 1
#define XMMTYPE_FPREG  6
#define XMMTYPE_FPACC  7
#define XMMTYPE_VFREG  8

// Shared with x86 iCore.h's x86type enum — needed by recBackpropCOP2's
// VI register tracking. Value matches the X86TYPE_VIREG enum entry.
#ifndef X86TYPE_VIREG
#define X86TYPE_VIREG 3
#endif

#define XMMGPR_LO  33
#define XMMGPR_HI  32
#define XMMFPU_ACC 32

//////////////////////
// Instruction Info //
//////////////////////
// Liveness information for the noobs :)
// Let's take I instructions that read from RN register set and write to
// WN register set.
// 1/ EEINST_USED will be set in register N of instruction I1, if and only if RN or WN is used in the insruction I2 with I2 >= I1.
// In others words, it will be set on [I0, ILast] with ILast the last instruction that use the register.
// 2/ EEINST_LASTUSE will be set in register N the last instruction that use the register.
// Note: EEINST_USED will be cleared after EEINST_LASTUSE
// My guess: both variable allow to detect register that can be flushed for free
//
// 3/ EEINST_LIVE* is cleared when register is written. And set again when register is read.
// My guess: the purpose is to detect the usage hole in the flow

#define EEINST_LIVE     1 // if var is ever used (read or write)
#define EEINST_LASTUSE   8 // if var isn't written/read anymore
#define EEINST_XMM    0x20 // var will be used in xmm ops
#define EEINST_USED   0x40

#define EEINST_COP2_DENORMALIZE_STATUS_FLAG 0x100
#define EEINST_COP2_NORMALIZE_STATUS_FLAG 0x200
#define EEINST_COP2_STATUS_FLAG 0x400
#define EEINST_COP2_MAC_FLAG 0x800
#define EEINST_COP2_CLIP_FLAG 0x1000
#define EEINST_COP2_SYNC_VU0 0x2000
#define EEINST_COP2_FINISH_VU0 0x4000
#define EEINST_COP2_FLUSH_VU0_REGISTERS 0x8000

struct EEINST
{
	u16 info; // extra info, if 1 inst is COP1, 2 inst is COP2. Also uses EEINST_XMM
	u8 regs[34]; // includes HI/LO (HI=32, LO=33)
	u8 fpuregs[33]; // ACC=32
	u8 vfregs[34]; // ACC=32, I=33
	u8 viregs[16];

	// uses XMMTYPE_ flags; if type == XMMTYPE_TEMP, not used
	u8 writeType[3], writeReg[3]; // reg written in this inst, 0 if no reg
	u8 readType[4], readReg[4];
};

extern EEINST* g_pCurInstInfo; // info for the cur instruction
extern void _recClearInst(EEINST* pinst);

// returns the number of insts + 1 until written (0 if not written)
extern u32 _recIsRegReadOrWritten(EEINST* pinst, int size, u8 xmmtype, u8 reg);

extern void _recFillRegister(EEINST& pinst, int type, int reg, int write);

// If unset, values which are not live will not be written back to memory.
// Tends to break stuff at the moment.
#define EE_WRITE_DEAD_VALUES 1

/// Returns true if the register is used later in the block, and this isn't the last instruction to use it.
/// In other words, the register is worth keeping in a host register/caching it.
static __fi bool EEINST_USEDTEST(u32 reg)
{
	return (g_pCurInstInfo->regs[reg] & (EEINST_USED | EEINST_LASTUSE)) == EEINST_USED;
}

/// Returns true if the register is used later in the block as an XMM/128-bit value.
static __fi bool EEINST_XMMUSEDTEST(u32 reg)
{
	return (g_pCurInstInfo->regs[reg] & (EEINST_USED | EEINST_XMM | EEINST_LASTUSE)) == (EEINST_USED | EEINST_XMM);
}

/// Returns true if the specified VF register is used later in the block.
static __fi bool EEINST_VFUSEDTEST(u32 reg)
{
	return (g_pCurInstInfo->vfregs[reg] & (EEINST_USED | EEINST_LASTUSE)) == EEINST_USED;
}

/// Returns true if the specified VI register is used later in the block.
static __fi bool EEINST_VIUSEDTEST(u32 reg)
{
	return (g_pCurInstInfo->viregs[reg] & (EEINST_USED | EEINST_LASTUSE)) == EEINST_USED;
}

/// Returns true if the value should be computed/written back.
/// Basically, this means it's either used before it's overwritten, or not overwritten by the end of the block.
static __fi bool EEINST_LIVETEST(u32 reg)
{
	return EE_WRITE_DEAD_VALUES || ((g_pCurInstInfo->regs[reg] & EEINST_LIVE) != 0);
}

/// Returns true if the register can be renamed into another.
static __fi bool EEINST_RENAMETEST(u32 reg)
{
	return (reg == 0 || !EEINST_USEDTEST(reg) || !EEINST_LIVETEST(reg));
}

static __fi bool FPUINST_ISLIVE(u32 reg)   { return !!(g_pCurInstInfo->fpuregs[reg] & EEINST_LIVE); }
static __fi bool FPUINST_LASTUSE(u32 reg)  { return !!(g_pCurInstInfo->fpuregs[reg] & EEINST_LASTUSE); }

/// Returns true if the register is used later in the block, and this isn't the last instruction to use it.
/// In other words, the register is worth keeping in a host register/caching it.
static __fi bool FPUINST_USEDTEST(u32 reg)
{
	return (g_pCurInstInfo->fpuregs[reg] & (EEINST_USED | EEINST_LASTUSE)) == EEINST_USED;
}

/// Returns true if the value should be computed/written back.
static __fi bool FPUINST_LIVETEST(u32 reg)
{
	return EE_WRITE_DEAD_VALUES || FPUINST_ISLIVE(reg);
}

/// Returns true if the register can be renamed into another.
static __fi bool FPUINST_RENAMETEST(u32 reg)
{
	return (!EEINST_USEDTEST(reg) || !EEINST_LIVETEST(reg));
}
