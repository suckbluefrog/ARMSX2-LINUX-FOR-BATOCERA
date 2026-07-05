// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#pragma once

#include "common/emitter/x86emitter.h"
#include "VUmicro.h"

// Namespace Note : iCore32 contains all of the Register Allocation logic, in addition to a handful
// of utility functions for emitting frequent code.

//#define RALOG(...) fprintf(stderr, __VA_ARGS__)
#define RALOG(...)

////////////////////////////////////////////////////////////////////////////////
// Shared Register allocation flags (apply to X86, XMM, MMX, etc).

#define MODE_READ        1
#define MODE_WRITE       2
#define MODE_CALLEESAVED  0x20 // can't flush reg to mem
#define MODE_COP2 0x40 // don't allow using reserved VU registers

#define PROCESS_EE_XMM 0x02

#define PROCESS_EE_S 0x04 // S is valid, otherwise take from mem
#define PROCESS_EE_T 0x08 // T is valid, otherwise take from mem
#define PROCESS_EE_D 0x10 // D is valid, otherwise take from mem

#define PROCESS_EE_LO         0x40 // lo reg is valid
#define PROCESS_EE_HI         0x80 // hi reg is valid
#define PROCESS_EE_ACC        0x40 // acc reg is valid

#define EEREC_S    (((info) >>  8) & 0xf)
#define EEREC_T    (((info) >> 12) & 0xf)
#define EEREC_D    (((info) >> 16) & 0xf)
#define EEREC_LO   (((info) >> 20) & 0xf)
#define EEREC_HI   (((info) >> 24) & 0xf)
#define EEREC_ACC  (((info) >> 20) & 0xf)

#define PROCESS_EE_SET_S(reg)   (((reg) <<  8) | PROCESS_EE_S)
#define PROCESS_EE_SET_T(reg)   (((reg) << 12) | PROCESS_EE_T)
#define PROCESS_EE_SET_D(reg)   (((reg) << 16) | PROCESS_EE_D)
#define PROCESS_EE_SET_LO(reg)  (((reg) << 20) | PROCESS_EE_LO)
#define PROCESS_EE_SET_HI(reg)  (((reg) << 24) | PROCESS_EE_HI)
#define PROCESS_EE_SET_ACC(reg) (((reg) << 20) | PROCESS_EE_ACC)

// special info not related to above flags
#define PROCESS_CONSTS 1
#define PROCESS_CONSTT 2

// XMM caching helpers
enum xmminfo : u16 
{
	XMMINFO_READLO = 0x001,
	XMMINFO_READHI = 0x002,
	XMMINFO_WRITELO = 0x004,
	XMMINFO_WRITEHI = 0x008,
	XMMINFO_WRITED = 0x010,
	XMMINFO_READD = 0x020,
	XMMINFO_READS = 0x040,
	XMMINFO_READT = 0x080,
	XMMINFO_READACC = 0x200,
	XMMINFO_WRITEACC = 0x400,
	XMMINFO_WRITET = 0x800,

	XMMINFO_64BITOP = 0x1000,
	XMMINFO_FORCEREGS = 0x2000,
	XMMINFO_FORCEREGT = 0x4000,
	XMMINFO_NORENAME = 0x8000 // disables renaming of Rs to Rt in Rt = Rs op imm
};

////////////////////////////////////////////////////////////////////////////////
//   X86 (32-bit) Register Allocation Tools

enum x86type : u8 
{
	X86TYPE_TEMP = 0,
	X86TYPE_GPR = 1,
	X86TYPE_FPRC = 2,
	X86TYPE_VIREG = 3,
	X86TYPE_PCWRITEBACK = 4,
	X86TYPE_PSX = 5,
	X86TYPE_PSX_PCWRITEBACK = 6
};

struct _x86regs
{
	u8 inuse;
	s8 reg;
	u8 mode;
	u8 needed;
	u8 type; // X86TYPE_
	u16 counter;
	u32 extra; // extra info assoc with the reg
};

extern _x86regs x86regs[iREGCNT_GPR], s_saveX86regs[iREGCNT_GPR];

bool _isAllocatableX86reg(int x86reg);
void _initX86regs();
int _getFreeX86reg(int mode);
int _allocX86reg(int type, int reg, int mode);
int _checkX86reg(int type, int reg, int mode);
bool _hasX86reg(int type, int reg, int required_mode = 0);
void _addNeededX86reg(int type, int reg);
void _clearNeededX86regs();
void _freeX86reg(const x86Emitter::xRegister32& x86reg);
void _freeX86reg(int x86reg);
void _freeX86regWithoutWriteback(int x86reg);
void _freeX86regs();
void _flushX86regs();
void _flushConstRegs(bool delete_const);
void _flushConstReg(int reg);
void _validateRegs();
void _writebackX86Reg(int x86reg);

void mVUFreeCOP2GPR(int hostreg);
bool mVUIsReservedCOP2(int hostreg);

////////////////////////////////////////////////////////////////////////////////
//   XMM (128-bit) Register Allocation Tools

#define XMMTYPE_TEMP   0 // has to be 0
#define XMMTYPE_GPRREG X86TYPE_GPR
#define XMMTYPE_FPREG  6
#define XMMTYPE_FPACC  7
#define XMMTYPE_VFREG  8

// lo and hi regs
#define XMMGPR_LO  33
#define XMMGPR_HI  32
#define XMMFPU_ACC 32

enum : int
{
	DELETE_REG_FREE = 0,
	DELETE_REG_FLUSH = 1,
	DELETE_REG_FLUSH_AND_FREE = 2,
	DELETE_REG_FREE_NO_WRITEBACK = 3
};

struct _xmmregs
{
	u8 inuse;
	s8 reg;
	u8 type;
	u8 mode;
	u8 needed;
	u16 counter;
};

void _initXMMregs();
int _getFreeXMMreg(u32 maxreg = iREGCNT_XMM);
int _allocTempXMMreg(XMMSSEType type);
int _allocFPtoXMMreg(int fpreg, int mode);
int _allocGPRtoXMMreg(int gprreg, int mode);
int _allocFPACCtoXMMreg(int mode);
void _reallocateXMMreg(int xmmreg, int newtype, int newreg, int newmode, bool writeback = true);
int _checkXMMreg(int type, int reg, int mode);
bool _hasXMMreg(int type, int reg, int required_mode = 0);
void _addNeededFPtoXMMreg(int fpreg);
void _addNeededFPACCtoXMMreg();
void _addNeededGPRtoX86reg(int gprreg);
void _addNeededPSXtoX86reg(int gprreg);
void _addNeededGPRtoXMMreg(int gprreg);
void _clearNeededXMMregs();
void _deleteGPRtoX86reg(int reg, int flush);
void _deletePSXtoX86reg(int reg, int flush);
void _deleteGPRtoXMMreg(int reg, int flush);
void _deleteFPtoXMMreg(int reg, int flush);
void _freeXMMreg(int xmmreg);
void _freeXMMregWithoutWriteback(int xmmreg);
void _writebackXMMreg(int xmmreg);
int _allocVFtoXMMreg(int vfreg, int mode);
void mVUFreeCOP2XMMreg(int hostreg);
void _flushCOP2regs();
void _flushXMMreg(int xmmreg);
void _flushXMMregs();

// Instruction liveness analysis (shared with ARM64 recompiler)
#include "iRecAnalysis.h"

extern _xmmregs xmmregs[iREGCNT_XMM], s_saveXMMregs[iREGCNT_XMM];

extern thread_local u8* j8Ptr[32];   // depreciated item.  use local u8* vars instead.
extern thread_local u32* j32Ptr[32]; // depreciated item.  use local u32* vars instead.

extern u16 g_x86AllocCounter;
extern u16 g_xmmAllocCounter;

// allocates only if later insts use this register
int _allocIfUsedGPRtoX86(int gprreg, int mode);
int _allocIfUsedVItoX86(int vireg, int mode);
int _allocIfUsedGPRtoXMM(int gprreg, int mode);
int _allocIfUsedFPUtoXMM(int fpureg, int mode);

//////////////////////////////////////////////////////////////////////////
// iFlushCall / _psxFlushCall Parameters

#define FLUSH_NONE             0x000 // frees caller saved registers
#define FLUSH_CONSTANT_REGS    0x001
#define FLUSH_FLUSH_XMM        0x002
#define FLUSH_FREE_XMM         0x004 // both flushes and frees
#define FLUSH_ALL_X86          0x020 // flush x86
#define FLUSH_FREE_TEMP_X86    0x040 // flush and free temporary x86 regs
#define FLUSH_FREE_NONTEMP_X86 0x080 // free all x86 regs, except temporary
#define FLUSH_FREE_VU0         0x100 // free all vu0 related regs
#define FLUSH_PC               0x200 // program counter
//#define FLUSH_CAUSE            0x000 // disabled for now: cause register, only the branch delay bit
#define FLUSH_CODE             0x800 // opcode for interpreter

#define FLUSH_EVERYTHING   0x1ff
//#define FLUSH_EXCEPTION		0x1ff   // will probably do this totally differently actually
#define FLUSH_INTERPRETER  0xfff
#define FLUSH_FULLVTLB 0x000

// no freeing, used when callee won't destroy xmm regs
#define FLUSH_NODESTROY (FLUSH_CONSTANT_REGS | FLUSH_FLUSH_XMM | FLUSH_ALL_X86)

