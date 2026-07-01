// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "VMManager.h"
#include "Elfheader.h"
#include "Cache.h"

#include "DebugTools/Breakpoints.h"

#include "common/FastJmp.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include "arm64/TraceBlocks.h"
#endif

#include <float.h>

#include "AndroidEEOpHist.h"
#include "common/Console.h"
#include "common/StringUtil.h"
#include <string>

using namespace R5900;		// for OPCODE and OpcodeImpl

#if ARMSX2_ANDROID_EE_OPHIST
namespace AndroidEEOpHist
{
	struct Hist { u64 primary[64]; u64 special[64]; u64 regimm[32]; u64 cop2[32]; u64 mmi[64]; };
	static Hist s_hist[2] = {}; // [0]=STEP, [1]=INLINE
	static u64 s_total = 0;
	static u64 s_trapsCompiled = 0;
	static const char* const s_pathName[2] = { "STEP", "INLINE" };

	// Print every non-zero entry of one sub-table as " idx=count" (hex idx).
	static void dumpTable(const char* path, const char* name, const u64* a, int n)
	{
		std::string line;
		for (int i = 0; i < n; i++)
			if (a[i])
				line += StringUtil::StdStringFromFormat(" %02x=%llu", i, (unsigned long long)a[i]);
		if (!line.empty())
			Console.WriteLnFmt("@@ANDROID_EE_OPHIST@@ {} {}{}", path, name, line);
	}

	static void Dump()
	{
		for (int p = 0; p < 2; p++)
		{
			dumpTable(s_pathName[p], "primary", s_hist[p].primary, 64);
			dumpTable(s_pathName[p], "special", s_hist[p].special, 64);
			dumpTable(s_pathName[p], "regimm", s_hist[p].regimm, 32);
			dumpTable(s_pathName[p], "cop2", s_hist[p].cop2, 32);
			dumpTable(s_pathName[p], "mmi", s_hist[p].mmi, 64);
		}
		Console.WriteLnFmt("@@ANDROID_EE_OPHIST@@ traps_compiled={} total={}",
			(unsigned long long)s_trapsCompiled, (unsigned long long)s_total);
		memset(s_hist, 0, sizeof(s_hist));
		s_total = 0;
	}

	void Record(int path, u32 op)
	{
		Hist& h = s_hist[path & 1];
		const u32 primary = op >> 26;
		h.primary[primary & 0x3f]++;
		if (primary == 0x00) h.special[op & 0x3f]++;
		else if (primary == 0x01) h.regimm[(op >> 16) & 0x1f]++;
		else if (primary == 0x12) h.cop2[(op >> 21) & 0x1f]++;
		else if (primary == 0x1c) h.mmi[op & 0x3f]++;
		if (++s_total >= 8000000ull)
			Dump();
	}

	void NoteTrapCompiled() { s_trapsCompiled++; }
}

// Inline-interp counting thunk (see header). recEmitInterpInline calls this instead
// of the raw handler when the histogram is on; cpuRegs.code was set just before.
void recOphistInlineThunk()
{
	AndroidEEOpHist::Record(1, cpuRegs.code);
	R5900::GetInstruction(cpuRegs.code).interpret();
}
#endif

extern int vu0branch, vu1branch;

static int branch2 = 0;
static u32 cpuBlockCycles = 0;		// 3 bit fixed point version of cycle count
static std::string disOut;
static bool intExitExecution = false;
static fastjmp_buf intJmpBuf;
static u32 intLastBranchTo;

void intEventTest();

void intUpdateCPUCycles()
{
	const bool lowcycles = (cpuBlockCycles <= 40);
	const s8 cyclerate = EmuConfig.Speedhacks.EECycleRate;
	u32 scale_cycles = 0;

	if (cyclerate == 0 || lowcycles || cyclerate < -99 || cyclerate > 3)
		scale_cycles = cpuBlockCycles >> 3;

	else if (cyclerate > 1)
		scale_cycles = cpuBlockCycles >> (2 + cyclerate);

	else if (cyclerate == 1)
		scale_cycles = (cpuBlockCycles >> 3) / 1.3f; // Adds a mild 30% increase in clockspeed for value 1.

	else if (cyclerate == -1) // the mildest value.
		// These values were manually tuned to yield mild speedup with high compatibility
		scale_cycles = (cpuBlockCycles <= 80 || cpuBlockCycles > 168 ? 5 : 7) * cpuBlockCycles / 32;

	else
		scale_cycles = ((5 + (-2 * (cyclerate + 1))) * cpuBlockCycles) >> 5;

	// Ensure block cycle count is never less than 1.
	cpuRegs.cycle += (scale_cycles < 1) ? 1 : scale_cycles;

	if (cyclerate > 1)
	{
		cpuBlockCycles &= (0x1 << (cyclerate + 2)) - 1;
	}
	else
	{
		cpuBlockCycles &= 0x7;
	}
}

// These macros are used to assemble the repassembler functions

void intBreakpoint(bool memcheck)
{
	const u32 pc = cpuRegs.pc;
 	if (CBreakPoints::CheckSkipFirst(BREAKPOINT_EE, pc) != 0)
	{
		CBreakPoints::ClearSkipFirst(BREAKPOINT_EE);
		return;
	}

	if (!memcheck)
	{
		auto cond = CBreakPoints::GetBreakPointCondition(BREAKPOINT_EE, pc);
		if (cond && !cond->Evaluate())
			return;
	}

	CBreakPoints::SetBreakpointTriggered(true, BREAKPOINT_EE);
	VMManager::SetPaused(true);
	Cpu->ExitExecution();
}

void intMemcheck(u32 op, u32 bits, bool store)
{
	// compute accessed address
	u32 start = cpuRegs.GPR.r[(op >> 21) & 0x1F].UL[0];
	if (static_cast<s16>(op) != 0)
		start += static_cast<s16>(op);
	if (bits == 128)
		start &= ~0x0F;

	start = standardizeBreakpointAddress(start);
	const u32 end = start + bits/8;

	auto checks = CBreakPoints::GetMemChecks(BREAKPOINT_EE);
	for (size_t i = 0; i < checks.size(); i++)
	{
		auto& check = checks[i];

		if (check.result == 0)
			continue;
		if ((check.memCond & MEMCHECK_WRITE) == 0 && store)
			continue;
		if ((check.memCond & MEMCHECK_READ) == 0 && !store)
			continue;

		if (check.hasCond)
		{
			if (!check.cond.Evaluate())
				continue;
		}

		if (start < check.end && check.start < end)
			intBreakpoint(true);
	}
}

void intCheckMemcheck()
{
	const u32 pc = cpuRegs.pc;
	const int needed = isMemcheckNeeded(pc);
	if (needed == 0)
		return;

	const u32 op = memRead32(needed == 2 ? pc + 4 : pc);
	const OPCODE& opcode = GetInstruction(op);

	const bool store = (opcode.flags & IS_STORE) != 0;
	switch (opcode.flags & MEMTYPE_MASK)
	{
		case MEMTYPE_BYTE:
			intMemcheck(op, 8, store);
			break;
		case MEMTYPE_HALF:
			intMemcheck(op, 16, store);
			break;
		case MEMTYPE_WORD:
			intMemcheck(op, 32, store);
			break;
		case MEMTYPE_DWORD:
			intMemcheck(op, 64, store);
			break;
		case MEMTYPE_QWORD:
			intMemcheck(op, 128, store);
			break;
	}
}

static void execI()
{
	// execI is called for every instruction so it must remains as light as possible.
	// If you enable the next define, Interpreter will be much slower (around
	// ~4fps on 3.9GHz Haswell vs ~8fps (even 10fps on dev build))
	// Extra note: due to some cycle count issue PCSX2's internal debugger is
	// not yet usable with the interpreter
//#define EXTRA_DEBUG
#if defined(EXTRA_DEBUG) || defined(PCSX2_DEVBUILD)
	// check if any breakpoints or memchecks are triggered by this instruction
	if (isBreakpointNeeded(cpuRegs.pc))
		intBreakpoint(false);

	intCheckMemcheck();

	CBreakPoints::CommitClearSkipFirst(BREAKPOINT_EE);
#endif

	const u32 pc = cpuRegs.pc;

	// We need to increase the pc before executing the memRead32. An exception could appears
	// and it expects the PC counter to be pre-incremented
	cpuRegs.pc += 4;

	// interprete instruction
	cpuRegs.code = memRead32( pc );

	const OPCODE& opcode = GetCurrentInstruction();
#if 0
	static long int runs = 0;
	//use this to find out what opcodes your game uses. very slow! (rama)
	runs++;
	 //leave some time to startup the testgame
	if (runs > 1599999999)
	{
		 //find all opcodes beginning with "L"
		if (opcode.Name[0] == 'L')
		{
			Console.WriteLn ("Load %s", opcode.Name);
		}
	}
#endif

#if 0
	static long int print_me = 0;
	// Based on cycle
	// if ( cpuRegs.cycle > 0x4f24d714 )
	// Or dump from a particular PC (useful to debug handler/syscall)
	if (pc == 0x80000000)
	{
		print_me = 2000;
	}
	if (print_me)
	{
		print_me--;
		disOut.clear();
		disR5900Fasm(disOut, cpuRegs.code, pc);
		CPU_LOG( disOut.c_str() );
	}
#endif


	cpuBlockCycles += opcode.cycles * (2 - ((cpuRegs.CP0.n.Config >> 18) & 0x1));
	opcode.interpret();
}

static __fi void _doBranch_shared(u32 tar)
{
	branch2 = cpuRegs.branch = 1;
	execI();

	// branch being 0 means an exception was thrown, since only the exception
	// handler should ever clear it.

	if( cpuRegs.branch != 0 )
	{
		if (Cpu == &intCpu)
		{
			if (intLastBranchTo == tar && EmuConfig.Speedhacks.WaitLoop)
			{
				intUpdateCPUCycles();
				bool can_skip = true;
				if (tar != 0x81fc0)
				{
					if ((cpuRegs.pc - tar) < (4 * 10))
					{
						for (u32 i = tar; i < cpuRegs.pc; i += 4)
						{
							if (PSM(i) != 0)
							{
								can_skip = false;
								break;
							}
						}
					}
					else
						can_skip = false;
				}

				if (can_skip)
				{
					if (static_cast<s64>(cpuRegs.nextEventCycle - cpuRegs.cycle) > 0)
						cpuRegs.cycle = cpuRegs.nextEventCycle;
					else
						cpuRegs.nextEventCycle = cpuRegs.cycle;
				}
			}
		}
		intLastBranchTo = tar;
		cpuRegs.pc = tar;
		cpuRegs.branch = 0;
	}
}

static void doBranch( u32 target )
{
	_doBranch_shared( target );
	intUpdateCPUCycles();
	intEventTest();
}

void intDoBranch(u32 target)
{
	//Console.WriteLn("Interpreter Branch ");
	_doBranch_shared( target );

	if( Cpu == &intCpu )
	{
		intUpdateCPUCycles();
		intEventTest();
	}
}

void intSetBranch()
{
	branch2 = /*cpuRegs.branch =*/ 1;
}

// Single-step entry used by the macOS-port arm64 recompiler when it hands an op
// back to the interpreter. Mirrors execI without the fastjmp exit (rec drives
// its own exits). Always flush cycles, including for branch ops, because the
// interpreter branch helpers only flush for intCpu. Without this, branch-only
// guest wait loops can freeze cpuRegs.cycle and starve pending events.
void intExecuteOneInst()
{
	const u32 thispc = cpuRegs.pc;
	cpuRegs.pc += 4;
	cpuRegs.code = memRead32(thispc);
#if ARMSX2_ANDROID_EE_OPHIST
	AndroidEEOpHist::Record(0, cpuRegs.code); // STEP (single-step) fallback profile
#endif

	const OPCODE& opcode = GetCurrentInstruction();
	cpuBlockCycles += opcode.cycles * (2 - ((cpuRegs.CP0.n.Config >> 18) & 0x1));

	opcode.interpret();

	intUpdateCPUCycles();
}

////////////////////////////////////////////////////////////////////
// R5900 Branching Instructions!
// These are the interpreter versions of the branch instructions.  Unlike other
// types of interpreter instructions which can be called safely from the recompilers,
// these instructions are not "recSafe" because they may not invoke the
// necessary branch test logic that the recs need to maintain sync with the
// cpuRegs.pc and delaySlot instruction and such.

namespace R5900 {
namespace Interpreter {
namespace OpcodeImpl {

/*********************************************************
* Jump to target                                         *
* Format:  OP target                                     *
*********************************************************/
// fixme: looking at the other branching code, shouldn't those _SetLinks in BGEZAL and such only be set
// if the condition is true? --arcum42

void J()
{
	doBranch(_JumpTarget_);
}

void JAL()
{
	// 0x3563b8 is the start address of the function that invalidate entry in TLB cache
	if (EmuConfig.Gamefixes.GoemonTlbHack) {
		if (_JumpTarget_ == 0x3563b8)
			GoemonUnloadTlb(cpuRegs.GPR.n.a0.UL[0]);
	}
	_SetLink(31);
	doBranch(_JumpTarget_);
}

/*********************************************************
* Register branch logic                                  *
* Format:  OP rs, rt, offset                             *
*********************************************************/

void BEQ()  // Branch if Rs == Rt
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] == cpuRegs.GPR.r[_Rt_].SD[0])
		doBranch(_BranchTarget_);
	else
		intEventTest();
}

void BNE()  // Branch if Rs != Rt
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] != cpuRegs.GPR.r[_Rt_].SD[0])
		doBranch(_BranchTarget_);
	else
		intEventTest();
}

/*********************************************************
* Register branch logic                                  *
* Format:  OP rs, offset                                 *
*********************************************************/

void BGEZ()    // Branch if Rs >= 0
{
	if(cpuRegs.GPR.r[_Rs_].SD[0] >= 0)
	{
		doBranch(_BranchTarget_);
	}
}

void BGEZAL() // Branch if Rs >= 0 and link
{
	_SetLink(31);
	if (cpuRegs.GPR.r[_Rs_].SD[0] >= 0)
	{
		doBranch(_BranchTarget_);
	}
}

void BGTZ()    // Branch if Rs >  0
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] > 0)
	{
		doBranch(_BranchTarget_);
	}
}

void BLEZ()   // Branch if Rs <= 0
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] <= 0)
	{
		doBranch(_BranchTarget_);
	}
}

void BLTZ()    // Branch if Rs <  0
{
	if (cpuRegs.GPR.r[_Rs_].SD[0] < 0)
	{
		doBranch(_BranchTarget_);
	}
}

void BLTZAL()  // Branch if Rs <  0 and link
{
	_SetLink(31);
	if (cpuRegs.GPR.r[_Rs_].SD[0] < 0)
	{
		doBranch(_BranchTarget_);
	}
}

/*********************************************************
* Register branch logic  Likely                          *
* Format:  OP rs, offset                                 *
*********************************************************/


void BEQL()    // Branch if Rs == Rt
{
	if(cpuRegs.GPR.r[_Rs_].SD[0] == cpuRegs.GPR.r[_Rt_].SD[0])
	{
		doBranch(_BranchTarget_);
	}
	else
	{
		cpuRegs.pc +=4;
		intEventTest();
	}
}

void BNEL()     // Branch if Rs != Rt
{
	if(cpuRegs.GPR.r[_Rs_].SD[0] != cpuRegs.GPR.r[_Rt_].SD[0])
	{
		doBranch(_BranchTarget_);
	}
	else
	{
		cpuRegs.pc +=4;
		intEventTest();
	}
}

void BLEZL()    // Branch if Rs <= 0
{
	if(cpuRegs.GPR.r[_Rs_].SD[0] <= 0)
	{
		doBranch(_BranchTarget_);
	}
	else
	{
		cpuRegs.pc +=4;
		intEventTest();
	}
}

void BGTZL()     // Branch if Rs >  0
{
	if(cpuRegs.GPR.r[_Rs_].SD[0] > 0)
	{
		doBranch(_BranchTarget_);
	}
	else
	{
		cpuRegs.pc +=4;
		intEventTest();
	}
}

void BLTZL()     // Branch if Rs <  0
{
	if(cpuRegs.GPR.r[_Rs_].SD[0] < 0)
	{
		doBranch(_BranchTarget_);
	}
	else
	{
		cpuRegs.pc +=4;
		intEventTest();
	}
}

void BGEZL()     // Branch if Rs >= 0
{
	if(cpuRegs.GPR.r[_Rs_].SD[0] >= 0)
	{
		doBranch(_BranchTarget_);
	}
	else
	{
		cpuRegs.pc +=4;
		intEventTest();
	}
}

void BLTZALL()   // Branch if Rs <  0 and link
{
	_SetLink(31);
	if(cpuRegs.GPR.r[_Rs_].SD[0] < 0)
	{
		doBranch(_BranchTarget_);
	}
	else
	{
		cpuRegs.pc +=4;
		intEventTest();
	}
}

void BGEZALL()   // Branch if Rs >= 0 and link
{
	_SetLink(31);
	if(cpuRegs.GPR.r[_Rs_].SD[0] >= 0)
	{
		doBranch(_BranchTarget_);
	}
	else
	{
		cpuRegs.pc +=4;
		intEventTest();
	}
}

/*********************************************************
* Register jump                                          *
* Format:  OP rs, rd                                     *
*********************************************************/
void JR()
{
	// 0x33ad48 and 0x35060c are the return address of the function (0x356250) that populate the TLB cache
	if (EmuConfig.Gamefixes.GoemonTlbHack) {
		const u32 add = cpuRegs.GPR.r[_Rs_].UL[0];
		if (add == 0x33ad48 || add == 0x35060c)
			GoemonPreloadTlb();
	}
	doBranch(cpuRegs.GPR.r[_Rs_].UL[0]);
}

void JALR()
{
	const u32 temp = cpuRegs.GPR.r[_Rs_].UL[0];

	if (_Rd_)  _SetLink(_Rd_);

	doBranch(temp);
}

} } }		// end namespace R5900::Interpreter::OpcodeImpl


// --------------------------------------------------------------------------------------
//  R5900cpu/intCpu interface (implementations)
// --------------------------------------------------------------------------------------

static void intReserve()
{
	// fixme : detect cpu for use the optimize asm code
}

static void intReset()
{
	cpuRegs.branch = 0;
	branch2 = 0;
}

extern bool s_eeTestMode;

void intEventTest()
{
	if (s_eeTestMode)
		return;

	// Perform counters, ints, and IOP updates:
	_cpuEventTest_Shared();

	if (intExitExecution)
	{
		intExitExecution = false;
		if (CHECK_EEREC)
			writebackCache();
		fastjmp_jmp(&intJmpBuf, 1);
	}
}

static void intSafeExitExecution()
{
	// If we're currently processing events, we can't safely jump out of the interpreter here, because we'll
	// leave things in an inconsistent state. So instead, we flag it for exiting once cpuEventTest() returns.
	if (eeEventTestIsActive)
		intExitExecution = true;
	else
	{
		if (CHECK_EEREC)
			writebackCache();
		fastjmp_jmp(&intJmpBuf, 1);
	}
}

static void intCancelInstruction()
{
	// See execute function.
	fastjmp_jmp(&intJmpBuf, 0);
}

static void intExecute()
{
	// This will come back as zero the first time it runs, or on instruction cancel.
	// It will come back as nonzero when we exit execution.
	if (fastjmp_set(&intJmpBuf) != 0)
		return;

	for (;;)
	{
		if (!VMManager::Internal::HasBootedELF())
		{
			// Avoid reloading every instruction.
			u32 elf_entry_point = VMManager::Internal::GetCurrentELFEntryPoint();
			u32 eeload_main = g_eeloadMain;
			u32 eeload_exec = g_eeloadExec;

			while (true)
			{
#ifdef TRACE_BLOCKS
				eeTraceBlock(cpuRegs.pc);
#endif
				execI();

				if (cpuRegs.pc == EELOAD_START)
				{
					// The EELOAD _start function is the same across all BIOS versions afaik
					const u32 mainjump = memRead32(EELOAD_START + 0x9c);
					if (mainjump >> 26 == 3) // JAL
						g_eeloadMain = ((EELOAD_START + 0xa0) & 0xf0000000U) | (mainjump << 2 & 0x0fffffffU);

					eeload_main = g_eeloadMain;
				}
				else if (cpuRegs.pc == eeload_main)
				{
					eeloadHook();
					if (VMManager::Internal::IsFastBootInProgress())
					{
						// See comments on this code in iR5900.cpp's recRecompile()
						const u32 typeAexecjump = memRead32(EELOAD_START + 0x470);
						const u32 typeBexecjump = memRead32(EELOAD_START + 0x5B0);
						const u32 typeCexecjump = memRead32(EELOAD_START + 0x618);
						const u32 typeDexecjump = memRead32(EELOAD_START + 0x600);
						if ((typeBexecjump >> 26 == 3) || (typeCexecjump >> 26 == 3) || (typeDexecjump >> 26 == 3)) // JAL to 0x822B8
							g_eeloadExec = EELOAD_START + 0x2B8;
						else if (typeAexecjump >> 26 == 3) // JAL to 0x82170
							g_eeloadExec = EELOAD_START + 0x170;
						else
							Console.WriteLn("intExecute: Could not enable launch arguments for fast boot mode; unidentified BIOS version! Please report this to the PCSX2 developers.");

						eeload_exec = g_eeloadExec;
					}

					elf_entry_point = VMManager::Internal::GetCurrentELFEntryPoint();
				}
				else if (cpuRegs.pc == eeload_exec)
				{
					eeloadHook2();
				}
				else if (cpuRegs.pc == elf_entry_point)
				{
					VMManager::Internal::EntryPointCompilingOnCPUThread();
					break;
				}
			}
		}
		else
		{
			while (true)
			{
#ifdef TRACE_BLOCKS
				eeTraceBlock(cpuRegs.pc);
#endif
				execI();
			}
		}
	}
}

static void intStep()
{
	execI();
}

static void intClear(u32 Addr, u32 Size)
{
}

static void intShutdown() {
}

R5900cpu intCpu =
{
	intReserve,
	intShutdown,

	intReset,
	intStep,
	intExecute,

	intSafeExitExecution,
	intCancelInstruction,

	intClear
};

//------------------------------------------------------------------
// Test Harness API (used by tests/ee/)
//------------------------------------------------------------------

#include "Memory.h"
#include "MemoryTypes.h"

static constexpr u32 EE_TEST_PC = 0x00100000u;
static bool s_eeTestOwnsMemory = false;
bool s_eeTestMode = false;

void EE_TestInit()
{
	if (!eeMem)
	{
		SysMemory::Allocate();
		SysMemory::Reset();
		s_eeTestOwnsMemory = true;
	}
	Cpu = &intCpu;
	s_eeTestMode = true;
}

void EE_TestShutdown()
{
	s_eeTestMode = false;
	if (s_eeTestOwnsMemory)
	{
		SysMemory::Release();
		s_eeTestOwnsMemory = false;
	}
}

void EE_TestWriteProg(const u32* words, u32 count)
{
	for (u32 i = 0; i < count; i++)
		memWrite32(EE_TEST_PC + i * 4, words[i]);
}

void EE_TestExec(u32 maxInstrs)
{
	cpuRegs.pc = EE_TEST_PC;
	cpuRegs.branch = 0;
	cpuRegs.cycle = 0;
	cpuRegs.nextEventCycle = UINT64_MAX; // prevent _cpuEventTest_Shared from running IOP/timers
	cpuBlockCycles = 0;

	for (u32 i = 0; i < maxInstrs; i++)
	{
		const u32 pc = cpuRegs.pc;
		const u32 code = memRead32(pc);

		// Halt: BEQ $0,$0,-1  (0x1000FFFF)
		if (code == 0x1000FFFFu)
			break;

		execI();
	}
}
