// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "R3000A.h"
#include "Common.h"
#include "Config.h"
#include "VMManager.h"

#include "R5900OpcodeTables.h"
#include "DebugTools/Breakpoints.h"
#include "IopBios.h"
#include "IopHw.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include "arm64/TraceBlocks.h"
#endif

using namespace R3000A;

// Used to flag delay slot instructions when throwig exceptions.
bool iopIsDelaySlot = false;

// During iop_shadow_verify's interp replay, suppress doBranch's iopEventTest
// call. The JIT block exit emits iPsxBranchTest which has already handled
// any pending event; replaying iopEventTest from inside the interp run would
// mutate cpuRegs / iopHw / counter scheduling a second time and corrupt the
// post-block state we're trying to compare against. Single-thread (IOP runs
// on EE thread).
bool iopShadowSuppressEventTest = false;

// =====================  IOP PER-INSTRUCTION CYCLE MULTIPLIER  =====================
// Real R3000A executes at ~0.5-0.7 effective IPC due to memory stalls /
// cache misses; PCSX2 advances psxRegs.cycle by exactly 1 per instruction
// (= 1.0 IPC), so per real-second PCSX2 runs 1.4-2x more game code than
// real PS1 — game logic outpaces audio (which is keyed correctly to
// psxRegs.cycle). This knob inflates the per-instruction cycle cost so
// the IOP runs proportionally fewer instructions per emulator-frame.
//
// SPU2, RTC, T0-T5 counters, and vsync ALL pace by psxRegs.cycle's
// wall-clock advance, which is set by the EE-side budget (unchanged) —
// they stay at correct wall-clock rate. Only game CODE slows.
//
//   1 = upstream behavior (1.0 IPC, ~1.4x speedup vs real PS1)
//   2 = 0.5 IPC — game runs ~half speed of upstream  (test/prove path)
//   3 = ~0.33 IPC — should slow game even more
//
// Find the value that brings game speed to match real PS1, then we'll
// tune more carefully (per-op cycle costs for loads/stores etc).
u32 g_iopCycleMultiplier = 1;

static bool branch2 = 0;
static u32 branchPC;

static void doBranch(s32 tar);	// forward declared prototype

/*********************************************************
* Register branch logic                                  *
* Format:  OP rs, offset                                 *
*********************************************************/

void psxBGEZ()         // Branch if Rs >= 0
{
	if (_i32(_rRs_) >= 0) doBranch(_BranchTarget_);
}

void psxBGEZAL()   // Branch if Rs >= 0 and link
{
	_SetLink(31);
	if (_i32(_rRs_) >= 0)
	{
		doBranch(_BranchTarget_);
	}
}

void psxBGTZ()          // Branch if Rs >  0
{
	if (_i32(_rRs_) > 0) doBranch(_BranchTarget_);
}

void psxBLEZ()         // Branch if Rs <= 0
{
	if (_i32(_rRs_) <= 0) doBranch(_BranchTarget_);
}
void psxBLTZ()          // Branch if Rs <  0
{
	if (_i32(_rRs_) < 0) doBranch(_BranchTarget_);
}

void psxBLTZAL()    // Branch if Rs <  0 and link
{
	_SetLink(31);
	if (_i32(_rRs_) < 0)
		{
			doBranch(_BranchTarget_);
		}
}

/*********************************************************
* Register branch logic                                  *
* Format:  OP rs, rt, offset                             *
*********************************************************/

void psxBEQ()   // Branch if Rs == Rt
{
	if (_i32(_rRs_) == _i32(_rRt_)) doBranch(_BranchTarget_);
}

void psxBNE()   // Branch if Rs != Rt
{
	if (_i32(_rRs_) != _i32(_rRt_)) doBranch(_BranchTarget_);
}

/*********************************************************
* Jump to target                                         *
* Format:  OP target                                     *
*********************************************************/
void psxJ()
{
	// check for iop module import table magic
	u32 delayslot = iopMemRead32(psxRegs.pc);
	if (delayslot >> 16 == 0x2400 && irxImportExec(irxImportTableAddr(psxRegs.pc), delayslot & 0xffff))
		return;

	doBranch(_JumpTarget_);
}

void psxJAL()
{
	_SetLink(31);
	doBranch(_JumpTarget_);
}

/*********************************************************
* Register jump                                          *
* Format:  OP rs, rd                                     *
*********************************************************/
void psxJR()
{
	doBranch(_u32(_rRs_));
}

void psxJALR()
{
	if (_Rd_)
	{
		_SetLink(_Rd_);
	}
	doBranch(_u32(_rRs_));
}

void psxBreakpoint(bool memcheck)
{
	u32 pc = psxRegs.pc;
	if (CBreakPoints::CheckSkipFirst(BREAKPOINT_IOP, pc) != 0)
	{
		CBreakPoints::ClearSkipFirst(BREAKPOINT_IOP);
		return;
	}

	if (!memcheck)
	{
		auto cond = CBreakPoints::GetBreakPointCondition(BREAKPOINT_IOP, pc);
		if (cond && !cond->Evaluate())
			return;
	}

	CBreakPoints::SetBreakpointTriggered(true, BREAKPOINT_IOP);
	VMManager::SetPaused(true);
	Cpu->ExitExecution();
}

void psxMemcheck(u32 op, u32 bits, bool store)
{
	// compute accessed address
	u32 start = psxRegs.GPR.r[(op >> 21) & 0x1F];
	if ((s16)op != 0)
		start += (s16)op;

	u32 end = start + bits / 8;

	auto checks = CBreakPoints::GetMemChecks(BREAKPOINT_IOP);
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
			psxBreakpoint(true);
	}
}

void psxCheckMemcheck()
{
	u32 pc = psxRegs.pc;
	int needed = psxIsMemcheckNeeded(pc);
	if (needed == 0)
		return;

	u32 op = iopMemRead32(needed == 2 ? pc + 4 : pc);
	// Yeah, we use the R5900 opcode table for the R3000
	const R5900::OPCODE& opcode = R5900::GetInstruction(op);

	bool store = (opcode.flags & IS_STORE) != 0;
	switch (opcode.flags & MEMTYPE_MASK)
	{
	case MEMTYPE_BYTE:
		psxMemcheck(op, 8, store);
		break;
	case MEMTYPE_HALF:
		psxMemcheck(op, 16, store);
		break;
	case MEMTYPE_WORD:
		psxMemcheck(op, 32, store);
		break;
	case MEMTYPE_DWORD:
		psxMemcheck(op, 64, store);
		break;
	}
}

///////////////////////////////////////////
// These macros are used to assemble the repassembler functions

static __fi void execI()
{
	// This function is called for every instruction.
	// Enabling the define below will probably, no, will cause the interpretor to be slower.
//#define EXTRA_DEBUG
#if defined(EXTRA_DEBUG) || defined(PCSX2_DEVBUILD)
	if (psxIsBreakpointNeeded(psxRegs.pc))
		psxBreakpoint(false);

	psxCheckMemcheck();
	
	CBreakPoints::CommitClearSkipFirst(BREAKPOINT_IOP);
#endif

	// Inject IRX hack
	if (psxRegs.pc == 0x1630 && EmuConfig.CurrentIRX.length() > 3) {
		if (iopMemRead32(0x20018) == 0x1F) {
			// FIXME do I need to increase the module count (0x1F -> 0x20)
			iopMemWrite32(0x20094, 0xbffc0000);
		}
	}

	psxRegs.code = iopMemRead32(psxRegs.pc);

		PSXCPU_LOG("%s", disR3000AF(psxRegs.code, psxRegs.pc));

	psxRegs.pc+= 4;
	psxRegs.cycle += g_iopCycleMultiplier;

	psxBSC[psxRegs.code >> 26]();
}

static void doBranch(s32 tar) {
	if (tar == 0x0)
		DevCon.Warning("[R3000 Interpreter] Warning: Branch to 0x0!");

	// When upgrading the IOP, there are two resets, the second of which is a 'fake' reset
	// This second 'reset' involves UDNL calling SYSMEM and LOADCORE directly, resetting LOADCORE's modules
	// This detects when SYSMEM is called and clears the modules then
	if(tar == 0x890)
	{
		DevCon.WriteLn(Color_Gray, "R3000 Debugger: Branch to 0x890 (SYSMEM). Clearing modules.");
		R3000SymbolGuardian.ClearIrxModules();
	}

	// Override the memory size argument to IOPBOOT
	if(tar == 0xbfc4a000) {
		psxRegs.GPR.n.a0 = Ps2MemSize::ExposedIopRam >> 20;
	}

	branch2 = iopIsDelaySlot = true;
	branchPC = tar;
	execI();
	PSXCPU_LOG( "\n" );
	iopIsDelaySlot = false;
	psxRegs.pc = branchPC;

	if (!iopShadowSuppressEventTest)
		iopEventTest();
}

// Per-instruction interp step used by the per-instruction IOP shadow
// harness. Runs exactly one `execI()` from current state. iopEventTest
// is suppressed inside doBranch — for branch instructions, doBranch
// also calls execI for the delay slot internally, but the per-instruction
// shadow doesn't fire on branch ops (their JIT codegen exits to dispatcher
// before reaching the post-emit BL), so this only ever runs straight-line
// non-branch ops in practice.
void psxInterpExecuteOne()
{
	iopShadowSuppressEventTest = true;
	branch2 = 0;
	execI();
	iopShadowSuppressEventTest = false;
}

// Single-step entry for the macOS-port IOP recompiler's fallback path.
void iopExecuteOneInst()
{
	execI();
}

// Block-level interp replay used by the IOP shadow-verify harness. Runs
// `execI()` until either the inner branch path fires (branch2 == true), pc
// reaches `endpc` (fall-through block boundary), or the instruction count
// safety cap is hit. Mirrors `intExecuteBlock`'s inner loop without the
// outer cycle-accounting / event-handling shell. iopEventTest is suppressed
// inside doBranch (see iopShadowSuppressEventTest) so the replay's only
// effect is psxRegs / iopMem / iopHw mutation from the block's actual ops.
//
// Caller (iop_shadow_verify) is responsible for snapshot/restore around
// this; here we just do the work.
void psxInterpReplayBlock(u32 endpc, u32 maxInstructions)
{
	branch2 = 0;
	iopShadowSuppressEventTest = true;
	u32 i = 0;
	// do-while so we always execute at least one instruction. If we
	// checked `pc != endpc` at loop entry, self-loop blocks (where the
	// branch target equals startpc, e.g. wait loops detected by nBlockFF)
	// would bail out immediately without running anything — entry pc
	// already equals endpc on those.
	//
	// Mask kseg0/kseg1 mirror bits when comparing pc vs endpc — the JIT
	// can emit a Str pc with kseg0 bit set (psxpc inherits it from a
	// kseg0 startpc) while the interp's per-instruction pc advance keeps
	// the block-entry mirror form. Both refer to the same physical
	// instruction; mask to compare logically.
	const u32 endpc_masked = endpc & 0x1FFFFFFFu;
	do
	{
		execI();
		i++;
	} while (!branch2 &&
	         (psxRegs.pc & 0x1FFFFFFFu) != endpc_masked &&
	         i < maxInstructions);
	if (branch2)
	{
		psxRegs.pc = branchPC;
		iopIsDelaySlot = false;
	}
	iopShadowSuppressEventTest = false;
}

static void intReserve() {
}

static void intAlloc() {
}

static void intReset() {
	intAlloc();
}

static s32 intExecuteBlock( s32 eeCycles )
{
	psxRegs.iopBreak = 0;
	psxRegs.iopCycleEE = eeCycles;
	u64 lastIOPCycle = 0;

	while (psxRegs.iopCycleEE > 0)
	{
		lastIOPCycle = psxRegs.cycle;
		if ((psxHu32(HW_ICFG) & 8) && ((psxRegs.pc & 0x1fffffffU) == 0xa0 || (psxRegs.pc & 0x1fffffffU) == 0xb0 || (psxRegs.pc & 0x1fffffffU) == 0xc0))
			psxBiosCall();

		branch2 = 0;
		while (!branch2)
		{
#ifdef TRACE_BLOCKS
			iopTraceBlock(psxRegs.pc);
#endif
			execI();
		}

		
		if ((psxHu32(HW_ICFG) & (1 << 3)))
		{
			// F = gcd(PS2CLK, PSXCLK) = 230400
			const u32 cnum = 1280; // PS2CLK / F
			const u32 cdenom = 147; // PSXCLK / F

			//One of the Iop to EE delta clocks to be set in PS1 mode.
			const u32 t = ((cnum * (psxRegs.cycle - lastIOPCycle)) + psxRegs.iopCycleEECarry);
			psxRegs.iopCycleEE -= t / cdenom;
			psxRegs.iopCycleEECarry = t % cdenom;
		}
		else
		{ 
			//default ps2 mode value
			psxRegs.iopCycleEE -= (psxRegs.cycle - lastIOPCycle) * 8;
		}
	}

	return psxRegs.iopBreak + psxRegs.iopCycleEE;
}

static void intClear(u32 Addr, u32 Size) {
}

static void intShutdown() {
}

R3000Acpu psxInt = {
	intReserve,
	intReset,
	intExecuteBlock,
	intClear,
	intShutdown
};
