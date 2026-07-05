// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — Branch & Jump Instructions
// BEQ, BNE, BGEZ, BLTZ, BLEZ, BGTZ, J, JAL, JR, JALR + Likely variants

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"

using namespace R5900;

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
#if defined(INTERP_BRANCH) || defined(INTERP_EE)
#define ISTUB_BEQ      1
#define ISTUB_BNE      1
#define ISTUB_BEQL     1
#define ISTUB_BNEL     1
#define ISTUB_BGEZ     1
#define ISTUB_BGTZ     1
#define ISTUB_BLEZ     1
#define ISTUB_BLTZ     1
#define ISTUB_BGEZL    1
#define ISTUB_BGTZL    1
#define ISTUB_BLEZL    1
#define ISTUB_BLTZL    1
#define ISTUB_BGEZAL   1
#define ISTUB_BLTZAL   1
#define ISTUB_BGEZALL  1
#define ISTUB_BLTZALL  1
#define ISTUB_J        1
#define ISTUB_JAL      1
#define ISTUB_JR       1
#define ISTUB_JALR     1
#define ISTUB_SYSCALL  1
#define ISTUB_BREAK    1
#else
#define ISTUB_BEQ      0
#define ISTUB_BNE      0
#define ISTUB_BEQL     0
#define ISTUB_BNEL     0
#define ISTUB_BGEZ     0
#define ISTUB_BGTZ     0
#define ISTUB_BLEZ     0
#define ISTUB_BLTZ     0
#define ISTUB_BGEZL    0
#define ISTUB_BGTZL    0
#define ISTUB_BLEZL    0
#define ISTUB_BLTZL    0
#define ISTUB_BGEZAL   0
#define ISTUB_BLTZAL   0
#define ISTUB_BGEZALL  0
#define ISTUB_BLTZALL  0
#define ISTUB_J        0
#define ISTUB_JAL      1  // GT4 regresses (wobbly wheels, reflections, loading spinner)
                          // — target-bit + sign-extend + _deleteEEreg fixes applied
                          // to the native body below but are necessary-not-sufficient.
                          // See armsx2_arm64_truth_is_x86jit memory file.
#define ISTUB_JR       0
#define ISTUB_JALR     0
#define ISTUB_SYSCALL  0
#define ISTUB_BREAK    0
#endif

// ============================================================================
//  Native codegen helpers (only compiled when at least one native branch exists)
// ============================================================================

#if !ISTUB_BEQ || !ISTUB_BNE || !ISTUB_BEQL || !ISTUB_BNEL || \
    !ISTUB_BGEZ || !ISTUB_BGTZ || !ISTUB_BLEZ || !ISTUB_BLTZ || \
    !ISTUB_BGEZL || !ISTUB_BGTZL || !ISTUB_BLEZL || !ISTUB_BLTZL || \
    !ISTUB_BGEZAL || !ISTUB_BLTZAL || !ISTUB_BGEZALL || !ISTUB_BLTZALL || \
    !ISTUB_J || !ISTUB_JAL || !ISTUB_JR || !ISTUB_JALR || \
    !ISTUB_SYSCALL || !ISTUB_BREAK

extern void recompileNextInstruction(bool delayslot, bool swapped_delay_slot);

// Helper: set PC to a known immediate and mark branch done
static void SetBranchImm(u32 imm)
{
	armAsm->Mov(RWSCRATCH, imm);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	g_branch = 1;
	g_cpuFlushedPC = true;
	// Signal to recRecompile's block-end link gate that the successor PC
	// is statically known. Covers J/JAL, the `_Rs_ == _Rt_` BEQ (MIPS's
	// `b label` encoding), and every const-folded conditional branch.
	g_eeStaticBranchPC = imm;
}

// Helper: for conditional branches with two GPR operands. Emits split
// taken/not-taken exit tails, each independently linkable via
// emitEELinkableExit. The taken tail is emitted here inline; the
// not-taken tail is staged for recRecompile's block-end iBranchTest via
// g_eeStaticBranchPC = fallthrough.
static void recBranch_GPR64(a64::Condition cond, int rs, int rt, u32 branchTarget, u32 fallthrough)
{
	if (GPR_IS_CONST2(rs, rt))
	{
		bool taken = false;
		switch (cond)
		{
			case a64::eq: taken = (g_cpuConstRegs[rs].SD[0] == g_cpuConstRegs[rt].SD[0]); break;
			case a64::ne: taken = (g_cpuConstRegs[rs].SD[0] != g_cpuConstRegs[rt].SD[0]); break;
			default: break;
		}
		recompileNextInstruction(true, false);
		SetBranchImm(taken ? branchTarget : fallthrough);
		return;
	}

	armLoadGPR64(RSCRATCHGPR, rs);
	if (rt == 0)
		armAsm->Cmp(RSCRATCHGPR, 0);
	else
	{
		armLoadGPR64(RSCRATCHGPR2, rt);
		armAsm->Cmp(RSCRATCHGPR, RSCRATCHGPR2);
	}

	// Cset into RDELAYSLOTGPR — survives the delay-slot emit so we can
	// test it again after the flush.
	armAsm->Cset(RDELAYSLOTGPR, cond);

	// Pre-DS flush — gated on DS opcode safety. Most non-faulting ALU
	// delay slots can skip this; only memory-access / trap-capable DSes
	// need GPR memory coherent before DS execution.
	armFlushConstRegsBeforeDS();
	recompileNextInstruction(true, false);
	armFlushConstRegs();

	// WaitLoop-candidate blocks keep the non-split CSEL exit: iBranchTest
	// will emit a fast-forward-and-jump-to-event tail, which direct-B
	// linking would bypass (defeating the speedhack). Budget > correctness-
	// preserving convergence with the interpreter.
	const bool waitloop_fire = EmuConfig.Speedhacks.WaitLoop && s_nBlockFF;
	if (waitloop_fire)
	{
		armAsm->Mov(RWSCRATCH, branchTarget);
		armAsm->Mov(RWSCRATCH2, fallthrough);
		armAsm->Cmp(RDELAYSLOTGPR, 0);
		armAsm->Csel(RWSCRATCH, RWSCRATCH, RWSCRATCH2, a64::ne);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));

		g_branch = 1;
		g_cpuFlushedPC = true;
		return;
	}

	// Split exits. RDELAYSLOTGPR == 0 → not taken. Cbz keeps the test
	// one insn and stays within range for the local label (±1 MB).
	a64::Label not_taken;
	armAsm->Cbz(RDELAYSLOTGPR, &not_taken);

	// Taken path: write cpuRegs.pc = branchTarget and emit a linkable
	// tail. emitEELinkableExit terminates (no fall-through) — the not-
	// taken label binds immediately after.
	armAsm->Mov(RWSCRATCH, branchTarget);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	emitEELinkableExit(branchTarget);

	// Not-taken path: write cpuRegs.pc = fallthrough and let the block-
	// end machinery (iBranchTest via s_eeLinkTarget = fallthrough) emit
	// the not-taken linkable tail.
	armAsm->Bind(&not_taken);
	armAsm->Mov(RWSCRATCH, fallthrough);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));

	g_branch = 1;
	g_cpuFlushedPC = true;
	g_eeStaticBranchPC = fallthrough;
}

// Helper: for conditional branches comparing one GPR against zero. Same
// split-exit strategy as recBranch_GPR64 — see that function's comments
// for the shape rationale.
static void recBranch_GPR64_vs_Zero(a64::Condition cond, int rs, u32 branchTarget, u32 fallthrough)
{
	if (GPR_IS_CONST1(rs))
	{
		bool taken = false;
		s64 val = g_cpuConstRegs[rs].SD[0];
		switch (cond)
		{
			case a64::ge: taken = (val >= 0); break;
			case a64::lt: taken = (val < 0); break;
			case a64::le: taken = (val <= 0); break;
			case a64::gt: taken = (val > 0); break;
			default: break;
		}
		recompileNextInstruction(true, false);
		SetBranchImm(taken ? branchTarget : fallthrough);
		return;
	}

	armLoadGPR64(RSCRATCHGPR, rs);
	armAsm->Cmp(RSCRATCHGPR, 0);
	armAsm->Cset(RDELAYSLOTGPR, cond);

	// Pre-DS flush gated on DS safety (see armFlushConstRegsBeforeDS).
	armFlushConstRegsBeforeDS();
	recompileNextInstruction(true, false);
	armFlushConstRegs();

	// WaitLoop guard — see recBranch_GPR64 for rationale.
	const bool waitloop_fire = EmuConfig.Speedhacks.WaitLoop && s_nBlockFF;
	if (waitloop_fire)
	{
		armAsm->Mov(RWSCRATCH, branchTarget);
		armAsm->Mov(RWSCRATCH2, fallthrough);
		armAsm->Cmp(RDELAYSLOTGPR, 0);
		armAsm->Csel(RWSCRATCH, RWSCRATCH, RWSCRATCH2, a64::ne);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));

		g_branch = 1;
		g_cpuFlushedPC = true;
		return;
	}

	a64::Label not_taken;
	armAsm->Cbz(RDELAYSLOTGPR, &not_taken);

	// Taken path.
	armAsm->Mov(RWSCRATCH, branchTarget);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	emitEELinkableExit(branchTarget);

	// Not-taken path — block-end iBranchTest emits its linkable tail.
	armAsm->Bind(&not_taken);
	armAsm->Mov(RWSCRATCH, fallthrough);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));

	g_branch = 1;
	g_cpuFlushedPC = true;
	g_eeStaticBranchPC = fallthrough;
}

// Helper: likely branch — if NOT taken, skip the delay slot entirely
static void recBranch_GPR64_Likely(a64::Condition cond, int rs, int rt, u32 branchTarget, u32 fallthrough)
{
	if (GPR_IS_CONST2(rs, rt))
	{
		bool taken = false;
		switch (cond)
		{
			case a64::eq: taken = (g_cpuConstRegs[rs].SD[0] == g_cpuConstRegs[rt].SD[0]); break;
			case a64::ne: taken = (g_cpuConstRegs[rs].SD[0] != g_cpuConstRegs[rt].SD[0]); break;
			default: break;
		}
		if (taken)
		{
			recompileNextInstruction(true, false);
			SetBranchImm(branchTarget);
		}
		else
		{
			SetBranchImm(fallthrough);
		}
		return;
	}

	// Flush unconditionally up front — the not-taken path also exits the block,
	// so any unflushed const-tracked GPRs must be written to memory regardless
	// of which side of the branch we take. Const tracking itself is preserved
	// (only the flushed bit is set), so subsequent armLoadGPR* still uses
	// Mov-imm for any const operands. Flushing before the load also protects
	// RSCRATCHGPR, which armFlushConstRegs uses internally as scratch.
	armFlushConstRegs();

	armLoadGPR64(RSCRATCHGPR, rs);
	if (rt == 0)
		armAsm->Cmp(RSCRATCHGPR, 0);
	else
	{
		armLoadGPR64(RSCRATCHGPR2, rt);
		armAsm->Cmp(RSCRATCHGPR, RSCRATCHGPR2);
	}

	// WaitLoop guard — see recBranch_GPR64 for rationale. Likely-variant
	// twist: delay slot is conditional (executed only on taken side).
	// We preserve that here by re-emitting the old combined path when
	// the block is a wait-loop candidate.
	const bool waitloop_fire = EmuConfig.Speedhacks.WaitLoop && s_nBlockFF;
	if (waitloop_fire)
	{
		a64::Label skipDS_wl, done_wl;
		armAsm->B(&skipDS_wl, a64::InvertCondition(cond));
		recompileNextInstruction(true, false);
		armFlushConstRegs();
		armAsm->Mov(RWSCRATCH, branchTarget);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
		armAsm->B(&done_wl);
		armAsm->Bind(&skipDS_wl);
		armAsm->Mov(RWSCRATCH, fallthrough);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
		armAsm->Bind(&done_wl);
		g_branch = 1;
		g_cpuFlushedPC = true;
		return;
	}

	// If condition is NOT met, skip the delay slot and go to fallthrough.
	// Both paths emit their own linkable exit — taken inline here, not-
	// taken via g_eeStaticBranchPC routed through block-end iBranchTest.
	a64::Label skipDelaySlot;
	armAsm->B(&skipDelaySlot, a64::InvertCondition(cond));

	// Condition met: execute delay slot, then exit to target with a
	// linkable tail. emitEELinkableExit terminates (no fall-through).
	recompileNextInstruction(true, false);
	armFlushConstRegs();
	armAsm->Mov(RWSCRATCH, branchTarget);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	emitEELinkableExit(branchTarget);

	// Not taken: skip delay slot, PC = fallthrough. Block-end iBranchTest
	// emits the linkable tail.
	armAsm->Bind(&skipDelaySlot);
	armAsm->Mov(RWSCRATCH, fallthrough);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));

	g_branch = 1;
	g_cpuFlushedPC = true;
	g_eeStaticBranchPC = fallthrough;
}

static void recBranch_GPR64_vs_Zero_Likely(a64::Condition cond, int rs, u32 branchTarget, u32 fallthrough)
{
	if (GPR_IS_CONST1(rs))
	{
		bool taken = false;
		s64 val = g_cpuConstRegs[rs].SD[0];
		switch (cond)
		{
			case a64::ge: taken = (val >= 0); break;
			case a64::lt: taken = (val < 0); break;
			case a64::le: taken = (val <= 0); break;
			case a64::gt: taken = (val > 0); break;
			default: break;
		}
		if (taken)
		{
			recompileNextInstruction(true, false);
			SetBranchImm(branchTarget);
		}
		else
		{
			SetBranchImm(fallthrough);
		}
		return;
	}

	// Flush unconditionally up front — see recBranch_GPR64_Likely for rationale.
	// Both paths exit the block, so consts must hit memory before either side.
	// Flushing before the load also protects RSCRATCHGPR (used internally by
	// the flush as scratch) so the Tbz/Tbnz below sees the loaded value.
	armFlushConstRegs();

	armLoadGPR64(RSCRATCHGPR, rs);

	// WaitLoop guard — see recBranch_GPR64 for rationale.
	const bool waitloop_fire = EmuConfig.Speedhacks.WaitLoop && s_nBlockFF;
	if (waitloop_fire)
	{
		a64::Label skipDS_wl, done_wl;
		if (cond == a64::lt)
			armAsm->Tbz(RSCRATCHGPR, 63, &skipDS_wl);
		else if (cond == a64::ge)
			armAsm->Tbnz(RSCRATCHGPR, 63, &skipDS_wl);
		else
		{
			armAsm->Cmp(RSCRATCHGPR, 0);
			armAsm->B(&skipDS_wl, a64::InvertCondition(cond));
		}
		recompileNextInstruction(true, false);
		armFlushConstRegs();
		armAsm->Mov(RWSCRATCH, branchTarget);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
		armAsm->B(&done_wl);
		armAsm->Bind(&skipDS_wl);
		armAsm->Mov(RWSCRATCH, fallthrough);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
		armAsm->Bind(&done_wl);
		g_branch = 1;
		g_cpuFlushedPC = true;
		return;
	}

	// For lt/ge, use Tbz/Tbnz to test sign bit directly (1 instruction vs 2)
	a64::Label skipDelaySlot;
	if (cond == a64::lt)
		armAsm->Tbz(RSCRATCHGPR, 63, &skipDelaySlot);
	else if (cond == a64::ge)
		armAsm->Tbnz(RSCRATCHGPR, 63, &skipDelaySlot);
	else
	{
		armAsm->Cmp(RSCRATCHGPR, 0);
		armAsm->B(&skipDelaySlot, a64::InvertCondition(cond));
	}

	// Condition met: execute delay slot, then exit to target with a
	// linkable tail. emitEELinkableExit terminates (no fall-through).
	recompileNextInstruction(true, false);
	armFlushConstRegs();
	armAsm->Mov(RWSCRATCH, branchTarget);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	emitEELinkableExit(branchTarget);

	// Not taken: skip delay slot, PC = fallthrough. Block-end iBranchTest
	// emits the linkable tail.
	armAsm->Bind(&skipDelaySlot);
	armAsm->Mov(RWSCRATCH, fallthrough);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));

	g_branch = 1;
	g_cpuFlushedPC = true;
	g_eeStaticBranchPC = fallthrough;
}

#endif // at least one native branch

// ============================================================================
//  Instruction implementations
// ============================================================================

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ---- BEQ ----
#if ISTUB_BEQ
void recBEQ() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BEQ); }
#else
void recBEQ()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (_Rs_ == _Rt_)
	{
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64(a64::eq, _Rs_, _Rt_, branchTarget, fallthrough);
}
#endif

// ---- BNE ----
#if ISTUB_BNE
void recBNE() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BNE); }
#else
void recBNE()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (_Rs_ == _Rt_)
	{
		// rs != rs is always false
		recompileNextInstruction(true, false);
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64(a64::ne, _Rs_, _Rt_, branchTarget, fallthrough);
}
#endif

// ---- BEQL ----
#if ISTUB_BEQL
void recBEQL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BEQL); }
#else
void recBEQL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (_Rs_ == _Rt_)
	{
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64_Likely(a64::eq, _Rs_, _Rt_, branchTarget, fallthrough);
}
#endif

// ---- BNEL ----
#if ISTUB_BNEL
void recBNEL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BNEL); }
#else
void recBNEL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (_Rs_ == _Rt_)
	{
		// Likely: not taken → skip delay slot
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64_Likely(a64::ne, _Rs_, _Rt_, branchTarget, fallthrough);
}
#endif

// ---- BGEZ ----
#if ISTUB_BGEZ
void recBGEZ() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BGEZ); }
#else
void recBGEZ()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		// 0 >= 0 is always true
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64_vs_Zero(a64::ge, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BGTZ ----
#if ISTUB_BGTZ
void recBGTZ() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BGTZ); }
#else
void recBGTZ()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		// 0 > 0 is always false
		recompileNextInstruction(true, false);
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64_vs_Zero(a64::gt, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BLEZ ----
#if ISTUB_BLEZ
void recBLEZ() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BLEZ); }
#else
void recBLEZ()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		// 0 <= 0 is always true
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64_vs_Zero(a64::le, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BLTZ ----
#if ISTUB_BLTZ
void recBLTZ() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BLTZ); }
#else
void recBLTZ()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		// 0 < 0 is always false
		recompileNextInstruction(true, false);
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64_vs_Zero(a64::lt, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BGEZL ----
#if ISTUB_BGEZL
void recBGEZL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BGEZL); }
#else
void recBGEZL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64_vs_Zero_Likely(a64::ge, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BGTZL ----
#if ISTUB_BGTZL
void recBGTZL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BGTZL); }
#else
void recBGTZL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64_vs_Zero_Likely(a64::gt, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BLEZL ----
#if ISTUB_BLEZL
void recBLEZL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BLEZL); }
#else
void recBLEZL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64_vs_Zero_Likely(a64::le, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BLTZL ----
#if ISTUB_BLTZL
void recBLTZL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BLTZL); }
#else
void recBLTZL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	if (!_Rs_)
	{
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64_vs_Zero_Likely(a64::lt, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BGEZAL ----
#if ISTUB_BGEZAL
void recBGEZAL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BGEZAL); }
#else
void recBGEZAL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	g_cpuConstRegs[31].SD[0] = (s32)(pc + 4);
	GPR_SET_CONST(31);

	if (!_Rs_)
	{
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64_vs_Zero(a64::ge, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BLTZAL ----
#if ISTUB_BLTZAL
void recBLTZAL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BLTZAL); }
#else
void recBLTZAL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	g_cpuConstRegs[31].SD[0] = (s32)(pc + 4);
	GPR_SET_CONST(31);

	if (!_Rs_)
	{
		recompileNextInstruction(true, false);
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64_vs_Zero(a64::lt, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BGEZALL ----
#if ISTUB_BGEZALL
void recBGEZALL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BGEZALL); }
#else
void recBGEZALL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	g_cpuConstRegs[31].SD[0] = (s32)(pc + 4);
	GPR_SET_CONST(31);

	if (!_Rs_)
	{
		recompileNextInstruction(true, false);
		SetBranchImm(branchTarget);
		return;
	}

	recBranch_GPR64_vs_Zero_Likely(a64::ge, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- BLTZALL ----
#if ISTUB_BLTZALL
void recBLTZALL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BLTZALL); }
#else
void recBLTZALL()
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	g_cpuConstRegs[31].SD[0] = (s32)(pc + 4);
	GPR_SET_CONST(31);

	if (!_Rs_)
	{
		SetBranchImm(fallthrough);
		return;
	}

	recBranch_GPR64_vs_Zero_Likely(a64::lt, _Rs_, branchTarget, fallthrough);
}
#endif

// ---- J ----
// Target: upper 4 bits come from the delay-slot address (= `pc` at entry here,
// since the recompiler's `pc` is already pointing at the delay slot), not from
// pc+4. Matches x86 recJ (iR5900Jump.cpp:34).
#if ISTUB_J
void recJ() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::J); }
#else
void recJ()
{
	u32 target = (_Target_ << 2) | (pc & 0xf0000000);
	recompileNextInstruction(true, false);
	SetBranchImm(target);
}
#endif

// ---- JAL ----
// Fully mirrors x86 recJAL (iR5900Jump.cpp:43-66), including its
// _deleteEEreg(31, 0) prologue which we'd previously skipped on the assumption
// the GPR cache / const tracker were clean at op entry. That assumption was
// load-bearing and GT4 proved it unreliable — something was leaving stale
// state for $ra across ops.
//
// Structural mapping x86 → arm64:
//   _deleteEEreg(31, 0) with flush=0:
//     - _deleteGPRtoX86reg(31, DELETE_REG_FLUSH_AND_FREE)   → armGprInvalidate(31)
//     - GPR_DEL_CONST(31)                                   → GPR_DEL_CONST(31)
//     - (no const flush emitted — we're about to overwrite)
//
// Value fixes vs. the old (disabled) arm64 implementation:
//   1. Target upper 4 bits come from `pc & 0xf0000000`, not `(pc+4) & …`.
//      In this recompiler pc at op entry IS the delay-slot address, so
//      the +4 pulled the wrong 256MB segment at boundaries.
//   2. Link address written as zero-extended u32 (UL[0] = pc+4, UL[1] = 0),
//      not sign-extended s32 → s64. The old SD[0] = (s32)(pc+4) stored
//      0xFFFFFFFF???????? in the upper half for any pc+4 ≥ 0x80000000
//      (BIOS / kseg0 / kseg1), silently corrupting $ra.
#if ISTUB_JAL
void recJAL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::JAL); }
#else
void recJAL()
{
	u32 target = (_Target_ << 2) | (pc & 0xf0000000);

	// Match x86's _deleteEEreg(31, 0): drop any host cache slot and clear the
	// const-tracker bit for $ra before installing the new link value.
	armGprInvalidate(31);
	GPR_DEL_CONST(31);

	g_cpuConstRegs[31].UL[0] = pc + 4;
	g_cpuConstRegs[31].UL[1] = 0;
	GPR_SET_CONST(31);

	recompileNextInstruction(true, false);
	SetBranchImm(target);
}
#endif

// ---- JR ----
#if ISTUB_JR
void recJR() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::JR); }
#else
void recJR()
{
	armLoadGPR64(RDELAYSLOTGPR, _Rs_);

	// Pre-DS flush gated on DS safety (see armFlushConstRegsBeforeDS).
	armFlushConstRegsBeforeDS();
	recompileNextInstruction(true, false);
	armFlushConstRegs();

	armAsm->Str(RWDELAYSLOT, a64::MemOperand(RCPUSTATE, PC_OFFSET));

	g_branch = 1;
	g_cpuFlushedPC = true;
}
#endif

// ---- JALR ----
// MIPS spec: rd is encoded in the instruction and can be any GPR, including
// $zero (r0). When rd == 0 the link write is discarded (hardware behavior on
// $zero) and the op degenerates to JR. Matches x86 recJALR (iR5900Jump.cpp:145):
// `if (_Rd_) { ...write link... }` — NO fallback to r31.
//
// Link write is zero-extended u32 (UL[0] = pc+4, UL[1] = 0), matching the
// same invariant fixed in recJAL above.
#if ISTUB_JALR
void recJALR() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::JALR); }
#else
void recJALR()
{
	armLoadGPR64(RDELAYSLOTGPR, _Rs_);

	if (_Rd_)
	{
		// Match x86 _deleteEEreg(_Rd_, 0): drop host cache + const tracker
		// for the link register before installing the new link value.
		armGprInvalidate(_Rd_);
		GPR_DEL_CONST(_Rd_);

		g_cpuConstRegs[_Rd_].UL[0] = pc + 4;
		g_cpuConstRegs[_Rd_].UL[1] = 0;
		GPR_SET_CONST(_Rd_);
	}

	// Pre-DS flush gated on DS safety (see armFlushConstRegsBeforeDS).
	armFlushConstRegsBeforeDS();
	recompileNextInstruction(true, false);
	armFlushConstRegs();

	armAsm->Str(RWDELAYSLOT, a64::MemOperand(RCPUSTATE, PC_OFFSET));

	g_branch = 1;
	g_cpuFlushedPC = true;
}
#endif

// ---- SYSCALL ----
#if ISTUB_SYSCALL
void recSYSCALL() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::SYSCALL); }
#else
void recSYSCALL()
{
	// Interpreter SYSCALL does cpuRegs.pc -= 4 internally, so flush pc as-is
	// (pc = SYSCALL_addr + 4, interpreter subtracts to get SYSCALL_addr)
	armFlushPC();
	armFlushCode();
	armFlushConstRegs();

	armEmitCall((const void*)R5900::Interpreter::OpcodeImpl::SYSCALL);

	// Interpreter may modify any GPR/PC — clear all const tracking
	g_cpuHasConstReg = 1;
	g_cpuFlushedConstReg = 1;
	g_branch = 2;
	g_cpuFlushedPC = true;
}
#endif

// ---- BREAK ----
#if ISTUB_BREAK
void recBREAK() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::BREAK); }
#else
void recBREAK()
{
	// Interpreter BREAK does cpuRegs.pc -= 4 internally, so flush pc as-is
	armFlushPC();
	armFlushCode();
	armFlushConstRegs();

	armEmitCall((const void*)R5900::Interpreter::OpcodeImpl::BREAK);

	// Interpreter may modify any GPR/PC — clear all const tracking
	g_cpuHasConstReg = 1;
	g_cpuFlushedConstReg = 1;
	g_branch = 2;
	g_cpuFlushedPC = true;
}
#endif

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
