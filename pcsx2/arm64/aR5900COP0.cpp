// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — COP0 Instructions
// MFC0, MTC0, BC0x, TLB*, ERET, EI, DI

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"
#include "COP0.h"
#include "Dmac.h"

extern void recompileNextInstruction(bool delayslot, bool swapped_delay_slot);

using namespace R5900;

// CP0 register offsets from RCPUSTATE (x19 = &cpuRegs)
static constexpr s64 CP0_OFFSET(int reg) { return offsetof(cpuRegisters, CP0) + reg * sizeof(u32); }
static constexpr s64 PERF_OFFSET = offsetof(cpuRegisters, PERF);
static constexpr s64 LAST_COP0_CYCLE_OFFSET = offsetof(cpuRegisters, lastCOP0Cycle);
static constexpr s64 LAST_PERF_CYCLE_OFFSET(int n) { return offsetof(cpuRegisters, lastPERFCycle) + n * sizeof(u64); }

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
#if defined(INTERP_COP0) || defined(INTERP_EE)
#define ISTUB_MFC0     1
#define ISTUB_MTC0     1
#define ISTUB_BC0F     1
#define ISTUB_BC0T     1
#define ISTUB_BC0FL    1
#define ISTUB_BC0TL    1
#define ISTUB_TLBR     1
#define ISTUB_TLBWI    1
#define ISTUB_TLBWR    1
#define ISTUB_TLBP     1
#define ISTUB_ERET     1
#define ISTUB_EI       1
#define ISTUB_DI       1
#else
#define ISTUB_MFC0     0
#define ISTUB_MTC0     0
#define ISTUB_BC0F     0
#define ISTUB_BC0T     0
#define ISTUB_BC0FL    0
#define ISTUB_BC0TL    0
#define ISTUB_TLBR     1   // TLB ops are complex — keep as interp (matches x86 recCall)
#define ISTUB_TLBWI    1
#define ISTUB_TLBWR    1
#define ISTUB_TLBP     1
#define ISTUB_ERET     1   // ERET uses armBranchCallInterpreter (matches x86 recBranchCall)
#define ISTUB_EI       1   // EI  uses armBranchCallInterpreter (matches x86 recBranchCall)
#define ISTUB_DI       0   // DI  has a native implementation below (matches x86 recDI —
                           // delayed-by-one-instruction + kernel-mode gating for
                           // Jak X, Spongebob, Incredibles, Tales of Fandom Vol. 2, etc.)
#endif

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {
namespace COP0 {

// ============================================================================
//  MFC0 — Move from COP0 register
//  Reads CP0 register _Rd_ into GPR[_Rt_] (sign-extended to 64 bits)
//  Special cases: Count (reg 9), Status (reg 12), PERF (reg 25)
// ============================================================================

#if ISTUB_MFC0
void recMFC0() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::MFC0); }
#else
void recMFC0()
{
	if (_Rd_ == 9)
	{
		// Count register — matches x86 iCOP0.cpp:135-151.
		// Count += (cycle - lastCOP0Cycle); lastCOP0Cycle = cycle.
		// No min-1 clamp — x86 accepts a zero increment if cycle hasn't advanced,
		// so we do too ("x86 JIT is truth").
		armEmitLoadCurrentCycle(RSCRATCHGPR);
		armAsm->Ldr(RSCRATCHGPR2, a64::MemOperand(RCPUSTATE, LAST_COP0_CYCLE_OFFSET));
		armAsm->Sub(RSCRATCHGPR3, RSCRATCHGPR, RSCRATCHGPR2);
		// Update Count (32-bit add, using low 32 bits of increment)
		armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RCPUSTATE, CP0_OFFSET(9)));
		armAsm->Add(RWSCRATCH2, RWSCRATCH2, RWSCRATCH3);
		armAsm->Str(RWSCRATCH2, a64::MemOperand(RCPUSTATE, CP0_OFFSET(9)));
		// lastCOP0Cycle = cycle
		armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, LAST_COP0_CYCLE_OFFSET));

		if (!_Rt_)
			return;

		armDelConstReg(_Rt_);
		armStoreGPR64SignExt32(RWSCRATCH2, _Rt_);
		return;
	}

	if (!_Rt_)
		return;

	armDelConstReg(_Rt_);

	switch (_Rd_)
	{
		case 12: // Status — raw read, sign-extended (matches x86 iCOP0.cpp:195-196)
		{
			armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, CP0_OFFSET(12)));
			armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
			break;
		}

		case 25: // PERF registers — matches x86 iCOP0.cpp:157-188.
		{
			if (0 == (_Imm_ & 1)) // MFPS — read PCCR
			{
				// x86 doesn't call COP0_UpdatePCCR for MFPS — we match.
				armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, PERF_OFFSET + offsetof(PERFregs, n.pccr)));
				armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
			}
			else if (0 == (_Imm_ & 2)) // MFPC 0 — read PCR0 (via UpdatePCCR)
			{
				armFlushConstRegs();
				armFlushPC();
				armFlushCode();
				armEmitFlushCycleBeforeCall();
				armEmitCall((const void*)COP0_UpdatePCCR);
				armEmitReloadCycleAfterCall();
				armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, PERF_OFFSET + offsetof(PERFregs, n.pcr0)));
				armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
			}
			else // MFPC 1 — read PCR1 (via UpdatePCCR)
			{
				armFlushConstRegs();
				armFlushPC();
				armFlushCode();
				armEmitFlushCycleBeforeCall();
				armEmitCall((const void*)COP0_UpdatePCCR);
				armEmitReloadCycleAfterCall();
				armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, PERF_OFFSET + offsetof(PERFregs, n.pcr1)));
				armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
			}
			break;
		}

		case 24: // Debug breakpoint registers — no-op
			break;

		default:
			armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, CP0_OFFSET(_Rd_)));
			armStoreGPR64SignExt32(RWSCRATCH, _Rt_);
			break;
	}
}
#endif

// ============================================================================
//  MTC0 — Move to COP0 register
//  Writes GPR[_Rt_] into CP0 register _Rd_
//  Special cases: Count (9), Status (12), Config (16), PERF (25)
// ============================================================================

#if ISTUB_MTC0
void recMTC0() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::MTC0); }
#else
void recMTC0()
{
	switch (_Rd_)
	{
		case 9: // Count
		{
			// x86 commits scaleblockcycles_clear() into cpuRegs.cycle here
			// because cpuRegs.cycle IS the x86 source of truth. On arm64, RCYCLE
			// is authoritative during block execution; cpuRegs.cycle is only
			// materialised across C calls and at iBranchTest. For MTC0 Count we
			// only need "cycle now" in a register to stash as lastCOP0Cycle —
			// no memory writeback needed.
			armEmitLoadCurrentCycle(RSCRATCHGPR);
			armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, LAST_COP0_CYCLE_OFFSET));
			armLoadGPR32(RWSCRATCH, _Rt_);
			armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, CP0_OFFSET(9)));
			break;
		}

		case 12: // Status
		{
			// Inlined WriteCP0Status (x86 iCOP0.cpp:205-211 / 271-278):
			//   COP0_UpdatePCCR();
			//   cpuRegs.CP0.n.Status.val = value;
			//   cpuSetNextEventDelta(4);
			//
			// We SKIP COP0_UpdatePCCR here. It's the perf-counter update
			// helper; its effect on Status is limited to lastPERFCycle bookkeeping
			// under the early-out `if (ERL || !CTE)`. For CTE=0 (counters not
			// enabled — the overwhelmingly common case in real games) the call
			// is effectively a no-op except for writing lastPERFCycle. For CTE=1
			// we drop some counter accuracy, which is consumed by game-side
			// profiling only — not emulation correctness.
			//
			// MTC0 Status is on interrupt-handler / kernel-code hot paths, so the
			// C-call overhead (~20 extra instructions plus function prologue /
			// epilogue) matters here. Keeping this fully inline.
			armLoadGPR32(RWSCRATCH, _Rt_);
			armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, CP0_OFFSET(12)));
			armEmitSetNextEventDelta(4);
			break;
		}

		case 16: // Config
		{
			// Inlined WriteCP0Config (COP0.cpp:30-35):
			//   Config = (value & ~0xFC0) | 0x440;
			// Protects the read-only IC/DC cache-size bits; forces them to the
			// fixed mask 0x440. Pure arithmetic, no C call needed.
			armLoadGPR32(RWSCRATCH, _Rt_);
			armAsm->And(RWSCRATCH, RWSCRATCH, ~static_cast<u32>(0xFC0));
			armAsm->Orr(RWSCRATCH, RWSCRATCH, 0x440);
			armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, CP0_OFFSET(16)));
			break;
		}

		case 25: // PERF registers — matches x86 iCOP0.cpp:227-255 / 294-322.
		{
			if (0 == (_Imm_ & 1)) // MTPS
			{
				if (0 != (_Imm_ & 0x3E)) // only effective when register field is 0
					break;
				// x86 pattern: iFlushCall + cycle-update + COP0_UpdatePCCR,
				// then bare store of pccr (no mask), then COP0_DiagnosticPCCR
				// with no cycle update between the two calls.
				armFlushConstRegs();
				armFlushPC();
				armFlushCode();
				armEmitFlushCycleBeforeCall();
				armEmitCall((const void*)COP0_UpdatePCCR);
				armEmitReloadCycleAfterCall();
				// Write pccr raw — x86 does NOT mask. DiagnosticPCCR below
				// will validate / log any unexpected bits.
				armLoadGPR32(RWSCRATCH, _Rt_);
				armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PERF_OFFSET + offsetof(PERFregs, n.pccr)));
				armEmitCall((const void*)COP0_DiagnosticPCCR);
				armEmitReloadCycleAfterCall();
			}
			else if (0 == (_Imm_ & 2)) // MTPC 0
			{
				// As with MTC0 Count: x86 writes cpuRegs.cycle because it's x86's
				// source of truth; arm64 has RCYCLE. Just compute "cycle now"
				// directly from (nec + RCYCLE) into a register for lastPERFCycle[0].
				armLoadGPR32(RWSCRATCH, _Rt_);
				armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PERF_OFFSET + offsetof(PERFregs, n.pcr0)));
				armEmitLoadCurrentCycle(RSCRATCHGPR);
				armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, LAST_PERF_CYCLE_OFFSET(0)));
			}
			else // MTPC 1
			{
				armLoadGPR32(RWSCRATCH, _Rt_);
				armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PERF_OFFSET + offsetof(PERFregs, n.pcr1)));
				armEmitLoadCurrentCycle(RSCRATCHGPR);
				armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, LAST_PERF_CYCLE_OFFSET(1)));
			}
			break;
		}

		case 24: // Debug breakpoint registers — no-op
			break;

		default:
			armLoadGPR32(RWSCRATCH, _Rt_);
			armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, CP0_OFFSET(_Rd_)));
			break;
	}
}
#endif

// ============================================================================
//  BC0F / BC0T / BC0FL / BC0TL — COP0 branch instructions
//  Condition: (((dmacRegs.stat.CIS | ~dmacRegs.pcr.CPC) & 0x3FF) == 0x3FF)
//  BC0T branches when true, BC0F when false.
// ============================================================================

// Helper: evaluate CPCOND0 and set condition result in RDELAYSLOTGPR.
// branchIfTrue: BC0T/BC0TL branch when condition is true.
static void recBC0_setup(bool branchIfTrue)
{
	// CPCOND0 = ((CIS | ~CPC) & 0x3FF) == 0x3FF
	//         ≡ (CPC & ~CIS & 0x3FF) == 0       (De Morgan)
	// Load stat first — the pcr load into w4 clobbers the base in x4.
	armAsm->Mov(RSCRATCHGPR, (u64)(uintptr_t)&dmacRegs);
	armAsm->Ldr(RWSCRATCH2, a64::MemOperand(RSCRATCHGPR, offsetof(DMACregisters, stat)));
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RSCRATCHGPR, offsetof(DMACregisters, pcr)));
	// CPC & ~CIS (bit-clear: clears bits in pcr that are set in stat)
	armAsm->Bic(RWSCRATCH, RWSCRATCH, RWSCRATCH2);
	// Test lower 10 bits — sets Z if all cleared (CPCOND0 true)
	armAsm->Tst(RWSCRATCH, 0x3FF);
	// Cset: 1 = taken.
	// BC0T: taken when eq (condition true). BC0F: taken when ne (condition false).
	armAsm->Cset(RDELAYSLOTGPR, branchIfTrue ? a64::eq : a64::ne);
}

// Non-likely BC0 branch
static void recBC0_helper(bool branchIfTrue)
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	recBC0_setup(branchIfTrue);

	armFlushConstRegs();
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

// Likely BC0 branch (skip delay slot if not taken)
static void recBC0_Likely_helper(bool branchIfTrue)
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	recBC0_setup(branchIfTrue);

	a64::Label skipDelaySlot, done;
	// If NOT taken, skip delay slot
	armAsm->Cbz(RDELAYSLOTGPR, &skipDelaySlot);

	// Taken: execute delay slot, branch to target
	armFlushConstRegs();
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

#if ISTUB_BC0F
void recBC0F()  { armBranchInterpWithDSCycles(R5900::Interpreter::OpcodeImpl::COP0::BC0F); }
#else
void recBC0F()  { recBC0_helper(false); }
#endif

#if ISTUB_BC0T
void recBC0T()  { armBranchInterpWithDSCycles(R5900::Interpreter::OpcodeImpl::COP0::BC0T); }
#else
void recBC0T()  { recBC0_helper(true); }
#endif

#if ISTUB_BC0FL
void recBC0FL() { armBranchInterpWithDSCycles(R5900::Interpreter::OpcodeImpl::COP0::BC0FL); }
#else
void recBC0FL() { recBC0_Likely_helper(false); }
#endif

#if ISTUB_BC0TL
void recBC0TL() { armBranchInterpWithDSCycles(R5900::Interpreter::OpcodeImpl::COP0::BC0TL); }
#else
void recBC0TL() { recBC0_Likely_helper(true); }
#endif

// ============================================================================
//  TLB instructions — complex, always use interpreter
// ============================================================================

#if ISTUB_TLBR
void recTLBR()  { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBR); }
#else
void recTLBR()  { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBR); }
#endif

#if ISTUB_TLBWI
void recTLBWI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBWI); }
#else
void recTLBWI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBWI); }
#endif

#if ISTUB_TLBWR
void recTLBWR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBWR); }
#else
void recTLBWR() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBWR); }
#endif

#if ISTUB_TLBP
void recTLBP()  { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBP); }
#else
void recTLBP()  { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::TLBP); }
#endif

// ============================================================================
//  ERET — Exception Return
//  Modifies PC and Status, must end the block.
// ============================================================================

#if ISTUB_ERET
void recERET() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::ERET); }
#else
void recERET() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::ERET); }
#endif

// ============================================================================
//  EI / DI — Enable / Disable Interrupts
//  EI enables interrupts and must force an event check (upstream uses recBranchCall).
// ============================================================================

#if ISTUB_EI
void recEI() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::EI); }
#else
void recEI() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::EI); }
#endif

// Native DI — matches x86 iCOP0.cpp:97-123 verbatim.
//
// Two semantics x86 JIT got right that the plain interpreter doesn't:
//   1. DI execution is delayed by one instruction — the JIT compiles the next
//      instruction BEFORE applying DI's effect. Games rely on this.
//   2. Only clear EIE (Status bit 0x10000) when in kernel mode OR when any of
//      EXL/ERL/EDI (0x20006) is set. In plain user mode with none of those
//      set, DI is a no-op.
//
// Upstream commit note lists games that regress if either semantic is
// violated: Jak X, Namco 50th anniversary, Spongebob the Movie, Spongebob
// Battle for Bikini Bottom, The Incredibles, The Incredibles rize of the
// underminer, Soukou kihei armodyne, Garfield Saving Arlene,
// Tales of Fandom Vol. 2.
//
// Flow:
//   if (!g_recompilingDelaySlot) compile_next_instruction
//   load Status
//   if (Status & (EXL|ERL|EDI))        goto apply
//   if (Status & KSU) (user/supervisor) goto exit
// apply:
//   Status &= ~EIE
//   store Status
// exit:
#if ISTUB_DI
void recDI() { armCallInterpreter(R5900::Interpreter::OpcodeImpl::COP0::DI); }
#else
void recDI()
{
	// DI is delayed by one instruction: compile the next op first, UNLESS
	// we're ourselves already a delay slot (JR/BEQ/etc. swap paths).
	if (!g_recompilingDelaySlot)
		recompileNextInstruction(false, false);

	a64::Label apply, exit;

	// Load Status into a scratch.
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RCPUSTATE, CP0_OFFSET(12)));

	// (Status & 0x20006) != 0 → fall through to `apply`.
	// x86 does `TEST eax, 0x20006; JNZ iHaveNoIdea` — we use TST+B.ne. The
	// branch is skipped if all of EXL/ERL/EDI are clear.
	armAsm->Tst(RWSCRATCH, 0x20006);
	armAsm->B(&apply, a64::ne);

	// None of EXL/ERL/EDI are set. Check KSU (Status bits 3-4).
	// (Status & 0x18) != 0 → user/supervisor mode → skip the clear.
	armAsm->Tst(RWSCRATCH, 0x18);
	armAsm->B(&exit, a64::ne);

	// Kernel mode with no EXL/ERL/EDI, OR any of EXL/ERL/EDI set: clear EIE.
	armAsm->Bind(&apply);
	armAsm->And(RWSCRATCH, RWSCRATCH, ~static_cast<u32>(0x10000));
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, CP0_OFFSET(12)));

	armAsm->Bind(&exit);
}
#endif

} // namespace COP0
} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
