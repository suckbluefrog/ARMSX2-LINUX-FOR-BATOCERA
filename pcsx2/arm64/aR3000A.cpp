// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 IOP (R3000A) Recompiler — Main JIT Engine
//
// Modeled on the x86 IOP JIT (x86/iR3000A.cpp) and the ARM64 EE JIT
// (arm64/iR5900_arm64.cpp). This file implements:
//   - Code buffer management (Reserve, Reset, Shutdown)
//   - ARM64 dispatcher stubs (DispatcherReg, JITCompile, Enter/Exit)
//   - Block compilation (iopRecRecompile, psxRecompileNextInstruction)
//   - Branch test & cycle counting (iPsxBranchTest, iPsxAddEECycles)
//   - Constant propagation helpers
//   - The psxRec CPU interface struct

#include "arm64/aR3000A.h"
#include "arm64/AsmHelpers.h"
#include "R3000A.h"
#include "x86/BaseblockEx.h"
#include "IopBios.h"
#include "IopHw.h"
#include "IopMem.h"
#include "Common.h"
#include "Config.h"
#include "VMManager.h"
#include "common/Console.h"

#include "arm64/TraceBlocks.h"
#include "common/HeapArray.h"
#include "common/Perf.h"
#include "DebugTools/Breakpoints.h"

#include "iRecAnalysis.h" // EEINST, g_pCurInstInfo

extern void armEndStackFrame(bool save_fpr);

using namespace vixl::aarch64;

// ============================================================================
//  Static globals
// ============================================================================

uptr psxRecLUT[0x10000];
static u32 psxhwLUT[0x10000];

static __fi u32 HWADDR(u32 mem) { return psxhwLUT[mem >> 16] + mem; }

static BASEBLOCK* recRAM = nullptr;
static BASEBLOCK* recROM = nullptr;
static BASEBLOCK* recROM1 = nullptr;
static BASEBLOCK* recROM2 = nullptr;
static BaseBlocks recBlocks;
static u8* recPtr = nullptr;
static u8* recPtrEnd = nullptr;
static ArmConstantPool s_iopConstPool;

u32 psxpc;            // recompiler pc
int psxbranch;        // branch state
u32 g_iopCyclePenalty;
u32 s_psxBlockCycles;
bool s_recompilingDelaySlot = false;

static EEINST* s_pInstCache = nullptr;
static u32 s_nInstCacheSize = 0;

static BASEBLOCK* s_pCurBlock = nullptr;
static BASEBLOCKEX* s_pCurBlockEx = nullptr;

static u32 s_nEndBlock = 0;
static u32 s_branchTo;
static bool s_nBlockFF;

u32 g_psxMaxRecMem = 0;

// Constant propagation state (defined in R3000A.cpp)

// Branch state save/restore
static u32 s_saveConstRegs[32];
static u32 s_saveHasConstReg = 0, s_saveFlushedConstReg = 0;
static EEINST* s_psaveInstInfo = nullptr;
static u32 s_savenBlockCycles = 0;

// PC/Code flush tracking
static bool g_psxFlushedPC, g_psxFlushedCode;

// ============================================================================
//  Dispatcher pointers (generated at init)
// ============================================================================

static const void* iopDispatcherEvent = nullptr;
const void* iopDispatcherReg = nullptr;  // Non-static: accessed by iR3000Atables_arm64.cpp
static const void* iopJITCompile = nullptr;
static const void* iopEnterRecompiledCode = nullptr;
static const void* iopExitRecompiledCode = nullptr;
static const void* iopUnmappedRecLUTPage = nullptr;

// ============================================================================
//  Forward declarations
// ============================================================================

static void iopRecRecompile(u32 startpc);
static void recEventTest();
static void iopRecError(int err);
static void iopClearRecLUT(BASEBLOCK* base, int count);
void recResetIOP();

#define PSX_GETBLOCK(x) PC_GETBLOCK_(x, psxRecLUT)

// ============================================================================
//  Constant Propagation Helpers
// ============================================================================

void iopArmFlushConstReg(int reg)
{
	if (PSX_IS_CONST1(reg) && !(g_psxFlushedConstReg & (1u << reg)))
	{
		armAsm->Mov(RWPSXSCRATCH, g_psxConstRegs[reg]);
		armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_GPR_OFFSET(reg)));
		g_psxFlushedConstReg |= (1u << reg);
	}
}

void iopArmFlushConstRegs()
{
	for (int i = 1; i < 32; i++)
	{
		if ((g_psxHasConstReg & (1u << i)) && !(g_psxFlushedConstReg & (1u << i)))
		{
			armAsm->Mov(RWPSXSCRATCH, g_psxConstRegs[i]);
			armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_GPR_OFFSET(i)));
			g_psxFlushedConstReg |= (1u << i);
		}
		if (g_psxHasConstReg == g_psxFlushedConstReg)
			break;
	}
}

void iopArmLoadGPR(const Register& dst, int gpr)
{
	if (gpr == 0)
	{
		armAsm->Mov(dst, wzr);
		return;
	}
	if (PSX_IS_CONST1(gpr))
	{
		armAsm->Mov(dst, g_psxConstRegs[gpr]);
		return;
	}
	armAsm->Ldr(dst, MemOperand(RPSXSTATE, PSX_GPR_OFFSET(gpr)));
}

void iopArmStoreGPR(const Register& src, int gpr)
{
	if (gpr == 0) return;
	PSX_DEL_CONST(gpr);
	armAsm->Str(src, MemOperand(RPSXSTATE, PSX_GPR_OFFSET(gpr)));
}

void iopArmFlushPC()
{
	if (!g_psxFlushedPC)
	{
		armAsm->Mov(RWPSXSCRATCH, psxpc);
		armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
		g_psxFlushedPC = true;
	}
}

void iopArmFlushCode()
{
	if (!g_psxFlushedCode)
	{
		armAsm->Mov(RWPSXSCRATCH, psxRegs.code);
		armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_CODE_OFFSET));
		g_psxFlushedCode = true;
	}
}

void iopArmCallInterpreter(void (*func)())
{
	iopArmFlushCode();
	iopArmFlushPC();
	iopArmFlushConstRegs();
	armEmitCall((const void*)func);
	// Conservative: after calling interpreter, invalidate all const tracking
	g_psxHasConstReg = g_psxFlushedConstReg = 1; // r0 is always const 0
}

void iopArmBranchCallInterpreter(void (*func)())
{
	iopArmFlushCode();
	iopArmFlushPC();
	iopArmFlushConstRegs();
	armEmitCall((const void*)func);
	g_psxHasConstReg = g_psxFlushedConstReg = 1;
	psxbranch = 2;

	// The interpreter's branch path (psxJ/psxBEQ/...) calls doBranch() which
	// internally runs execI() for the delay slot — that consumes one extra IOP
	// cycle that the JIT block epilogue does not see (s_psxBlockCycles only
	// counts the branch instruction itself). Without compensation, iopCycleEE
	// is undercharged by 8 per stubbed branch, the IOP runs hot relative to
	// the EE, and IOP-side timing (SPU, counters) drifts faster than real.
	// psxRegs.cycle is already correct: interpreter execI bumps it by 1 for
	// the delay slot, and the JIT epilogue bumps it by 1 for the branch.
	// Only iopCycleEE needs the missing 8.
	armAsm->Ldr(w1, MemOperand(RPSXSTATE, PSX_IOPCYCLEEE_OFFSET));
	armAsm->Sub(w1, w1, 8);
	armAsm->Str(w1, MemOperand(RPSXSTATE, PSX_IOPCYCLEEE_OFFSET));
}

// ============================================================================
//  Branch state save/restore
// ============================================================================

void psxSaveBranchState()
{
	s_savenBlockCycles = s_psxBlockCycles;
	memcpy(s_saveConstRegs, g_psxConstRegs, sizeof(g_psxConstRegs));
	s_saveHasConstReg = g_psxHasConstReg;
	s_saveFlushedConstReg = g_psxFlushedConstReg;
	s_psaveInstInfo = g_pCurInstInfo;
}

void psxLoadBranchState()
{
	s_psxBlockCycles = s_savenBlockCycles;
	memcpy(g_psxConstRegs, s_saveConstRegs, sizeof(g_psxConstRegs));
	g_psxHasConstReg = s_saveHasConstReg;
	g_psxFlushedConstReg = s_saveFlushedConstReg;
	g_pCurInstInfo = s_psaveInstInfo;
}

// ============================================================================
//  IOP shadow-verify harness (IOP_SHADOW_VERIFY)
//
//  Mirrors the VU0/VU1 design — pre-block snapshot of psxRegs, post-block
//  snapshot, restore + interp re-run via psxInterpReplayBlock, register-by-
//  register compare, halt-on-first-divergence with state dump and native
//  backtrace.
//
//  Constraints (full discussion in InterpFlags.h):
//    - Validates GPR + CP0 + pc + hi/lo. Cycle-related fields are masked
//      because the JIT batches cycle add at block end while interp bumps
//      per-instruction; CP2D/CP2C (GTE) is also masked because GTE is
//      interp-stubbed.
//    - Memory (iopMem->Main, iopHw, scratchpad) is NOT rolled back. Blocks
//      that write to HW registers will see double-writes during interp
//      replay; for those blocks the harness may go silent or fire
//      spuriously. Best for ALU/branch-heavy bug investigation.
//    - Interp-side iopEventTest is suppressed during replay (see
//      iopShadowSuppressEventTest in R3000AInterpreter.cpp) so we don't
//      mutate cpuRegs/iopHw a second time.
// ============================================================================
#ifdef IOP_SHADOW_VERIFY
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <unwind.h>

alignas(16) static u8 s_iop_shadow_pre [sizeof(psxRegisters)];
alignas(16) static u8 s_iop_shadow_post[sizeof(psxRegisters)];

// (NOTE: full iopMem snapshot/restore was attempted and reverted — even at
// ExposedIopRam-sized buffers (~2MB), the per-block memcpy storm during IOP
// BIOS init's high block-frequency phase pegs DRAM bandwidth and stalls the
// VM. The cleaner fix for iopMem-loop false positives is the SKIP_PC macro
// or relying on per-instruction shadow which doesn't need iopMem rollback.)

// Latched on first divergence — halt path sets it before aborting; the
// verify call also checks it on entry to avoid log spam if another verify
// path got there first. Atomic because halt path may abort mid-verify.
static std::atomic<bool> s_iop_shadow_diverged{false};

// Compile-time flag: set by iopRecRecompile when the block contains a
// load+store with matching (base reg, offset) — those blocks false-positive
// on per-block shadow without iopMem rollback (which we don't have because
// it's too expensive). Read at the verify-hook emit sites; if set, the
// per-block verify BL is skipped for this block. Per-instruction shadow
// stays active and remains accurate.
static bool s_iop_shadow_block_skip = false;

// Runtime flag: set by iop_per_instr_verify when a dynamic-base load/store
// resolves to the IOP HW window. Cleared at block entry by iop_shadow_snapshot.
// Read at iop_shadow_verify entry; if set, per-block compare is skipped
// (HW reads have side effects → JIT consumes the value, replay reads the
// post-consume state — the divergence reflects HW state-machine progression,
// not a JIT codegen bug). Only fires when IOP_SHADOW_VERIFY_PER_INSTR is on
// (per-instr is what actually does the runtime address check); without it
// per-block dynamic-base HW divergences still need manual SKIP_PC.
static bool s_iop_shadow_block_had_hw_runtime = false;

static void iop_shadow_snapshot()
{
	if (s_iop_shadow_diverged.load(std::memory_order_relaxed))
		return;
	std::memcpy(s_iop_shadow_pre, &psxRegs, sizeof(psxRegisters));
	// Reset the runtime HW flag at block entry; per-instr verify will set
	// it during the block's execution if any dynamic-base load/store hits
	// the HW window.
	s_iop_shadow_block_had_hw_runtime = false;
}

static void iop_shadow_dump_state(const char* label, const psxRegisters* state)
{
	Console.Error("--- %s ---", label);
	Console.Error("  pc=0x%08x  code=0x%08x", state->pc, state->code);
	for (u32 r = 0; r < 32; r++)
	{
		Console.Error("  GPR[%2u] = 0x%08x", r, state->GPR.r[r]);
	}
	Console.Error("  hi=0x%08x  lo=0x%08x", state->GPR.n.hi, state->GPR.n.lo);
	Console.Error("  CP0.Status=0x%08x  Cause=0x%08x  EPC=0x%08x  BadVAddr=0x%08x",
		state->CP0.n.Status, state->CP0.n.Cause, state->CP0.n.EPC, state->CP0.n.BadVAddr);
	Console.Error("  cycle=%llu  iopCycleEE=%d  interrupt=0x%08x  pcWriteback=0x%08x",
		(unsigned long long)state->cycle, state->iopCycleEE,
		state->interrupt, state->pcWriteback);
}

struct IopShadowUnwindCtx
{
	u32 frames_seen = 0;
	static constexpr u32 kMaxFrames = 32;
};

static _Unwind_Reason_Code iop_shadow_unwind_cb(struct _Unwind_Context* ctx, void* arg)
{
	auto* state = static_cast<IopShadowUnwindCtx*>(arg);
	if (state->frames_seen >= IopShadowUnwindCtx::kMaxFrames)
		return _URC_END_OF_STACK;

	const uintptr_t pc = _Unwind_GetIP(ctx);
	if (!pc)
		return _URC_END_OF_STACK;

	Dl_info info{};
	const char* sym = "?";
	uintptr_t off = 0;
	if (dladdr(reinterpret_cast<void*>(pc), &info) && info.dli_sname)
	{
		sym = info.dli_sname;
		off = pc - reinterpret_cast<uintptr_t>(info.dli_saddr);
	}
	else if (info.dli_fname)
	{
		sym = info.dli_fname;
		off = pc - reinterpret_cast<uintptr_t>(info.dli_fbase);
	}
	Console.Error("  #%2u  pc=0x%016lx  %s+0x%lx",
		state->frames_seen, (unsigned long)pc, sym, (unsigned long)off);
	state->frames_seen++;
	return _URC_NO_REASON;
}

static void iop_shadow_halt(u32 startpc, u32 endpc,
	const char* first_field, const char* first_detail)
{
	if (s_iop_shadow_diverged.exchange(true, std::memory_order_acq_rel))
		return;

	Console.Error("============================================================");
	Console.Error(" IOP SHADOW DIVERGENCE  startpc=0x%08x  endpc=0x%08x",
		startpc, endpc);
	Console.Error("   first divergent field: %s  %s", first_field, first_detail);
	Console.Error("============================================================");

	const psxRegisters* pre = reinterpret_cast<const psxRegisters*>(s_iop_shadow_pre);
	const psxRegisters* jit = reinterpret_cast<const psxRegisters*>(s_iop_shadow_post);
	iop_shadow_dump_state("PRE-BLOCK (input state)", pre);
	iop_shadow_dump_state("JIT RESULT", jit);
	iop_shadow_dump_state("INTERP RESULT (truth)", &psxRegs);

	// Dump the block's actual instruction encodings so we can see what
	// codegen we're chasing. Walk from startpc until endpc OR a sane cap
	// (in case startpc==endpc for per-instruction halts, we still log the
	// single instruction). Bound the walk to keep logcat usable.
	{
		Console.Error("--- Block instructions ---");
		const u32 first = pre->pc;
		const u32 last = (endpc > first && (endpc - first) < 256u) ? endpc : (first + 32u);
		for (u32 p = first; p <= last; p += 4)
		{
			const u32 enc = iopMemRead32(p);
			Console.Error("  0x%08x: 0x%08x  (op=0x%02x rs=%2u rt=%2u rd=%2u sa=%2u funct=0x%02x imm=0x%04x)",
				p, enc,
				(enc >> 26) & 0x3F,
				(enc >> 21) & 0x1F,
				(enc >> 16) & 0x1F,
				(enc >> 11) & 0x1F,
				(enc >>  6) & 0x1F,
				enc & 0x3F,
				enc & 0xFFFF);
		}
	}

	Console.Error("--- Native C++ backtrace ---");
	IopShadowUnwindCtx ctx;
	_Unwind_Backtrace(&iop_shadow_unwind_cb, &ctx);

	Console.Error("============================================================");
	Console.Error(" Aborting — tombstone will follow");
	Console.Error("============================================================");
	std::abort();
}

static void iop_shadow_log(u32 startpc, u32 endpc,
	const char* field, const char* fmt, ...)
{
	char detail[224];
	va_list ap;
	va_start(ap, fmt);
	std::vsnprintf(detail, sizeof(detail), fmt, ap);
	va_end(ap);
	iop_shadow_halt(startpc, endpc, field, detail);
}

// Called at every block exit (post-instructions, pre-cycle/event/dispatch).
// `endpc` is the post-block pc the JIT exits with — used to terminate the
// interp replay at the same boundary the JIT did. Cases:
//   - psxSetBranchImm(imm): endpc = imm. Taken branches fire branch2 with
//     branchPC = imm (loop terminates on branch2, then pc = imm). Not-taken
//     conditional branches DON'T fire branch2 — they fall through; replay
//     terminates when pc reaches imm (== psxpc on the not-taken path).
//   - psxSetBranchReg (dynamic JR/JALR): always taken; branch2 fires;
//     sentinel 0xFFFFFFFF is fine, we never compare against pc.
//   - iopArmBranchCallInterpreter (psxbranch==2): endpc = psxpc = delay-
//     slot pc, for the same reason as psxSetBranchImm not-taken path.
//   - Fall-through (link_next_block): endpc = psxpc = block end. Replay
//     terminates on pc reaching endpc; branch2 never fires.
//
// startpc is recovered from s_iop_shadow_pre->pc (captured at block entry).
static void iop_shadow_verify(u32 endpc)
{
	if (s_iop_shadow_diverged.load(std::memory_order_relaxed))
	{
		// Already halted (or about to). Don't re-enter the comparison;
		// just leave psxRegs as the JIT left it so the VM doesn't drift.
		return;
	}

	// Per-instruction verify sets this flag if any dynamic-base load/store
	// in the block resolved to the IOP HW window at runtime. HW reads have
	// side effects (HW_ICTRL etc. auto-clear bits on read), so the JIT's
	// runtime read consumed the value — replay'd interp would read the
	// post-consume state. Comparison would diverge for reasons unrelated to
	// JIT codegen.
	if (s_iop_shadow_block_had_hw_runtime)
		return;

	// Cycle-window gate: snapshot/verify are emitted unconditionally; this
	// runtime check skips the dominant cost (interp replay + memcmp) when
	// outside the user's window. Use to bypass long boot/menu sequences.
#if defined(IOP_SHADOW_VERIFY_FROM_CYCLE) || defined(IOP_SHADOW_VERIFY_TO_CYCLE)
	{
		const u64 cur = psxRegs.cycle;
#ifdef IOP_SHADOW_VERIFY_FROM_CYCLE
		if (cur < (IOP_SHADOW_VERIFY_FROM_CYCLE)) return;
#endif
#ifdef IOP_SHADOW_VERIFY_TO_CYCLE
		if (cur > (IOP_SHADOW_VERIFY_TO_CYCLE))   return;
#endif
	}
#endif

	// PC-range skip: silence known false-positive blocks (typically iopMem-
	// loop blocks that need IOP_SHADOW_VERIFY_IOPMEM rollback to compare
	// correctly). Compare against the snapshot's pc which is the actual
	// block startpc.
#if defined(IOP_SHADOW_VERIFY_SKIP_PC_LO) && defined(IOP_SHADOW_VERIFY_SKIP_PC_HI)
	{
		const u32 spc = reinterpret_cast<const psxRegisters*>(s_iop_shadow_pre)->pc;
		if (spc >= (IOP_SHADOW_VERIFY_SKIP_PC_LO) && spc < (IOP_SHADOW_VERIFY_SKIP_PC_HI))
			return;
	}
#endif

	// Snapshot post-JIT state. Capture full psxRegisters; we'll selectively
	// compare against interp result below.
	std::memcpy(s_iop_shadow_post, &psxRegs, sizeof(psxRegisters));

	const u32 startpc = reinterpret_cast<const psxRegisters*>(s_iop_shadow_pre)->pc;

	// Restore pre-block psxRegs and run interp. iopEventTest is suppressed
	// inside doBranch for the duration of the replay (see
	// iopShadowSuppressEventTest). iopMem and iopHw are NOT rolled back —
	// blocks that read+write the same address (the IOPBOOT counter pattern
	// at 0xbfc4ab68: LW/ADDU/J/SW) will false-positive because replay's LW
	// reads the JIT's already-stored value. Use the SKIP_PC macro to silence
	// known cases or rely on per-instruction shadow (which is immune to this
	// since each replay's window is a single instruction). Safety cap of
	// 256 instructions covers any realistic IOP block (typical 5-30 insns,
	// max maybe ~64).
	std::memcpy(&psxRegs, s_iop_shadow_pre, sizeof(psxRegisters));
	psxInterpReplayBlock(endpc, 256);

	// Now: psxRegs == interp result, s_iop_shadow_post == JIT result.
	const psxRegisters* jit  = reinterpret_cast<const psxRegisters*>(s_iop_shadow_post);
	const psxRegisters* iref = &psxRegs;
	bool diverged = false;

	auto check_u32 = [&](const char* name, u32 j, u32 i) -> bool {
		if (j != i) {
			iop_shadow_log(startpc, endpc, name,
				"jit=0x%08x interp=0x%08x", j, i);
			return true;
		}
		return false;
	};

	// pc — primary signal that control flow diverged. Mask kseg0/kseg1
	// mirror bits: JIT may emit Str pc with kseg0 set (compile-time psxpc
	// inherited it from a kseg0 startpc), while interp's per-instruction
	// pc advance from a non-mirror entry stays raw. Same physical instr.
	if (!diverged) diverged = check_u32("pc",
		jit->pc & 0x1FFFFFFFu, iref->pc & 0x1FFFFFFFu);

	// GPR[1..31] — r0 is hardwired zero. hi/lo at indices 32/33 (per the
	// GPRRegs union comment "hi needs to be at index 32!").
	for (u32 r = 1; r < 32 && !diverged; r++)
	{
		if (jit->GPR.r[r] != iref->GPR.r[r])
		{
			char fname[16];
			std::snprintf(fname, sizeof(fname), "GPR[%u]", r);
			diverged = check_u32(fname, jit->GPR.r[r], iref->GPR.r[r]);
		}
	}
	if (!diverged) diverged = check_u32("hi", jit->GPR.n.hi, iref->GPR.n.hi);
	if (!diverged) diverged = check_u32("lo", jit->GPR.n.lo, iref->GPR.n.lo);

	// CP0 — Status / Cause / EPC / BadVAddr / Count are the ones that
	// matter for IOP exceptions. Everything else is rarely written.
	for (u32 c = 0; c < 32 && !diverged; c++)
	{
		if (jit->CP0.r[c] != iref->CP0.r[c])
		{
			char fname[16];
			std::snprintf(fname, sizeof(fname), "CP0[%u]", c);
			diverged = check_u32(fname, jit->CP0.r[c], iref->CP0.r[c]);
		}
	}

	// CP2D / CP2C (GTE) is masked — INTERP_IOP_COP2 keeps GTE stubbed in
	// the JIT, so JIT and interp both go through the same fallback path
	// and any divergence here would be in the stub, not codegen. cycle /
	// iopCycleEE / iopBreak / iopNextEventCycle / sCycle / eCycle /
	// interrupt / pcWriteback are all masked because batched JIT cycle
	// accounting diverges from per-instruction interp accounting by
	// design.

	// Restore JIT state so the VM continues from the JIT's actual result,
	// not the interp replay's. Includes iopMem + iopHw to undo the
	// rollback we did before replay. Must run on success (no divergence)
	// AND on the diverged-but-already-halted-elsewhere early return above.
	if (!diverged)
		std::memcpy(&psxRegs, s_iop_shadow_post, sizeof(psxRegisters));
}

#ifdef IOP_SHADOW_VERIFY_PER_INSTR
// Per-instruction shadow buffers — separate from the per-block ones because
// the two harnesses run at independent points (snapshot/verify pairs are
// strictly nested: per-instruction inside per-block).
alignas(16) static u8 s_iop_per_instr_pre [sizeof(psxRegisters)];
alignas(16) static u8 s_iop_per_instr_post[sizeof(psxRegisters)];

static void iop_per_instr_snapshot()
{
	if (s_iop_shadow_diverged.load(std::memory_order_relaxed))
		return;
	std::memcpy(s_iop_per_instr_pre, &psxRegs, sizeof(psxRegisters));
}

// `pc` is the compile-time-known pc of the instruction whose JIT-emitted
// code just executed. Replay runs exactly one execI() starting from
// s_iop_per_instr_pre, compares against the JIT post-state.
static void iop_per_instr_verify(u32 pc)
{
	if (s_iop_shadow_diverged.load(std::memory_order_relaxed))
		return;

	// Honor the same cycle window as per-block.
#if defined(IOP_SHADOW_VERIFY_FROM_CYCLE) || defined(IOP_SHADOW_VERIFY_TO_CYCLE)
	{
		const u64 cur = psxRegs.cycle;
#ifdef IOP_SHADOW_VERIFY_FROM_CYCLE
		if (cur < (IOP_SHADOW_VERIFY_FROM_CYCLE)) return;
#endif
#ifdef IOP_SHADOW_VERIFY_TO_CYCLE
		if (cur > (IOP_SHADOW_VERIFY_TO_CYCLE))   return;
#endif
	}
#endif

	std::memcpy(s_iop_per_instr_post, &psxRegs, sizeof(psxRegisters));

	std::memcpy(&psxRegs, s_iop_per_instr_pre, sizeof(psxRegisters));
	// Patch pc to the instruction's compile-time-known address. The JIT
	// does NOT update psxRegs.pc per-instruction inside a block — it stays
	// at startpc until block-end batched store. Without this patch, the
	// pre-instruction snapshot has psxRegs.pc = block startpc, so
	// execI's `iopMemRead32(psxRegs.pc)` would re-fetch the FIRST
	// instruction of the block instead of the instruction we're verifying
	// — replay runs the wrong op every time after instr 1.
	psxRegs.pc = pc;

	// Runtime HW-range check for loads/stores with a non-const-tracked
	// base. The compile-time check in psxRecompileNextInstruction catches
	// const-base HW memrefs but misses dynamic bases (e.g. r4 loaded from
	// a prior block). At this point psxRegs holds the per-instruction PRE
	// state, so GPR.r[base] is the runtime value the JIT load/store used.
	// If the effective address falls in the IOP HW window, skip the
	// comparison — HW reads have side effects that would diverge.
	{
		const u32 op  = iopMemRead32(pc);
		const u32 opc = op >> 26;
		const bool is_load  = (opc >= 0x20 && opc <= 0x26);
		const bool is_store = (opc == 0x28 || opc == 0x29 || opc == 0x2A ||
		                       opc == 0x2B || opc == 0x2E);
		if (is_load || is_store)
		{
			const u32 base    = (op >> 21) & 0x1F;
			const u32 baseval = psxRegs.GPR.r[base];
			const int32_t off = static_cast<int16_t>(op & 0xFFFF);
			const u32 effective = baseval + static_cast<u32>(off);
			const u32 masked    = effective & 0x1FFFFFFFu;
			// Same definition as the compile-time check: HW = anything
			// outside IOP RAM and BIOS ROM. Covers SPEED/DEV9 (0x1F400000),
			// scratchpad+iopHw (0x1F800000), SPU2 (0x1F900000), CDVD,
			// expansion regions, etc.
			const bool is_iop_ram  = (masked < Ps2MemSize::TotalIopRam);
			const bool is_bios_rom = (masked >= 0x1FC00000u);
			if (!is_iop_ram && !is_bios_rom)
			{
				// Mark the block as having a runtime HW memref so the
				// per-block verify also bails. Then restore JIT's POST
				// state and skip the per-instruction comparison.
				s_iop_shadow_block_had_hw_runtime = true;
				std::memcpy(&psxRegs, s_iop_per_instr_post, sizeof(psxRegisters));
				return;
			}
		}
	}

	psxInterpExecuteOne();

	const psxRegisters* jit  = reinterpret_cast<const psxRegisters*>(s_iop_per_instr_post);
	const psxRegisters* iref = &psxRegs;
	bool diverged = false;

	// Stale-block heuristic: if JIT POST matches PRE bit-for-bit on all
	// the fields we'd compare (GPR/hi/lo/CP0), the JIT-emit produced no
	// observable runtime effect. Real instructions that mutate state
	// (ADDIU, LW, MFC0, etc.) ALWAYS produce a JIT change in the masked
	// field set. Zero-change-from-JIT means one of:
	//   a) The instruction is genuinely a no-op (NOP, branch, or store-
	//      only — those don't write GPR/CP0/hi/lo). Then INTERP also
	//      produces no change and the divergence wouldn't fire.
	//   b) The JIT block is stale: cached compiled code reflects compile-
	//      time memory contents (e.g. zeros = NOPs), not current runtime
	//      contents. SMC invalidation didn't fire, JIT runs no-ops while
	//      interp reads current memory and runs the real instruction.
	//   c) An actual JIT codegen bug producing no observable effect —
	//      vanishingly rare; the field-wise compare below would still
	//      catch most of those, just not at this instruction.
	// Case (b) is the dominant cause of false-positive halts here. Skip
	// when we detect zero JIT-side change (and INTERP made changes —
	// otherwise no divergence would have fired).
	{
		const psxRegisters* pre = reinterpret_cast<const psxRegisters*>(s_iop_per_instr_pre);
		bool jit_made_change = false;
		for (u32 r = 1; r < 32 && !jit_made_change; r++)
			if (pre->GPR.r[r] != jit->GPR.r[r]) jit_made_change = true;
		if (!jit_made_change && pre->GPR.n.hi != jit->GPR.n.hi) jit_made_change = true;
		if (!jit_made_change && pre->GPR.n.lo != jit->GPR.n.lo) jit_made_change = true;
		for (u32 c = 0; c < 32 && !jit_made_change; c++)
			if (pre->CP0.r[c] != jit->CP0.r[c]) jit_made_change = true;
		if (!jit_made_change)
		{
			// JIT did nothing observable. Skip — restore JIT post so the
			// VM keeps progressing (with the no-op'd state, which will
			// likely keep diverging until SMC invalidation fires or the
			// block gets recompiled).
			std::memcpy(&psxRegs, s_iop_per_instr_post, sizeof(psxRegisters));
			return;
		}
	}

	auto check_u32 = [&](const char* name, u32 j, u32 i) -> bool {
		if (j != i) {
			// Reuse the per-block halt path — it dumps PRE / JIT / INTERP
			// based on s_iop_shadow_pre/post, but for per-instruction we
			// want the per-instruction PRE/POST. Patch them in temporarily.
			std::memcpy(s_iop_shadow_pre,  s_iop_per_instr_pre,  sizeof(psxRegisters));
			std::memcpy(s_iop_shadow_post, s_iop_per_instr_post, sizeof(psxRegisters));
			iop_shadow_log(pc, pc, name, "jit=0x%08x interp=0x%08x", j, i);
			return true;
		}
		return false;
	};

	// pc / code / cycle are masked: JIT doesn't update psxRegs.pc per-
	// instruction (block-end batched store); interp's execI does pc += 4
	// per call. Same for cycle (interp +=1, JIT batched). And `code` may
	// or may not be flushed by iopArmFlushCode. The per-instruction
	// shadow is for catching REGISTER-LEVEL codegen bugs; pc/cycle
	// accounting is verified at block-level scope where both sides agree.
	for (u32 r = 1; r < 32 && !diverged; r++)
	{
		if (jit->GPR.r[r] != iref->GPR.r[r])
		{
			char fname[16];
			std::snprintf(fname, sizeof(fname), "GPR[%u]", r);
			diverged = check_u32(fname, jit->GPR.r[r], iref->GPR.r[r]);
		}
	}
	if (!diverged) diverged = check_u32("hi", jit->GPR.n.hi, iref->GPR.n.hi);
	if (!diverged) diverged = check_u32("lo", jit->GPR.n.lo, iref->GPR.n.lo);
	for (u32 c = 0; c < 32 && !diverged; c++)
	{
		if (jit->CP0.r[c] != iref->CP0.r[c])
		{
			char fname[16];
			std::snprintf(fname, sizeof(fname), "CP0[%u]", c);
			diverged = check_u32(fname, jit->CP0.r[c], iref->CP0.r[c]);
		}
	}

	if (!diverged)
		std::memcpy(&psxRegs, s_iop_per_instr_post, sizeof(psxRegisters));
}
#endif // IOP_SHADOW_VERIFY_PER_INSTR
#endif // IOP_SHADOW_VERIFY

// ============================================================================
//  Event test (called from dispatcher)
// ============================================================================

static void recEventTest()
{
	_cpuEventTest_Shared();
}

// ============================================================================
//  ARM64 Dispatcher Generation
// ============================================================================

// Dispatcher: jump to compiled block at psxRegs.pc
static const void* _DynGen_DispatcherReg()
{
	u8* retval = armStartBlock();

	// w0 = psxRegs.pc
	armAsm->Ldr(w0, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
	// x1 = psxRecLUT[pc >> 16]
	armAsm->Lsr(w1, w0, 16);
	armAsm->Ldr(x1, MemOperand(RPSXRECLUT, x1, LSL, 3));
	// block = psxRecLUT[pc>>16] + pc * 2 (sizeof(BASEBLOCK)=8, /4=2)
	armAsm->Lsl(x2, x0, 1);
	armAsm->Ldr(x3, MemOperand(x1, x2));
	armAsm->Br(x3);

	armEndBlock();
	return retval;
}

// Event dispatcher: call recEventTest then jump to DispatcherReg
static const void* _DynGen_DispatcherEvent()
{
	pxAssert(iopDispatcherReg);
	u8* retval = armStartBlock();

	armEmitCall((const void*)recEventTest);
	armEmitJmp(iopDispatcherReg);

	armEndBlock();
	return retval;
}

// JIT compile stub: called when hitting uncompiled block
static const void* _DynGen_JITCompile()
{
	pxAssert(iopDispatcherReg);
	u8* retval = armStartBlock();

	// arg1 = psxRegs.pc
	armAsm->Ldr(RWARG1, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
	armEmitCall((const void*)iopRecRecompile);

	// Re-dispatch to newly compiled block
	armAsm->Ldr(w0, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
	armAsm->Lsr(w1, w0, 16);
	armAsm->Ldr(x1, MemOperand(RPSXRECLUT, x1, LSL, 3));
	armAsm->Lsl(x2, x0, 1);
	armAsm->Ldr(x3, MemOperand(x1, x2));
	armAsm->Br(x3);

	armEndBlock();
	return retval;
}

// Enter recompiled code: called from C++ (recExecuteBlock)
static const void* _DynGen_EnterRecompiledCode()
{
	u8* retval = armStartBlock();

	// Save callee-saved registers
	armBeginStackFrame(false);

	// Pin state registers
	armMoveAddressToReg(RPSXSTATE, &psxRegs);
	armMoveAddressToReg(RPSXRECLUT, psxRecLUT);

	// Jump to dispatcher
	armEmitJmp(iopDispatcherReg);

	// Exit point: blocks jump here when iopCycleEE <= 0
	// This must immediately follow so iopExitRecompiledCode is addressable.
	armEndBlock();

	// Generate iopExitRecompiledCode as a separate block
	iopExitRecompiledCode = armStartBlock();
	armEndStackFrame(false);
	armAsm->Ret();
	armEndBlock();

	return retval;
}

// Unmapped page error handler
static const void* _DynGen_UnmappedRecLUTPage()
{
	u8* retval = armStartBlock();

	armAsm->Mov(RWARG1, 0);
	armEmitCall((const void*)iopRecError);
	armEmitJmp(iopExitRecompiledCode);

	armEndBlock();
	return retval;
}

static void _DynGen_Dispatchers()
{
	// Register each dispatcher with simpleperf so it shows up by name in
	// profiler reports. armAsmPtr is the bump pointer that advances after
	// each armEndBlock(); sample it before/after each generator to get
	// the size of the just-emitted block.
	const auto reg_dispatcher = [](const void* start, const char* name) {
		Perf::iop.Register(start,
			static_cast<size_t>(armAsmPtr - reinterpret_cast<u8*>(const_cast<void*>(start))),
			name);
	};

	iopDispatcherReg = _DynGen_DispatcherReg();
	reg_dispatcher(iopDispatcherReg, "IOP_DispatcherReg");

	iopDispatcherEvent = _DynGen_DispatcherEvent();
	reg_dispatcher(iopDispatcherEvent, "IOP_DispatcherEvent");

	iopJITCompile = _DynGen_JITCompile();
	reg_dispatcher(iopJITCompile, "IOP_JITCompile");

	iopEnterRecompiledCode = _DynGen_EnterRecompiledCode();
	reg_dispatcher(iopEnterRecompiledCode, "IOP_EnterRecompiledCode");
	// iopExitRecompiledCode is set inside _DynGen_EnterRecompiledCode

	iopUnmappedRecLUTPage = _DynGen_UnmappedRecLUTPage();
	reg_dispatcher(iopUnmappedRecLUTPage, "IOP_UnmappedRecLUTPage");

	recBlocks.SetJITCompile(iopJITCompile);
}

// ============================================================================
//  Error handler
// ============================================================================

static void iopRecError(int err)
{
	switch (err)
	{
		case 0:
			Console.Error("[IOP ARM64 Rec] Jump to unmapped recLUT page (PC: 0x%08x)", psxRegs.pc);
			break;
		case 1:
			Console.Error("[IOP ARM64 Rec] Block execution at 0x%08x with zero fnptr (code buffer overflow?)", psxRegs.pc);
			break;
		default:
			Console.Error("[IOP ARM64 Rec] Unknown error %d at PC 0x%08x", err, psxRegs.pc);
			break;
	}

	Cpu->ExitExecution();
}

// ============================================================================
//  Cycle counting & branch test
// ============================================================================

static __fi u32 psxScaleBlockCycles()
{
	return s_psxBlockCycles;
}

// Emit code to subtract scaled IOP cycles from iopCycleEE.
// In normal mode (not PS1): iopCycleEE -= blockCycles * 8
// In PS1 mode: uses cnum/cdenom ratio (emitted as a C++ helper call for now).
static void iPsxAddEECycles(u32 blockCycles)
{
	// Normal mode (most common): iopCycleEE -= blockCycles * 8
	// PS1 mode check is done at compile time for this block.
	// For simplicity, always emit the normal-mode path.
	// PS1 mode is extremely rare in PS2 context.

	if (blockCycles == 0) return;

	u32 eeCycles = blockCycles * 8;
	armAsm->Ldr(w1, MemOperand(RPSXSTATE, PSX_IOPCYCLEEE_OFFSET));
	if (Assembler::IsImmAddSub(eeCycles))
	{
		armAsm->Subs(w1, w1, eeCycles);
	}
	else
	{
		armAsm->Mov(w0, eeCycles);
		armAsm->Subs(w1, w1, w0);
	}
	armAsm->Str(w1, MemOperand(RPSXSTATE, PSX_IOPCYCLEEE_OFFSET));
}

// Emit the branch test at the end of a compiled block.
// Adds block cycles, subtracts from iopCycleEE, checks for events,
// and exits if the IOP timeslice is exhausted.
static void iPsxBranchTest(u32 newpc, u32 cpuBranch)
{
	u32 blockCycles = psxScaleBlockCycles();

	if (EmuConfig.Speedhacks.WaitLoop && s_nBlockFF && newpc == s_branchTo)
	{
		// Wait loop optimization: fast-forward cycle to iopNextEventCycle
		// cycle += (iopCycleEE + 7) >> 3
		// cycle = min(cycle, iopNextEventCycle)
		armAsm->Ldr(x0, MemOperand(RPSXSTATE, PSX_CYCLE_OFFSET));
		armAsm->Mov(x4, x0); // save original cycle
		armAsm->Ldr(w1, MemOperand(RPSXSTATE, PSX_IOPCYCLEEE_OFFSET));
		armAsm->Add(w1, w1, 7);
		armAsm->Lsr(w1, w1, 3);
		armAsm->Add(x0, x0, x1);
		armAsm->Ldr(x2, MemOperand(RPSXSTATE, PSX_IOPNEXTEVENTCYCLE_OFFSET));
		armAsm->Cmp(x0, x2);
		armAsm->Csel(x0, x2, x0, hs); // clamp to iopNextEventCycle
		armAsm->Str(x0, MemOperand(RPSXSTATE, PSX_CYCLE_OFFSET));

		// Compute how many IOP cycles we actually advanced, subtract *8 from iopCycleEE
		armAsm->Sub(x0, x0, x4); // delta cycles
		armAsm->Lsl(x0, x0, 3); // * 8
		armAsm->Ldr(w1, MemOperand(RPSXSTATE, PSX_IOPCYCLEEE_OFFSET));
		armAsm->Subs(w1, w1, w0);
		armAsm->Str(w1, MemOperand(RPSXSTATE, PSX_IOPCYCLEEE_OFFSET));

		// Exit if iopCycleEE <= 0
		armEmitCondBranch(le, iopExitRecompiledCode);

		// Call iopEventTest
		armEmitCall((const void*)iopEventTest);

		// If PC changed, re-dispatch
		if (newpc != 0xffffffff)
		{
			armAsm->Ldr(w0, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
			armAsm->Mov(w1, newpc);
			armAsm->Cmp(w0, w1);
			armEmitCondBranch(ne, iopDispatcherReg);
		}
	}
	else
	{
		// Normal path: add block cycles to psxRegs.cycle
		armAsm->Ldr(x0, MemOperand(RPSXSTATE, PSX_CYCLE_OFFSET));
		if (Assembler::IsImmAddSub(blockCycles))
		{
			armAsm->Add(x0, x0, blockCycles);
		}
		else
		{
			armAsm->Mov(x4, (u64)blockCycles);
			armAsm->Add(x0, x0, x4);
		}
		armAsm->Str(x0, MemOperand(RPSXSTATE, PSX_CYCLE_OFFSET));

		// Subtract from iopCycleEE
		iPsxAddEECycles(blockCycles);

		// Exit if iopCycleEE <= 0
		// w1 still holds the updated iopCycleEE from iPsxAddEECycles
		armEmitCondBranch(le, iopExitRecompiledCode);

		// Check if an event is pending: cycle >= iopNextEventCycle
		armAsm->Ldr(x2, MemOperand(RPSXSTATE, PSX_IOPNEXTEVENTCYCLE_OFFSET));
		armAsm->Cmp(x0, x2);

		Label noEventDone;
		armAsm->B(&noEventDone, lo); // branch if cycle < iopNextEventCycle (no event)

		// Event pending: call iopEventTest
		armEmitCall((const void*)iopEventTest);

		// If PC changed due to exception/interrupt, re-dispatch
		if (newpc != 0xffffffff)
		{
			armAsm->Ldr(w0, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
			armAsm->Mov(w1, newpc);
			armAsm->Cmp(w0, w1);
			armEmitCondBranch(ne, iopDispatcherReg);
		}

		armAsm->Bind(&noEventDone);
	}
}

// ============================================================================
//  Branch target helpers (called from instruction implementations)
// ============================================================================

// Dynamic branch (JR/JALR): PC was set by instruction, go through dispatcher
void psxSetBranchReg()
{
	psxbranch = 1;

	// Flush state
	iopArmFlushConstRegs();

#ifdef IOP_SHADOW_VERIFY
	// Verify BEFORE iPsxBranchTest (iPsxBranchTest runs iopEventTest which
	// can mutate iopMem and skew the comparison). Skip emit on aliased-
	// memref blocks (auto-detected at compile time).
	if (!s_iop_shadow_block_skip)
	{
		armAsm->Mov(RWARG1, 0xffffffffu);
		armEmitCall((const void*)iop_shadow_verify);
	}
#endif

	// iPsxBranchTest with dynamic target
	iPsxBranchTest(0xffffffff, 1);

	// Dispatch
	armEmitJmp(iopDispatcherReg);
}

// Static branch: PC = imm
void psxSetBranchImm(u32 imm)
{
	psxbranch = 1;

	// Store PC
	armAsm->Mov(RWPSXSCRATCH, imm);
	armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
	iopArmFlushConstRegs();

#ifdef IOP_SHADOW_VERIFY
	// Verify BEFORE iPsxBranchTest (see psxSetBranchReg comment for the
	// iopMem-mutation rationale). Pass `imm` as endpc — for TAKEN branches
	// the interp's doBranch fires branch2; for NOT-TAKEN branches (rpsxBEQ
	// etc. emit psxSetBranchImm(psxpc) on the fall-through path) the interp
	// handler does NOT set branch2 — it just falls through. `imm` works
	// for both: branch target if taken, block end if not. Skip on aliased-
	// memref blocks (auto-detected at compile time).
	if (!s_iop_shadow_block_skip)
	{
		armAsm->Mov(RWARG1, imm);
		armEmitCall((const void*)iop_shadow_verify);
	}
#endif

	iPsxBranchTest(imm, imm <= psxpc);

	// Dispatch (could be block-linked in the future)
	armEmitJmp(iopDispatcherReg);
}

// ============================================================================
//  Delay slot swap attempt
// ============================================================================

bool psxTrySwapDelaySlot(u32 rs, u32 rt, u32 rd)
{
	if (s_recompilingDelaySlot)
		return false;

	const u32 opcode_encoded = iopMemRead32(psxpc);
	if (opcode_encoded == 0) // NOP
	{
		psxRecompileNextInstruction(true, true);
		return true;
	}

	const u32 opcode_rs = ((opcode_encoded >> 21) & 0x1F);
	const u32 opcode_rt = ((opcode_encoded >> 16) & 0x1F);
	const u32 opcode_rd = ((opcode_encoded >> 11) & 0x1F);

	switch (opcode_encoded >> 26)
	{
		case 8: // ADDI
		case 9: // ADDIU
		case 10: // SLTI
		case 11: // SLTIU
		case 12: // ANDI
		case 13: // ORI
		case 14: // XORI
		case 15: // LUI
		{
			// Rt = Rs op Imm -- safe if Rt doesn't conflict with branch regs
			if (opcode_rt == rs || opcode_rt == rt || opcode_rt == rd)
				break;
			if ((rs != 0 && opcode_rt == rs) || (rt != 0 && opcode_rt == rt))
				break;
			psxRecompileNextInstruction(true, true);
			return true;
		}

		case 0: // SPECIAL
		{
			const u32 funct = opcode_encoded & 0x3F;
			switch (funct)
			{
				case 0: // SLL
				case 2: // SRL
				case 3: // SRA
				{
					if (opcode_rd == rs || opcode_rd == rt || opcode_rd == rd)
						break;
					psxRecompileNextInstruction(true, true);
					return true;
				}
				case 33: // ADDU
				case 35: // SUBU
				case 36: // AND
				case 37: // OR
				case 38: // XOR
				case 39: // NOR
				case 42: // SLT
				case 43: // SLTU
				{
					if (opcode_rd == rs || opcode_rd == rt || opcode_rd == rd)
						break;
					if ((rs != 0 && opcode_rd == rs) || (rt != 0 && opcode_rd == rt))
						break;
					psxRecompileNextInstruction(true, true);
					return true;
				}
				default:
					break;
			}
			break;
		}
		default:
			break;
	}

	return false;
}

// ============================================================================
//  Instruction compilation
// ============================================================================

void psxRecompileNextInstruction(bool delayslot, bool swapped_delayslot)
{
	const int old_code = psxRegs.code;
	EEINST* old_inst_info = g_pCurInstInfo;
	s_recompilingDelaySlot = delayslot;

	psxRegs.code = iopMemRead32(psxpc);
	// Per-instruction cycle multiplier — see g_iopCycleMultiplier in
	// R3000AInterpreter.cpp. JIT bakes this in at compile time. NOTE:
	// changing the value requires invalidating the JIT cache (psxCpu->Reset)
	// for already-compiled blocks to pick up the new multiplier.
	extern u32 g_iopCycleMultiplier;
	s_psxBlockCycles += g_iopCycleMultiplier;
	psxpc += 4;

	g_pCurInstInfo++;

	g_psxFlushedPC = false;
	g_psxFlushedCode = false;
	g_iopCyclePenalty = 0;

#ifdef IOP_PROFILE_OPS
	const u8* _iop_op_start = armGetCurrentCodePointer();
	const u32 _iop_op_pc    = psxpc - 4;
	const u32 _iop_op_code  = psxRegs.code;
#endif

#if defined(IOP_SHADOW_VERIFY) || defined(IOP_SHADOW_VERIFY_PER_INSTR)
	// Auto-skip loads/stores that target the IOP hardware register window.
	// Reading HW regs has side effects (e.g. HW_ICTRL at 0x1F801078 auto-
	// clears bits on read), so JIT's runtime load consumes the value while
	// the comparison shadow's load reads the post-consume value — diverg-
	// ence that doesn't reflect any JIT codegen bug. Detect via the JIT's
	// compile-time const tracking on the base register: if base is const-
	// known (typically from LUI+ORI/ADDIU) and base+offset falls in the
	// IOP HW range, set the per-instr local skip and (if per-block VERIFY
	// is also enabled) the per-block skip flag.
	//
	// Runs under EITHER macro so per-instruction shadow standalone gets
	// the skip too; under both, per-block also gets it via
	// s_iop_shadow_block_skip.
	[[maybe_unused]] bool _iop_skip_per_instr = false;
	{
		const u32 op    = psxRegs.code;
		const u32 opc   = op >> 26;
		const bool is_load  = (opc >= 0x20 && opc <= 0x26);
		const bool is_store = (opc == 0x28 || opc == 0x29 || opc == 0x2A ||
		                       opc == 0x2B || opc == 0x2E);
		if (is_load || is_store)
		{
			const u32 base = (op >> 21) & 0x1F;
			if (PSX_IS_CONST1(base))
			{
				const u32 effective = g_psxConstRegs[base] +
				                       static_cast<u32>(static_cast<int32_t>(
				                           static_cast<int16_t>(op & 0xFFFF)));
				const u32 masked = effective & 0x1FFFFFFFu;
				// IOP HW window: anything outside IOP RAM (0..8MB max for
				// dev mode) and outside BIOS ROM (0x1FC00000+). The PS2
				// IOP has hardware spread across 0x1F400000 (SPEED/DEV9),
				// 0x1F800000 (scratchpad+iopHw), 0x1F900000 (SPU2),
				// 0x1FA00000 (CDVD/etc.). All have side-effect-having
				// reads/writes that the shadow comparison can't roll back.
				const bool is_iop_ram  = (masked < Ps2MemSize::TotalIopRam);
				const bool is_bios_rom = (masked >= 0x1FC00000u);
				if (!is_iop_ram && !is_bios_rom)
				{
					_iop_skip_per_instr = true;
#ifdef IOP_SHADOW_VERIFY
					s_iop_shadow_block_skip = true;
#endif
				}
			}
		}
	}
#endif

#ifdef IOP_SHADOW_VERIFY_PER_INSTR
	// Flush const-tracked GPRs to memory BEFORE snapshot so PRE captures
	// the JIT's logical state (memory coherent with g_psxConstRegs). Without
	// this, instructions later in the block see stale memory r26 (e.g.
	// from a deferred LUI) and the interp replay reads the wrong inputs.
	if (!_iop_skip_per_instr)
	{
		iopArmFlushConstRegs();
		armEmitCall((const void*)iop_per_instr_snapshot);
	}
	const u32 _iop_per_instr_pc = psxpc - 4;
#endif

	// Dispatch to instruction recompiler
	rpsxBSC[psxRegs.code >> 26]();
	s_psxBlockCycles += g_iopCyclePenalty;

#ifdef IOP_SHADOW_VERIFY_PER_INSTR
	// Same flush AFTER the op's emit: any new const tracked by this
	// instruction (e.g. LUI/ADDIU writing a literal) gets Str'd to memory
	// so POST captures the JIT's logical post-instruction state. Without
	// this, ops that const-fold their result (no Str emit) make POST
	// memory disagree with interp's actual write — every const-folding
	// op fires a spurious divergence.
	if (!_iop_skip_per_instr)
	{
		iopArmFlushConstRegs();
		// Verify post-emit. Only reached on non-branch ops. Pass the
		// instruction's pc so the halt log identifies the exact op.
		armAsm->Mov(RWARG1, _iop_per_instr_pc);
		armEmitCall((const void*)iop_per_instr_verify);
	}
#endif
#ifdef IOP_PROFILE_OPS
	{
		const u8* _iop_op_end = armGetCurrentCodePointer();
		if (_iop_op_end > _iop_op_start)
		{
			char _iop_op_name[48];
			std::snprintf(_iop_op_name, sizeof(_iop_op_name),
				"IOP_%02x_0x%08x", _iop_op_code >> 26, _iop_op_pc);
			Perf::iop.Register(_iop_op_start,
				static_cast<size_t>(_iop_op_end - _iop_op_start), _iop_op_name);
		}
	}
#endif

	if (swapped_delayslot)
	{
		psxRegs.code = old_code;
		g_pCurInstInfo = old_inst_info;
	}
}

// ============================================================================
//  Block compilation
// ============================================================================

static void iopRecRecompile(const u32 startpc)
{
	u32 i;
	u32 link_next_block = 0;

	// SYSMEM module detection
	if (startpc == 0x890)
	{
		DevCon.WriteLn(Color_Gray, "R3000 Debugger: Branch to 0x890 (SYSMEM). Clearing modules.");
		R3000SymbolGuardian.ClearIrxModules();
	}

	// IRX injection hack
	if (startpc == 0x1630 && EmuConfig.CurrentIRX.length() > 3)
	{
		if (iopMemRead32(0x20018) == 0x1F)
			iopMemWrite32(0x20094, 0xbffc0000);
	}

	// IOPBOOT memory size override
	if (startpc == 0xbfc4a000)
		psxRegs.GPR.n.a0 = Ps2MemSize::ExposedIopRam >> 20;

	// PS1 mode can drive the IOP into garbage code paths (e.g. JR into a
	// register that ended up holding 0). Real hardware raises an Address
	// Error on instruction fetch at PC=0; mirror that here instead of
	// asserting, so the BIOS exception handler at 0x80000080 can take over.
	// Upstream relies on pxAssert being compiled out in release; our Android
	// debug builds keep it live, which turned this into a SIGABRT.
	if (startpc == 0)
	{
		Console.Warning("[IOP rec] startpc == 0, raising AdEL exception");
		psxException(0x10 /* AdEL: address error, instruction fetch */, 0);
		return;
	}
	pxAssert(startpc);

	// Check code buffer space
	if (recPtr >= recPtrEnd)
	{
		recResetIOP();
	}

	// Set up assembler
	size_t capacity = recPtrEnd - recPtr;
	armSetAsmPtr(recPtr, capacity, &s_iopConstPool);
	recPtr = armStartBlock();

	s_pCurBlock = PSX_GETBLOCK(startpc);
	pxAssert(s_pCurBlock->GetFnptr() == (uptr)iopJITCompile);

	s_pCurBlockEx = recBlocks.Get(HWADDR(startpc));
	if (!s_pCurBlockEx || s_pCurBlockEx->startpc != HWADDR(startpc))
		s_pCurBlockEx = recBlocks.New(HWADDR(startpc), (uptr)recPtr);

	psxbranch = 0;
	s_pCurBlock->SetFnptr((uptr)armGetCurrentCodePointer());
	s_psxBlockCycles = 0;

	// Reset recompiler state
	psxpc = startpc;
	g_psxHasConstReg = g_psxFlushedConstReg = 1; // r0 is always const 0

	// BIOS call check
	if ((psxHu32(HW_ICFG) & 8) && (HWADDR(startpc) == 0xa0 || HWADDR(startpc) == 0xb0 || HWADDR(startpc) == 0xc0))
	{
		armEmitCall((const void*)psxBiosCall);
		// If psxBiosCall returns non-zero, BIOS handled it — jump to dispatcher
		armEmitCbnz(RWRET, iopDispatcherReg);
	}

	// Scan for block end
	i = startpc;
	s_nEndBlock = 0xffffffff;
	s_branchTo = (u32)-1;

	while (1)
	{
		BASEBLOCK* pblock = PSX_GETBLOCK(i);
		if (i != startpc && pblock->GetFnptr() != (uptr)iopJITCompile)
		{
			link_next_block = 1;
			s_nEndBlock = i;
			break;
		}

		psxRegs.code = iopMemRead32(i);

		switch (psxRegs.code >> 26)
		{
			case 0: // SPECIAL
				if (_psxFunct_ == 8 || _psxFunct_ == 9) // JR, JALR
				{
					s_nEndBlock = i + 8;
					goto StartRecomp;
				}
				break;

			case 1: // REGIMM
				if (_psxRt_ == 0 || _psxRt_ == 1 || _psxRt_ == 16 || _psxRt_ == 17)
				{
					s_branchTo = _psxImm_ * 4 + i + 4;
					if (s_branchTo > startpc && s_branchTo < i)
						s_nEndBlock = s_branchTo;
					else
						s_nEndBlock = i + 8;
					goto StartRecomp;
				}
				break;

			case 2: // J
			case 3: // JAL
				s_branchTo = (_psxTarget_ << 2) | ((i + 4) & 0xf0000000);
				s_nEndBlock = i + 8;
				goto StartRecomp;

			case 4: // BEQ
			case 5: // BNE
			case 6: // BLEZ
			case 7: // BGTZ
				s_branchTo = _psxImm_ * 4 + i + 4;
				if (s_branchTo > startpc && s_branchTo < i)
					s_nEndBlock = s_branchTo;
				else
					s_nEndBlock = i + 8;
				goto StartRecomp;
		}

		i += 4;
	}

StartRecomp:

#ifdef IOP_SHADOW_VERIFY
	// Auto-detect blocks that read+write the same (base reg, offset) pair —
	// these false-positive on per-block shadow without iopMem rollback (the
	// IOPBOOT counter pattern: LW r5,0(r6) / ADDU r5,r5,r16 / SW r5,0(r6)).
	// JIT runtime's SW updates memory, replay's later LW reads the post-SW
	// value and computes a one-iteration-ahead result. Conservative match
	// (encoded base+offset, not runtime address) — false skip on legitimate
	// non-aliased blocks where the base register changed mid-block, but
	// per-instruction shadow stays active so we still catch instruction-
	// level codegen bugs there. Saves the user from manually maintaining
	// SKIP_PC ranges across IOPBOOT (which has many such blocks).
	{
		s_iop_shadow_block_skip = false;
		// Single pass: detect SYSCALL/BREAK, AND track whether the block
		// has any load and any store. If both → skip. Without iopMem
		// rollback (which we can't afford during BIOS init's high block
		// frequency), any combination of load + store can hide indirect
		// aliasing through different base registers — e.g. LW r4,0x1200(r4)
		// at 0xb4d0 and SW r2,-4(r17) at 0xb500 (with r17=0x11204) both
		// touch mem[0x11200], but encoded (base,offset) pairs differ. The
		// stricter aliased-detector misses this. Per-instruction shadow is
		// unaffected — its single-instruction window has no cross-instr
		// memory carry, so it stays accurate on these blocks.
		bool has_load = false;
		bool has_store = false;
		for (u32 j = startpc; j < s_nEndBlock; j += 4)
		{
			const u32 op = iopMemRead32(j);
			const u32 opc = op >> 26;
			if (opc == 0)
			{
				const u32 funct = op & 0x3F;
				if (funct == 0x0c || funct == 0x0d)
				{
					s_iop_shadow_block_skip = true;
					break;
				}
			}
			// Loads: LB(0x20), LH(0x21), LWL(0x22), LW(0x23),
			// LBU(0x24), LHU(0x25), LWR(0x26).
			if (opc >= 0x20 && opc <= 0x26)
				has_load = true;
			// Stores: SB(0x28), SH(0x29), SWL(0x2A), SW(0x2B), SWR(0x2E).
			if (opc == 0x28 || opc == 0x29 || opc == 0x2A ||
			    opc == 0x2B || opc == 0x2E)
				has_store = true;
		}
		if (!s_iop_shadow_block_skip && has_load && has_store)
			s_iop_shadow_block_skip = true;
	}
#endif

	// Detect wait loops (branch-to-self with only NOPs)
	s_nBlockFF = false;
	if (s_branchTo == startpc)
	{
		s_nBlockFF = true;
		for (i = startpc; i < s_nEndBlock; i += 4)
		{
			if (i != s_nEndBlock - 8)
			{
				switch (iopMemRead32(i))
				{
					case 0: // NOP
						break;
					default:
						s_nBlockFF = false;
				}
			}
		}
	}

	// Build EEINST liveness info
	{
		EEINST* pcur;

		if (s_nInstCacheSize < (s_nEndBlock - startpc) / 4 + 1)
		{
			free(s_pInstCache);
			s_nInstCacheSize = (s_nEndBlock - startpc) / 4 + 10;
			s_pInstCache = (EEINST*)malloc(sizeof(EEINST) * s_nInstCacheSize);
			pxAssert(s_pInstCache != nullptr);
		}

		pcur = s_pInstCache + (s_nEndBlock - startpc) / 4;
		_recClearInst(pcur);
		pcur->info = 0;

		for (i = s_nEndBlock; i > startpc; i -= 4)
		{
			psxRegs.code = iopMemRead32(i - 4);
			pcur[-1] = pcur[0];
			rpsxpropBSC(pcur - 1, pcur);
			pcur--;
		}
	}

	// Compile instructions
	g_pCurInstInfo = s_pInstCache;

#ifdef TRACE_BLOCKS
	armAsm->Mov(RWARG1, startpc);
	armEmitCall((void*)iopTraceBlock);
#endif

#ifdef IOP_SHADOW_VERIFY
	// Snapshot psxRegs at block entry, AFTER the BIOS Cbnz above so we
	// only fire for blocks that actually run their JIT'd instructions
	// (BIOS-handled calls exit to dispatcher before reaching here).
	armEmitCall((const void*)iop_shadow_snapshot);
#endif

	while (!psxbranch && psxpc < s_nEndBlock)
	{
		psxRecompileNextInstruction(false, false);
	}

	pxAssert((psxpc - startpc) >> 2 <= 0xffff);
	s_pCurBlockEx->size = (psxpc - startpc) >> 2;

	if (!(psxpc & 0x10000000))
		g_psxMaxRecMem = std::max((psxpc & ~0xa0000000), g_psxMaxRecMem);

	// Emit block epilogue
	if (psxbranch == 2)
	{
		// Dynamic branch (JR/JALR)
		iopArmFlushConstRegs();
#ifdef IOP_SHADOW_VERIFY
		// Verify BEFORE iPsxBranchTest (see psxSetBranchReg comment).
		// psxbranch == 2 = iopArmBranchCallInterpreter (fully-stubbed branch).
		// For TAKEN branches the interp handler's doBranch fires branch2 and
		// pc = target. For NOT-TAKEN conditional branches (psxBEQ etc.
		// returning without doBranch) pc stays at the delay-slot pc that
		// FlushPC wrote — which equals psxpc here. Pass psxpc as endpc so
		// replay terminates at the same pc the JIT did even when not taken.
		if (!s_iop_shadow_block_skip)
		{
			armAsm->Mov(RWARG1, psxpc);
			armEmitCall((const void*)iop_shadow_verify);
		}
#endif
		iPsxBranchTest(0xffffffff, 1);
		armEmitJmp(iopDispatcherReg);
	}
	else
	{
		if (psxbranch)
			pxAssert(!link_next_block);
		else
		{
			// Fall-through: add cycles
			u32 blockCycles = psxScaleBlockCycles();

			armAsm->Ldr(x0, MemOperand(RPSXSTATE, PSX_CYCLE_OFFSET));
			if (Assembler::IsImmAddSub(blockCycles))
				armAsm->Add(x0, x0, blockCycles);
			else
			{
				armAsm->Mov(x4, (u64)blockCycles);
				armAsm->Add(x0, x0, x4);
			}
			armAsm->Str(x0, MemOperand(RPSXSTATE, PSX_CYCLE_OFFSET));
			iPsxAddEECycles(blockCycles);
		}

		if (link_next_block || !psxbranch)
		{
			pxAssert(psxpc == s_nEndBlock);
			iopArmFlushConstRegs();

			// Store PC and dispatch (no block linking for now)
			armAsm->Mov(RWPSXSCRATCH, psxpc);
			armAsm->Str(RWPSXSCRATCH, MemOperand(RPSXSTATE, PSX_PC_OFFSET));
#ifdef IOP_SHADOW_VERIFY
			// Fall-through block — pass psxpc (== s_nEndBlock) as endpc so
			// replay terminates at the same boundary the JIT did. Skip on
			// aliased-memref blocks (auto-detected at compile time).
			if (!s_iop_shadow_block_skip)
			{
				armAsm->Mov(RWARG1, psxpc);
				armEmitCall((const void*)iop_shadow_verify);
			}
#endif
			armEmitJmp(iopDispatcherReg);
			psxbranch = 3;
		}
	}

	// Finalize block
	u8* blockEnd = armEndBlock();

	pxAssert(blockEnd < SysMemory::GetIOPRecEnd());
	s_pCurBlockEx->x86size = blockEnd - recPtr;

	Perf::iop.RegisterPC((void*)s_pCurBlockEx->fnptr, s_pCurBlockEx->x86size, s_pCurBlockEx->startpc);

	recPtr = blockEnd;

	pxAssert((g_psxHasConstReg & g_psxFlushedConstReg) == g_psxHasConstReg);

	s_pCurBlock = nullptr;
	s_pCurBlockEx = nullptr;
}

// ============================================================================
//  Reserve / Reset / Execute / Clear / Shutdown
// ============================================================================

static DynamicHeapArray<BASEBLOCK, 4096> recLutReserve;
static DynamicHeapArray<BASEBLOCK, 4096> recLutUnmapped;
static size_t recLutEntries;
static bool extraRam = false;

static void recReserveRAM()
{
	recLutEntries =
		((Ps2MemSize::ExposedIopRam + Ps2MemSize::Rom + Ps2MemSize::Rom1 + Ps2MemSize::Rom2) / 4);

	if (recLutReserve.size() != recLutEntries)
		recLutReserve.resize(recLutEntries);

	recLutUnmapped.resize(_64kb / 4);

	BASEBLOCK* curpos = recLutReserve.data();
	recRAM = curpos;
	curpos += (Ps2MemSize::ExposedIopRam / 4);
	recROM = curpos;
	curpos += (Ps2MemSize::Rom / 4);
	recROM1 = curpos;
	curpos += (Ps2MemSize::Rom1 / 4);
	recROM2 = curpos;
	curpos += (Ps2MemSize::Rom2 / 4);
}

static void recReserve()
{
	recPtr = SysMemory::GetIOPRec();
	recPtrEnd = SysMemory::GetIOPRecEnd() - _64kb;

	s_iopConstPool.Init(recPtrEnd, _64kb);

	recReserveRAM();

	if (!s_pInstCache)
	{
		s_nInstCacheSize = 128;
		s_pInstCache = (EEINST*)malloc(sizeof(EEINST) * s_nInstCacheSize);
		if (!s_pInstCache)
			pxFailRel("Failed to allocate IOP InstCache array.");
	}
}

void recResetIOP()
{
	DevCon.WriteLn("iR3000A ARM64 Recompiler reset.");

	if (CHECK_EXTRAMEM != extraRam)
	{
		recReserveRAM();
		extraRam = !extraRam;
	}

	// Set up assembler at start of code buffer
	size_t capacity = recPtrEnd - SysMemory::GetIOPRec();
	armSetAsmPtr(SysMemory::GetIOPRec(), capacity, &s_iopConstPool);
	s_iopConstPool.Reset();

	// Generate dispatchers
	_DynGen_Dispatchers();
	// armEndBlock() already advanced armAsmPtr past emitted code and set armAsm=nullptr,
	// so we can't use armGetCurrentCodePointer() (which dereferences armAsm).
	recPtr = armAsmPtr;

	// Clear all block entries
	iopClearRecLUT(reinterpret_cast<BASEBLOCK*>(recLutReserve.data()),
		Ps2MemSize::ExposedIopRam + Ps2MemSize::Rom + Ps2MemSize::Rom1 + Ps2MemSize::Rom2);

	BASEBLOCK* unmapped = recLutUnmapped.data();

	for (int i = 0; i < 0x10000; i++)
		recLUT_SetPage(psxRecLUT, psxhwLUT, unmapped, i, 0, 0);

	for (int i = 0; i < (int)(_64kb / 4); i++)
		unmapped[i].SetFnptr((uptr)iopUnmappedRecLUTPage);

	// Map IOP RAM (mirrored at 0x0000, 0x8000, 0xa000)
	for (int i = 0; i < 0x80; i++)
	{
		u32 mask = (Ps2MemSize::ExposedIopRam / _64kb) - 1;
		recLUT_SetPage(psxRecLUT, psxhwLUT, recRAM, 0x0000, i, i & mask);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recRAM, 0x8000, i, i & mask);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recRAM, 0xa000, i, i & mask);
	}

	// Map ROM
	for (int i = 0x1fc0; i < 0x2000; i++)
	{
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM, 0x0000, i, i - 0x1fc0);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM, 0x8000, i, i - 0x1fc0);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM, 0xa000, i, i - 0x1fc0);
	}

	// Map ROM1
	for (int i = 0x1e00; i < 0x1e40; i++)
	{
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM1, 0x0000, i, i - 0x1e00);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM1, 0x8000, i, i - 0x1e00);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM1, 0xa000, i, i - 0x1e00);
	}

	// Map ROM2
	for (int i = 0x1e40; i < 0x1e48; i++)
	{
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM2, 0x0000, i, i - 0x1e40);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM2, 0x8000, i, i - 0x1e40);
		recLUT_SetPage(psxRecLUT, psxhwLUT, recROM2, 0xa000, i, i - 0x1e40);
	}

	if (s_pInstCache)
		memset(s_pInstCache, 0, sizeof(EEINST) * s_nInstCacheSize);

	recBlocks.Reset();
	g_psxMaxRecMem = 0;
	psxbranch = 0;
}

static void iopClearRecLUT(BASEBLOCK* base, int count)
{
	for (int i = 0; i < count / 4; i++)
		base[i].SetFnptr((uptr)iopJITCompile);
}

static __noinline s32 recExecuteBlock(s32 eeCycles)
{
	psxRegs.iopBreak = 0;
	psxRegs.iopCycleEE = eeCycles;

	((void (*)())iopEnterRecompiledCode)();

	return psxRegs.iopBreak + psxRegs.iopCycleEE;
}

// Returns the offset to the next instruction after any cleared memory
static __fi u32 psxRecClearMem(u32 pc)
{
	BASEBLOCK* pblock;

	pblock = PSX_GETBLOCK(pc);
	if (pblock->GetFnptr() == (uptr)iopJITCompile)
		return 4;

	pc = HWADDR(pc);

	u32 lowerextent = pc, upperextent = pc + 4;
	int blockidx = recBlocks.Index(pc);
	pxAssert(blockidx != -1);

	while (BASEBLOCKEX* pexblock = recBlocks[blockidx - 1])
	{
		if (pexblock->startpc + pexblock->size * 4 <= lowerextent)
			break;
		lowerextent = std::min(lowerextent, pexblock->startpc);
		blockidx--;
	}

	int toRemoveFirst = blockidx;

	while (BASEBLOCKEX* pexblock = recBlocks[blockidx])
	{
		if (pexblock->startpc >= upperextent)
			break;
		lowerextent = std::min(lowerextent, pexblock->startpc);
		upperextent = std::max(upperextent, pexblock->startpc + pexblock->size * 4);
		blockidx++;
	}

	if (toRemoveFirst != blockidx)
		recBlocks.Remove(toRemoveFirst, (blockidx - 1));

	// Clear all BASEBLOCK entries in range
	for (u32 addr = lowerextent; addr < upperextent; addr += 4)
	{
		BASEBLOCK* p = PSX_GETBLOCK(addr);
		p->SetFnptr((uptr)iopJITCompile);
	}

	return upperextent - pc;
}

#define PSXREC_CLEARM(mem) \
	(((mem) < g_psxMaxRecMem && (psxRecLUT[(mem) >> 16] + (mem))) ? \
			psxRecClearMem(mem) : \
			4)

static void recClearIOP(u32 addr, u32 size)
{
	u32 upperLimit = addr + size;
	for (u32 i = addr; i < upperLimit; i += PSXREC_CLEARM(i))
		;
}

static void recShutdown()
{
	recLutReserve.deallocate();
	s_iopConstPool.Destroy();

	safe_free(s_pInstCache);
	s_nInstCacheSize = 0;

	recPtr = nullptr;
	recPtrEnd = nullptr;
}

// ============================================================================
//  psxRec CPU interface
// ============================================================================

R3000Acpu psxRec = {
	recReserve,
	recResetIOP,
	recExecuteBlock,
	recClearIOP,
	recShutdown,
};
