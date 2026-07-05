// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — Main file
// Provides block compilation, dispatchers, and execution loop.

#include "Common.h"
#include "CDVD/CDVD.h"
#include "Elfheader.h"
#include "GS.h"
#include "Host.h"
#include "Memory.h"
#include "Patch.h"
#include "R3000A.h"
#include "R5900OpcodeTables.h"
#include "VMManager.h"
#include "vtlb.h"

#include "arm64/arm64Emitter.h"
#include "arm64/AsmHelpers.h"
#include "x86/BaseblockEx.h"

#include "arm64/TraceBlocks.h"

// EEINST struct + EEINST_USED/LIVE bits + g_pCurInstInfo extern declaration.
// Shared with the IOP recompiler — iRecAnalysis.h is the arm64-friendly
// version of the x86/iCore.h analysis pieces (no x86emitter dependency).
#include "iRecAnalysis.h"

// Defined in arm64/iR5900Backprop_arm64.cpp (a copy of x86/iR5900Analysis.cpp's
// machinery, since that TU drags in x86emitter.h via iR5900Analysis.h).
extern void recBackpropBSC(u32 code, EEINST* prev, EEINST* pinst);

#include "common/AlignedMalloc.h"
#include "common/Console.h"
#include "common/FastJmp.h"
#include "common/HeapArray.h"
#include "common/Perf.h"

#include <unordered_map>

using namespace R5900;

// ============================================================================
//  State — mirrors the x86 recompiler's state
// ============================================================================

static bool eeRecNeedsReset = false;
static bool eeCpuExecuting = false;
static bool eeRecExitRequested = false;
static bool g_resetEeScalingStats = false;

#define PC_GETBLOCK(x) PC_GETBLOCK_(x, recLUT)

u32 maxrecmem = 0;
alignas(16) static uptr recLUT[_64kb];
alignas(16) static u32 hwLUT[_64kb];

static __fi u32 HWADDR(u32 mem) { return hwLUT[mem >> 16] + mem; }

u32 s_nBlockCycles = 0;
bool s_nBlockInterlocked = false;
u32 pc;
int g_branch;

alignas(16) GPR_reg64 g_cpuConstRegs[32] = {};
u32 g_cpuHasConstReg = 0, g_cpuFlushedConstReg = 0;
bool g_cpuFlushedPC, g_cpuFlushedCode, g_recompilingDelaySlot;

eeProfiler EE::Profiler;

// Defined in x86/iCore.cpp on x86 — we need our own on ARM64
EEINST* g_pCurInstInfo = nullptr;

// ============================================================================
//  Code buffer and block management
// ============================================================================

static DynamicHeapArray<u8, 4096> recRAMCopy;
static DynamicHeapArray<BASEBLOCK, 4096> recLutReserve_RAM;
static DynamicHeapArray<BASEBLOCK, 4096> recLutUnmapped;
static size_t recLutEntries;
static bool extraRam;

static BASEBLOCK* recRAM = nullptr;
static BASEBLOCK* recROM = nullptr;
static BASEBLOCK* recROM1 = nullptr;
static BASEBLOCK* recROM2 = nullptr;

static BaseBlocks recBlocks;
static u8* recPtr = nullptr;
static u8* recPtrEnd = nullptr;
EEINST* s_pInstCache = nullptr;
static u32 s_nInstCacheSize = 0;
// Past-the-end pointer set after the backprop prep pass — points to the
// sentinel EEINST whose `regs[]` is LIVE-filled but has no USED bits. The
// Belady evictor uses this to bound its forward next-use scan; reads
// past this point would hit memory left over from a previous block.
static EEINST* s_pInstCacheLast = nullptr;

static BASEBLOCK* s_pCurBlock = nullptr;
static BASEBLOCKEX* s_pCurBlockEx = nullptr;
u32 s_nEndBlock = 0;
u32 s_branchTo;
static bool s_branchIsUnconditional; // true for J/JAL, false for BEQ/BNE etc.
bool s_nBlockFF; // wait-loop candidate — promoted to global so branch emitters
                 // can fall back to the non-split CSEL path (preserves WaitLoop
                 // fast-forward).

// Timeout-loop detection. Matches the `addiu rN, rN, -K; bne rN, $0, self; nop`
// idiom — the only legal block shape is exactly those three instructions, and
// rN must be the same on the addiu and the bne. When set, recRecompile emits
// an O(1) skip in place of the loop body that decrements rN by the cycles
// consumed and exits to the dispatcher. Mirrors x86 recSkipTimeoutLoop in
// ix86-32/iR5900.cpp:2118-2157.
static s32 s_eeTimeoutReg = -1;

// See arm64Emitter.h for the contract. SetBranchImm sets this; recRecompile
// reads it in the block-end link gate and resets it at block entry.
u32 g_eeStaticBranchPC = 0;

static int* s_pCode = nullptr;

GPR_reg64 s_saveConstRegs[32];
static u32 s_saveHasConstReg = 0, s_saveFlushedConstReg = 0;
static EEINST* s_psaveInstInfo = nullptr;

static u32 s_savenBlockCycles = 0;

// Constant pool for the recompiler code buffer
static ArmConstantPool s_recConstPool;

// ============================================================================
//  Dispatcher pointers (filled by _DynGen_Dispatchers)
// ============================================================================

static const void* DispatcherEvent = nullptr;
static const void* DispatcherReg = nullptr;
static const void* JITCompile = nullptr;
static const void* EnterRecompiledCode = nullptr;
static const void* DispatchBlockDiscard = nullptr;
static const void* DispatchPageReset = nullptr;
static const void* UnmappedRecLUTPage = nullptr;

// ============================================================================
//  Block linking — direct-B tail chaining across the EE code cache
// ============================================================================
//
// Each linkable block exit records a patch_site (the address of the
// unconditional B emitted inside iBranchTest's successor slot). When the
// static target PC later compiles (or is already compiled), we rewrite that
// B to jump directly to the target block's entry point, bypassing the
// DispatcherReg LUT lookup. The cycle-budget check (ADDS RCYCLE + B.PL
// DispatcherEvent) stays at every block exit, so event delivery remains
// prompt — linking only removes the memory round-trip through the LUT
// when the budget survives.
//
// Only two block-exit shapes are linkable today:
//   1. Unconditional J/JAL (static target = s_branchTo, g_branch==1 with
//      s_branchIsUnconditional=true).
//   2. Block fall-through at s_nEndBlock / page boundary (static target =
//      natural next PC, g_branch==0).
// Conditional branches (BEQ/BNE/etc.) use CSEL + runtime PC dispatch in
// iR5900Branch_arm64.cpp and are NOT linkable without restructuring those
// emitters into split taken/not-taken tails. JR/JALR and syscalls/traps
// are indirect — cpuRegs.pc is runtime-determined, so they stay on the
// DispatcherReg path.
struct BlockLinkExit
{
	u32  target_pc;      // statically-known successor PC
	u8*  patch_site;     // address of the unconditional B to rewrite
	u8*  fallthrough;    // unlinked target (DispatcherReg) — used by unpatch
	u8*  current_target; // where patch_site currently points
};

struct BlockLinks
{
	u8*           entry;         // block's compiled fnptr — linked callers jump here
	BlockLinkExit exits[2];      // up to 2 static exits: conditional branches emit
	                             // both taken and not-taken linkable tails.
	u32           num_exits;     // 0, 1, or 2 — how many `exits[]` slots are live.
};

// Sidecar registry: startpc (hwaddr) → link info. Blocks register here when
// they compile; recClear drops entries for invalidated blocks (and unpatches
// incoming edges). Lookup is by block-start PC, matching the way
// tryForwardLink / patchWaitingPredecessors resolve exit targets.
static std::unordered_map<u32, BlockLinks> s_blockLinks;

// Reverse index: target_hw → list of predecessor startpc(hw)s. A block is
// added to the list for each of its UNIQUE exit target HWs when it compiles.
// patchWaitingEEPredecessors uses this to find candidate predecessors in
// O(preds_at_target) instead of O(total_blocks) — without it, every new
// compile scanned every existing block's exits, costing 35% of CPU during
// first-cache lag spikes (FFXII / GTASA / Darkwatch).
//
// Stale entries (predecessor block invalidated by recClear) are skipped
// lazily during the patch walk via an s_blockLinks.find() check; the lists
// shrink only on full reset (recResetRaw) since per-block GC requires
// remembering each block's exit targets and EE blocks' BlockLinks already
// hold them — but lazy cleanup keeps the patch loop's hot path branch-light.
static std::unordered_map<u32, std::vector<u32>> s_eeWaitingForHw;

// ============================================================================
//  Forward declarations
// ============================================================================

static void recRecompile(const u32 startpc);
static void recResetRaw();
static void recError(u32 error);
static void iBranchTest(u32 newpc = 0xffffffff);
static void ClearRecLUT(BASEBLOCK* base, int count);
static u32 scaleblockcycles();
static void recExitExecution();
static void dyna_block_discard(u32 start, u32 sz);
static void dyna_page_reset(u32 start, u32 sz);
static void recClear(u32 addr, u32 size);
static void memory_protect_recompiled_code(u32 startpc, u32 size);

// ============================================================================
//  ARM64 codegen helpers
// ============================================================================

void armFlushConstRegs()
{
	// Only flush regs that are const AND not yet in memory — these are the
	// regs that mutated since the last flush. Common-case fast-path: dirty
	// mask is 0 and we exit immediately without iterating 1..31.
	u32 dirty = g_cpuHasConstReg & ~g_cpuFlushedConstReg & 0xFFFFFFFEu; // skip r0
	if (!dirty)
		return;

	// Iterate set bits via ctz instead of scanning 31 indices. With typical
	// dirty counts of 1-3, this is 1-3 iterations vs 31 mask-tests.
	while (dirty)
	{
		const int i = __builtin_ctz(dirty);
		dirty &= dirty - 1; // clear lowest set bit

		// Write constant value to cpuRegs.GPR[i].SD[0] (lower 64 bits only).
		// Upper 64 bits (UD[1]) are left untouched — matches interpreter behavior.
		const s64 val = g_cpuConstRegs[i].SD[0];
		if (val == 0)
		{
			armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(i)));
		}
		else
		{
			armAsm->Mov(RSCRATCHGPR, static_cast<u64>(val));
			armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, GPR_OFFSET(i)));
		}
	}
	// All previously-dirty consts are now in memory.
	g_cpuFlushedConstReg = g_cpuHasConstReg;
}

// Classify whether an EE opcode is a "safe" branch delay slot for the
// pre-DS flush skip optimization. Returns true for non-faulting,
// non-memory-accessing, non-interp-calling ops.
//
// The post-DS armFlushConstRegs always runs (it's the block-exit flush,
// needed for direct-B linking to a successor that reads GPR memory). The
// pre-DS flush is only needed if the DS could:
//   1. Take an exception and longjmp before its own flush would happen
//      (true for memory loads/stores via fastmem, since the fastmem fault
//      thunk does not flush const tracker).
//   2. Call an interp helper that reads GPR memory before the helper's
//      own armCallInterpreter flush runs (cannot happen — armCallInterpreter
//      always flushes first).
//
// So we only need to flush before DS if (1) applies — i.e., the DS is a
// memory access or other faulting op. For pure ALU/shift/MFHI/etc., the
// pre-DS flush is redundant and the post-DS flush is sufficient.
bool armDelaySlotIsSafe(u32 opcode)
{
	// Canonical NOP — never emits anything.
	if (opcode == 0u)
		return true;

	const u32 op = (opcode >> 26) & 0x3Fu;

	switch (op)
	{
		case 0x00: // SPECIAL — funct in low 6 bits
		{
			const u32 funct = opcode & 0x3Fu;
			switch (funct)
			{
				// Shifts (no memory, no exceptions)
				case 0x00: case 0x02: case 0x03:           // SLL, SRL, SRA
				case 0x04: case 0x06: case 0x07:           // SLLV, SRLV, SRAV
				case 0x14: case 0x16: case 0x17:           // DSLLV, DSRLV, DSRAV
				case 0x38: case 0x3A: case 0x3B:           // DSLL, DSRL, DSRA
				case 0x3C: case 0x3E: case 0x3F:           // DSLL32, DSRL32, DSRA32
				// HI/LO move (no memory)
				case 0x10: case 0x11: case 0x12: case 0x13: // MFHI, MTHI, MFLO, MTLO
				// Mult/Div (no memory; EE doesn't trap on /0 in JIT)
				case 0x18: case 0x19: case 0x1A: case 0x1B: // MULT, MULTU, DIV, DIVU
				// MFSA/MTSA (no memory)
				case 0x28: case 0x29:                       // MFSA, MTSA
				// Non-trapping register ALU (ADDU/SUBU/AND/OR/XOR/NOR/SLT/SLTU/DADDU/DSUBU)
				case 0x21: case 0x23: case 0x24: case 0x25: // ADDU, SUBU, AND, OR
				case 0x26: case 0x27: case 0x2A: case 0x2B: // XOR, NOR, SLT, SLTU
				case 0x2D: case 0x2F:                       // DADDU, DSUBU
				// MOVZ/MOVN
				case 0x0A: case 0x0B:                       // MOVZ, MOVN
				// SYNC
				case 0x0F:                                  // SYNC
					return true;
				// Trapping/exception ops — NOT safe:
				//   0x08 JR, 0x09 JALR (branches in DS are illegal anyway)
				//   0x0C SYSCALL, 0x0D BREAK
				//   0x20 ADD, 0x22 SUB, 0x2C DADD, 0x2E DSUB (overflow trap)
				//   0x30-0x36 TGE/TGEU/TLT/TLTU/TEQ/TNE (trap)
				default:
					return false;
			}
		}

		case 0x01: // REGIMM — branches; illegal in DS regardless
			return false;

		// Top-level immediate ALU — no memory, no exceptions
		case 0x09: case 0x0A: case 0x0B:                   // ADDIU, SLTI, SLTIU
		case 0x0C: case 0x0D: case 0x0E: case 0x0F:        // ANDI, ORI, XORI, LUI
		case 0x19:                                          // DADDIU
			return true;

		// 0x08 ADDI, 0x18 DADDI — overflow trap. NOT safe.
		// 0x02-0x07 J/JAL/BEQ/BNE/BLEZ/BGTZ — branches.
		// 0x10 COP0 — TLB ops can fault, mode changes, ERET. NOT safe.
		// 0x11 COP1, 0x12 COP2 — arith mostly safe but interp paths can be
		//   tricky. Conservative: not safe (FPU exceptions, VU0 macro side
		//   effects).
		// 0x14-0x17 BEQL/BNEL/BLEZL/BGTZL — branches.
		// 0x1A-0x1B LDL/LDR — memory.
		// 0x1C MMI — mostly arith but some can be ISTUB; conservative skip.
		// 0x1E LQ, 0x1F SQ — memory.
		// 0x20-0x2F loads/stores.
		// 0x30+ LL/SC/coprocessor loads/stores — memory.
		default:
			return false;
	}
}

void armFlushConstRegsBeforeDS()
{
	// Peek at the DS opcode at the current pc. The branch emitter has
	// already advanced pc past the branch instruction itself (via the
	// pre-increment in recompileNextInstruction's caller), so pc points
	// at the DS. Skip the flush if the DS is provably safe.
	const u32 ds_code = *reinterpret_cast<const u32*>(PSM(pc));
	if (!armDelaySlotIsSafe(ds_code))
		armFlushConstRegs();
}

// Flush a single constant register to memory if it has a const value that
// hasn't been written back yet. No-op otherwise. Used by call sites that
// bypass armLoadGPR* and read GPR memory directly (128-bit MMI/QMTC2 loads,
// PLZCW upper-half access) so the const value is coherent before the read.
void armFlushConstReg(int reg)
{
	if (reg <= 0 || reg >= 32)
		return;
	const u32 bit = 1u << reg;
	if (!(g_cpuHasConstReg & bit) || (g_cpuFlushedConstReg & bit))
		return;

	const s64 val = g_cpuConstRegs[reg].SD[0];
	if (val == 0)
	{
		armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(reg)));
	}
	else
	{
		armAsm->Mov(RSCRATCHGPR, static_cast<u64>(val));
		armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, GPR_OFFSET(reg)));
	}
	g_cpuFlushedConstReg |= bit;
}

// Drop a register's const tracking AFTER first committing its value to memory.
// The early `GPR_DEL_CONST(_Rd_)` pattern at the top of an op is unsafe when
// _Rd_ aliases _Rs_/_Rt_: clearing the const flag causes the subsequent
// armLoadGPR* on the operand to fall through to an LDR that reads stale
// memory (the const value never reached cpuRegs.GPR). Flushing first ensures
// memory is coherent regardless of whether _Rd_ aliases an operand. The
// follow-up armStoreGPR* still overwrites _Rd_; the wasted store is
// harmless. Use this anywhere you would have written GPR_DEL_CONST(_Rd_)
// before loading source operands.
void armDelConstReg(int reg)
{
	armFlushConstReg(reg);
	GPR_DEL_CONST(reg);
}

// ============================================================================
//  Tier 1 GPR cache — Phase C plumbing (NOT wired to ops yet)
// ============================================================================
// See arm64Emitter.h for the API contract and rationale. This is pure
// infrastructure: no instruction emitter currently calls armGprAlloc/Flush/
// Invalidate. Block entry (recRecompile main loop) calls armGprCacheReset
// so the slot table starts in a defined state once Phase D begins wiring
// individual ops to the cache.

ArmGprCacheSlot g_armGprCache[32];
u16 g_armGprCachePoolUsed;

// Monotonic per-block clock for LRU stamps. Reset at block entry along with
// the slot table; bumped on every armGprAlloc hit/miss so eviction can pick
// the least-recently-used slot.
static u32 g_armGprUseClock;

// MIPS GPRs we want to keep resident under pressure: r1 (at), r2/r3 (v0/v1),
// r4-r7 (a0-a3), r24/r25 (t8/t9). These are the busiest regs in
// compiler-generated code; without a bias the linear-low-first eviction
// would spill them on every pool exhaustion. Score-based evictor below adds
// kArmGprHighPrioBonus to their LRU stamp so a low-priority slot is always
// preferred as victim, falling back to LRU within each class.
static constexpr u32 kArmGprHighPrioMask = (0xFEu) | (1u << 24) | (1u << 25);
static constexpr u32 kArmGprHighPrioBonus = 1u << 24;

// Returns true if the cached value of MIPS GPR `g` is dead at g_pCurInstInfo
// (the LIVE bit on the state coming INTO the current emit point is clear,
// meaning the next event for this reg is either nothing or a write that
// discards the cached value). Eviction can drop these slots without
// flushing dirty contents — a free win.
//
// Conservative when no analysis is available: returns false so the LRU
// fallback path drives the choice. False if `g` is out of range.
static bool armGprIsDeadAt(int g)
{
	if (g <= 0 || g >= 32)
		return false;
	if (!g_pCurInstInfo || !s_pInstCacheLast)
		return false;
	return (g_pCurInstInfo->regs[g] & EEINST_LIVE) == 0;
}

// Pool of caller-saved host regs available for cache slots. Order is the
// allocation order (front-first). Indices into this array are what get
// stored in g_armGprCachePoolUsed.
//   x4..x6  : RSCRATCHGPR{,2,3}            — used by per-instruction codegen
//   x9..x10 : armEmit{Set,Flush,Reload}Cycle scratch
//   x16..x17: vixl IP0/IP1
//   x19..x28: pinned state regs (RCPUSTATE, RCYCLE, RMEMBASE, RRECLUT,
//             RFASTMEMBASE, RDELAYSLOTGPR + reserved Phase D-H slots)
// Everything else in 0..30 is either an arg (x0..x3), the FP/LR (x29/x30),
// or already excluded above. That leaves x7, x8, x11..x15.
static constexpr u8 kArmGprCachePool[] = {7, 8, 11, 12, 13, 14, 15};
static constexpr int kArmGprCachePoolSize = static_cast<int>(sizeof(kArmGprCachePool) / sizeof(kArmGprCachePool[0]));
static_assert(kArmGprCachePoolSize <= 16, "g_armGprCachePoolUsed bitmask is u16");

// Linear scan: pool host_code → index in kArmGprCachePool. -1 if not in pool.
static int armGprPoolIndex(u8 host_code)
{
	for (int i = 0; i < kArmGprCachePoolSize; i++)
	{
		if (kArmGprCachePool[i] == host_code)
			return i;
	}
	return -1;
}

// Acquire a free pool slot, or evict a victim. Eviction policy:
//
//   1. Dead-reg first: if any cached slot's MIPS reg is provably dead at
//      the current emit point (backprop says LIVE-clear coming in — the
//      next event is either a write that discards the value or block
//      end), prefer that slot. Picking the dead reg avoids penalising a
//      live one whose value would have to be reloaded later. Pick the
//      LRU dead slot for determinism.
//
//      The dirty flush is NOT skipped even when LIVE-clear: iRecAnalysis.h
//      keeps EE_WRITE_DEAD_VALUES=1 with a "tends to break stuff at the
//      moment" note, so we mirror that caution. The win here is the
//      eviction *choice*, not avoiding the spill.
//
//   2. Operand protection: skip slots whose last_use stamp falls within
//      kProtectRecent ticks of g_armGprUseClock. Those were allocated as
//      operands of the in-flight op, and reusing their host code for a
//      different MIPS reg would alias the XRegister returned earlier in
//      the same op.
//
//   3. LRU + priority among the remaining slots. Score = last_use +
//      kArmGprHighPrioBonus for hot ABI regs (kArmGprHighPrioMask). Lowest
//      score loses.
//
//   4. If every slot is protected (would only happen if the in-flight op
//      asks for more operands than the pool can hold), fall back to LRU
//      ignoring the protection band — a correctness fault either way, but
//      LRU at least picks the oldest slot.
static u8 armGprAcquirePoolSlot()
{
	for (int i = 0; i < kArmGprCachePoolSize; i++)
	{
		const u16 bit = static_cast<u16>(1u << i);
		if (!(g_armGprCachePoolUsed & bit))
		{
			g_armGprCachePoolUsed |= bit;
			return kArmGprCachePool[i];
		}
	}

	// 1. Dead-reg first. Pick the LRU dead slot.
	int victim = -1;
	{
		u32 oldest = UINT32_MAX;
		for (int g = 1; g < 32; g++)
		{
			const ArmGprCacheSlot& slot = g_armGprCache[g];
			if (slot.host_code == 0xff) continue;
			if (armGprPoolIndex(slot.host_code) < 0) continue;
			if (!armGprIsDeadAt(g)) continue;
			if (slot.last_use < oldest)
			{
				oldest = slot.last_use;
				victim = g;
			}
		}
	}

	// 2 + 3. LRU + priority among live slots not currently held as
	// operands of the in-flight op. kProtectRecent of 4 covers the
	// typical 1-3 reg reads + 1 reg write of an EE op with headroom.
	if (victim < 0)
	{
		constexpr u32 kProtectRecent = 4;
		const u32 protect_floor = (g_armGprUseClock > kProtectRecent)
			? (g_armGprUseClock - kProtectRecent) : 0;

		u64 best_score = ~0ull;
		for (int g = 1; g < 32; g++)
		{
			const ArmGprCacheSlot& slot = g_armGprCache[g];
			if (slot.host_code == 0xff) continue;
			if (armGprPoolIndex(slot.host_code) < 0) continue;
			if (slot.last_use >= protect_floor) continue;
			const u64 bonus = (kArmGprHighPrioMask & (1u << g)) ? kArmGprHighPrioBonus : 0u;
			const u64 score = static_cast<u64>(slot.last_use) + bonus;
			if (score < best_score)
			{
				best_score = score;
				victim = g;
			}
		}
	}

	// 4. Every slot is protected — fall back to plain LRU.
	if (victim < 0)
	{
		u32 oldest = UINT32_MAX;
		for (int g = 1; g < 32; g++)
		{
			const ArmGprCacheSlot& slot = g_armGprCache[g];
			if (slot.host_code == 0xff) continue;
			if (armGprPoolIndex(slot.host_code) < 0) continue;
			if (slot.last_use < oldest)
			{
				oldest = slot.last_use;
				victim = g;
			}
		}
	}

	if (victim < 0)
	{
		pxFailRel("armGprAcquirePoolSlot: no slots free and nothing to evict");
		return 0xff;
	}

	ArmGprCacheSlot& vslot = g_armGprCache[victim];
	if (vslot.dirty)
	{
		armAsm->Str(a64::XRegister(vslot.host_code),
			a64::MemOperand(RCPUSTATE, GPR_OFFSET(victim)));
	}
	const u8 host = vslot.host_code;
	vslot.host_code = 0xff;
	vslot.dirty = false;
	vslot.sxw = false;
	vslot.last_use = 0;
	return host;
}

void armGprCacheReset()
{
	for (int i = 0; i < 32; i++)
	{
		g_armGprCache[i].host_code = 0xff;
		g_armGprCache[i].dirty = false;
		g_armGprCache[i].sxw = false;
		g_armGprCache[i].last_use = 0;
	}
	g_armGprCachePoolUsed = 0;
	g_armGprUseClock = 0;
}

bool armGprIsCached(int gpr)
{
	return gpr >= 0 && gpr < 32 && g_armGprCache[gpr].host_code != 0xff;
}

a64::XRegister armGprAlloc(int gpr, bool for_write)
{
	pxAssertMsg(gpr > 0 && gpr < 32, "armGprAlloc: GPR0 has no cache slot (use xzr)");

	ArmGprCacheSlot& slot = g_armGprCache[gpr];

	if (slot.host_code != 0xff)
	{
		slot.last_use = ++g_armGprUseClock;
		// Coherence with the const tracker.  If GPR_SET_CONST(gpr) was called
		// after this slot was populated (e.g. LUI/ALU-const-path setting a new
		// const while an earlier op still has the slot cached with an old
		// runtime value), the cache is stale — the const is authoritative.
		// For reads: re-materialise the const into the slot, drop dirty.
		// For writes: caller is about to overwrite anyway, but we still need
		// to clear GPR_IS_CONST1 so later readers don't keep preferring the
		// now-stale const.  Covers both sides of the const/cache sync gap
		// without requiring every emitter to explicitly invalidate.
		if (GPR_IS_CONST1(gpr))
		{
			if (for_write)
			{
				// New runtime value coming — const tracker is about to be stale.
				GPR_DEL_CONST(gpr);
			}
			else
			{
				// Read: const value wins over stale cache.
				armAsm->Mov(a64::XRegister(slot.host_code),
					static_cast<u64>(g_cpuConstRegs[gpr].SD[0]));
				slot.dirty = false;
				slot.sxw = false;
			}
		}
		if (for_write)
		{
			slot.dirty = true;
			slot.sxw = false;
		}
		return a64::XRegister(slot.host_code);
	}

	// Not cached — pick a pool slot and (for reads) populate it.
	const u8 host = armGprAcquirePoolSlot();
	a64::XRegister reg(host);

	if (!for_write)
	{
		if (GPR_IS_CONST1(gpr))
		{
			// Const-tracked: emit MOV-imm. The const tracker is left alone;
			// other read sites in this block can still see the const.
			armAsm->Mov(reg, static_cast<u64>(g_cpuConstRegs[gpr].SD[0]));
		}
		else
		{
			armAsm->Ldr(reg, a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
		}
	}
	else if (GPR_IS_CONST1(gpr))
	{
		// Writing a new runtime value to a previously-const GPR.  Clear
		// const tracking so armLoadGPR* readers don't keep returning the
		// now-stale compile-time value.  Mirrors the explicit armDelConstReg
		// that well-behaved emitters call before armGprAlloc(for_write) —
		// this is the belt-and-suspenders safety net when they don't.
		GPR_DEL_CONST(gpr);
	}
	// for_write: caller will store next; no load needed.

	slot.host_code = host;
	slot.dirty = for_write;
	slot.sxw = false;
	slot.last_use = ++g_armGprUseClock;
	return reg;
}

a64::XRegister armGprAllocTmp()
{
	const u8 host = armGprAcquirePoolSlot();
	return a64::XRegister(host);
}

void armGprReleaseTmp(const a64::Register& reg)
{
	const int idx = armGprPoolIndex(static_cast<u8>(reg.GetCode()));
	pxAssertMsg(idx >= 0, "armGprReleaseTmp: register is not in the cache pool");
	g_armGprCachePoolUsed &= ~static_cast<u16>(1u << idx);
}

void armGprFlush(int gpr)
{
	if (gpr <= 0 || gpr >= 32)
		return;
	ArmGprCacheSlot& slot = g_armGprCache[gpr];
	if (slot.host_code == 0xff || !slot.dirty)
		return;
	armAsm->Str(a64::XRegister(slot.host_code),
		a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
	slot.dirty = false;
}

void armGprInvalidate(int gpr)
{
	if (gpr <= 0 || gpr >= 32)
		return;
	ArmGprCacheSlot& slot = g_armGprCache[gpr];
	if (slot.host_code == 0xff)
		return;
	if (slot.dirty)
	{
		armAsm->Str(a64::XRegister(slot.host_code),
			a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
	}
	const int idx = armGprPoolIndex(slot.host_code);
	if (idx >= 0)
		g_armGprCachePoolUsed &= ~static_cast<u16>(1u << idx);
	slot.host_code = 0xff;
	slot.dirty = false;
	slot.sxw = false;
}

void armGprFlushAll()
{
	for (int g = 1; g < 32; g++)
	{
		ArmGprCacheSlot& slot = g_armGprCache[g];
		if (slot.host_code != 0xff && slot.dirty)
			armGprFlush(g);
	}
}

void armGprInvalidateAll()
{
	for (int g = 1; g < 32; g++)
	{
		if (g_armGprCache[g].host_code != 0xff)
			armGprInvalidate(g);
	}
	// Pool should be empty now; reset explicitly in case any tmps leaked.
	// (Phase C: tmp leaks are a bug — Phase D ops MUST release every tmp.)
	g_armGprCachePoolUsed = 0;
}

// ============================================================================
//  Cycle delta helpers (Phase B)
// ============================================================================
// RCYCLE (x20) holds (s64)(cpuRegs.cycle - cpuRegs.nextEventCycle).
// Negative = budget remaining; >= 0 = event due.
// Per-block accounting becomes a single ADDS + B.MI continue, replacing the
// 7-instruction LDR/ADD/STR/LDR/CMP/B.LO/B sequence used previously.
//
// The in-memory cpuRegs.cycle is stale during JIT execution (the in-flight
// delta lives in x20). It is reconciled at every "exit to C++" point:
//   - DispatcherEvent (before recEventTest)
//   - armBranchCallInterpreter (before the BL into the interp branch helper)
// armCallInterpreter does NOT writeback — per-instruction interp helpers do
// not read cpuRegs.cycle, matching the prior s_nBlockCycles model.

// Emit: cpuRegs.cycle = cpuRegs.nextEventCycle + RCYCLE
//       (i.e. flush the in-flight delta back to memory)
static void armWritebackCycle()
{
	armAsm->Ldr(a64::x1, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Add(a64::x0, a64::x1, RCYCLE);
	armAsm->Str(a64::x0, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
}

// Emit: RCYCLE = cpuRegs.cycle - cpuRegs.nextEventCycle
//       (i.e. reload the delta from memory after C++ may have changed either)
static void armReloadCycle()
{
	armAsm->Ldr(a64::x0, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
	armAsm->Ldr(a64::x1, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Sub(RCYCLE, a64::x0, a64::x1);
}

// Emit: dst = cpuRegs.nextEventCycle + RCYCLE
// JIT-current cycle for ops that need to read "cycle now" (e.g. COP0 Count).
// Reading CYCLE_OFFSET directly would be stale by potentially many blocks
// since RCYCLE only writes back at iBranchTest / DispatcherEvent.
void armEmitLoadCurrentCycle(const a64::Register& dst)
{
	armAsm->Ldr(dst, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Add(dst, dst, RCYCLE);
}

// Inline cpuSetNextEventDelta(delta) — see header for math derivation.
//   if (RCYCLE + delta) < 0:
//     nextEventCycle += (RCYCLE + delta)
//     RCYCLE = -delta
// Clobbers x9, x10.
void armEmitSetNextEventDelta(s32 delta)
{
	a64::Label skip;
	// x9 = RCYCLE + delta, sets flags. ADDS supports a 12-bit unsigned imm
	// (optionally LSL #12); the macro assembler spills to a temp if needed.
	armAsm->Adds(a64::x9, RCYCLE, delta);
	// If result >= 0 (signed), the existing schedule is already at or before
	// (cycle + delta), so leave it.
	armAsm->B(&skip, a64::ge);
	// nextEventCycle += x9 (pulls the event forward by the missing budget)
	armAsm->Ldr(a64::x10, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Add(a64::x10, a64::x10, a64::x9);
	armAsm->Str(a64::x10, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	// RCYCLE = -delta
	armAsm->Mov(RCYCLE, -static_cast<s64>(delta));
	armAsm->Bind(&skip);
}

// Flush RCYCLE → cpuRegs.cycle BEFORE a C++ call that reads cycle or
// reschedules events. Required because Phase B keeps the JIT-current cycle
// only in RCYCLE; cpuRegs.cycle in memory is stale by the block's accumulated
// cycles. Without this flush, callees like CPU_INT compute new nec from a
// stale "now" and schedule events too late.
//   cpuRegs.cycle = cpuRegs.nextEventCycle + RCYCLE
// Uses x9 as scratch (caller-saved, free pre-call).
void armEmitFlushCycleBeforeCall()
{
	armAsm->Ldr(a64::x9, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Add(a64::x9, a64::x9, RCYCLE);
	armAsm->Str(a64::x9, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
}

// Reload RCYCLE from memory AFTER a C++ call that may have mutated nec
// (and that was preceded by armEmitFlushCycleBeforeCall, so cpuRegs.cycle
// is current). Reconstructs the Phase B invariant:
//   RCYCLE = cpuRegs.cycle - cpuRegs.nextEventCycle
// Uses x9/x10 as scratch — does NOT clobber x0/w0 (vtlb_memRead's return).
void armEmitReloadCycleAfterCall()
{
	armAsm->Ldr(a64::x9, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
	armAsm->Ldr(a64::x10, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Sub(RCYCLE, a64::x9, a64::x10);
}

void armLoadGPR64(const a64::Register& dst, int gpr)
{
	if (gpr == 0)
		armAsm->Mov(dst, a64::xzr);
	else if (GPR_IS_CONST1(gpr))
		armAsm->Mov(dst, static_cast<u64>(g_cpuConstRegs[gpr].SD[0]));
	else
		armAsm->Ldr(dst, a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
}

void armLoadGPR32(const a64::Register& dst, int gpr)
{
	if (gpr == 0)
		armAsm->Mov(dst.IsX() ? dst : a64::Register(dst.GetCode(), 64), a64::xzr);
	else if (GPR_IS_CONST1(gpr))
		armAsm->Mov(dst, g_cpuConstRegs[gpr].UL[0]);
	else
		armAsm->Ldr(a64::WRegister(dst.GetCode()), a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
}

void armStoreGPR64SignExt32(const a64::Register& src_w, int gpr)
{
	if (gpr == 0)
		return;

	// Sign-extend 32-bit value to 64-bit, store to SD[0] only.
	// Upper 64 bits (UD[1]) left untouched — matches interpreter behavior.
	a64::XRegister src_x(src_w.GetCode());
	armAsm->Sxtw(src_x, a64::WRegister(src_w.GetCode()));
	armAsm->Str(src_x, a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
	GPR_DEL_CONST(gpr);
}

void armStoreGPR64(const a64::Register& src_x, int gpr)
{
	if (gpr == 0)
		return;
	// Store to SD[0] only. Upper 64 bits (UD[1]) left untouched.
	armAsm->Str(src_x, a64::MemOperand(RCPUSTATE, GPR_OFFSET(gpr)));
	GPR_DEL_CONST(gpr);
}

void armFlushPC()
{
	if (!g_cpuFlushedPC)
	{
		armAsm->Mov(RWSCRATCH, pc);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
		g_cpuFlushedPC = true;
	}
}

void armFlushCode()
{
	if (!g_cpuFlushedCode)
	{
		armAsm->Mov(RWSCRATCH, cpuRegs.code);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, CODE_OFFSET));
		g_cpuFlushedCode = true;
	}
}

void armCallInterpreter(void (*func)())
{
    armFlushPC();
    armFlushCode();
    armFlushConstRegs();

    // Phase B: the JIT-current cycle lives in RCYCLE (x20); cpuRegs.cycle in
    // memory is stale. Many interpreter stubs (COP2/VU0 CFC2/CTC2/QMFC2, MMI,
    // etc.) transitively call CPU_INT / cpuSetNextEventDelta, which read
    // cpuRegs.cycle to schedule events. Without flushing first, events are
    // scheduled relative to an old "now" and nextEventCycle goes wrong;
    // without reloading after, the RCYCLE invariant breaks and the next
    // iBranchTest miscomputes budget — either way, a hang.
    armEmitFlushCycleBeforeCall();
    armEmitCall((const void*)func);
    armEmitReloadCycleAfterCall();

    // RMEMBASE (x21) is callee-saved — no reload needed.
    g_cpuHasConstReg = 1;
    g_cpuFlushedConstReg = 1;
}

// Branch-call variant: used for interpreter branch/syscall/trap stubs.
// Matches the upstream x86 recBranchCall:
//   1) Writes the in-flight cycle delta back and forces an event check
//      (cpuRegs.cycle = nextEventCycle + RCYCLE; nextEventCycle = cycle)
//   2) Flushes state and calls the standard interpreter function
//      (which calls doBranch → intUpdateCPUCycles → intEventTest,
//       properly counting delay slot cycles and processing events)
//   3) Reloads RCYCLE from memory (interp may have advanced cycle and/or
//      scheduled a new nextEventCycle)
//   4) Sets g_branch = 2
void armBranchCallInterpreter(void (*func)())
{
	// cpuRegs.cycle = nextEventCycle + RCYCLE  (writeback the in-flight delta)
	armAsm->Ldr(a64::x1, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));
	armAsm->Add(a64::x0, a64::x1, RCYCLE);
	armAsm->Str(a64::x0, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
	// nextEventCycle = cycle  (force the post-call iBranchTest into DispatcherEvent)
	armAsm->Str(a64::x0, a64::MemOperand(RCPUSTATE, NEXT_EVENT_CYCLE_OFFSET));

	// Flush PC, code, and const regs — but NOT cycle (already flushed above).
	// We cannot delegate to armCallInterpreter here because it calls
	// armEmitFlushCycleBeforeCall, which would re-read the nec we just set
	// to cycle and add RCYCLE again, double-counting the accumulated delta.
	armFlushPC();
	armFlushCode();
	armFlushConstRegs();

	armEmitCall((const void*)func);
	armEmitReloadCycleAfterCall();

	g_cpuHasConstReg = 1;
	g_cpuFlushedConstReg = 1;
	g_branch = 2;
}

// ============================================================================
//  Branch-interp fallback with DS cycle accounting
// ============================================================================
//
// For branch ops whose interpreter stub internally runs the delay-slot
// instruction via _doBranch_shared (BC0F/T/FL/TL, BC1F/T/FL/TL, BC2F/T/FL/TL):
// the rec never recompiles the DS, so s_nBlockCycles misses its cost. The
// interp DOES consume the cycles into cpuBlockCycles, but that's only flushed
// to cpuRegs.cycle when Cpu == &intCpu — from the rec's call, it stays
// pending. Manually account for the DS here using the same formula
// recompileNextInstruction uses at the top of the function.
//
// pc at entry is the DS address (branch_addr + 4), because
// recompileNextInstruction pre-incremented pc before dispatching the branch op.
// pc is advanced past the DS on return so s_pCurBlockEx->size covers both the
// branch and its delay slot — needed for SMC invalidation bounds to match the
// native branch path (which advances pc via recompileNextInstruction(true)).
void armBranchInterpWithDSCycles(void (*func)())
{
	const u32 ds_code = *(const u32*)PSM(pc);
	const R5900::OPCODE& ds = R5900::GetInstruction(ds_code);
	const u32 ds_cycles = (ds_code == 0) ? 9 : ds.cycles;
	s_nBlockCycles += ds_cycles * (2 - ((cpuRegs.CP0.n.Config >> 18) & 0x1));

	armBranchCallInterpreter(func);

	pc += 4;
}

// ============================================================================
//  Cycle scaling (same algorithm as x86)
// ============================================================================

#define DEFAULT_SCALED_BLOCKS() (s_nBlockCycles >> 3)

static u32 scaleblockcycles_calculation()
{
	const bool lowcycles = (s_nBlockCycles <= 40);
	const s8 cyclerate = EmuConfig.Speedhacks.EECycleRate;
	u32 scale_cycles = 0;

	if (cyclerate == 0 || lowcycles || cyclerate < -99 || cyclerate > 3)
		scale_cycles = DEFAULT_SCALED_BLOCKS();
	else if (cyclerate > 1)
		scale_cycles = s_nBlockCycles >> (2 + cyclerate);
	else if (cyclerate == 1)
		scale_cycles = DEFAULT_SCALED_BLOCKS() / 1.3f;
	else if (cyclerate == -1)
		scale_cycles = (s_nBlockCycles <= 80 || s_nBlockCycles > 168 ? 5 : 7) * s_nBlockCycles / 32;
	else
		scale_cycles = ((5 + (-2 * (cyclerate + 1))) * s_nBlockCycles) >> 5;

	return (scale_cycles < 1) ? 1 : scale_cycles;
}

static u32 scaleblockcycles()
{
	return scaleblockcycles_calculation();
}

u32 scaleblockcycles_clear()
{
	u32 scaled = scaleblockcycles_calculation();
	const s8 cyclerate = EmuConfig.Speedhacks.EECycleRate;
	const bool lowcycles = (s_nBlockCycles <= 40);

	if (!lowcycles && cyclerate > 1)
		s_nBlockCycles &= (0x1 << (cyclerate + 2)) - 1;
	else
		s_nBlockCycles &= 0x7;

	return scaled;
}

// ============================================================================
//  ARM64 Dispatchers
// ============================================================================

static void recEventTest()
{
	_cpuEventTest_Shared();

	if (eeRecExitRequested)
	{
		eeRecExitRequested = false;
		recExitExecution();
	}
}

// Dispatcher: jump to block at cpuRegs.pc
static const void* _DynGen_DispatcherReg()
{
	u8* retval = armStartBlock();

	// w0 = cpuRegs.pc
	armAsm->Ldr(a64::w0, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	// x1 = recLUT[pc >> 16] (has negative offset baked in by recLUT_SetPage)
	armAsm->Lsr(a64::w1, a64::w0, 16);
	armAsm->Ldr(a64::x1, a64::MemOperand(RRECLUT, a64::x1, a64::LSL, 3));
	// PC_GETBLOCK_ = recLUT[pc>>16] + pc * (sizeof(BASEBLOCK) / 4)
	// sizeof(BASEBLOCK) = 8, so scale = 2. Must use full pc, not masked.
	armAsm->Lsl(a64::x2, a64::x0, 1); // x2 = (u64)pc * 2 (w0 was zero-extended by ldr)
	armAsm->Ldr(a64::x3, a64::MemOperand(a64::x1, a64::x2));
	armAsm->Br(a64::x3);

	armEndBlock();
	return retval;
}

// Event dispatcher: writeback the cycle delta, call recEventTest, reload, then
// jump to DispatcherReg.
// On x86, DispatcherEvent falls through into DispatcherReg (contiguous code).
// On ARM64 each armStartBlock/armEndBlock pair introduces alignment padding,
// so we must use an explicit jump instead of relying on fallthrough.
static const void* _DynGen_DispatcherEvent()
{
	pxAssert(DispatcherReg); // must be generated first
	u8* retval = armStartBlock();

	// recEventTest reads cpuRegs.cycle and may schedule a new nextEventCycle.
	// Flush the in-flight delta before the call, reload after.
	armWritebackCycle();
	armEmitCall((const void*)recEventTest);
	armReloadCycle();
	armEmitJmp(DispatcherReg);

	armEndBlock();
	return retval;
}

// JIT compile: called when we hit an uncompiled block
static const void* _DynGen_JITCompile()
{
	u8* retval = armStartBlock();

	// arg1 = cpuRegs.pc
	armAsm->Ldr(RWARG1, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	armEmitCall((const void*)recRecompile);

	// Now dispatch to the newly compiled block
	armAsm->Ldr(a64::w0, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	armAsm->Lsr(a64::w1, a64::w0, 16);
	armAsm->Ldr(a64::x1, a64::MemOperand(RRECLUT, a64::x1, a64::LSL, 3));
	armAsm->Lsl(a64::x2, a64::x0, 1); // x2 = (u64)pc * 2
	armAsm->Ldr(a64::x3, a64::MemOperand(a64::x1, a64::x2));
	armAsm->Br(a64::x3);

	armEndBlock();
	return retval;
}

// Enter recompiled code: called from C++ to start execution
static const void* _DynGen_EnterRecompiledCode()
{
	u8* retval = armStartBlock();

	// Save callee-saved registers and set up stack frame
	armBeginStackFrame(false);

	// Load pinned state registers
	// (fpuRegs is reached via RCPUSTATE + FPUREGS_BASE — no dedicated reg.)
	armMoveAddressToReg(RCPUSTATE, &cpuRegs);
	armMoveAddressToReg(RRECLUT, recLUT);

	// Load RAM base if available
	if (eeMem)
		armMoveAddressToReg(RMEMBASE, eeMem->Main);

	// Load fastmem base for VTLB direct access
	if (vtlb_private::vtlbdata.fastmem_base)
		armMoveAddressToReg(RFASTMEMBASE, (void*)vtlb_private::vtlbdata.fastmem_base);

	// Initialize RCYCLE = cpuRegs.cycle - cpuRegs.nextEventCycle
	armReloadCycle();

	// Jump to the dispatcher
	armEmitJmp(DispatcherReg);

	armEndBlock();
	return retval;
}

static const void* _DynGen_DispatchBlockDiscard()
{
	u8* retval = armStartBlock();
	armEmitCall((const void*)dyna_block_discard);
	armEmitJmp(DispatcherReg);
	armEndBlock();
	return retval;
}

static const void* _DynGen_DispatchPageReset()
{
	u8* retval = armStartBlock();
	armEmitCall((const void*)dyna_page_reset);
	armEmitJmp(DispatcherReg);
	armEndBlock();
	return retval;
}

static const void* _DynGen_UnmappedRecLUTPage()
{
	u8* retval = armStartBlock();
	// Pass the actual cpuRegs.pc so recError can report it
	armAsm->Ldr(RWARG1, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	armEmitCall((const void*)recError);
	armEndBlock();
	return retval;
}

static void _DynGen_Dispatchers()
{
	// DispatcherReg must be generated before DispatcherEvent (Event jumps to Reg).
	// On x86 these are contiguous (fallthrough), but on ARM64 each block has
	// alignment padding, so DispatcherEvent uses an explicit jump instead.
	//
	// Register each dispatcher with simpleperf so it shows up by name in
	// profiler reports. armAsmPtr is the bump pointer that advances after
	// each armEndBlock(); we sample it before/after each generator to get
	// the size of the just-emitted block.
	const auto reg_dispatcher = [](const void* start, const char* name) {
		Perf::ee.Register(start,
			static_cast<size_t>(armAsmPtr - reinterpret_cast<u8*>(const_cast<void*>(start))),
			name);
	};

	DispatcherReg = _DynGen_DispatcherReg();
	reg_dispatcher(DispatcherReg, "EE_DispatcherReg");

	DispatcherEvent = _DynGen_DispatcherEvent();
	reg_dispatcher(DispatcherEvent, "EE_DispatcherEvent");

	JITCompile = _DynGen_JITCompile();
	reg_dispatcher(JITCompile, "EE_JITCompile");

	EnterRecompiledCode = _DynGen_EnterRecompiledCode();
	reg_dispatcher(EnterRecompiledCode, "EE_EnterRecompiledCode");

	DispatchBlockDiscard = _DynGen_DispatchBlockDiscard();
	reg_dispatcher(DispatchBlockDiscard, "EE_DispatchBlockDiscard");

	DispatchPageReset = _DynGen_DispatchPageReset();
	reg_dispatcher(DispatchPageReset, "EE_DispatchPageReset");

	UnmappedRecLUTPage = _DynGen_UnmappedRecLUTPage();
	reg_dispatcher(UnmappedRecLUTPage, "EE_UnmappedRecLUTPage");

	recBlocks.SetJITCompile(JITCompile);
}

// ============================================================================
//  Block linking — patch helpers
// ============================================================================

// Rewrite a patch_site to jump directly to `target`. No-op when the site
// already points at target. The underlying armEmitJmpPtr overwrites a
// single 4-byte `B` instruction (26-bit signed offset, ±128 MB reach —
// comfortably covers the 64 MB EE code cache).
static void patchEELinkSite(BlockLinkExit& exit, u8* target)
{
	if (!exit.patch_site)
		return;
	if (exit.current_target == target)
		return;
	armEmitJmpPtr(exit.patch_site, target, true);
	exit.current_target = target;
}

// Restore a patch_site to its unlinked fallthrough (DispatcherReg). Called
// when the exit's target block is invalidated by recClear so the caller
// doesn't B into freed/stale code.
static void unpatchEELinkSite(BlockLinkExit& exit)
{
	patchEELinkSite(exit, exit.fallthrough);
}

// Look up a compiled block's entry point by hwaddr(startpc). Returns
// nullptr when the block isn't linkable (not compiled yet, or a block type
// without a link entry recorded).
static u8* findEEBlockEntry(u32 target_pc)
{
	auto it = s_blockLinks.find(HWADDR(target_pc));
	if (it == s_blockLinks.end())
		return nullptr;
	return it->second.entry;
}

// Called right after a block compiles: for each live static exit, if its
// target is already compiled, wire the exit up. Mirrors tryForwardLink
// in iVU1micro_arm64.cpp.
static void tryForwardLinkEE(BlockLinks& block)
{
	for (u32 e = 0; e < block.num_exits; e++)
	{
		BlockLinkExit& exit = block.exits[e];
		u8* target_entry = findEEBlockEntry(exit.target_pc);
		if (target_entry)
			patchEELinkSite(exit, target_entry);
	}
}

// Add a freshly-registered block to the reverse index for each of its UNIQUE
// exit-target HWs. Caller has already inserted the block into s_blockLinks.
// Dedup matters because a self-loop (cond branch with both arms identical)
// would otherwise add the same pred twice — patchWaitingEEPredecessors
// already filters by exit.target match, so dup pred entries waste only
// memory, but keep the index tight.
static void indexEEBlockExits(u32 my_pc, const BlockLinks& bl)
{
	const u32 my_hw = HWADDR(my_pc);
	for (u32 e = 0; e < bl.num_exits; e++)
	{
		const u32 target_hw = HWADDR(bl.exits[e].target_pc);
		bool dup = false;
		for (u32 j = 0; j < e; j++)
		{
			if (HWADDR(bl.exits[j].target_pc) == target_hw)
			{
				dup = true;
				break;
			}
		}
		if (!dup)
			s_eeWaitingForHw[target_hw].push_back(my_hw);
	}
}

// Called right after a block compiles at `my_pc` with entry `my_entry`:
// for any previously-compiled block whose static exit target is `my_pc`,
// patch that exit's B site to jump directly to us.
//
// Walks the reverse index s_eeWaitingForHw[my_hw] — typically a handful
// of preds — instead of every block in s_blockLinks. Pre-index, this loop
// dominated CPU during first-cache lag bursts (35% of one trace on
// FFXII / GTASA / Darkwatch when the block table held thousands of entries).
// Lazily skips stale pred entries (block invalidated by recClear); they
// stay in the index until recResetRaw flushes everything.
static void patchWaitingEEPredecessors(u32 my_pc, u8* my_entry)
{
	if (!my_entry)
		return;
	const u32 my_hw = HWADDR(my_pc);
	auto wit = s_eeWaitingForHw.find(my_hw);
	if (wit == s_eeWaitingForHw.end())
		return;
	const auto& preds = wit->second;
	for (u32 pred_hw : preds)
	{
		auto bit = s_blockLinks.find(pred_hw);
		if (bit == s_blockLinks.end())
			continue; // stale entry — pred block was invalidated
		BlockLinks& pred = bit->second;
		for (u32 e = 0; e < pred.num_exits; e++)
		{
			BlockLinkExit& exit = pred.exits[e];
			if (HWADDR(exit.target_pc) != my_hw)
				continue;
			if (exit.current_target == my_entry)
				continue;
			patchEELinkSite(exit, my_entry);
		}
	}
}

// Called by recClear when a PC range is invalidated. Walks every
// registered block's exits and unpatches any that target the cleared
// range — preventing a dangling direct-B into code that's about to be
// (or has just been) recycled by the bump allocator on next compile.
// Then removes blocks whose entry is itself in the cleared range.
static void invalidateEELinks(u32 start_hw, u32 end_hw)
{
	// Pass 1: unpatch any exit pointing into the cleared range. This
	// includes exits of blocks OUTSIDE the cleared range — they're still
	// live but their link target is going away.
	for (auto& kv : s_blockLinks)
	{
		BlockLinks& pred = kv.second;
		for (u32 e = 0; e < pred.num_exits; e++)
		{
			BlockLinkExit& exit = pred.exits[e];
			const u32 t = HWADDR(exit.target_pc);
			if (t >= start_hw && t < end_hw)
				unpatchEELinkSite(exit);
		}
	}

	// Pass 2: drop registry entries for blocks whose own startpc is in
	// the cleared range. Their compiled code is being discarded; next
	// dispatch to that PC will recompile and re-register.
	for (auto it = s_blockLinks.begin(); it != s_blockLinks.end(); )
	{
		if (it->first >= start_hw && it->first < end_hw)
			it = s_blockLinks.erase(it);
		else
			++it;
	}
}

// ============================================================================
//  Block end — cycle counting and dispatch
// ============================================================================

// Link target for the current block's main exit (the one emitted by
// iBranchTest at block-end). Nonzero → emit the linkable form. Zero →
// emit the original dispatcher-based form (conditional-runtime path,
// JR/JALR, syscall/break, WaitLoop-optimized blocks).
static u32 s_eeLinkTarget = 0;

// Staging for patch-site metadata that recRecompile will commit into
// the new block's BlockLinks record. Filled by emitEELinkableExit — both
// iBranchTest's linkable tail AND conditional-branch emitters in
// iR5900Branch_arm64.cpp push their exits here. Up to 2 slots: a
// conditional branch emits a taken and a not-taken tail; an
// unconditional branch or fall-through emits just one.
static BlockLinkExit s_eeExitsStaging[2];
static u32           s_eeExitsStagingCount = 0;

// Emits the linkable block-exit tail for `target_pc` and stages the patch
// metadata. Defined here so both the internal iBranchTest path and the
// external branch emitters (via the arm64Emitter.h extern) share one
// implementation — keeping the wire format identical and reducing
// surface for drift bugs.
void emitEELinkableExit(u32 target_pc)
{
	// Shape:
	//     ADDS RCYCLE, RCYCLE, cycles
	//     B.PL <event>          ; budget exhausted
	//     B    <patch_site>     ; initially DispatcherReg, patched to target
	//   event:
	//     <armEmitJmp(DispatcherEvent)>
	//
	// B.cond has only ±1 MB range but <event> is a local label 2-3
	// instructions away — always reachable. The patch B is 26-bit (±128
	// MB), covers any block in the 64 MB cache.
	const u32 cycles = scaleblockcycles();
	armAsm->Adds(RCYCLE, RCYCLE, cycles);

#ifdef EE_FORCE_CYCLE_FLUSH
	// Flush cpuRegs.cycle = nec + RCYCLE before the patched B. Async hardware
	// (VU1/MTGS/GIF) reads cpuRegs.cycle through the handshake; without this,
	// long direct-B chains can leave it stale across many blocks. See
	// InterpFlags.h for the rationale.
	armWritebackCycle();
#endif

	a64::Label event_path;
	armAsm->B(&event_path, a64::pl);

	u8* patch_site = armGetCurrentCodePointer();
	{
		const s64 disp = static_cast<s64>(
			reinterpret_cast<intptr_t>(DispatcherReg)
			- reinterpret_cast<intptr_t>(patch_site));
		pxAssert((disp & 3) == 0 && vixl::IsInt26(disp >> 2));
		a64::SingleEmissionCheckScope guard(armAsm);
		armAsm->b(disp >> 2);
	}

	armAsm->Bind(&event_path);
	armEmitJmp(DispatcherEvent);

	pxAssert(s_eeExitsStagingCount < 2);
	BlockLinkExit& e = s_eeExitsStaging[s_eeExitsStagingCount++];
	e.target_pc      = target_pc;
	e.patch_site     = patch_site;
	e.fallthrough    = const_cast<u8*>(static_cast<const u8*>(DispatcherReg));
	e.current_target = const_cast<u8*>(static_cast<const u8*>(DispatcherReg));
}

static void iBranchTest(u32 newpc)
{
	// RCYCLE += scaleblockcycles();
	// if (RCYCLE < 0)  goto DispatcherReg (or linked block);  // still have budget
	// else             goto DispatcherEvent;                  // event due
	//
	// RCYCLE = (cycle - nextEventCycle), so result-negative (N=1) means
	// cycle is still behind nextEventCycle, i.e. budget remaining. B.MI catches
	// that. The fall-through path covers both result==0 and result>0.

	u32 cycles = scaleblockcycles();

	if (EmuConfig.Speedhacks.WaitLoop && s_nBlockFF && newpc == s_branchTo)
	{
		// Wait loop optimization: cycle = max(cycle + n, nextEventCycle)
		// In delta form: RCYCLE = max(RCYCLE + n, 0), then jump to DispatcherEvent.
		armAsm->Adds(RCYCLE, RCYCLE, cycles);
		armAsm->Csel(RCYCLE, a64::xzr, RCYCLE, a64::mi);
		armEmitJmp(DispatcherEvent);
	}
	else if (s_eeLinkTarget != 0)
	{
		// Linkable main-block exit — delegate to the shared helper that
		// also services conditional-branch emitters. Stages a patch site
		// into s_eeExitsStaging for recRecompile to commit.
		emitEELinkableExit(s_eeLinkTarget);
	}
	else
	{
		// Unlinked path: bump the delta, branch on sign. Preserved for
		// cases where we couldn't stage a link (non-const conditional
		// branches that didn't split their exits, JR/JALR, syscalls)
		// or suppressed it (WaitLoop's event-only exit).
		armAsm->Adds(RCYCLE, RCYCLE, cycles);
#ifdef EE_FORCE_CYCLE_FLUSH
		// Flush cpuRegs.cycle to memory before dispatch — see comment in
		// emitEELinkableExit and InterpFlags.h for the rationale.
		armWritebackCycle();
#endif
		armEmitCondBranch(a64::mi, DispatcherReg); // N=1: still have budget
		armEmitJmp(DispatcherEvent);               // fall through: event due
	}
}

// ============================================================================
//  Instruction recompilation
// ============================================================================

void recompileNextInstruction(bool delayslot, bool swapped_delay_slot)
{
	if (EmuConfig.EnablePatches)
		Patch::ApplyDynamicPatches(pc);

	s_pCode = (int*)PSM(pc);
	pxAssert(s_pCode);

	const int old_code = cpuRegs.code;
	EEINST* old_inst_info = g_pCurInstInfo;

	cpuRegs.code = *(int*)s_pCode;

	// Pre-increment `pc` unconditionally so armFlushPC / cpuRegs.pc match the
	// interpreter's post-increment invariant (see execI in Interpreter.cpp:176).
	// For a delay-slot op this means cpuRegs.pc reads as DS_addr+4 while the op
	// runs, which cpuException relies on for EPC = pc-4 on the BD path.
	// Native branch emitters never run in DS mode (the guard below rejects
	// branches-in-DS), so their pc+4 / (pc & 0xf0000000) math — which expects
	// pc = branch_addr+4 at entry — is unaffected.
	pc += 4;
	g_cpuFlushedPC = false;
	g_cpuFlushedCode = false;
	if (delayslot)
	{
		g_recompilingDelaySlot = true;
		// Emit cpuRegs.branch = 1 so any exception raised during this DS op
		// (overflow, TLB miss via vtlb helpers, etc.) takes the BD path in
		// cpuException and produces the correct EPC / CAUSE.BD. Matches the
		// interpreter's _doBranch_shared (Interpreter.cpp:230). Done for every
		// DS op (native + ISTUB) since native LOAD/STORE can TLB-miss too.
		//
		// We deliberately do NOT speculatively set CAUSE.BD here. cpuException
		// itself sets `cpuRegs.CP0.n.Cause |= 0x80000000` whenever its `bd`
		// argument is nonzero, and every JIT exception path passes
		// cpuRegs.branch as that argument (see R5900OpcodeImpl.cpp ovrfl
		// helpers and cpuTlbMissR/W). The pre-set was redundant — matches
		// x86, where FLUSH_CAUSE (iR5900.cpp:1236-1242) is `#if 0`'d for
		// the same reason — and burned 6 insns per DS (3 set + 3 clear).
		armAsm->Mov(RWSCRATCH, 1);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, offsetof(cpuRegisters, branch)));
	}

	g_pCurInstInfo++;

	// Branch-in-delay-slot guard (matches x86 iR5900.cpp:1743-1803 — FlatOut
	// PR #1783). MIPS UB when a branch sits inside another branch's delay slot;
	// x86 logs and skips the inner emission so the outer branch does not end
	// up with a half-built target from the inner branch's g_branch = 1.
	if (delayslot)
	{
		bool check_branch_delay = false;
		switch (_Opcode_)
		{
			case 0: // SPECIAL
				if (_Funct_ == 8 || _Funct_ == 9) // JR, JALR
					check_branch_delay = true;
				break;
			case 1: // REGIMM
				if (_Rt_ < 4 || (_Rt_ >= 0x10 && _Rt_ < 0x14))
					check_branch_delay = true;
				break;
			case 2: case 3:                             // J, JAL
			case 4: case 5: case 6: case 7:             // BEQ, BNE, BLEZ, BGTZ
			case 0x14: case 0x15: case 0x16: case 0x17: // BEQL, BNEL, BLEZL, BGTZL
				check_branch_delay = true;
				break;
		}
		if (check_branch_delay)
		{
			DevCon.Warning("Branch %x in delay slot!", cpuRegs.code);
			// Undo the pre-DS cpuRegs.branch = 1 store: we're skipping the
			// inner branch emission entirely, so leave branch=0 at block exit.
			// CAUSE.BD is no longer speculatively set, so nothing to clear.
			armAsm->Str(a64::wzr, a64::MemOperand(RCPUSTATE, offsetof(cpuRegisters, branch)));
			g_recompilingDelaySlot = false;
			cpuRegs.code = old_code;
			g_pCurInstInfo = old_inst_info;
			return;
		}
	}

	const OPCODE& opcode = GetCurrentInstruction();

	// Check for branch in delay slot - if so, skip recompiling it to avoid infinite recursion.
	// Based on x86 JIT code by FlatOut, see https://github.com/PCSX2/pcsx2/pull/1783
	if (delayslot)
	{
		bool is_branch = false;
		switch (_Opcode_)
		{
			case 0:
				switch (_Funct_)
				{
					case 8: // jr
					case 9: // jalr
						is_branch = true;
						break;
				}
				break;
			case 1:
				switch (_Rt_)
				{
					case 0:  // bltz
					case 1:  // bgez
					case 2:  // bltzl
					case 3:  // bgezl
					case 0x10: // bltzal
					case 0x11: // bgezal
					case 0x12: // bltzall
					case 0x13: // bgezall
						is_branch = true;
						break;
				}
				break;
			case 2:  // j
			case 3:  // jal
			case 4:  // beq
			case 5:  // bne
			case 6:  // blez
			case 7:  // bgtz
			case 0x14: // beql
			case 0x15: // bnel
			case 0x16: // blezl
			case 0x17: // bgtzl
				is_branch = true;
				break;
			case 0x11: // COP1
				if (_Rs_ == 8) // BC1
				{
					switch (_Rt_)
					{
						case 0: // bc1f
						case 1: // bc1t
						case 2: // bc1fl
						case 3: // bc1tl
							is_branch = true;
							break;
					}
				}
				break;
		}
		if (is_branch)
		{
			DevCon.Warning("Branch %08x in delay slot!", cpuRegs.code);
			pc += 4;
			g_cpuFlushedPC = false;
			g_cpuFlushedCode = false;
			g_recompilingDelaySlot = false;
			cpuRegs.code = old_code;
			g_pCurInstInfo = old_inst_info;
			return;
		}
	}

	// NOP check
	if (cpuRegs.code == 0x00000000)
	{
		s_nBlockCycles += 9 * (2 - ((cpuRegs.CP0.n.Config >> 18) & 0x1));
	}
	else
	{
		s_nBlockCycles += opcode.cycles * (2 - ((cpuRegs.CP0.n.Config >> 18) & 0x1));

#ifdef EE_PROFILE_OPS
		const u8* _ee_op_start = armGetCurrentCodePointer();
#endif
		if (opcode.recompile)
			opcode.recompile();
		else
		{
			// No recompiler implementation — fall back to interpreter
			armCallInterpreter(opcode.interpret);
		}
#ifdef EE_PROFILE_OPS
		{
			const u8* _ee_op_end = armGetCurrentCodePointer();
			if (_ee_op_end > _ee_op_start)
			{
				char _ee_op_name[48];
				std::snprintf(_ee_op_name, sizeof(_ee_op_name),
					"EE_%s_0x%08x", opcode.Name ? opcode.Name : "OP", pc - 4);
				Perf::ee.Register(_ee_op_start,
					static_cast<size_t>(_ee_op_end - _ee_op_start), _ee_op_name);
			}
		}
#endif

		// Phase D: op-local GPR cache discipline.
		// Migrated ops (ALU only as of Phase D) use armGprAlloc to bind MIPS
		// GPRs to host pool regs within their codegen. Commit any dirty slots
		// to memory and drop the cache before moving on to the next opcode —
		// non-migrated ops still read/write memory directly via armLoadGPR /
		// armStoreGPR, so the cache must not survive across the boundary.
		// No-op when nothing was cached (the common case).
		armGprInvalidateAll();
	}

	if (delayslot)
	{
		// Clear cpuRegs.branch after the DS completes so subsequent ops in the
		// block that raise exceptions are not misattributed to the BD path.
		// CAUSE.BD is set by cpuException itself when bd != 0, so we don't
		// need a paired BD-clear here.
		armAsm->Str(a64::wzr, a64::MemOperand(RCPUSTATE, offsetof(cpuRegisters, branch)));
		g_recompilingDelaySlot = false;
	}

	cpuRegs.code = old_code;
	g_pCurInstInfo = old_inst_info;
}

// ============================================================================
//  Block compilation
// ============================================================================

static void recError(u32 error)
{
	static u32 lastError = ~0u;
	static int errorCount = 0;

	if (error == lastError)
	{
		if (++errorCount > 5)
		{
			Console.Error("EE ARM64 recError: pc=%08X repeated %d times, halting", error, errorCount);
			VMManager::SetPaused(true);
			cpuRegs.branch = 0;
			recExitExecution();
			return;
		}
	}
	else
	{
		lastError = error;
		errorCount = 1;
	}

	Console.Error("EE ARM64 Recompiler Error: pc=%08X (recLUT page %04X unmapped)", error, error >> 16);
	cpuRegs.branch = 0;
	recExitExecution();
}

u8* recBeginThunk()
{
	if (recPtr >= recPtrEnd)
		eeRecNeedsReset = true;

	armSetAsmPtr(recPtr, recPtrEnd - recPtr, nullptr);
	recPtr = armStartBlock();
	return recPtr;
}

u8* recEndThunk()
{
	u8* block_end = armEndBlock();
	pxAssert(block_end < SysMemory::GetEERecEnd());
	recPtr = block_end;
	return block_end;
}

static void recRecompile(const u32 startpc)
{
	pxAssert(startpc);

	// Check if we need to reset the code buffer
	if (recPtr >= recPtrEnd)
		eeRecNeedsReset = true;

	if (HWADDR(startpc) == VMManager::Internal::GetCurrentELFEntryPoint())
		VMManager::Internal::EntryPointCompilingOnCPUThread();

	if (eeRecNeedsReset)
	{
		eeRecNeedsReset = false;
		recResetRaw();
	}

	// Set up the assembler to write to the code buffer
	armSetAsmPtr(recPtr, recPtrEnd - recPtr, &s_recConstPool);
	u8* blockStart = armStartBlock();

	s_pCurBlock = PC_GETBLOCK(startpc);
	pxAssert(s_pCurBlock->GetFnptr() == (uptr)JITCompile);

	s_pCurBlockEx = recBlocks.Get(HWADDR(startpc));
	pxAssert(!s_pCurBlockEx || s_pCurBlockEx->startpc != HWADDR(startpc));
	s_pCurBlockEx = recBlocks.New(HWADDR(startpc), (uptr)blockStart);
	pxAssert(s_pCurBlockEx);

	// EELOAD hooks for fast boot
	if (HWADDR(startpc) == EELOAD_START)
	{
		const u32 mainjump = memRead32(EELOAD_START + 0x9c);
		if (mainjump >> 26 == 3) // JAL
			g_eeloadMain = ((EELOAD_START + 0xa0) & 0xf0000000U) | (mainjump << 2 & 0x0fffffffU);
	}

	if (g_eeloadMain && HWADDR(startpc) == HWADDR(g_eeloadMain))
	{
		armEmitCall((void*)eeloadHook);
		if (VMManager::Internal::IsFastBootInProgress())
		{
			const u32 typeAexecjump = memRead32(EELOAD_START + 0x470);
			const u32 typeBexecjump = memRead32(EELOAD_START + 0x5B0);
			const u32 typeCexecjump = memRead32(EELOAD_START + 0x618);
			const u32 typeDexecjump = memRead32(EELOAD_START + 0x600);
			if ((typeBexecjump >> 26 == 3) || (typeCexecjump >> 26 == 3) || (typeDexecjump >> 26 == 3))
				g_eeloadExec = EELOAD_START + 0x2B8;
			else if (typeAexecjump >> 26 == 3)
				g_eeloadExec = EELOAD_START + 0x170;
			else
				Console.WriteLn("recRecompile: Could not enable launch arguments for fast boot mode.");
		}
	}

	if (g_eeloadExec && HWADDR(startpc) == HWADDR(g_eeloadExec))
		armEmitCall((void*)eeloadHook2);

	g_branch = 0;
	g_eeStaticBranchPC = 0; // set by SetBranchImm if the branch emitter picks a compile-time successor

	// Reset block-link exit staging — conditional branch emitters push
	// their taken tail here during the compile, and iBranchTest pushes
	// the block-end (not-taken / unconditional / fall-through) tail.
	s_eeExitsStagingCount = 0;

	// Reset recompiler state
	s_nBlockCycles = 0;
	s_nBlockInterlocked = false;
	pc = startpc;
	g_cpuHasConstReg = g_cpuFlushedConstReg = 1; // r0 is always const 0
	pxAssert(g_cpuConstRegs[0].UD[0] == 0);

	// Determine block boundaries
	u32 i = startpc;
	s_nEndBlock = 0xffffffff;
	s_branchTo = -1;
	s_branchIsUnconditional = false;

	while (1)
	{
		BASEBLOCK* pblock = PC_GETBLOCK(i);

		if (i != startpc)
		{
			if ((i & 0xffc) == 0x0) // page boundary
			{
				s_nEndBlock = i;
				break;
			}

			if (pblock->GetFnptr() != (uptr)JITCompile)
			{
				s_nEndBlock = i;
				break;
			}
		}

		cpuRegs.code = *(int*)PSM(i);

		switch (cpuRegs.code >> 26)
		{
			case 0: // special
				if (_Funct_ == 8 || _Funct_ == 9) // JR, JALR
				{
					s_nEndBlock = i + 8;
					goto StartRecomp;
				}
				else if (_Funct_ == 12 || _Funct_ == 13) // SYSCALL, BREAK
				{
					s_nEndBlock = i + 4;
					goto StartRecomp;
				}
				break;
			case 1: // regimm
				if (_Rt_ < 4 || (_Rt_ >= 16 && _Rt_ < 20))
				{
					s_branchTo = _Imm_ * 4 + i + 4;
					s_nEndBlock = i + 8;
					goto StartRecomp;
				}
				break;
			case 2: // J
			case 3: // JAL
				s_branchTo = (_Target_ << 2) | ((i + 4) & 0xf0000000);
				s_branchIsUnconditional = true;
				s_nEndBlock = i + 8;
				goto StartRecomp;
			case 4: case 5: case 6: case 7: // BEQ, BNE, BLEZ, BGTZ
			case 20: case 21: case 22: case 23: // BEQL, BNEL, BLEZL, BGTZL
				s_branchTo = _Imm_ * 4 + i + 4;
				s_nEndBlock = i + 8;
				goto StartRecomp;
			case 16: // COP0
				if (_Rs_ == 16) // COP_FUNC
				{
					if (_Funct_ == 24) // ERET
					{
						s_nEndBlock = i + 4;
						goto StartRecomp;
					}
				}
				// Fall through! COP0's BC0F/BC0T/BC0FL/BC0TL encode at rs=8,
				// which lines up with COP1's BC1* and COP2's BC2*.
				[[fallthrough]];
			case 17: // COP1
			case 18: // COP2
				if (_Rs_ == 8) // BC{0,1,2}{F,T,FL,TL}
				{
					s_branchTo = _Imm_ * 4 + i + 4;
					s_nEndBlock = i + 8;
					goto StartRecomp;
				}
				break;
		}

		i += 4;
	}

StartRecomp:

	// Detect wait/spin loops: if the block branches back to its own start
	// and only does loads, immediate arithmetic, nops, or mfc/cfc (i.e. it
	// polls a hardware register), we can fast-forward the cycle counter to
	// the next event instead of spinning.
	s_nBlockFF = false;
	if (s_branchTo == startpc)
	{
		s_nBlockFF = true;

		u32 reads = 0, loads = 1;

		for (i = startpc; i < s_nEndBlock; i += 4)
		{
			if (i == s_nEndBlock - 8)
				continue;
			cpuRegs.code = *(u32*)PSM(i);
			// nop
			if (cpuRegs.code == 0)
				continue;
			// cache, sync
			else if (_Opcode_ == 057 || (_Opcode_ == 0 && _Funct_ == 017))
				continue;
			// imm arithmetic
			else if ((_Opcode_ & 070) == 010 || (_Opcode_ & 076) == 030)
			{
				if (loads & 1 << _Rs_)
				{
					loads |= 1 << _Rt_;
					continue;
				}
				else
					reads |= 1 << _Rs_;
				if (reads & 1 << _Rt_)
				{
					s_nBlockFF = false;
					break;
				}
			}
			// common register arithmetic instructions
			else if (_Opcode_ == 0 && (_Funct_ & 060) == 040 && (_Funct_ & 076) != 050)
			{
				if (loads & 1 << _Rs_ && loads & 1 << _Rt_)
				{
					loads |= 1 << _Rd_;
					continue;
				}
				else
					reads |= 1 << _Rs_ | 1 << _Rt_;
				if (reads & 1 << _Rd_)
				{
					s_nBlockFF = false;
					break;
				}
			}
			// loads
			else if ((_Opcode_ & 070) == 040 || (_Opcode_ & 076) == 032 || _Opcode_ == 067)
			{
				if (loads & 1 << _Rs_)
				{
					loads |= 1 << _Rt_;
					continue;
				}
				else
					reads |= 1 << _Rs_;
				if (reads & 1 << _Rt_)
				{
					s_nBlockFF = false;
					break;
				}
			}
			// mfc*, cfc*
			else if ((_Opcode_ & 074) == 020 && _Rs_ < 4)
			{
				loads |= 1 << _Rt_;
			}
			else
			{
				s_nBlockFF = false;
				break;
			}
		}
	}

	// Timeout-loop detection. Block must be exactly:
	//   addi/addiu rN, rN, -K   (negative immediate, same source/dest reg)
	//   bne rN, $0, self        (branch back to start)
	//   nop                     (delay slot)
	// Anything else (or any non-zero op outside this triplet) disqualifies.
	// Mirrors x86 detection at ix86-32/iR5900.cpp:2280-2345.
	s_eeTimeoutReg = -1;
	if (s_branchTo == startpc && (s_nEndBlock - startpc) <= 12)
	{
		bool is_timeout_loop = true;
		s32 timeout_reg = -1;
		for (u32 j = startpc; j < s_nEndBlock && is_timeout_loop; j += 4)
		{
			cpuRegs.code = *(u32*)PSM(j);
			const u32 op = cpuRegs.code >> 26;
			if (op == 8 || op == 9) // addi / addiu
			{
				if (timeout_reg >= 0 || _Rs_ != _Rt_ || _Imm_ >= 0)
					is_timeout_loop = false;
				else
					timeout_reg = _Rs_;
			}
			else if (op == 5) // bne
			{
				if (timeout_reg != static_cast<s32>(_Rs_) || _Rt_ != 0
				    || j + 4 >= s_nEndBlock || *(u32*)PSM(j + 4) != 0)
				{
					is_timeout_loop = false;
				}
			}
			else if (cpuRegs.code != 0)
			{
				is_timeout_loop = false;
			}
		}
		if (is_timeout_loop && timeout_reg >= 1)
			s_eeTimeoutReg = timeout_reg;
	}

	// Build instruction info cache + run backprop liveness analysis. The
	// pass mirrors x86 ix86-32/iR5900.cpp:2532-2545: walk the block in
	// REVERSE from a past-the-end sentinel (all regs assumed live across
	// the block boundary), back-propagating USED/LIVE/LASTUSE bits per
	// MIPS register. After this, s_pInstCache[k] holds the state coming
	// INTO instruction k-1 (matches the x86 g_pCurInstInfo convention),
	// which the GPR cache evictor consults for next-use info.
	{
		u32 numinsts = (s_nEndBlock - startpc) / 4;
		if (numinsts + 1 > s_nInstCacheSize)
		{
			free(s_pInstCache);
			s_nInstCacheSize = numinsts + 1;
			s_pInstCache = (EEINST*)malloc(sizeof(EEINST) * s_nInstCacheSize);
		}
		memset(s_pInstCache, 0, sizeof(EEINST) * (numinsts + 1));

		EEINST* pcur = s_pInstCache + numinsts;
		_recClearInst(pcur);
		pcur->info = 0;
		s_pInstCacheLast = pcur;

		for (s32 i = static_cast<s32>(s_nEndBlock); i != static_cast<s32>(startpc); i -= 4)
		{
			cpuRegs.code = *(u32*)PSM(i - 4);
			pcur[-1] = pcur[0];
			recBackpropBSC(cpuRegs.code, pcur - 1, pcur);
			pcur--;
		}
	}

	// Emit SMC protection (integrity checks / write-protect) at the top of the block
	memory_protect_recompiled_code(startpc, (s_nEndBlock - startpc) >> 2);

	// Now emit code for each instruction
	g_pCurInstInfo = s_pInstCache;
	g_cpuFlushedPC = false;
	g_cpuFlushedCode = false;
	armGprCacheReset();

#ifdef TRACE_BLOCKS
	armEmitFlushCycleBeforeCall();
	armAsm->Mov(RWARG1, startpc);
	armEmitCall((void*)eeTraceBlock);
	armEmitReloadCycleAfterCall();
#endif

	// Timeout-loop skip. The block is `addi rN,rN,-K; bne rN,$0,self; nop`,
	// which decrements rN once per loop iteration (one iter per scaleblockcycles
	// worth of EE cycles, hardcoded as 8 here to match x86 recSkipTimeoutLoop).
	// Instead of running the loop, decrement rN by the iterations that fit in
	// the remaining cycle budget and jump to the dispatcher.
	//
	// Math (delta form, RCYCLE = cycle - nextEventCycle, < 0 = budget left):
	//   if RCYCLE >= 0:           DispatcherEvent  (event already due)
	//   proj = (u64)rN * 8 + RCYCLE
	//   new_RCYCLE = min(proj, 0)
	//   iters = (new_RCYCLE - RCYCLE) / 8
	//   rN -= iters; RCYCLE = new_RCYCLE
	//   if rN != 0:               DispatcherEvent  (event hit before loop done)
	//   else:                     pc = s_nEndBlock; DispatcherReg
	if (EmuConfig.Speedhacks.WaitLoop && s_eeTimeoutReg >= 1)
	{
		// Early out — RCYCLE >= 0 means the event is already due. Skip the
		// timeout math entirely; let the dispatcher service the event.
		armAsm->Cmp(RCYCLE, a64::xzr);
		armEmitCondBranch(a64::ge, DispatcherEvent);

		// w0 = rN (32-bit unsigned), x1 = (u64)rN * 8 + RCYCLE
		armAsm->Ldr(a64::w0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(s_eeTimeoutReg)));
		armAsm->Add(a64::x1, RCYCLE, a64::Operand(a64::x0, a64::LSL, 3));

		// new_RCYCLE = min(x1, 0). Cmp x1 vs 0 then csel: if x1 >= 0 (loop
		// would overrun budget) take 0, else keep x1.
		armAsm->Cmp(a64::x1, a64::xzr);
		armAsm->Csel(a64::x2, a64::xzr, a64::x1, a64::ge);

		// iters = (new_RCYCLE - old_RCYCLE) / 8. The diff is non-negative
		// here (new >= old in both branches), so ASR vs LSR are equivalent.
		armAsm->Sub(a64::x3, a64::x2, RCYCLE);
		armAsm->Lsr(a64::x3, a64::x3, 3);

		// rN -= iters; store back; install new RCYCLE.
		armAsm->Sub(a64::w0, a64::w0, a64::w3);
		armAsm->Str(a64::w0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(s_eeTimeoutReg)));
		armAsm->Mov(RCYCLE, a64::x2);

		// If rN != 0, the cycle budget hit nextEventCycle before rN reached
		// zero — service the event. If rN == 0, the loop completed; advance
		// pc past the loop and go through DispatcherReg (no event due).
		a64::Label completed;
		armAsm->Cbz(a64::w0, &completed);
		armEmitJmp(DispatcherEvent);

		armAsm->Bind(&completed);
		armAsm->Mov(RWSCRATCH, s_nEndBlock);
		armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
		g_cpuFlushedPC = true;
		armEmitJmp(DispatcherReg);

		// Short-circuit the per-instruction emit loop. The block-end handler
		// will still emit its standard tail (iBranchTest etc.), but it's
		// dead code after the unconditional jumps above. Suppressing block
		// linking via WaitLoop semantics matches the existing s_nBlockFF
		// path's intent.
		g_branch = 1;
		s_nBlockFF = true;
		pc = s_nEndBlock;
	}

	while (!g_branch && pc < s_nEndBlock)
	{
		recompileNextInstruction(false, false);
	}

	pxAssert((pc - startpc) >> 2 <= 0xffff);
	s_pCurBlockEx->size = (pc - startpc) >> 2;

	// Handle block ending. Compute the static link target (nonzero →
	// linkable successor) here so iBranchTest knows which exit shape to
	// emit. Leave zero for conditional branches, JR/JALR, syscalls and
	// WaitLoop-optimized blocks; those keep the pre-link dispatcher path.
	//
	// s_eeExitsStaging may already have 1 entry staged by a conditional
	// branch emitter (the taken tail); don't clear it here.
	s_eeLinkTarget = 0;

	if (g_branch == 2) // syscall/break — event check, indirect dispatch
	{
		armFlushConstRegs();
		iBranchTest();
	}
	else
	{
		// Branch or fall-through
		if (!g_cpuFlushedPC)
		{
			armAsm->Mov(RWSCRATCH, pc);
			armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
		}
		armFlushConstRegs();

		// Linkable shapes:
		//   - Any branch emitter that called SetBranchImm(x) — captured via
		//     g_eeStaticBranchPC. That covers J/JAL, the `_Rs_ == _Rt_`
		//     BEQ form (MIPS's `b label`), and every const-folded
		//     conditional branch. cpuRegs.pc is guaranteed to end up == x.
		//   - Fall-through (no branch emitted, block capped at s_nEndBlock):
		//     cpuRegs.pc ends up == `pc` (just written above).
		// Non-const conditional branches use CSEL between two PCs at
		// runtime (g_branch==1, g_eeStaticBranchPC==0) — single successor
		// PC isn't known, so we can't pick a patch target and they stay
		// on the DispatcherReg fallback.
		//
		// Suppress linking when the WaitLoop speedhack will fire. Its
		// iBranchTest path jumps unconditionally to DispatcherEvent (no
		// B.MI successor slot exists to patch).
		const bool waitloop_fire = EmuConfig.Speedhacks.WaitLoop && s_nBlockFF;
		if (!waitloop_fire)
		{
			if (g_branch == 1 && g_eeStaticBranchPC != 0)
				s_eeLinkTarget = g_eeStaticBranchPC;
			else if (g_branch == 0)
				s_eeLinkTarget = pc;
		}

		iBranchTest(s_branchTo);
	}

	armEndBlock();

	// armEndBlock() advances armAsmPtr and sets armAsm=nullptr,
	// so use armAsmPtr directly instead of armGetCurrentCodePointer().
	recPtr = armAsmPtr;
	pxAssert((g_cpuHasConstReg & g_cpuFlushedConstReg) == g_cpuHasConstReg);

	// Register the compiled block with simpleperf/perfetto so JIT'd code
	// shows up as `EE_<startpc>` in profiler reports instead of "unknown
	// unknown". Mirrors x86 ix86-32/iR5900.cpp's per-block Register call.
	// Cost: one map insert per block compile, no runtime hit.
	Perf::ee.RegisterPC(blockStart, static_cast<size_t>(recPtr - blockStart), startpc);

	// Point the BASEBLOCK at the compiled code so the dispatcher jumps directly
	// to it. Without this, the block stays pointed at JITCompile and gets
	// recompiled on every dispatch — burning through the code buffer.
	s_pCurBlock->SetFnptr((uptr)blockStart);

	// Register this block in the link graph. `num_exits` reflects how many
	// patchable B sites emitEELinkableExit staged during this block — 0
	// (JR/JALR/syscall/CSEL branch that didn't split), 1 (J/JAL, const-
	// folded branch, fall-through), or 2 (conditional branch with split
	// taken/not-taken tails). tryForwardLinkEE wires each outgoing exit
	// to an already-compiled target; patchWaitingEEPredecessors rewires
	// any previously-compiled blocks whose static target is us.
	{
		BlockLinks bl = {};
		bl.entry      = reinterpret_cast<u8*>(blockStart);
		bl.num_exits  = s_eeExitsStagingCount;
		for (u32 i = 0; i < s_eeExitsStagingCount; i++)
			bl.exits[i] = s_eeExitsStaging[i];

		const u32 hw = HWADDR(startpc);
		s_blockLinks[hw] = bl;

		// Reverse index entry — must happen BEFORE patchWaitingEEPredecessors
		// so any predecessors discovered for THIS block can find us, and after
		// s_blockLinks insert so a self-loop's tryForwardLinkEE finds the entry.
		indexEEBlockExits(startpc, s_blockLinks[hw]);

		tryForwardLinkEE(s_blockLinks[hw]);
		patchWaitingEEPredecessors(startpc, bl.entry);
	}

	if (!(pc & 0x10000000))
		maxrecmem = std::max((pc & ~0xa0000000), maxrecmem);

	s_pCurBlock = nullptr;
	s_pCurBlockEx = nullptr;
}

// ============================================================================
//  Memory management and lifecycle
// ============================================================================

static void recReserveRAM()
{
	recLutEntries = (Ps2MemSize::ExposedRam + Ps2MemSize::Rom + Ps2MemSize::Rom1 + Ps2MemSize::Rom2) / 4;

	if (recRAMCopy.size() != Ps2MemSize::ExposedRam)
		recRAMCopy.resize(Ps2MemSize::ExposedRam);

	if (recLutReserve_RAM.size() != recLutEntries)
		recLutReserve_RAM.resize(recLutEntries);

	recLutUnmapped.resize(_64kb / 4);

	BASEBLOCK* basepos = recLutReserve_RAM.data();
	recRAM = basepos; basepos += (Ps2MemSize::ExposedRam / 4);
	recROM = basepos; basepos += (Ps2MemSize::Rom / 4);
	recROM1 = basepos; basepos += (Ps2MemSize::Rom1 / 4);
	recROM2 = basepos; basepos += (Ps2MemSize::Rom2 / 4);

	BASEBLOCK* unmapped = recLutUnmapped.data();
	for (int j = 0; j < 0x10000; j++)
		recLUT_SetPage(recLUT, hwLUT, unmapped, j, 0, 0);

	for (int j = 0x0000; j < (int)(Ps2MemSize::ExposedRam / 0x10000); j++)
	{
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0x0000, j, j);
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0x2000, j, j);
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0x3000, j, j);
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0x8000, j, j);
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0xa000, j, j);
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0xb000, j, j);
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0xc000, j, j);
		recLUT_SetPage(recLUT, hwLUT, recRAM, 0xd000, j, j);
	}

	for (int j = 0x1fc0; j < 0x2000; j++)
	{
		recLUT_SetPage(recLUT, hwLUT, recROM, 0x0000, j, j - 0x1fc0);
		recLUT_SetPage(recLUT, hwLUT, recROM, 0x8000, j, j - 0x1fc0);
		recLUT_SetPage(recLUT, hwLUT, recROM, 0xa000, j, j - 0x1fc0);
	}

	for (int j = 0x1e00; j < 0x1e40; j++)
	{
		recLUT_SetPage(recLUT, hwLUT, recROM1, 0x0000, j, j - 0x1e00);
		recLUT_SetPage(recLUT, hwLUT, recROM1, 0x8000, j, j - 0x1e00);
		recLUT_SetPage(recLUT, hwLUT, recROM1, 0xa000, j, j - 0x1e00);
	}

	for (int j = 0x1e40; j < 0x1e80; j++)
	{
		recLUT_SetPage(recLUT, hwLUT, recROM2, 0x0000, j, j - 0x1e40);
		recLUT_SetPage(recLUT, hwLUT, recROM2, 0x8000, j, j - 0x1e40);
		recLUT_SetPage(recLUT, hwLUT, recROM2, 0xa000, j, j - 0x1e40);
	}
}

static void recReserve()
{
	Console.WriteLn("ARM64 recReserve: GetEERec=%p GetEERecEnd=%p", SysMemory::GetEERec(), SysMemory::GetEERecEnd());
	recPtr = SysMemory::GetEERec();
	recPtrEnd = SysMemory::GetEERecEnd() - _64kb;
	Console.WriteLn("ARM64 recReserve: recPtr=%p recPtrEnd=%p (capacity=%zu)", recPtr, recPtrEnd, (size_t)(recPtrEnd - recPtr));

	// Initialize constant pool at the end of the code buffer
	s_recConstPool.Init(recPtrEnd, _64kb);
	Console.WriteLn("ARM64 recReserve: constant pool initialized");

	recReserveRAM();
	Console.WriteLn("ARM64 recReserve: RAM reserved");

	pxAssertRel(!s_pInstCache, "InstCache not allocated");
	s_nInstCacheSize = 128;
	s_pInstCache = (EEINST*)malloc(sizeof(EEINST) * s_nInstCacheSize);
	if (!s_pInstCache)
		pxFailRel("Failed to allocate R5900 InstCache array");
	Console.WriteLn("ARM64 recReserve: done");
}

alignas(16) static u16 manual_page[Ps2MemSize::TotalRam >> 12];
alignas(16) static u8 manual_counter[Ps2MemSize::TotalRam >> 12];

// Emit SMC protection at the top of a compiled block.
//
// ProtMode_None / ProtMode_Write:
//   Install OS write-protection on the page at compile time.  No code emitted.
//   The vtlb fault handler clears stale blocks when the game writes to the page.
//
// ProtMode_Manual:
//   Emit per-DWORD integrity checks inline.  If any word differs from the
//   compile-time snapshot, the block was stale and we jump to DispatchBlockDiscard.
//   Counted blocks also accumulate a weighted value into manual_page[]; when the
//   16-bit accumulator overflows the page switches to OS write-protection via
//   DispatchPageReset.
static void memory_protect_recompiled_code(u32 startpc, u32 size)
{
	const u32 inpage_ptr = HWADDR(startpc);
	const u32 inpage_sz  = size * 4; // bytes

	// Kernel thread-stack pages must always use manual protection
	const bool contains_thread_stack =
		((startpc >> 12) == 0x81) || ((startpc >> 12) == 0x80001);
	const vtlb_ProtectionMode PageType =
		contains_thread_stack ? ProtMode_Manual : mmap_GetRamPageInfo(inpage_ptr);

	switch (PageType)
	{
		case ProtMode_NotRequired:
			break;

		case ProtMode_None:
		case ProtMode_Write:
			// Switch to OS write-protection now; the vtlb fault handler will
			// call recClear when the game writes to this page.  No emitted code.
			mmap_MarkCountedRamPage(inpage_ptr);
			manual_page[inpage_ptr >> 12] = 0;
			break;

		case ProtMode_Manual:
		{
			// Pre-load dyna_block_discard args before the integrity loop so
			// they are live at DispatchBlockDiscard regardless of which word
			// fails the check.
			armAsm->Mov(RWARG1, inpage_ptr);
			armAsm->Mov(RWARG2, inpage_sz / 4);

			// Emit one compare per DWORD in the block.
			u32 lpc = inpage_ptr;
			u32 stg = inpage_sz;
			while (stg > 0)
			{
				const u32 snapshot = *(const u32*)PSM(lpc);

				// Load current value from PS2 RAM (RMEMBASE = eeMem->Main)
				armAsm->Mov(RSCRATCHGPR, (uint64_t)lpc);
				armAsm->Ldr(RWSCRATCH, a64::MemOperand(RMEMBASE, RSCRATCHGPR));

				// Compare with compile-time snapshot
				armAsm->Mov(RSCRATCHGPR2, (uint64_t)(u32)snapshot);
				armAsm->Cmp(RWSCRATCH, RWSCRATCH2);
				armEmitCondBranch(a64::ne, DispatchBlockDiscard);

				stg -= 4;
				lpc += 4;
			}

			// Counted blocks: accumulate block size into manual_page[page].
			// When the 16-bit accumulator overflows the page switches to OS
			// write-protection (cheaper than continued inline checks).
			if (!contains_thread_stack && manual_counter[inpage_ptr >> 12] <= 3)
			{
				armMoveAddressToReg(RSCRATCHGPR, &manual_page[inpage_ptr >> 12]);
				armAsm->Ldrh(RWSCRATCH2, a64::MemOperand(RSCRATCHGPR));
				armAsm->Add(RWSCRATCH2, RWSCRATCH2, size);
				armAsm->Strh(RWSCRATCH2, a64::MemOperand(RSCRATCHGPR));
				armAsm->Cmp(RWSCRATCH2, 0x10000);
				armEmitCondBranch(a64::ge, DispatchPageReset);
			}
			break;
		}
	}
}

static void ClearRecLUT(BASEBLOCK* base, int count)
{
	for (int i = 0; i < count / 4; i++)
		base[i].SetFnptr((uptr)JITCompile);
}

static void recResetRaw()
{
	Console.WriteLn(Color_StrongBlack, "EE/ARM64 Recompiler Reset");

	if (CHECK_EXTRAMEM != extraRam)
	{
		Console.WriteLn("ARM64 recReset: extra RAM changed, re-reserving");
		recReserveRAM();
		extraRam = !extraRam;
	}

	EE::Profiler.Reset();
	Console.WriteLn("ARM64 recReset: profiler reset, setting up asm ptr=%p capacity=%zu", SysMemory::GetEERec(), (size_t)(recPtrEnd - SysMemory::GetEERec()));

	// Set up assembler at the beginning of the code buffer
	armSetAsmPtr(SysMemory::GetEERec(), recPtrEnd - SysMemory::GetEERec(), &s_recConstPool);
	s_recConstPool.Reset();
	Console.WriteLn("ARM64 recReset: asm ptr set, generating dispatchers");

	_DynGen_Dispatchers();
	// armEndBlock() already advanced armAsmPtr past emitted code and set armAsm=nullptr,
	// so we can't use armGetCurrentCodePointer() (which dereferences armAsm).
	recPtr = armAsmPtr;

	ClearRecLUT(recLutReserve_RAM.data(),
		Ps2MemSize::ExposedRam + Ps2MemSize::Rom + Ps2MemSize::Rom1 + Ps2MemSize::Rom2);

	for (int j = 0; j < _64kb / 4; j++)
		recLutUnmapped.data()[j].SetFnptr((uptr)UnmappedRecLUTPage);

	recRAMCopy.fill(0);
	maxrecmem = 0;

	if (s_pInstCache)
		memset(s_pInstCache, 0, sizeof(EEINST) * s_nInstCacheSize);

	recBlocks.Reset();
	vtlb_ClearLoadStoreInfo();

	// Block-link graph is invalidated wholesale on reset — all the patch
	// sites we recorded point into code that's about to be overwritten,
	// and the entry pointers are about to be recycled by the bump
	// allocator. Drop everything; blocks re-register as they recompile.
	s_blockLinks.clear();
	s_eeWaitingForHw.clear();

	g_branch = 0;
	g_resetEeScalingStats = true;

	memset(manual_page, 0, sizeof(manual_page));
	memset(manual_counter, 0, sizeof(manual_counter));
}

static void recShutdown()
{
	recRAMCopy.deallocate();
	recLutReserve_RAM.deallocate();

	recBlocks.Reset();
	recRAM = recROM = recROM1 = recROM2 = nullptr;

	safe_free(s_pInstCache);
	s_nInstCacheSize = 0;

	recPtr = nullptr;
	recPtrEnd = nullptr;
}

static void recStep()
{
}

static fastjmp_buf m_SetJmp_StateCheck;

static void recExitExecution()
{
	fastjmp_jmp(&m_SetJmp_StateCheck, 1);
}

static void recSafeExitExecution()
{
	eeRecExitRequested = true;

	if (!eeEventTestIsActive)
	{
		cpuRegs.nextEventCycle = 0;
	}
	else
	{
		if (psxRegs.iopCycleEE > 0)
		{
			psxRegs.iopBreak += psxRegs.iopCycleEE;
			psxRegs.iopCycleEE = 0;
		}
	}
}

static void recResetEE()
{
	Console.WriteLn("ARM64 recResetEE: eeCpuExecuting=%d", (int)eeCpuExecuting);
	if (eeCpuExecuting)
	{
		eeRecNeedsReset = true;
		recSafeExitExecution();
		return;
	}

	recResetRaw();
}

static fastjmp_buf m_SetJmp_CancelInstruction;

static void recCancelInstruction()
{
	// An interpreter stub (called via armCallInterpreter) raised an error
	// (Address Error, TLB miss, etc.). The interpreter pre-increments
	// cpuRegs.pc before executing, so armFlushPC already wrote pc+4.
	// Don't exit recExecute — just longjmp back to the dispatch loop so
	// cycles keep accumulating and events fire. This matches the
	// interpreter's intCancelInstruction which longjmps back into its
	// for(;;) loop without exiting intExecute.
	//
	// The x86 JIT marks this as "should never happen" because it compiles
	// loads/stores natively. We hit it because we use interpreter stubs.
	fastjmp_jmp(&m_SetJmp_CancelInstruction, 1);
}

static void recExecute()
{
	// Console.WriteLn("ARM64 recExecute: enter, eeRecNeedsReset=%d EnterRecompiledCode=%p", (int)eeRecNeedsReset, EnterRecompiledCode);
	if (eeRecNeedsReset)
	{
		eeRecNeedsReset = false;
		recResetRaw();
	}

	if (!fastjmp_set(&m_SetJmp_StateCheck))
	{
		eeCpuExecuting = true;

		// Set up the cancel-instruction landing pad. When an interpreter stub
		// calls Cpu->CancelInstruction(), we longjmp here instead of fully
		// exiting recExecute. This keeps us inside the execution loop so we
		// can bump cycles, check events, and re-dispatch — matching the
		// interpreter's behavior where intCancelInstruction re-enters the
		// for(;;) loop.
		if (fastjmp_set(&m_SetJmp_CancelInstruction))
		{
			// Landed here from recCancelInstruction, triggered by an interp
			// stub calling Cpu->CancelInstruction() mid-block (Address Error,
			// TLB miss, etc.). The interp has already written cpuRegs.pc via
			// the pre-increment in execI, so we don't need to restore PC.
			//
			// Invariant on entry: the faulting op's armEmitFlushCycleBeforeCall
			// already wrote `cpuRegs.cycle = nextEventCycle + RCYCLE`, so cycle
			// reflects the block's progress up to (but not including) whatever
			// the interp would have consumed if it had completed normally.
			//
			// Bump by 8 = "one scaled-instruction worth" (DEFAULT_SCALED_BLOCKS
			// right-shifts s_nBlockCycles by 3, so scaling factor is 8×; a base
			// 1-cycle op contributes 8 to cpuRegs.cycle). This guarantees
			// forward progress so we can't loop forever on a persistently
			// faulting address, and gives any hardware-state-dependent events
			// a chance to fire via the check below.
			//
			// The re-dispatch into EnterRecompiledCode reads the current
			// cpuRegs.pc (written by the interp before the cancel) and rebuilds
			// RCYCLE from (cycle - nextEventCycle), restoring the Phase B cycle
			// invariant even after this manual bump.
			cpuRegs.cycle += 8;
			if (cpuRegs.cycle >= cpuRegs.nextEventCycle)
				_cpuEventTest_Shared();

			if (eeRecExitRequested)
			{
				eeRecExitRequested = false;
				goto exit;
			}
		}

		// Console.WriteLn("ARM64 recExecute: jumping to EnterRecompiledCode");
		((void (*)())EnterRecompiledCode)();
	}

exit:
	eeCpuExecuting = false;
	EE::Profiler.Print();
}

static void dyna_block_discard(u32 start, u32 sz)
{
	DevCon.WriteLn(Color_StrongGray, "Clearing Manual Block @ 0x%08X  [size=%d]", start, sz * 4);
	recClear(start, sz);
}

static void dyna_page_reset(u32 start, u32 sz)
{
	recClear(start & ~0xfffUL, 0x400);
	manual_counter[start >> 12]++;
	mmap_MarkCountedRamPage(start);
}

static void recClear(u32 addr, u32 size)
{
	if ((addr) >= maxrecmem || !(recLUT[(addr) >> 16] + (addr & ~0xFFFFUL)))
		return;
	addr = HWADDR(addr);

	int blockidx = recBlocks.LastIndex(addr + size * 4 - 4);

	if (blockidx == -1)
		return;

	u32 lowerextent = static_cast<u32>(-1), upperextent = 0, ceiling = static_cast<u32>(-1);

	BASEBLOCKEX* pexblock = recBlocks[blockidx + 1];
	if (pexblock)
		ceiling = pexblock->startpc;

	int toRemoveLast = blockidx;

	while ((pexblock = recBlocks[blockidx]))
	{
		u32 blockstart = pexblock->startpc;
		u32 blockend = pexblock->startpc + pexblock->size * 4;
		BASEBLOCK* pblock = PC_GETBLOCK(blockstart);

		if (pblock == s_pCurBlock)
		{
			if (toRemoveLast != blockidx)
				recBlocks.Remove((blockidx + 1), toRemoveLast);
			toRemoveLast = --blockidx;
			continue;
		}

		if (blockend <= addr)
		{
			lowerextent = std::max(lowerextent, blockend);
			break;
		}

		lowerextent = std::min(lowerextent, blockstart);
		upperextent = std::max(upperextent, blockend);

		pblock->SetFnptr((uptr)JITCompile);

		blockidx--;
	}

	if (toRemoveLast != blockidx)
		recBlocks.Remove((blockidx + 1), toRemoveLast);

	upperextent = std::min(upperextent, ceiling);

	for (u32 cleared = lowerextent; cleared < upperextent; cleared += 4)
	{
		BASEBLOCK* pblock = PC_GETBLOCK(cleared);
		pblock->SetFnptr((uptr)JITCompile);
	}

	// Invalidate block-link graph over the same range. Any predecessor
	// holding a direct-B into a freed block has its patch site rewritten
	// back to DispatcherReg so subsequent dispatches fall through to the
	// LUT (where the now-JITCompile pointer will recompile). Entries for
	// blocks inside the cleared range are dropped from the registry.
	if (upperextent > lowerextent)
		invalidateEELinks(lowerextent, upperextent);
}

R5900cpu recCpu = {
	recReserve,
	recShutdown,
	recResetEE,
	recStep,
	recExecute,
	recSafeExitExecution,
	recCancelInstruction,
	recClear,
};

