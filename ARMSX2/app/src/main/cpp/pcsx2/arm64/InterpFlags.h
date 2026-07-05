// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0

#pragma once

// Bisect: Commented = native
// Un-comment a group to find a general problem area
// Un-comment indiviudal ops to narrow it down to a specific op
//
// These flags are also consumed by shared headers (Config.h) to keep config
// macros like REC_VU1 / THREAD_VU1 consistent with the JIT bisect state.

//EE
//#define INTERP_EE        // Master
//#define INTERP_BRANCH    // BEQ, BNE, J, JAL, JR, JALR, SYSCALL, BREAK, etc.
//#define INTERP_MOVE      // LUI, MFHI/LO, MTHI/LO, MOVZ, MOVN, MFSA, MTSA, etc.
//#define INTERP_COP0      // MFC0, MTC0, BC0x, TLB*, ERET, EI, DI
//#define INTERP_COP1      // MFC1, MTC1, CFC1, CTC1, BC1x, FPU arith/cmp/cvt
//#define INTERP_COP2      // QMFC2/QMTC2 (native 128b), CFC2/CTC2, BC2x, all VU0 macro math ops
//#define INTERP_ALU       // ADDU, SUBU, ADDIU, DADDU, DSUBU, DADDIU, AND/OR/XOR/NOR, SLT/U, etc.
//#define INTERP_SHIFT     // SLL, SRL, SRA, SLLV, SRLV, SRAV, DSLL/DSRL/DSRA + 32 variants
//#define INTERP_MMI       // All packed SIMD ops: PADD*/PSUB*, PCGT*, PMAX/MIN*, PCEQ*, PABS*, PSxx shifts, etc.
//#define INTERP_LOAD      // LB, LBU, LH, LHU, LW, LWU, LD, LQ, LWL/R, LDL/R, LWC1, LQC2
//#define INTERP_STORE     // SB, SH, SW, SD, SQ, SWL/R, SDL/R, SWC1, SQC2
//#define INTERP_TRAP      // TGEI, TGEIU, TLTI, TLTIU, TEQI, TNEI, TGE, TGEU, TLT, TLTU, TEQ, TNE
//#define INTERP_MULTDIV

//VU0
// Pair-level bisect: comment out a line to force that class of pairs to fall back to vu0Exec.
//#define INTERP_VU0            // Master
//#define INTERP_VU0_PAIR       // Force every pair to fall back to vu0Exec (kills all per-pair native machinery)
//#define INTERP_VU0_HAZARD     // (Dormant — VF/CLIP hazards always fall back; native save/restore not yet implemented)
//#define INTERP_VU0_MBIT       // Fall back to vu0Exec when M-bit (bit 29) is set on the upper instruction
//#define INTERP_VU0_DTBITS     // Fall back to vu0Exec when D-bit or T-bit is set
//#define INTERP_VU0_EBIT       // Fall back to vu0Exec when E-bit is set
//#define INTERP_VU0_BRANCH     // Fall back to vu0Exec when the pair contains a branch lower op


//#define INTERP_VU0_UPPER     // FMAC arith (ADD/SUB/MUL/MADD/MSUB xyzwqi), accum, MAX/MINI, ABS, CLIP, FTOI/ITOF, NOP
//#define INTERP_VU0_LOWER_FDIV       // DIV, SQRT, RSQRT, WAITQ, WAITP
//#define INTERP_VU0_LOWER_IALU       // IADD, ISUB, IADDI, IADDIU, ISUBIU, IAND, IOR
//#define INTERP_VU0_LOWER_LOADSTORE  // LQ, LQD, LQI, SQ, SQD, SQI, ILW, ISW, ILWR, ISWR
//#define INTERP_VU0_LOWER_BRANCH     // B, BAL, JR, JALR, IBEQ, IBNE, IBLTZ, IBGTZ, IBLEZ, IBGEZ
//#define INTERP_VU0_LOWER_MISC       // MOVE, MR32, MFIR, MTIR, MFP, flag ops, random, EFU, XITOP, XTOP, XGKICK

//VU1
//#define INTERP_VU1            // Master
//#define INTERP_VU_UPPER      // FMAC arith (ADD/SUB/MUL/MADD/MSUB xyzwqi), accum, MAX/MINI, ABS, CLIP, FTOI/ITOF, NOP
//#define INTERP_VU_FDIV       // DIV, SQRT, RSQRT, WAITQ, WAITP
//#define INTERP_VU_IALU       // IADD, ISUB, IADDI, IADDIU, ISUBIU, IAND, IOR
//#define INTERP_VU_LOADSTORE  // LQ, LQD, LQI, SQ, SQD, SQI, ILW, ISW, ILWR, ISWR
//#define INTERP_VU_BRANCH     // B, BAL, JR, JALR, IBEQ, IBNE, IBLTZ, IBGTZ, IBLEZ, IBGEZ
//#define INTERP_VU_MISC       // MOVE, MR32, MFIR, MTIR, MFP, flag ops, random, EFU, XITOP, XTOP, XGKICK

// Diagnostic: emit per-block exec counter at VU1 block linkEntry and dump the
// top-20 hottest blocks (with per-pair disassembly) on per-game shutdown —
// fires from VMManager::Shutdown via recArmVU1::DumpProfile, so the overlay's
// Exit button surfaces the dump (not just full app exit). Adds ~5 insns per
// block entry (measurable cost on linked-chain entries), keep off unless profiling.
//#define VU1_PROFILE_BLOCKS

// EXPERIMENT: force cpuRegs.cycle writeback to memory at every EE block-end
// iBranchTest path (both linked and unlinked exits). Native EE branches keep
// cycle as a delta in RCYCLE (x19); cpuRegs.cycle in memory only updates when
// DispatcherEvent fires (event budget exhausted). On long direct-B linked
// chains, that's many blocks of stale memory — async hardware (VU1, GIF,
// MTGS) reading cpuRegs.cycle through the EE↔VU1 handshake sees an outdated
// "now" and may decide nothing's due, even when work has piled up.
//
// armBranchCallInterpreter (used under INTERP_BRANCH) writes cycle to memory
// AND sets nec=cycle, forcing event dispatch on the very next iBranchTest.
// armCallInterpreter (used under INTERP_LOAD) writes cycle to memory before
// each BL. Either flag alone restores enough cycle-coherence to unblock
// async work.
//
// Hypothesis under test: Crash Twinsanity main-menu ocean water requires
// either INTERP_BRANCH or INTERP_LOAD enabled to render. With both native,
// the water vanishes (Cortex floats in nothing). Theory: the EE↔VU1 hand-
// shake that triggers ocean upload reads cpuRegs.cycle and decides "no work"
// because RCYCLE-in-register hasn't been flushed.
//
// Add: 3 insns (Ldr nec / Add cycle = nec+RCYCLE / Str cycle) at every block
// end. Roughly +1-2% block-end overhead. Toggle in CompileBlock-ish path.
//#define EE_FORCE_CYCLE_FLUSH


// Diagnostic: shadow-compare every native VU0 pair against a parallel
// _vu0Exec interp run on a snapshot of VU0 state. Logs the first divergent
// field per pair (jit value vs interp value + pc) to Console.Error. Fires
// only on the native pair path (fallback pairs use vu0Exec themselves and
// would always agree). Slows VU0 emulation drastically — debug-only.
// Use this to localize "JIT pair body produces wrong VF/ACC/macflag"
// bugs that don't show up in INTERP_VU0_PAIR mode.
//#define VU0_SHADOW_VERIFY

// Diagnostic: gate the shadow-verify comparison to a cycle window so the
// harness doesn't tank perf during long boot sequences. Uncomment ONE
// (or both) to limit when the comparison fires. Snapshots/verify
// emit ALWAYS at compile time; the GATE is at runtime in vu0_shadow_verify
// — out-of-window pairs skip the vu0Exec call + memcmp entirely.
//
//   FROM: skip until VU->cycle reaches this value. Use to bypass init / BIOS.
//   TO:   stop after VU->cycle exceeds this value. Use to abort once past
//         the area of interest.
//
// Find your target cycle via the OSD (no exact cycle counter shown there)
// or by enabling shadow-verify with a permissive window first, noting the
// cycle in the divergence log, then narrowing on the next run.
//#define VU0_SHADOW_VERIFY_FROM_CYCLE 24148000000ULL
//#define VU0_SHADOW_VERIFY_TO_CYCLE   24148999999ULL

// Diagnostic: VU1 shadow-compare. Mirrors the VU0 design — pre-pair snapshot
// of VURegs+VU1.Mem(16KB), post-pair restore + interp re-run via vu1Exec,
// field-by-field compare, halt-on-first-divergence with state dump and
// native backtrace. Use this to validate Phase-2-deferred-VF-write attempts
// or any other JIT-body change suspected of producing wrong per-pair output.
//
// Constraints:
//   - MUST run with THREAD_VU1 = false (MTVU off). The harness re-runs
//     interp on the same thread that ran JIT; under MTVU the comparison
//     would race with cross-thread state mutations.
//   - XGKICK pairs are SKIPPED — re-running interp would duplicate GIF
//     writes. Tradeoff: vertex-upload pairs aren't validated by the harness.
//   - Hazard-fallback pairs (VF/CLIP read-after-write that go through
//     vu1Exec) are NOT verified — interp result trivially equals JIT.
//   - Slows VU1 emulation drastically. Debug-only.
//
// Recommended workflow when chasing a Phase-2 deferred-write coherence bug:
//   1. Set EmuConfig.Speedhacks.vuThread = false (MTVU off) in your run.
//   2. Rebuild with VU1_SHADOW_VERIFY defined.
//   3. Reproduce the symptom. The harness aborts on the first per-pair
//      divergence and dumps PRE / JIT / INTERP state for the offending pair.
//   4. The first divergent field maps to the JIT bug (e.g. VF[N] mismatch
//      → wrong writeback for that pair's upper op).
//#define VU1_SHADOW_VERIFY

// Cycle-window gate for VU1_SHADOW_VERIFY. Same shape as the VU0 version —
// snapshot and compare are emitted unconditionally; the runtime gate inside
// vu1_shadow_verify skips the vu1Exec re-run + memcmp when VU1.cycle is
// outside [FROM, TO]. Use to bypass long init/menu boots when reproducing.
//#define VU1_SHADOW_VERIFY_FROM_CYCLE 0ULL
//#define VU1_SHADOW_VERIFY_TO_CYCLE   0ULL

// Phase 2: VU1 deferred VF writes. The current Phase 2.5 cache is write-
// through — every FMAC writeback emits an immediate Str q to VF memory
// alongside the slot update. With deferred writes enabled, the Str is
// elided and the slot is marked dirty; the actual memory write is deferred
// to the next flush site (LRU eviction, BL via vfCacheFlushAndInvalidate,
// inline-bypass ops via vfCacheFlushOne, block end). Saves 1 Str per FMAC
// pair when consecutive ops touch the same VF.
//
// HISTORY: a prior attempt was reverted with an unidentified coherence bug
// (GoW2 graphics dropped — see armsx2_vu_regalloc_gap memory). The bug
// surfaced as silent corruption with no specific PC. Re-enabling now is
// gated on running with VU1_SHADOW_VERIFY too — the harness will halt at
// the first divergent pair, exposing whatever bypassed the cache flush.
//
// Recommended workflow:
//   1. Set EmuConfig.Speedhacks.vuThread = false (MTVU off — required by
//      the shadow harness).
//   2. Uncomment BOTH VU1_DEFER_VF_WRITES and VU1_SHADOW_VERIFY below.
//   3. Rebuild + run a transform-heavy scene (GoW2 / FFX battle / racing).
//   4. The first divergent pair surfaces in logcat with full PRE/JIT/INTERP
//      dump — fix the missed flush site, repeat.
//
// Once the harness has gone silent through GoW2 specifically (the prior
// regression), Phase 2 ships safely. Until then, leave commented out.
//#define VU1_DEFER_VF_WRITES

// VU1 BLOCK-LEVEL shadow verify. Companion to VU1_SHADOW_VERIFY (per-pair).
//
// Per-pair shadow flushes the VF (NEON) cache before each snapshot/verify
// BL — that's correct for catching per-pair semantic bugs but it MASKS
// cross-pair coherence bugs. Specifically: under VU1_DEFER_VF_WRITES, pair
// K's FMAC writeback stays in a NEON cache slot until a flush site fires.
// If pair K+M reads VF[X] memory directly (e.g. inline Ldr in MTIR / MFP /
// MFIR / LQ family) without a paired vfCacheFlushOne(X), it sees stale
// memory. Per-pair shadow's pre-snapshot flush forces memory coherent at
// every pair boundary, so K+M's emit reads correct memory under shadow but
// stale memory in real execution. Bug invisible to per-pair.
//
// Block-level shadow runs the WHOLE block end-to-end with the cache living
// across pairs (matching real execution), then compares end-of-block state.
// Block epilogue's vfCacheFlushAndInvalidate commits all deferred writes
// before snapshot-post. A coherence bug shows up as a divergent VF/VI/Mem
// in the end state.
//
// Use when:
// - Per-pair shadow runs silent through a graphical bug (e.g., SH2 under
//   VU1_DEFER_VF_WRITES) that ONLY manifests in real execution.
// - You need to catch missed vfCacheFlushOne / vfCacheInvalidate sites in
//   the JIT's inline-emit lower ops.
//
// Requires VU1_SHADOW_VERIFY also defined (shares the divergence latch and
// log helpers). MTVU must be off.
//#define VU1_BLOCK_SHADOW_VERIFY

// Diagnostic: TPC-range fallback bisection. Pairs whose pc falls in
// [INTERP_VU0_PC_LOW, INTERP_VU0_PC_HIGH] route to vu0Exec; others go
// through the native JIT body. Use to BINARY-SEARCH a buggy pair when
// INTERP_VU0_PAIR fixes the bug but per-pair shadow-verify is silent.
//
// Workflow:
//   1. Set range = whole 4KB program (LOW=0, HIGH=0xFFF). Verify physics
//      are correct → equivalent to INTERP_VU0_PAIR.
//   2. Halve the range. Re-run. If physics break, the bug is in the OTHER
//      half. Otherwise it's in this half.
//   3. Repeat until you've narrowed to ~8 bytes (one pair).
//   4. Disable VU0_SHADOW_VERIFY and INTERP_VU0_PC range; manually inspect
//      the JIT emit for that pair against x86 microVU.
//
// The bug is in a pair the harness can't observe — likely a hazard pair
// fallback (the harness skips fallback pairs entirely) or external state
// the per-pair compare doesn't capture (vif0Regs, EE memory, etc.).
// Symptom #1 (MGS2 ledge teleport + back-facing invisible walls): pinned
// to pc=0x4D8 — keep this range stable across symptom #2 bisect rounds.
//#define INTERP_VU0_PC_LOW  0x04D8u
//#define INTERP_VU0_PC_HIGH 0x04DFu

// Symptom #2 (MGS2 bullets hit invisible walls, cannot descend stairs):
// INTERACTING-PAIR bug at pc=0x168 + pc=0x170. Routing either pair alone
// is NOT enough — both must take the vu0Exec fallback simultaneously.
// Bisect cascade summary:
//   0x140-0x17F OK; 0x140-0x15F BROKEN; 0x160-0x17F OK
//   0x160-0x16F BROKEN; 0x170-0x17F BROKEN; 0x168-0x17F OK
//   0x168-0x177 OK (covers both 0x168 + 0x170 pairs, drops 0x178)
// → both 0x168 and 0x170 pairs are critical and interact. Likely one
//   writes a VF/VI/ACC value the other reads, with a JIT-emit bug at the
//   data handoff. To investigate: disassemble VU0 micro at:
//     pc=0x168: lower at Micro[0x168], upper at Micro[0x16C]
//     pc=0x170: lower at Micro[0x170], upper at Micro[0x174]
//   and audit the JIT emit vs x86 microVU.
//
// Range 2 below pins both pairs through interp until the JIT bug is fixed.
// Range1 (above) pins symptom #1 at 0x4D8.
//#define INTERP_VU0_PC_LOW2  0x0168u
//#define INTERP_VU0_PC_HIGH2 0x0177u

// Diagnostic: per-op JIT symbol registration with simpleperf. When defined,
// each emitted op is registered as a separate symbol (e.g. `EE_OP_lui_0x123`,
// `VU1_U_05_0x0040`). The next simpleperf trace then attributes samples
// directly to specific MIPS/VU op emits inside hot blocks.
//
// Cost: one snprintf + one Perf::Register call per op AT COMPILE TIME (not
// per execution). For typical workloads this is a few thousand calls per
// block-cache-fill, negligible compared to actual emit cost. The runtime
// JIT'd code is unchanged. Each Register call appends a line to the
// /data/local/tmp/perf-<PID>.map file (~50 bytes/op), so a long session
// can grow that file to several MB — flush the map when starting a new
// trace if needed.
//
// Recommended workflow: enable one or two of these at a time, rebuild,
// record perf.data, analyze, disable. Don't ship with these on.
//#define EE_PROFILE_OPS
//#define IOP_PROFILE_OPS
//#define VU0_PROFILE_OPS
//#define VU1_PROFILE_OPS

//DMAC
//#define INTERP_DMAC          // VIF0, VIF1, GIF, IPU0/1, SIF0/1/2, SPR0/1 + interrupt handlers

// Diagnostic: IOP shadow-compare. Mirrors the VU0/VU1 design — pre-block
// snapshot of psxRegs, post-block snapshot, restore + interp re-run via
// psxInterpReplayBlock, register-by-register compare, halt-on-first-
// divergence with PRE/JIT/INTERP dump and native backtrace. Use this to
// validate IOP JIT codegen changes (e.g. const-prop tightening, new ALU
// op natives, or LWL/LWR/SWL/SWR fastpaths) against the interpreter.
//
// Constraints:
//   - Validates GPR + CP0 + pc + hi/lo. CYCLE-RELATED FIELDS ARE MASKED
//     (cycle, iopCycleEE, iopBreak, iopNextEventCycle, sCycle/eCycle,
//     interrupt, pcWriteback) — JIT uses block-end batched cycle add,
//     interp uses per-instruction increment, so they diverge by-design.
//     CP2D/CP2C (GTE) is also masked since GTE is interp-stubbed.
//   - Memory (iopMem->Main, iopHw, scratchpad) is NOT rolled back. Blocks
//     that write to HW registers will see double-writes during interp
//     replay; for those blocks the harness may go silent or fire
//     spuriously. Best for ALU/branch-heavy bug investigation.
//   - Slows IOP emulation drastically. Debug-only.
//
// Recommended workflow when chasing an IOP JIT codegen bug:
//   1. Rebuild with IOP_SHADOW_VERIFY defined.
//   2. Reproduce the symptom. The harness aborts on the first per-block
//      divergence and dumps PRE / JIT / INTERP state for the offending
//      block.
//   3. The first divergent field maps to the JIT bug (e.g. GPR[t0]
//      mismatch → wrong codegen for whatever op writes t0 in that block).
//#define IOP_SHADOW_VERIFY

// Cycle-window gate for IOP_SHADOW_VERIFY. Same shape as VU0/VU1 — both
// snapshot and verify BLs are emitted unconditionally; the runtime gate
// inside iop_shadow_verify skips the interp re-run + memcmp when
// psxRegs.cycle is outside [FROM, TO]. Use to bypass long boot sequences
// when reproducing a specific scene.
//#define IOP_SHADOW_VERIFY_FROM_CYCLE 0ULL
//#define IOP_SHADOW_VERIFY_TO_CYCLE   0ULL

// PC-range skip for per-block IOP shadow. Blocks whose startpc falls in
// [LO, HI) are excluded from the per-block comparison. Use to silence
// known false positives — typically iopMem-loop blocks like 0xbfc4ab68
// (LW r5,0(r6) / ADDU r5,r5,r16 / SW r5,0(r6) increment a counter in
// memory; the replay reads the JIT's already-stored value and computes a
// one-iteration-ahead result). Per-instruction shadow is unaffected by
// this skip and stays accurate on those same blocks.
//
// Define both macros to enable. Example to skip the IOPBOOT counter block:
//#define IOP_SHADOW_VERIFY_SKIP_PC_LO 0xbfc4ab68u
//#define IOP_SHADOW_VERIFY_SKIP_PC_HI 0xbfc4ab80u

// Per-instruction shadow for IOP. Requires IOP_SHADOW_VERIFY also defined
// (shares halt path / divergence latch). When per-block divergence reports
// the block but not the instruction, this narrows it: snapshot before each
// non-branch op emit, run ONE execI from the snapshot post-emit, compare.
// Halts at the EXACT pc of the divergent op. Doesn't fire on branch ops
// (their codegen exits to dispatcher before reaching the post-emit BL) —
// branch-codegen bugs still surface via per-block shadow.
//
// Cost: 2 BLs per non-branch op (snapshot + verify). Per-block shadow
// already costs O(block) to replay; per-instruction adds another factor
// of N instructions × 1 execI per BL. Slow but precise — debug only.
//#define IOP_SHADOW_VERIFY_PER_INSTR

//IOP
// INTERP_IOP master: flips ALL sub-categories to interp stubs AND disables
// delay-slot swap (psxTrySwapDelaySlot returns false unconditionally). This
// makes the IOP JIT a thin dispatcher around the interpreter — every op
// goes through iopArmCallInterpreter(psxXxx). Const-prop is effectively
// disabled (resets after each BL), execution order matches MIPS+interp
// exactly. Use for shadow-verify when chasing native-only hangs/bugs:
// running native+shadow with INTERP_IOP turned on means JIT side is
// effectively interp, divergences should be near-zero. To narrow: turn
// INTERP_IOP off and bring sub-categories back to native one at a time.
//#define INTERP_IOP             // Master
//#define INTERP_IOP_ALU         // BISECT: uncommented → per-instruction ISTUBs in iR3000Atables_arm64.cpp
//#define INTERP_IOP_BRANCH      // BEQ/BNE/BLEZ/BGTZ/BLTZ/BGEZ/BLTZAL/BGEZAL/J/JAL/JR/JALR
//#define INTERP_IOP_SHIFT       // SLL/SRL/SRA/SLLV/SRLV/SRAV
//#define INTERP_IOP_MULTDIV     // MULT/MULTU/DIV/DIVU/MFHI/MTHI/MFLO/MTLO
//#define INTERP_IOP_MOVE        // MFHI/MTHI/MFLO/MTLO
//#define INTERP_IOP_LOADSTORE   // LB/LBU/LH/LHU/LW/LWL/LWR/SB/SH/SW/SWL/SWR
//#define INTERP_IOP_COP0        // MFC0/MTC0/CFC0/CTC0/RFE
#define INTERP_IOP_COP2        // All GTE (keep stubbed)
#define INTERP_IOP_SYSTEM      // SYSCALL/BREAK

// =====================  PS1DRV / PS1-mode tracing toggles  =====================
// Per-category PS1-mode runtime tracing. Off by default. Each toggle gates a
// small set of `Console.WriteLn("[PS1DRV-<cat>] ...")` calls in specific
// files. Captured via `adb logcat -s STDOUT:W` from a connected device.
//
// All sites use the rate-limit helper in PS1DrvTrace.h so noisy events
// (every-frame stuff) don't flood the log; one-shot transitions log freely.
//
// Workflow:
//   1. Uncomment the relevant toggle below.
//   2. Rebuild + redeploy.
//   3. Reproduce the bug; capture logs.
//   4. Re-comment when done.
//
// Convention: tag is [PS1DRV-CDROM], [PS1DRV-DMA], [PS1DRV-IRQ], etc. Match
// the toggle name's suffix. Filter logs with: adb logcat -s STDOUT:W | grep PS1DRV
//#define PS1DRV_TRACE_CDROM        // cdrInterrupt / cdrReadInterrupt state transitions, RErr, IRQ delivery
//#define PS1DRV_TRACE_DMA          // psxDma0/1/2 entry/exit, deferred completion
//#define PS1DRV_TRACE_IRQ          // iopIntcIrq fires, ISTAT writes, IMASK writes
//#define PS1DRV_TRACE_GPU          // GP0/GP1 commands flowing through pgif
//#define PS1DRV_TRACE_SIO            // SIO0 (controller/memcard) transactions
//#define PS1DRV_TRACE_MDEC         // MDEC command/data writes, DMA completions
