// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU1 microIR.
//
// Mirrors the data model of pcsx2/x86/microVU_IR.h so the analyzer + emit
// passes can be ported mechanically from x86 microVU. The arm port's
// pre-existing peephole-derived fields (mac_cluster_*, batch_*, etc.) are
// preserved as extensions on top of the upstream data model — they remain
// the arm port's source of truth for NEON-specific fusions while the
// upstream-shaped fields (uOp / lOp / sFlag / mFlag / cFlag) host the work
// that needs to come back: real flag-instance pipelining, Q/P instance
// tracking, microRegInfo block-match keying, and per-op stall counting.
//
// Lifecycle: zeroed at CompileBlock entry, populated by mvu1AnalyzeBlock
// once the _VURegsNum arrays are filled, consumed by the emit pass.

#pragma once

#include "VUmicro.h"

namespace armvu1ir
{

// ============================================================================
//  Branch encoding — mirrors microLowerOp::branch in x86 microVU.
// ============================================================================
// Keeps a single vocabulary for "what kind of branch is this lower" across
// all consumers. The numeric values match x86 microLowerOp::branch so any
// ported analyzer code that compares `branch == 2` for BAL keeps working.
enum BranchKind : u8
{
	BR_NONE  = 0,
	BR_B     = 1,  // unconditional B
	BR_BAL   = 2,  // unconditional with link
	BR_IBEQ  = 3,
	BR_IBNE  = 4,
	BR_IBLTZ = 5,
	BR_IBGTZ = 6,
	BR_IBLEZ = 7,
	BR_IBGEZ = 8,
	BR_JR    = 9,  // indirect (target = VI[Is])
	BR_JALR  = 10, // indirect with link
};

// ============================================================================
//  Upstream-mirroring building blocks
// ============================================================================
// These structs match x86 microVU_IR.h field-for-field so ported analyzer
// helpers (mVUanalyzeFMAC1/2/3/4, analyzeReg1..6, analyzeBranchVI, etc.)
// can be transcribed mechanically.

// VF reg with per-lane read/write tracking. analyzeReg* mutate one of these
// in-place: read paths set the lane to 1, write paths set it to 4 (matches
// the cycle-decrement encoding the pipeline-state tracker expects).
struct microVFreg
{
	u8 reg; // Reg index (0..31)
	u8 x;
	u8 y;
	u8 z;
	u8 w;
};

// VI reg with used-as-read or used-as-write counter.
struct microVIreg
{
	u8 reg;  // Reg index (0..15 cacheable, 16..31 special)
	u8 used; // 1 for read, aCycles for write
};

// Const-prop info per VI reg. Populated by analyzeIADDI / analyzeJump etc.
// Consumed by analyzeJump to enable constJump.isValid → JIT skips the
// VI[Is] read and emits a direct B/BL to the constant target.
struct microConstInfo
{
	u8  isValid;
	u32 regValue;
};

// Flag instance ring slot — one per pair, per flag (s/m/c). The instance
// ring lets the JIT emit ONE flag store per "next reader", not one per
// FMAC writeback. arm port currently emits flags unconditionally; filling
// this struct enables future skip-on-no-reader gating.
struct microFlagInst
{
	bool doFlag;       // Update flag on this instruction
	bool doNonSticky;  // Update non-sticky (O/U/S/Z) on this instruction (status only)
	u8   write;        // Instance the s-stage write targets
	u8   lastWrite;    // Most-up-to-date instance (used by mid-block lookups)
	u8   read;         // Instance the t-stage read sources from
};

// Per-block flag-cycle bookkeeping used by mVUsetFlags-equivalent passes.
// xStatus/xMac/xClip slots track which cycle each instance was written at.
struct microFlagCycles
{
	int xStatus[4];
	int xMac[4];
	int xClip[4];
	int cycles;
};

// Per-pair upper-op summary. Drives FMAC emit decisions + flag elision.
struct microUpperOp
{
	bool eBit;
	bool iBit;
	bool mBit;
	bool tBit;
	bool dBit;
	microVFreg VF_write;
	microVFreg VF_read[2];
};

// Per-pair lower-op summary. Mirrors x86 microLowerOp exactly; the numeric
// `branch` field shadows the top-level BranchKind so ported analyzers don't
// need re-targeting.
struct microLowerOp
{
	microVFreg VF_write;
	microVFreg VF_read[2];
	microVIreg VI_write;
	microVIreg VI_read[2];
	microConstInfo constJump;
	u32  branch;
	u32  kickcycles;
	bool badBranch;
	bool evilBranch;
	bool isNOP;
	bool isFSSET;
	bool noWriteVF;
	bool backupVI;
	bool memReadIs;
	bool memReadIt;
	bool readFlags;
	bool isMemWrite;
	bool isKick;
};

// Per-pair pipeline-state slot used by the analyzer for in-flight scratch
// (the upper and lower of the SAME pair don't observe each other's writes,
// so the analyzer accumulates VF/VI/Q/P cycle deltas here and folds them
// into the post-pair pipeline state at end-of-pair).
struct regCycleInfo
{
	u8 x : 4;
	u8 y : 4;
	u8 z : 4;
	u8 w : 4;
};

struct microTempRegInfo
{
	regCycleInfo VF[2]; // [0] = upper write, [1] = lower write
	u8 VFreg[2];
	u8 VI;
	u8 VIreg;
	u8 q;
	u8 p;
	u8 r;
	u8 xgkick;
};

// Pipeline state at block boundaries. Used as the block-match key for the
// VU1 block cache (variant-cache content key). 96 bytes, 16-byte aligned so
// the host compare path can use vector compares.
//
// NOTE: porting microVU's content-keyed block cache to arm64 is a future
// slice. Today the arm port matches blocks by startPC alone, with a single
// "best variant" deque per slot. Holding microRegInfo in microIR enables
// the future move to upstream-style needExactMatch / pState compares.
union alignas(16) microRegInfo
{
	struct
	{
		union
		{
			struct
			{
				u8 needExactMatch; // If set, block needs exact match of pipeline state
				u8 flagInfo;       // xC*2 | xM*2 | xS*2 | 0*1 | fullFlag Valid*1
				u8 q;
				u8 p;
				u8 xgkick;
				u8 viBackUp;       // VI reg written in branch-delay slot
				u8 blockType;      // 0 = Normal; 1,2 = single-pair (E-bit/Branch Ending)
				u8 r;
			};
			u64 quick64[1];
			u32 quick32[2];
		};

		u32 xgkickcycles;
		u8 unused;
		u8 vi15v; // 'vi15' constant is valid
		u16 vi15; // Const-prop info for vi15

		struct
		{
			u8 VI[16];
			regCycleInfo VF[32];
		};
	};

	u128 full128[96 / sizeof(u128)];
	u64  full64[96 / sizeof(u64)];
	u32  full32[96 / sizeof(u32)];
};
static_assert(sizeof(microRegInfo) == 96, "microRegInfo was not 96 bytes");

// ============================================================================
//  Per-pair record (microOp)
// ============================================================================
// Compact layout, alignas(16) so the emit loop can prefetch ahead cheaply.
// Holds BOTH the arm port's existing per-pair derived flags (peephole gates,
// hazard summary, cluster-fusion tags) AND the upstream x86 microOp fields
// (uOp / lOp / sFlag / mFlag / cFlag / stall / Q/P instances). The two
// halves coexist while emit-side consumers migrate one-by-one off the arm
// shorthand and onto the upstream-shaped views.
struct alignas(16) microOp
{
	// ------- Raw context -------
	u32  upper;
	u32  lower;
	u32  pc;            // Block-relative PC of this pair (== lower's PC).
	BranchKind branch;  // BR_NONE if no branch in lower.

	// ------- E/I/M/T/D bits (arm-shorthand — duplicated in uOp for ported code) -------
	bool eBit;
	bool iBit;
	bool mBit;
	bool tBit;
	bool dBit;

	// ------- Block position flags -------
	bool isEOB;     // Last pair in the block.
	bool isBdelay;  // This pair is the delay slot of an earlier branch.

	// ------- Lower-pipe classification (arm-shorthand; duplicated in lOp) -------
	bool isNOP;       // Lower decoded to a NOP (or an I-bit's missing lower).
	bool isFSSET;     // Lower is FSSET.
	bool isFlagRead;  // Lower is FSAND/FSEQ/FSOR/FMAND/FMEQ/FMOR/FCAND/FCEQ/FCOR/FCGET.
	bool isMemWrite;  // Lower is SQ/SQI/SQD.
	bool isKick;      // Lower is XGKICK.

	// ------- Hazard summary -------
	// Lane-aware: vf_write_collision OR vf_read_after_write means the pair's
	// upper writes a VF lane that the lower also touches (write or read).
	// CompileBlock's hazard gate routes these to vu1Exec.
	bool vf_write_collision;
	bool vf_read_after_write;
	bool clip_write_collision;
	bool clip_read_after_write;

	// swapOps: set when the lower is a flag-reader op writing a non-zero
	// VI target — those need to read s/m/c flag instance from BEFORE the
	// upper writes its new flag. Currently informational on arm (no flag-
	// instance pipelining yet); the doSwapOp native fast-path uses it.
	bool swapOps;

	// noWriteVF / backupVF — reserved for native handling of those cases.
	bool noWriteVF;
	bool backupVF;

	// analyzeBranchVI (audit item #12): when set, this pair writes a VI
	// that some downstream branch (within 4 pairs) reads. The VI-writing
	// emit must call emitBackupVI to snapshot the OLD value before
	// overwriting. When false, the emitBackupVI BL is elided.
	bool needs_vi_backup;

	// Dead VF write elision (FMAC opt #14).
	bool dead_vf_write_upper;
	bool dead_vf_write_lower;

	// Same-VF different-lane batching (FMAC opt #17).
	bool batch_with_next;
	bool batch_from_prev;

	// ABS-of-known-positive elimination (FMAC opt #4).
	bool abs_src_known_non_neg;

	// FMAC opt #19: matrix-vector MAC cluster fusion (MULAx → MADDAy →
	// MADDAz → MADDw). lead pair emits the entire 4-pair NEON sequence;
	// the 3 follower pairs no-op at emit time.
	bool mac_cluster_lead;
	bool mac_cluster_member;
	bool mac_cluster_xyz_only;

	// FMAC opt #20: OPMULA+OPMSUB 2-pair cross-product cluster.
	bool opmac_cluster_lead;
	bool opmac_cluster_member;

	// ------- x86-port additions (upstream microOp fields) -------
	// Populated by the upstream-mirroring analyzer. Consumers that get
	// migrated off the arm shorthand read from these directly.
	u8   stall;        // Stall cycles introduced by this pair.
	bool isBadOp;      // Bad opcode (not a legal VU1 instruction).
	bool doXGKICK;     // Do XGKICK transfer on this instruction.
	u32  XGKICKPC;     // PC the XGKICK fired at (early-exit safety).
	bool doDivFlag;    // Transfer div flag to status on this instruction.
	int  readQ;        // Q instance read by this pair.
	int  writeQ;       // Q instance written by this pair.
	int  readP;        // P instance read by this pair.
	int  writeP;       // P instance written by this pair.

	microFlagInst sFlag; // Status flag instance info.
	microFlagInst mFlag; // MAC    flag instance info.
	microFlagInst cFlag; // Clip   flag instance info.

	microUpperOp uOp;    // Upper op classification (full upstream shape).
	microLowerOp lOp;    // Lower op classification (full upstream shape).
};

// ============================================================================
//  Block-level IR
// ============================================================================
// Lives alongside the existing skip_info[] / pair_needs_flags[] arrays in
// CompileBlock. info[] is caller-owned storage sized to VU1_MAX_BLOCK_PAIRS
// to avoid a heap alloc on the hot compile path.
//
// VU1_MAX_BLOCK_PAIRS is defined in microVU.cpp. We don't import it here
// to keep the header dependency-free; the consumer in the .cpp passes an
// externally-sized array via the analyze entry point.
struct microIR
{
	microOp* info;   // Caller-owned storage, sized to numPairs.
	u32      count;  // == numPairs for the block.
	u32      startPC;

	// Block-level summary flags (mirror the existing block_has_* locals
	// in CompileBlock — exposed here so future passes can consume them).
	bool has_ebit;
	bool has_branch;
	bool has_dbit_or_tbit;
	bool has_ibxx;
	bool has_vi_backup_set;
	bool has_xgkick;

	// ------- x86-port additions -------
	// VI const-prop within the block. Populated by analyzeIADDI etc.,
	// consumed by analyzeJump to enable constJump fast-path.
	microConstInfo constReg[16];

	// Pipeline state at block start and end. block-end is needed by
	// JR/JALR's per-target variant cache (jumpCache equivalent). Today
	// these are storage-only; consumers in a future slice.
	microRegInfo block;
	microRegInfo blockEnd;

	// Per-pair scratch pipeline state (analyzer-internal).
	microTempRegInfo regsTemp;

	// Block-aggregate cycle count + sFlag-hack viability.
	u32 cycles;
	u32 sFlagHack;

	// Per-block branch summary (matches microIR::branch in x86).
	u8 branchKind;
};

// ============================================================================
//  Pass 1 — analyze
// ============================================================================
//
// Populates `ir.info[0..numPairs-1]` from the already-filled _VURegsNum
// arrays (uregs_data / lregs_data) plus the raw VU1.Micro words, then runs
// the upstream-mirroring flag-instance / Q/P-instance / stall walks to
// fill the x86-port additions (uOp, lOp, sFlag, mFlag, cFlag, stall, etc.).
//
// The arm64 CompileBlock today walks the block twice for analysis (once
// for the block_has_* flags, once for skip_info). This entry point is a
// third independent walk for now — it's cheap (~256 pairs max) and additive.
// A future cleanup can fold the three walks together once all consumers are
// migrated to the IR.

void mvu1AnalyzeBlock(
	u32 startPC,
	u32 numPairs,
	const _VURegsNum* uregs_data,
	const _VURegsNum* lregs_data,
	microIR& ir);

} // namespace armvu1ir
