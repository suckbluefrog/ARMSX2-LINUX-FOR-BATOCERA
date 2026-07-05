// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU1 Recompiler — Upper Instruction Stubs
// FMAC arithmetic (ADD/SUB/MUL/MADD/MSUB with xyzwqi broadcasts),
// accumulator variants (ADDA/SUBA/MULA/MADDA/MSUBA),
// MAX, MINI, ABS, CLIP, FTOI/ITOF, OPMULA, OPMSUB, NOP

// Included from aVU.cpp. All headers and `using namespace vixl::aarch64;` are
// already in effect there; VU1_*_REG / vfOff / viOff are also defined in the
// parent. accOff / vfOffStatic0 / vfIndexFromDstOff are Upper-only and stay
// here.

// ============================================================================
//  Flag-deferral state
//
//  Set by CompileBlock before each recVU1_UpperTable[] dispatch. When false,
//  emitBinaryFmac/emitTernaryFmac skip the BL vu1_fmac_writeback (which costs
//  ~50ns per FMAC for MAC/STAT flag updates) and instead emit an inline
//  NEON clamp + store. The analysis is performed in iVU1micro_arm64.cpp:
//  an FMAC pair "needs flags" if either (a) it's one of the last 4 FMAC
//  pairs in the block (cross-block conservative — flags must reach VI[FLAG]
//  for the next block to read them) or (b) some later pair in the same
//  block reads MAC/STATUS/CLIP via FMxxx/FSxxx/FCxxx.
//
//  Default is true (safe / pre-existing behavior) for any path that bypasses
//  the analysis (e.g., direct interpreter execution).
// ============================================================================
bool g_vu1NeedsFlags = true;

// ============================================================================
//  U/O flag-bits gate
//
//  Set by CompileBlock once per block. When false, emitFmacInlineWriteback
//  skips the per-lane denormal (U) / inf (O) detection in Step A, the
//  denormal-flush in Step B, the U/O lane-pack ops in Step C, and the U/O
//  status-bit Tst/Cset/Orr in Step F. Net savings: ~12 NEON + ~6 GPR insn
//  per writeback.
//
//  Block-level analysis: true iff any in-block op has VIread of MAC_FLAG or
//  STATUS_FLAG, OR `CHECK_VU_OVERFLOW(1)` (gamefix forces full computation
//  for the inf-clamp path), OR vuFlagHack is off (exact mode).
//
//  Cross-block soundness: this matches OLD's mVUupdateFlags semantics —
//  upstream gates the entire O-bit ladder behind CHECK_VUOVERFLOWHACK and
//  never computes U at all. Games that read MAC/STATUS only in cross-block
//  fashion get the same OLD-equivalent semantic gap (very rare; CFC2/FMxxx
//  on flag regs is uncommon).
//
//  Default true matches pre-existing behavior for any path that bypasses
//  the per-block analysis (e.g., direct interpreter execution).
// ============================================================================
bool g_vu1NeedsUOFlags = true;

// ============================================================================
//  Dead VF write elision (FMAC opt #14)
//
//  Set per-pair-per-emit by CompileBlock: true iff this op's VF write is
//  proven dead by Pass 1 (forward-walk found a covering overwrite within
//  4 pairs with no intervening read of any live lane). When true,
//  emitFmacInlineWriteback / emitFmacStoreMasked still compute flags but
//  skip the VF cache store (saves 1 force-load + 1-4 lane Movs + 1-4 Strs
//  per dead writer; flag pipeline stays correct).
//
//  Default false (safe / pre-existing behavior). ACC writes are NEVER
//  treated as dead — the pinned VU1_ACC_REG persists across pairs and
//  the analysis only marks VF (not ACC) writes.
// ============================================================================
bool g_vu1DeadVFWrite = false;

// ============================================================================
//  Same-VF different-lane batching (FMAC opt #17)
//
//  Pass 1 detects adjacent FMAC pairs writing the same VF on disjoint X/Y
//  lanes (K writes vfX.x, K+1 writes vfX.y). K's writeback defers — instead
//  of Str s5 it copies v5's relevant lane into the d10 stash. K+1's
//  writeback overlays its lane into d10 and emits one Str d covering the
//  combined XY half (via vfCacheStore with mask 0xC, which the partial-
//  lane peephole encodes as a single 64-bit store at base+0).
//
//  Stash uses d10 (lower 64 bits of v10). AAPCS64 marks d8-d15 as callee-
//  saved, so BLs in K's tail or K+1's prologue preserve it. Lower ops
//  don't touch v10 (verified by grep). The cache slot's lane state stays
//  coherent because vfCacheStore on the combined mask updates both the
//  slot and memory in one shot — between K and K+1 the slot still holds
//  pre-K values, which is correct for any lane that wasn't M_K (forbidden
//  to read by Pass 1's gate).
//
//  g_vu1BatchWithNext / g_vu1BatchFromPrev are set per-emit by CompileBlock
//  before dispatching emitVU1Upper.
// ============================================================================
bool g_vu1BatchWithNext = false;
bool g_vu1BatchFromPrev = false;

// FMAC opt #4: ABS source is provably non-negative for the lanes the ABS
// op writes. Set per-emit by CompileBlock; consumed by emitAbsFmac to
// swap Fabs for a direct slot→v5 load (saves 1 Fabs insn).
bool g_vu1AbsSrcKnownNonNeg = false;

// FMAC opt #19: matrix-vector MAC cluster fusion. Set per-emit by
// CompileBlock from microIR Pass 1 detection.
//
//   g_vu1MacClusterLead   — this pair is the MULAx that leads a confirmed
//                           4-pair MULAx+MADDAy+MADDAz+MADDw chain. The
//                           MULAx recVU1 emitter handles the WHOLE chain
//                           and stashes the matrix cols + writes back ACC
//                           + writes VFd of the trailing MADDw.
//   g_vu1MacClusterMember — this pair is one of the three MADD followers.
//                           recVU1 emitter no-ops the math (lead already
//                           did it); the existing per-pair cycle/fmaccount
//                           accounting in CompileBlock still increments.
//   g_vu1MacClusterFt     — VF[ft] = the shared broadcast vector for the
//                           chain. Lead emit uses this; followers ignore.
bool g_vu1MacClusterLead    = false;
bool g_vu1MacClusterMember  = false;
bool g_vu1MacClusterXyzOnly = false; // true when all 4 pairs use mask 0xE (preserve W)
u32  g_vu1MacClusterFt      = 0;     // shared broadcast vector
u32  g_vu1MacClusterFs0     = 0;     // MULAx fs  = matrix col 0
u32  g_vu1MacClusterFs1     = 0;     // MADDAy fs = matrix col 1
u32  g_vu1MacClusterFs2     = 0;     // MADDAz fs = matrix col 2
u32  g_vu1MacClusterFs3     = 0;     // MADDw fs  = matrix col 3
u32  g_vu1MacClusterFd      = 0;     // MADDw fd  = result destination (non-zero, gated by Pass 1)

// FMAC opt #20: OPMULA + OPMSUB cross-product cluster.
bool g_vu1OpmacClusterLead   = false;
bool g_vu1OpmacClusterMember = false;
u32  g_vu1OpmacClusterFs     = 0; // OPMULA fs (= a in cross(a, b))
u32  g_vu1OpmacClusterFt     = 0; // OPMULA ft (= b in cross(a, b))
u32  g_vu1OpmacClusterFd     = 0; // OPMSUB fd (cross product destination)

// ============================================================================
//  Deferred VF writeback for vf_read_after_write hazard pairs.
//
//  PS2 VU pairs run upper/lower simultaneously, so when lower reads VF[X] that
//  upper writes, lower must see the PRE-upper value. The previous BL-into-
//  vu1Exec fallback handled this via interp's save/restore semantics — costly:
//  5 flushes + BL + 5 reloads per hazard pair.
//
//  This deferred-writeback path keeps everything native: upper computes into
//  v5 normally and updates the pinned macflag/statusflag/clipflag regs, but
//  instead of committing v5 to VF[X] cache + memory, it spills v5 to a 16-byte
//  scratch and records dst_vf + xyzw. Lower then emits — its VF[X] reads see
//  the original memory value (broadcast cache holds pre-upper too). After
//  lower, the caller reloads from scratch and calls vfCacheStore to commit.
//
//  Mechanism:
//    - CompileBlock sets g_vu1DeferVfWriteback = true before the upper emit,
//      and resets g_vu1DeferredVfIdx = -1 (sentinel).
//    - emitFmacInlineWriteback / emitFmacStoreMasked / emitNoFlagWriteback
//      consult the flag in their Step G (after the ACC redirect / dead-write
//      / fd==0 short circuits). When set, they Str q5 to s_vu1DeferredVfStash,
//      record idx + mask, and skip the cache store.
//    - After emitVU1Upper returns, CompileBlock clears g_vu1DeferVfWriteback.
//    - After emitVU1Lower returns, CompileBlock — if g_vu1DeferredVfIdx > 0 —
//      reloads q5 from the scratch and calls vfCacheStore + broadcast-cache
//      invalidate to commit. Idx == -1 means upper wrote ACC / VF[0] /
//      dead-elided — nothing to commit.
//
//  Scratch lives in a static 16-byte buffer addressed via armMoveAddressToReg
//  rather than a stack slot — avoids resizing the prologue's 96-byte frame.
//  Single-threaded (VU1 emit runs on one thread at a time, MTVU or EE).
//
//  Lower ops in vf_read_after_write hazards are by construction VF-readers
//  (SQ/SQI/SQD, MFP/MFIR, MTIR-from-VF, MR32-of-VF), never flag-readers
//  (FSAND/FSEQ/etc. don't have a VFread overlap with upper's VFwrite). So
//  upper's flag updates running before lower is correct here — flag-reader
//  lowers can't co-occur with this hazard class.
// ============================================================================
bool g_vu1DeferVfWriteback   = false;  // set per-emit by CompileBlock.
int  g_vu1DeferredVfIdx      = -1;     // -1 = none. Filled by writeback emit.
u8   g_vu1DeferredXyzw       = 0;      // captured by writeback emit.
alignas(16) u8 g_vu1DeferredVfStash[16] = {};

namespace
{
	bool s_batchPendingActive    = false;
	int  s_batchPendingVf        = 0;
	u8   s_batchPendingFirstMask = 0;
}

void vu1BatchCacheReset()
{
	s_batchPendingActive = false;
}

// ============================================================================
//  Broadcast operand cache (FMAC opt #16)
//
//  emitLoadBroadcast loads VF[ft][comp] broadcasted into v1. For consecutive
//  pairs broadcasting the same (ft, comp) — common in lighting code where
//  ADDx-then-MULx on the same scalar appears in chains — the second load
//  is redundant. Track the last (ft, comp) emitted and skip the Dup (and
//  upstream Ldr if the slot was already resident) on a hit.
//
//  Invalidated on:
//    - Block start (vu1BroadcastCacheReset).
//    - Any BL — wired into vfCacheFlushAndInvalidate; v1 is caller-saved
//      per AAPCS64 so the BL clobbers it.
//    - VF[cached_ft] is written by some op — wired into vfCacheStore via
//      vu1BroadcastCacheNoteVfWritten; the broadcast value would be stale.
//    - v1 is overwritten by another helper (vfCacheLoadInto with scratch=v1,
//      emitLoadQI, emitVuAddSubHack, FTOI/ITOF, MR32). Each writer calls
//      vu1BroadcastCacheReset directly.
//
//  ACC and clamp interactions: emitVuClampVec on v1 keeps the cached value
//  semantically valid (clamp is idempotent for valid floats; +MAX/-MAX
//  bit-patterns don't change under repeated clamps), so we DO NOT
//  invalidate around clamp. emitFmacWriteback / emitFmacInlineWriteback
//  use v2-v8 + v5; v1 is preserved across the whole writeback.
// ============================================================================
namespace
{
	bool s_broadcastCacheValid = false;
	u32  s_broadcastCacheFt    = 0;
	int  s_broadcastCacheComp  = -1;

	// FMAC opt #18: deferred-Dup broadcast plan. emitLoadBroadcast plans the
	// source (VF cache slot + lane, or VF[0] constant, or Q/I) but defers the
	// Dup-into-v1 until a consumer actually needs v1. The matrix-transform
	// hot path (MULAx/MADDAy/MADDAz/MADDw chain — top block in GoW2 at
	// pc=0x05d8 was 30% of VU1 time) only needs an FMUL where ARM64 supports
	// `Fmul Vd.4s, Vn.4s, Vm.s[lane]` directly — emitting the Dup is wasted.
	//
	// Lifetime: planned at op start, consumed by emitBinary/Ternary/MaxFmac,
	// invalidated on any code path that touches v1 (BL, vfCacheLoadInto into
	// v1, emitLoadQI, addSubHack, FTOI/ITOF, MR32) — same hooks as the v1
	// broadcast cache above. None means "v1 already holds whatever it should;
	// don't try lane-indexed shortcuts" — used by FMAC_*_FULL paths and Q/I.
	// CacheLane: VF cache slot (v17..v24), lane 0..3 (X..W).
	// QIScalar:  Q or I scalar pre-loaded into v1.s[0] only — lane-FMUL via
	//            v1.s[0]. Materialization Dups v1.s[0] into v1.4s. Saves one
	//            Dup per Q/I-broadcast FMUL (MULq, MULi, MADDq, etc.).
	// FullSlot:  full VF[ft] vector resident in a cache slot — used by FMAC_*_FULL
	//            (MUL.xyzw, ADD.w, MULA.xyz, etc.). Consumers emit the ALU op
	//            directly against the slot reg, skipping the prior `Mov v1, slot`.
	//            Materialization is `Mov v1.16b, slot.16b` for clamp/hack/MAX.
	enum class BroadcastPlanKind { None, V1Held, ConstZero, ConstOne, CacheLane, QIScalar, FullSlot };
	BroadcastPlanKind s_broadcastPlanKind = BroadcastPlanKind::None;
	int               s_broadcastPlanSlotReg = 0; // CacheLane/QIScalar/FullSlot: NEON reg code
	int               s_broadcastPlanLane    = 0; // CacheLane: 0..3 (X..W); QIScalar: always 0; FullSlot: unused
}

void vu1BroadcastCacheReset()
{
	s_broadcastCacheValid = false;
	s_broadcastPlanKind = BroadcastPlanKind::None;
}

void vu1BroadcastCacheNoteVfWritten(int vfreg)
{
	if (s_broadcastCacheValid && static_cast<int>(s_broadcastCacheFt) == vfreg)
		s_broadcastCacheValid = false;
	// If plan references a slot whose VF was just written, the slot still
	// holds the up-to-date value (vfCacheStore writes the slot then optionally
	// to memory) — but the lane index we recorded is for the OLD value. In
	// practice this never fires mid-op (writes happen at writeback, after the
	// FMUL consumer); reset defensively.
	if (s_broadcastPlanKind == BroadcastPlanKind::CacheLane
		&& static_cast<int>(s_broadcastCacheFt) == vfreg)
	{
		s_broadcastPlanKind = BroadcastPlanKind::None;
	}
}

// ============================================================================
//  Native NEON codegen helpers
// ============================================================================

// VU1_BASE_REG / VU1_MACFLAG_REG / VU1_STATUSFLAG_REG / VU1_CLIPFLAG_REG /
// VU1_ACC_REG are defined in the parent aVU.cpp.

// vfOff / viOff are defined in the parent aVU.cpp. Only accOff is Upper-only.
static constexpr int64_t accOff()
{
	return static_cast<int64_t>(offsetof(VURegs, ACC));
}

// Byte offset of VF[0]. The inline writeback uses this to detect the "drop
// the result store, flags still update" case that matches the C helper's
// `dst != &VU->VF[0]` check.
static constexpr int64_t vfOffStatic0 = static_cast<int64_t>(offsetof(VURegs, VF));

// Phase 2: writeback helpers receive an int64_t dst_off that's either
// accOff() or vfOff(fd). For VF writes, we defer the Str into the cache
// (vfCacheStore); for ACC writes, the existing pinned-VU1_ACC_REG path
// applies. This helper extracts the VF index, returning -1 if dst_off
// targets ACC or another field.
static inline int vfIndexFromDstOff(int64_t dst_off)
{
	if (dst_off >= vfOffStatic0 && dst_off < vfOffStatic0 + 32 * static_cast<int64_t>(sizeof(VECTOR)))
		return static_cast<int>((dst_off - vfOffStatic0) / static_cast<int64_t>(sizeof(VECTOR)));
	return -1;
}

// ============================================================================
//  Inline MAC/STATUS flag computation + result clamping + writeback.
//  (Replaces the vu1_fmac_writeback / vu1_opmula_writeback / vu1_opmsub_writeback
//  C helpers. Eliminates the BL overhead + function-call frame for what runs
//  on every FMAC pair where g_vu1NeedsFlags is true.)
//
//  The interpreter's VU_MAC_UPDATE (VUflags.cpp) categorises each lane's
//  float result as zero / denormal / inf-or-NaN / normal and sets four flag
//  bits per lane in VU->macflag; it also flushes denormals to ±0 and
//  (under CHECK_VU_OVERFLOW) clamps inf/NaN to ±0x7F7FFFFF before storing.
//  VU_STAT_UPDATE then derives statusflag as a nibble-by-nibble "any bit set"
//  reduction of macflag.
//
//  Flag bit layout (16 bits, one nibble per type):
//    bits  0- 3  Z (zero)     — lane W/Z/Y/X at bit 0/1/2/3
//    bits  4- 7  S (sign)
//    bits  8-11  U (underflow — denormal)
//    bits 12-15  O (overflow — inf/NaN)
//  VU_MACx_UPDATE uses shift=3 for X, 2 for Y, 1 for Z, 0 for W — matching
//  v5.S[0]=X .. v5.S[3]=W.
//
//  For lanes NOT in xyzw the four flag bits get cleared (matches
//  VU_MAC*_CLEAR). For lanes IN xyzw the result is written back with VF[0]
//  special-cased to drop the write (flags still update).
//
//  Per-lane result clamp (independent of flags):
//    denormal → sign only (flush to ±0)
//    inf/NaN + CHECK_VU_OVERFLOW(1) → sign | 0x7F7FFFFF
//    otherwise → pass through.
// ============================================================================

// Static constant for per-lane left-shift amounts used when packing per-lane
// 4-bit flag values (at bits 0, 4, 8, 12 within each lane's u32) into the
// 16-bit macflag. Lane 0 (X) shifts left 3, lane 3 (W) shifts left 0.
// Loaded once per emit call via armLoadConstant128 (1-insn literal pool).
alignas(16) static const u32 kFmacShiftVec[4] = {3, 2, 1, 0};

// FMAC opt #6 (NEON-vectorized compare): per-lane bit-pack mask for CLIP's
// positive compare results. Lane 0 (fs.x) → bit 0, lane 1 (fs.y) → bit 2,
// lane 2 (fs.z) → bit 4, lane 3 unused (W lane is the value, not compared).
// The sign-flipped mask {2, 8, 32, 0} is derived at emit time as
// (kClipMaskPos << 1) to save one constant pool entry.
alignas(16) static const u32 kClipMaskPos[4] = {0x01u, 0x04u, 0x10u, 0x00u};

// Mode for the inline writeback helper:
//   GenericFmac : normal ADD/SUB/MUL/MADD/MSUB + accum variants.
//                 Lanes not in xyzw get their flag bits cleared.
//   OpmXYZ      : OPMULA / OPMSUB. Operates on xyz only; W lane's existing
//                 flag bits are preserved (interpreter never touches macflag[w]).
enum class FmacWritebackMode { GenericFmac, OpmXYZ };

// Inline replacement for vu1_fmac_writeback / vu1_opmula_writeback / vu1_opmsub_writeback.
//
//   v5  : float32x4_t result, lanes [X, Y, Z, W] — CLOBBERED (receives the
//         clamp-adjusted value that gets stored to dst).
//   dst_off : byte offset from VU1_BASE to the destination VECTOR. For
//         OPMULA this is accOff(); for OPMSUB / generic it's vfOff(fd) or
//         accOff() (matches the prior C-helper call sites).
//   xyzw : write mask (0..0xF). OpmXYZ mode forces effective xyzw=0xE (xyz).
//   mode : see FmacWritebackMode above.
//
// Scratch: v2, v3, v4, v6, v7, v8; w4, w5, w6, w7. All caller-saved.
// Writes VU->macflag and VU->statusflag via the pinned regs (VU1_MACFLAG_REG
// / VU1_STATUSFLAG_REG) rather than memory — Phase-7 caches all three flag
// fields in callee-saved registers for the duration of the block.

// Partial-lane store peephole. Replaces the old Ldr+Mov+Str merge pattern
// (2+N insns for N set lanes) with targeted stores that touch only the
// memory lanes selected by `xyzw`. Lanes not in `xyzw` are untouched in
// memory — equivalent semantics to the merge-load-store, since the old
// code merged selected lanes into a full quadword but the non-selected
// lanes came straight from memory and went straight back.
//
// Key vixl gotcha: `st1 {vt.<T>}[lane], [xn, #imm]` requires imm==0
// (LoadStoreStructVerify asserts in debug; silently miscompiles in
// release). So for any store at a non-zero offset, we either use a
// scalar Str form (s/d/q — those DO support immediate offsets) or
// rearrange the source lanes into lane 0/low-64 of a scratch first.
//
// Scratch: v2 (caller-saved, unused by writeback step G / emitBinaryFmac
// aftermath / MIN/MAX/ABS tail — all three call sites treat v2 as dead
// after this function returns).
//
// Expects result in v5 (the FMAC pipeline output). base_off is the byte
// offset from VU1_BASE_REG to the destination VECTOR (vfOff(fd) /
// accOff() / etc.). xyzw must be nonzero and not 0xF — the caller
// handles those via full-store / no-op respectively.
//
// xyzw bit layout: bit 3 = X (lane 0 / offset 0),
//                  bit 2 = Y (lane 1 / offset 4),
//                  bit 1 = Z (lane 2 / offset 8),
//                  bit 0 = W (lane 3 / offset 12).
//
// Instruction count per mask (vs old 2+N pattern):
//   X only  (0x8):              1 insn  (was 3)
//   XY only (0xC):              1 insn  (was 4)
//   other single (0x1/2/4):    2 insns (was 3)
//   ZW only  (0x3):             2 insns (was 4)
//   split doubles / triples:   2–3 insns (was 4–5)
static void emitPartialLaneStore(int64_t base_off, u32 xyzw)
{
	// Helper: store v5's lane L to [base + L*4] via `Mov v2.s[0], v5.s[L]
	// + Str s2, [base + L*4]`. Two insns; used for non-lane-0 singles.
	auto emitLaneS = [&](int lane) {
		armAsm->Mov(v2.V4S(), 0, v5.V4S(), lane);
		armAsm->Str(s2, MemOperand(VU1_BASE_REG, base_off + lane * 4));
	};

	switch (xyzw)
	{
		// --- Single-lane stores ---
		case 0x8: // X: Str s5 at +0 (1 insn, lane 0 maps straight to offset 0)
			armAsm->Str(s5, MemOperand(VU1_BASE_REG, base_off + 0));
			return;
		case 0x4: emitLaneS(1); return; // Y
		case 0x2: emitLaneS(2); return; // Z
		case 0x1: emitLaneS(3); return; // W

		// --- Adjacent dual-lane stores ---
		case 0xC: // XY: Str d5 at +0 (1 insn, low 64 maps straight)
			armAsm->Str(d5, MemOperand(VU1_BASE_REG, base_off + 0));
			return;
		case 0x3: // ZW: rotate v5 high 64 into low 64 of v2, then Str d2 at +8
			armAsm->Ext(v2.V16B(), v5.V16B(), v5.V16B(), 8);
			armAsm->Str(d2, MemOperand(VU1_BASE_REG, base_off + 8));
			return;

		// --- Triple-lane stores ---
		case 0xE: // XYZ = XY (Str d5) + Z
			armAsm->Str(d5, MemOperand(VU1_BASE_REG, base_off + 0));
			emitLaneS(2);
			return;
		case 0xD: // XYW = XY (Str d5) + W
			armAsm->Str(d5, MemOperand(VU1_BASE_REG, base_off + 0));
			emitLaneS(3);
			return;
		case 0xB: // XZW = X (Str s5) + ZW (Ext + Str d)
			armAsm->Str(s5, MemOperand(VU1_BASE_REG, base_off + 0));
			armAsm->Ext(v2.V16B(), v5.V16B(), v5.V16B(), 8);
			armAsm->Str(d2, MemOperand(VU1_BASE_REG, base_off + 8));
			return;
		case 0x7: // YZW = Y (lane move + Str s) + ZW (Ext + Str d)
			emitLaneS(1);
			armAsm->Ext(v2.V16B(), v5.V16B(), v5.V16B(), 8);
			armAsm->Str(d2, MemOperand(VU1_BASE_REG, base_off + 8));
			return;

		// --- Non-adjacent dual-lane stores ---
		case 0x9: // XW
			armAsm->Str(s5, MemOperand(VU1_BASE_REG, base_off + 0));
			emitLaneS(3);
			return;
		case 0xA: // XZ
			armAsm->Str(s5, MemOperand(VU1_BASE_REG, base_off + 0));
			emitLaneS(2);
			return;
		case 0x5: // YW
			emitLaneS(1);
			emitLaneS(3);
			return;
		case 0x6: // YZ
			emitLaneS(1);
			emitLaneS(2);
			return;
	}
}

// FMAC opt #17: batched VF cache store. When g_vu1BatchWithNext is set we
// stash v5's relevant lane (X = lane 0, or Y = lane 1) into d10 and skip
// the cache store; the partner pair's emit (g_vu1BatchFromPrev) overlays
// its lane and emits a single combined store. d10's lower 64 bits are
// AAPCS64-callee-saved, so BLs between K and K+1 don't clobber it.
//
// Returns true if it consumed the write (deferred or batch-completed),
// false if the caller should fall through to vfCacheStore.
static bool tryBatchedVfCacheStore(int dst_vf, u32 mask)
{
	if (g_vu1BatchWithNext)
	{
		// Pass 1 detects only the X-only / Y-only adjacent pattern, so
		// the only meaningful lane bits at this point are 0x8 (X→lane 0)
		// and 0x4 (Y→lane 1).
		if (mask & 0x8u) armAsm->Mov(v10.V4S(), 0, v5.V4S(), 0);
		if (mask & 0x4u) armAsm->Mov(v10.V4S(), 1, v5.V4S(), 1);
		s_batchPendingActive    = true;
		s_batchPendingVf        = dst_vf;
		s_batchPendingFirstMask = static_cast<u8>(mask);
		return true;
	}

	if (g_vu1BatchFromPrev && s_batchPendingActive && s_batchPendingVf == dst_vf)
	{
		if (mask & 0x4u) armAsm->Mov(v10.V4S(), 1, v5.V4S(), 1);
		if (mask & 0x8u) armAsm->Mov(v10.V4S(), 0, v5.V4S(), 0);
		const u8 combined = static_cast<u8>(s_batchPendingFirstMask | mask);
		s_batchPendingActive = false;
		// vfCacheStore on mask 0xC routes through vfCacheEmitPartialLaneStore's
		// `Str slotReg.D() at base+0` fast path — one 64-bit store covering
		// XY in a single insn. The slot's lanes 0/1 are merged from v10's
		// lanes 0/1 (which we just populated), so the slot ends up with the
		// correct combined value.
		vfCacheStore(dst_vf, v10, combined);
		return true;
	}

	return false;
}

static void emitFmacInlineWriteback(int64_t dst_off, u32 xyzw, FmacWritebackMode mode)
{
	const bool  skip_dst_write   = (dst_off == vfOffStatic0); // VF[0] hardwired
	const bool  overflow_clamp   = CHECK_VU_OVERFLOW(1);
	// compute_uo: emit U (denormal) and O (inf/NaN) flag bits + denormal flush
	// + inf clamp. False when no in-block op reads MAC/STATUS and the user has
	// not enabled the overflow gamefix — matches OLD's mVUupdateFlags pattern
	// (which never computes U and only computes O behind CHECK_VUOVERFLOWHACK).
	// Saves ~12 NEON + ~6 GPR insn per writeback for typical blocks.
	const bool  compute_uo       = g_vu1NeedsUOFlags || overflow_clamp;
	const bool  opm_mode         = (mode == FmacWritebackMode::OpmXYZ);
	// In OPMULA/OPMSUB the ISA treats xyz as the active lanes and never
	// touches macflag[w] — force effective xyzw to 0xE so our lane loop
	// stays uniform, and remember to preserve the prior W flag bits when
	// rebuilding macflag.
	const u32   eff_xyzw         = opm_mode ? 0xEu : xyzw;

	// ------------------------------------------------------------------
	// Step A: extract exp/sign and compute per-lane category masks.
	// ------------------------------------------------------------------

	// v2 = v5 << 1 — doubles magnitude, strips sign. Zero iff v5 ∈ {+0,-0}.
	armAsm->Shl(v2.V4S(), v5.V4S(), 1);
	// v3 = exp (8-bit per lane).
	armAsm->Ushr(v3.V4S(), v2.V4S(), 24);
	// v4 = sign bit (0 or 1 per lane).
	armAsm->Ushr(v4.V4S(), v5.V4S(), 31);

	// v7 = v_exp0_mask : 0xFFFFFFFF if (exp == 0) — covers both zero AND denormal.
	// This is the Z bit source (PS2 spec: Z fires for zero OR denormal).
	armAsm->Cmeq(v7.V4S(), v3.V4S(), 0);

	if (compute_uo)
	{
		// v6 = v_zero_mask : 0xFFFFFFFF if (v5 == ±0), else 0.
		armAsm->Cmeq(v6.V4S(), v2.V4S(), 0);
		// v6 = v_denorm_mask : (exp == 0) AND NOT zero.
		armAsm->Bic(v6.V16B(), v7.V16B(), v6.V16B());
		// v8 = 0xFF broadcast, then v8 = v_inf_mask : (exp == 0xFF).
		armAsm->Movi(v8.V4S(), 0xFF);
		armAsm->Cmeq(v8.V4S(), v3.V4S(), v8.V4S());
	}

	// ------------------------------------------------------------------
	// Step B: apply result clamping to v5.
	//   denormal lanes → sign-only.
	//   inf/NaN lanes (with overflow gate)  → sign | 0x7F7FFFFF.
	// ------------------------------------------------------------------

	if (compute_uo)
	{
		// v3 = 0x80000000 per lane (sign-bit-placed).
		armAsm->Shl(v3.V4S(), v4.V4S(), 31);
		// v5 ← denorm ? v3 : v5 (flush denormals to ±0).
		// Bit Vd,Vn,Vm : Vd = (Vd & ~Vm) | (Vn & Vm).
		armAsm->Bit(v5.V16B(), v3.V16B(), v6.V16B());

		if (overflow_clamp)
		{
			// v2 = 0x7F7FFFFF broadcast (MAX_FLOAT bit pattern).
			armAsm->Mov(w4, 0xFFFFu);
			armAsm->Movk(w4, 0x7F7F, 16);
			armAsm->Dup(v2.V4S(), w4);
			// v2 = sign | MAX_FLOAT.
			armAsm->Orr(v2.V16B(), v3.V16B(), v2.V16B());
			// v5 ← inf ? (sign|MAX) : v5.
			armAsm->Bit(v5.V16B(), v2.V16B(), v8.V16B());
		}
	}

	// ------------------------------------------------------------------
	// Step C: build per-lane 4-bit flag word in v3:
	//   v3.S[i] = Z | (S<<4) | (U<<8) | (O<<12)
	// Each of Z/S/U/O is 0 or 1.
	//   Z (zero-or-denorm) = (exp == 0) = v7 narrowed to 0/1.
	//   S                  = v4 (already 0/1 from step A).
	//   U (denormal)       = v6 (mask) narrowed to 0/1.
	//   O (inf/NaN)        = v8 (mask) narrowed to 0/1.
	// ------------------------------------------------------------------

	// v7 narrows mask → 0/1. Shift the 0xFFFFFFFF pattern right by 31.
	armAsm->Ushr(v7.V4S(), v7.V4S(), 31);   // Z bits at bit 0
	// Shift sign to its position within the per-lane nibble tower.
	armAsm->Shl(v4.V4S(), v4.V4S(), 4);     // S bit moves to bit 4
	// Combine Z and S.
	armAsm->Orr(v3.V16B(), v7.V16B(), v4.V16B());

	if (compute_uo)
	{
		armAsm->Ushr(v6.V4S(), v6.V4S(), 31);   // U bits at bit 0
		armAsm->Ushr(v8.V4S(), v8.V4S(), 31);   // O bits at bit 0
		armAsm->Shl(v6.V4S(), v6.V4S(), 8);     // U bit moves to bit 8
		armAsm->Shl(v8.V4S(), v8.V4S(), 12);    // O bit moves to bit 12
		armAsm->Orr(v3.V16B(), v3.V16B(), v6.V16B());
		armAsm->Orr(v3.V16B(), v3.V16B(), v8.V16B());
	}

	// ------------------------------------------------------------------
	// Step D: pack v3's 4 lanes into a single 16-bit macflag value.
	// Lane i shifts left by (3-i) so lane 0 (X) lands at bits 3/7/11/15,
	// lane 3 (W) at bits 0/4/8/12. Horizontal sum then extract scalar.
	// ------------------------------------------------------------------

	// Load [3, 2, 1, 0] into v2 as 4x32-bit shift amounts (1 insn via literal pool).
	armLoadConstant128(v2.Q(), kFmacShiftVec);
	armAsm->Ushl(v3.V4S(), v3.V4S(), v2.V4S());
	armAsm->Addv(s3, v3.V4S());              // horizontal sum → lane 0
	armAsm->Umov(w6, v3.S(), 0);             // w6 = raw packed macflag (16 meaningful bits)

	// ------------------------------------------------------------------
	// Step E: apply xyzw mask to macflag (clear lanes not in eff_xyzw).
	// Each lane contributes 4 bits, one in each nibble. The combined mask
	// is `eff_xyzw * 0x1111` — replicates the per-lane bit pattern into
	// all four flag nibbles.
	// ------------------------------------------------------------------

	if (eff_xyzw != 0xFu)
	{
		armAsm->And(w6, w6, static_cast<u32>(eff_xyzw) * 0x1111u);
	}

	// For OPMULA/OPMSUB, preserve the existing W lane flag bits (bits
	// 0/4/8/12 — the 0x1111 mask). Merge prior macflag[W_bits] with the
	// newly-computed XYZ bits. Read from the pinned reg, not memory.
	if (opm_mode)
	{
		armAsm->And(w4, VU1_MACFLAG_REG, 0x1111u);  // keep only old W bits
		armAsm->Orr(w6, w6, w4);                    // combine with fresh XYZ bits
	}

	// Macflag lives in the pinned reg for the block; no memory store here.
	armAsm->Mov(VU1_MACFLAG_REG, w6);

	// ------------------------------------------------------------------
	// Step F: derive statusflag. Bit i of statusflag = "any bit set in
	// macflag nibble i". Tst + Cset + Orr shifted.
	// ------------------------------------------------------------------

	if (compute_uo)
	{
		// 6-insn nibble-OR + compress: replaces the 11-insn Tst+Cset+Orr
		// chain that used to dominate this step. w6 is dead after this
		// (macflag was already stored to VU1_MACFLAG_REG above), so we
		// destructively reuse it as the OR-fold accumulator.
		//
		// Steps 1-2: collapse each 4-bit nibble of w6 to its OR.
		//   bit 0/4/8/12 of result = (any bit set in Z/S/U/O nibble)
		//   bits between are contaminated (cross-nibble OR-fold leaks),
		//   so step 3 masks them out.
		// 0x11111111 is the encodable equivalent of 0x1111: bits 16-31 of
		// w6 are 0 (Step D Umov sourced from a 16-bit-meaningful s-reg),
		// so Anding with the upper-half-also-set 0x11111111 has the same
		// effect on bits 0-15 as Anding with 0x1111. (0x1111 itself is
		// not a valid ARM64 logical immediate — would need 2 insn to load.)
		// Steps 4-5: spread the 4 bit-0/4/8/12 bits down to positions 0-3
		// via shift+OR (no carry/multiply, no latency hazard).
		// Step 6: ubfx 4 bits into the statusflag reg.
		armAsm->Orr(w6, w6, Operand(w6, LSR, 1));
		armAsm->Orr(w6, w6, Operand(w6, LSR, 2));
		armAsm->And(w6, w6, 0x11111111u);
		armAsm->Orr(w6, w6, Operand(w6, LSR, 3));
		armAsm->Orr(w6, w6, Operand(w6, LSR, 6));
		armAsm->Ubfx(VU1_STATUSFLAG_REG, w6, 0, 4);
	}
	else
	{
		// !compute_uo: U/O macflag bits are always 0, so only Z+S contribute
		// to statusflag. The original 5-insn Tst chain matches this exactly
		// and beats the 6-insn nibble-OR approach by 1 insn here, so keep it.
		armAsm->Tst(w6, 0x000Fu);
		armAsm->Cset(VU1_STATUSFLAG_REG, ne);
		armAsm->Tst(w6, 0x00F0u);
		armAsm->Cset(w4, ne);
		armAsm->Orr(VU1_STATUSFLAG_REG, VU1_STATUSFLAG_REG, Operand(w4, LSL, 1));
	}
	// statusflag lives in the pinned reg; no memory store here.

	// ------------------------------------------------------------------
	// Step G: write clamped v5 to dst with xyzw (eff_xyzw) mask. Skip when
	// dst is VF[0] (hardwired; the C helper's `write` check).
	//
	// Phase-8: ACC writes redirect into the pinned VU1_ACC_REG (q16) —
	// no memory store until block epilogue / vu1Exec hazard.
	// ------------------------------------------------------------------

	if (skip_dst_write)
		return;

	const bool to_acc = (dst_off == accOff());

	if (to_acc)
	{
		// ACC stays pinned in v16 — Mov lanes in place, no cache or memory.
		if (eff_xyzw == 0xFu)
		{
			armAsm->Mov(VU1_ACC_REG.V16B(), v5.V16B());
			return;
		}
		if (eff_xyzw & 8) armAsm->Mov(VU1_ACC_REG.V4S(), 0, v5.V4S(), 0);
		if (eff_xyzw & 4) armAsm->Mov(VU1_ACC_REG.V4S(), 1, v5.V4S(), 1);
		if (eff_xyzw & 2) armAsm->Mov(VU1_ACC_REG.V4S(), 2, v5.V4S(), 2);
		if (eff_xyzw & 1) armAsm->Mov(VU1_ACC_REG.V4S(), 3, v5.V4S(), 3);
		return;
	}

	// FMAC opt #14: dead VF write elision. Pass 1 proved a covering
	// overwrite within 4 pairs forward with no live-lane read in between,
	// so the cache store + write-through Str are wasted work. Skipped
	// AFTER the to_acc branch above so ACC writes (never dead) still fire.
	if (g_vu1DeadVFWrite)
		return;

	const int dst_vf = vfIndexFromDstOff(dst_off);

	// vf_read_after_write hazard: spill v5 to the deferred-stash scratch
	// instead of committing to VF[X]. CompileBlock reloads after lower's
	// emit and calls vfCacheStore to commit. Lower (a VF-reader) sees the
	// original memory value, matching interp's save/restore semantics
	// without the BL-into-vu1Exec roundtrip.
	if (g_vu1DeferVfWriteback && dst_vf > 0)
	{
		armMoveAddressToReg(x4, &g_vu1DeferredVfStash[0]);
		armAsm->Str(v5.Q(), a64::MemOperand(x4));
		g_vu1DeferredVfIdx = dst_vf;
		g_vu1DeferredXyzw  = static_cast<u8>(eff_xyzw);
		return;
	}

	// Phase 2: defer the VF[fd] Str into the cache slot. The actual memory
	// store fires at block end / hazard / BL via vfCacheFlushAndInvalidate.
	if (dst_vf > 0)
	{
		// FMAC opt #17: try the same-VF different-lane batching helper
		// first. On a matched K it stashes v5's lane and skips the
		// store; on a matched K+1 it merges and Strs the combined XY.
		if (!tryBatchedVfCacheStore(dst_vf, eff_xyzw))
			vfCacheStore(dst_vf, v5, static_cast<u8>(eff_xyzw));
	}
	else
		emitPartialLaneStore(dst_off, eff_xyzw); // fallback for unrecognized dst
}

// Generic FMAC writeback — result must be in v5.4S.
// dst_off = byte offset of destination from VU1_BASE_REG.
static void emitFmacWriteback(int64_t dst_off, u32 xyzw)
{
	emitFmacInlineWriteback(dst_off, xyzw, FmacWritebackMode::GenericFmac);
}

// Store v5 to dst_off with xyzw mask, used by the flag-skipping fast path.
// Identical merge logic to emitNoFlagWriteback below, but takes a raw byte
// offset (so it can target ACC as well as VF[fd]) and skips the fd==0 check
// only when dst_off matches VF[0]. Output clamping is the caller's job.
// vfOffStatic0 is defined near the top of the file.
static void emitFmacStoreMasked(int64_t dst_off, u32 xyzw)
{
	// VF[0] is hardwired to {0,0,0,1}; the C helper drops the write but
	// still updates flags — in the no-flag path there's nothing to do.
	if (dst_off == vfOffStatic0)
		return;

	// Phase-8: ACC is pinned in VU1_ACC_REG (q16). Redirect stores to the
	// pinned register — no cache, no memory round-trip.
	const bool to_acc = (dst_off == accOff());
	if (to_acc)
	{
		if (xyzw == 0xF)
		{
			armAsm->Mov(VU1_ACC_REG.V16B(), v5.V16B());
			return;
		}
		if (xyzw & 8) armAsm->Mov(VU1_ACC_REG.V4S(), 0, v5.V4S(), 0);
		if (xyzw & 4) armAsm->Mov(VU1_ACC_REG.V4S(), 1, v5.V4S(), 1);
		if (xyzw & 2) armAsm->Mov(VU1_ACC_REG.V4S(), 2, v5.V4S(), 2);
		if (xyzw & 1) armAsm->Mov(VU1_ACC_REG.V4S(), 3, v5.V4S(), 3);
		return;
	}

	// FMAC opt #14: dead VF write elision. Same gate as the flag-path
	// writeback — checked AFTER the to_acc branch so ACC writes are
	// unaffected. No flags are computed in this path, so the saving here
	// is purely the cache store + write-through Str.
	if (g_vu1DeadVFWrite)
		return;

	const int dst_vf = vfIndexFromDstOff(dst_off);

	// vf_read_after_write deferred writeback — see comment on
	// g_vu1DeferVfWriteback. Spill v5 to the scratch instead of committing.
	if (g_vu1DeferVfWriteback && dst_vf > 0)
	{
		armMoveAddressToReg(x4, &g_vu1DeferredVfStash[0]);
		armAsm->Str(v5.Q(), a64::MemOperand(x4));
		g_vu1DeferredVfIdx = dst_vf;
		g_vu1DeferredXyzw  = static_cast<u8>(xyzw);
		return;
	}

	// Phase 2: defer the VF[fd] write into the cache. The Str fires at
	// block end / hazard / BL via vfCacheFlushAndInvalidate.
	if (dst_vf > 0)
	{
		// FMAC opt #17: same-VF different-lane batching.
		if (!tryBatchedVfCacheStore(dst_vf, xyzw))
			vfCacheStore(dst_vf, v5, static_cast<u8>(xyzw));
		return;
	}

	// Fallback for any dst that isn't VF[1..31] or ACC — keep the direct
	// store path. Shouldn't trigger in practice (writebacks are always
	// either ACC or VF), but preserved for safety.
	emitPartialLaneStore(dst_off, xyzw);
}

// Writeback for ops that DO NOT update MAC/Status flags (MAX, MINI, ABS).
// The interpreter's _vuMAX/_vuMINI*/_vuABS write directly to VF[fd] via
// applyMinMax/applyUnaryFunction, neither of which touches macflag/statusflag.
// Going through vu1_fmac_writeback would corrupt MAC flags that subsequent
// FMAND/FCAND/etc. read — observed as geometry dropouts in San Andreas.
//
// fd: destination register index (compile-time known via VU1.code).
//     Interpreter returns immediately when fd==0, so no write *and* no flag
//     update — match that exactly (cannot accidentally write VF[0]).
// xyzw: write mask (compile-time known via VU1.code bits 21-24).
// Result must be in v5.4S.
static void emitNoFlagWriteback(u32 fd, u32 xyzw)
{
	if (fd == 0) return; // VF[0] hardwired; interpreter no-ops the whole insn
	// FMAC opt #14: skip dead writes. MAX/MINI/ABS don't touch flags, so
	// there's nothing else to do — pure cache-store skip.
	if (g_vu1DeadVFWrite) return;
	// vf_read_after_write deferred writeback — see g_vu1DeferVfWriteback.
	if (g_vu1DeferVfWriteback)
	{
		armMoveAddressToReg(x4, &g_vu1DeferredVfStash[0]);
		armAsm->Str(v5.Q(), a64::MemOperand(x4));
		g_vu1DeferredVfIdx = static_cast<int>(fd);
		g_vu1DeferredXyzw  = static_cast<u8>(xyzw);
		return;
	}
	// FMAC opt #17: same-VF different-lane batching. MAX/MINI/ABS are
	// FMAC-pipe ops, so Pass 1 may flag them as a batch endpoint.
	if (tryBatchedVfCacheStore(static_cast<int>(fd), xyzw))
		return;
	// Phase 2: defer the write into the cache. Memory store fires at the
	// next flush point. Mirrors emitFmacStoreMasked's VF path.
	vfCacheStore(static_cast<int>(fd), v5, static_cast<u8>(xyzw));
}

// ============================================================================
//  NEON FMAC emit patterns
//
//  Binary (ADD/SUB/MUL):  dst = VF[fs] OP src2
//  Ternary (MADD/MSUB):   dst = ACC ± VF[fs] * src2
//
//  src2 is loaded into v1:
//    Broadcast x/y/z/w: ldr s1 + dup v1.4s
//    Q register:        ldr w0 from VI[REG_Q], dup v1.4s, w0
//    I register:        ldr w0 from VI[REG_I], dup v1.4s, w0
//    Full vector:       ldr q1 from VF[ft]
// ============================================================================

// Load second operand broadcast from VF[ft] component into v1.4S.
// Uses the VF cache: a hit lets us Dup directly from a resident slot
// without touching memory. A miss loads the full 16-byte VF into a slot
// (vs the prior 4-byte single-lane Ldr) — slightly more dcache traffic on
// the first read, but every subsequent broadcast across the chain is a
// register-only Dup. Matters most for matrix-transform chains that read
// the same vertex VF four times in a row.
static void emitLoadBroadcast(u32 ft, int comp) // comp: 0=x,1=y,2=z,3=w
{
	// FMAC opt #16: broadcast operand cache. v1 already holds
	// broadcast(VF[ft], comp) from the prior pair AND no invalidation
	// has fired since — skip the Dup/Fmov entirely. Saves 1 Dup per
	// reused pair, plus 1 Ldr in the rare case that VF[ft]'s cache slot
	// had been evicted (vfCacheLoadResident's miss path).
	if (s_broadcastCacheValid && s_broadcastCacheFt == ft && s_broadcastCacheComp == comp)
	{
		s_broadcastPlanKind = BroadcastPlanKind::V1Held;
		return;
	}

	// FMAC opt #18: defer the Dup. Plan the broadcast source so a downstream
	// FMUL consumer can use ARM64's lane-indexed Fmul (`Fmul Vd, Vn, Vm.s[lane]`)
	// and skip the Dup entirely. Consumers that need v1 as a vector
	// (FADD/FSUB, clamp on v1, addSubHack) call emitMaterializeBroadcast first.
	if (ft == 0)
	{
		// VF[0] is the hardwired constant (0, 0, 0, 1) — comp 0/1/2 → 0.0f,
		// comp 3 → 1.0f. Recorded as a const-kind plan; materialization builds
		// the constant into v1. Lane-FMUL fast-path doesn't apply here.
		s_broadcastPlanKind = (comp == 3) ? BroadcastPlanKind::ConstOne
		                                  : BroadcastPlanKind::ConstZero;
	}
	else
	{
		// vfCacheLoadResident bumps the slot's LRU clock — a single subsequent
		// vfCacheLoadInto miss inside the same op cannot evict this slot
		// (an LRU eviction picks the LEAST-recently-used slot), so the plan
		// stays valid through the FMUL consumer.
		const auto resident = vfCacheLoadResident(static_cast<int>(ft));
		s_broadcastPlanKind    = BroadcastPlanKind::CacheLane;
		s_broadcastPlanSlotReg = resident.GetCode();
		s_broadcastPlanLane    = comp;
	}

	// v1 doesn't actually hold (ft, comp) yet — materialization will set this.
	s_broadcastCacheValid = false;
	s_broadcastCacheFt    = ft;
	s_broadcastCacheComp  = comp;
}

// FMAC opt #18: materialize the deferred broadcast into v1. Idempotent — a
// V1Held plan is a no-op. Called by every consumer that needs v1 as a real
// vector (FADD/FSUB, clamp-on-v1, addSubHack, MAX/MINI). Lane-indexed FMUL
// callers skip this and use tryEmitFmulLaneBroadcast instead.
static void emitMaterializeBroadcast()
{
	// Track whether the materialize result is a VF (ft, comp) broadcast — only
	// then is the (ft, comp) cache below valid for next-pair hits. QI stays
	// off the cache because it's not keyed by VF.
	bool isVfBroadcast = true;
	switch (s_broadcastPlanKind)
	{
		case BroadcastPlanKind::None:
		case BroadcastPlanKind::V1Held:
			return;
		case BroadcastPlanKind::ConstZero:
			armAsm->Movi(v1.V4S(), 0);
			break;
		case BroadcastPlanKind::ConstOne:
			armAsm->Fmov(v1.V4S(), 1.0f);
			break;
		case BroadcastPlanKind::CacheLane:
		{
			const a64::VRegister slotReg(s_broadcastPlanSlotReg, 128);
			armAsm->Dup(v1.V4S(), slotReg.V4S(), s_broadcastPlanLane);
			break;
		}
		case BroadcastPlanKind::QIScalar:
			// Scalar pre-loaded in v1.s[0]; broadcast it into all 4 lanes.
			// Same-source-dst Dup is well-defined (read-then-write).
			armAsm->Dup(v1.V4S(), v1.V4S(), 0);
			isVfBroadcast = false;
			break;
		case BroadcastPlanKind::FullSlot:
		{
			// Full VF[ft] vector lives in a cache slot. Mov it into v1 for
			// any consumer that can't read the slot directly (clamp on v1,
			// MAX/MINI's Smax/Smin operands, addSubHack mutation).
			const a64::VRegister slotReg(s_broadcastPlanSlotReg, 128);
			if (slotReg.GetCode() != v1.GetCode())
				armAsm->Mov(v1.V16B(), slotReg.V16B());
			isVfBroadcast = false; // Not a (ft, comp) broadcast — keep cache invalid.
			break;
		}
	}
	s_broadcastPlanKind   = BroadcastPlanKind::V1Held;
	s_broadcastCacheValid = isVfBroadcast;
}

// FMAC opt #18: emit `Fmul vd.4s, vn.4s, slot.s[lane]` when the broadcast plan
// has a NEON-resident source (VF cache slot OR Q/I scalar pre-loaded into
// v1.s[0]). Returns true on emit, false if the plan is None/V1Held/Const*
// (caller falls back to materialize + regular Fmul). Same single-rounded
// semantics as Dup-then-Fmul, so PS2's two-rounding (separate FMUL + FADD)
// remains preserved by the caller emitting an FADD/FSUB after this.
//
// vixl signature: Fmul(vd, vn, vm, vm_index) requires vm to be the 1-element
// single-precision scalar form (.S(), Is1S) — not .V4S(). Passing .V4S()
// triggers vixl's assert at assembler-aarch64.cc:4659.
static bool tryEmitFmulLaneBroadcast(const a64::VRegister& vd, const a64::VRegister& vn)
{
	if (s_broadcastPlanKind != BroadcastPlanKind::CacheLane
		&& s_broadcastPlanKind != BroadcastPlanKind::QIScalar)
		return false;
	const a64::VRegister slotReg(s_broadcastPlanSlotReg, 128);
	armAsm->Fmul(vd.V4S(), vn.V4S(), slotReg.S(), s_broadcastPlanLane);
	return true;
}

// FMAC opt #18 (FullSlot extension): plan a full-vector `src2 = VF[ft]` for
// FMAC_*_FULL paths. Replaces the old `vfCacheLoadInto(ft, v1)` pre-load —
// FMUL/FADD/FSUB consumers can read the cache slot directly (`Fmul Vd.4s,
// Vn.4s, slot.4s`) and skip the Mov-into-v1 entirely. Saves 1 Mov per
// full-vector FMAC op when the slot is resident; same Ldr cost on a miss.
//
// VF[0] is hardwired memory-only (vfCacheLoadInto special-cases ft=0 to
// Ldr from memory and never allocates a cache slot). For the rare ft==0
// FULL path, fall back to the legacy v1 path so consumers still see a
// well-formed v1 vector.
static void emitPlanFullSlot(u32 ft)
{
	// Reset prior plan/v1-broadcast — caller is about to overwrite ft's slot
	// state, and any stale broadcast in v1 is no longer safe to reuse.
	s_broadcastCacheValid = false;

	if (ft == 0)
	{
		// Legacy path: VF[0] not in the slot pool. Load full vector to v1.
		vfCacheLoadInto(0, v1);
		s_broadcastPlanKind = BroadcastPlanKind::V1Held;
		return;
	}

	const auto resident = vfCacheLoadResident(static_cast<int>(ft));
	s_broadcastPlanKind    = BroadcastPlanKind::FullSlot;
	s_broadcastPlanSlotReg = resident.GetCode();
	// Lane unused for FullSlot.
}

// Load Q or I register broadcast.
//
// FMAC opt #18 (extended): defer the Dup. Ldr the scalar into v1.s[0] only
// and record a QIScalar plan. FMUL consumers use lane-indexed `Fmul Vd.4s,
// Vn.4s, v1.s[0]` directly. ADD/SUB/clamp consumers materialize via
// `Dup v1.4s, v1.s[0]` (idempotent). Saves one Dup per MULq/MULi pair.
static void emitLoadQI(int64_t vi_off)
{
	// v1 broadcast cache: invalid — v1 no longer holds a VF broadcast.
	s_broadcastCacheValid = false;
	// Ldr s1 zero-extends the upper 96 bits of v1 — only v1.s[0] is meaningful.
	armAsm->Ldr(s1, MemOperand(VU1_BASE_REG, vi_off));
	s_broadcastPlanKind    = BroadcastPlanKind::QIScalar;
	s_broadcastPlanSlotReg = v1.GetCode();
	s_broadcastPlanLane    = 0;
}

// --- vuDouble-style input clamping for NEON vectors ---
// The interpreter calls vuDouble() on every FMAC input, which:
//   exp=0 (denormal): flush to ±0 (ARM64 FZ mode handles this)
//   exp=0xFF (inf/NaN): clamp to ±MAX_FLOAT (gated on CHECK_VU_SIGN_OVERFLOW)
// Mirrors the x86 JIT's sign-preserving operand clamp (mVUclamp2 in
// microVU_Clamp.inl): SMIN against +MAX_FLOAT handles positive overflow,
// UMIN against -MAX_FLOAT (as unsigned bit pattern 0xFF7FFFFF) handles
// negative overflow, leaving the sign bit of -NaN / -INF intact.
// Setup: load +MAX_FLOAT into v6, -MAX_FLOAT into v7.

static void emitVuClampSetup()
{
	armAsm->Mov(w0, 0xFFFF);
	armAsm->Movk(w0, 0x7F7F, 16); // w0 = 0x7F7FFFFF = +MAX_FLOAT
	armAsm->Dup(v6.V4S(), w0);
	armAsm->Fneg(v7.V4S(), v6.V4S()); // v7 = 0xFF7FFFFF = -MAX_FLOAT
}

static void emitVuClampVec(const VRegister& vn)
{
	// Signed 32-bit min clamps +overflow (+NaN/+INF > +MAX as signed int).
	armAsm->Smin(vn, vn, v6.V4S());
	// Unsigned 32-bit min clamps -overflow (-NaN/-INF unsigned > -MAX's
	// unsigned bit pattern 0xFF7FFFFF). Positive values (<= 0x7F7FFFFF)
	// are smaller than 0xFF7FFFFF unsigned and pass through unchanged.
	armAsm->Umin(vn, vn, v7.V4S());
}

// VuAddSubHack (Tri-ace games): when |exp(to) - exp(from)| >= 25, the
// smaller-exponent operand contributes nothing to the sum at the bit-exact
// level the game's encryption check expects. Zero its mantissa+exponent
// (preserving sign) before the FADD. Mirrors x86 ADD_SS_TriAceHack at
// microVU_Misc.inl:509-535. Applied per lane via NEON; only the active xyzw
// lane(s) write back so non-active lane outcomes are discarded harmlessly.
//
// Per upstream: only ADDi/ADDq scalar ops with single-lane mask apply this.
// Multi-lane ADDi/ADDq uses SSE_ADD2PS which doesn't include the hack. Other
// ops (SUB, MUL, ADD broadcast, full-vector ADD) never apply it.
//
// Scratch: uses v6-v9. Caller must emit emitVuClampSetup BEFORE this if
// clamps are also needed (clamp leaves v6/v7 with constants we then
// overwrite).
static void emitVuAddSubHack(const VRegister& to, const VRegister& from)
{
	// FMAC opt #16: this conditionally zeroes lanes of `to`/`from`. When
	// called with v1 as the `from` operand (the typical site in
	// emitBinaryFmac), v1 is mutated to a non-broadcast value — even if
	// the runtime mask doesn't fire, the cache state would no longer
	// correspond to "v1 == broadcast(VF[ft], comp)" because the lane
	// mask was applied. Conservatively invalidate.
	if (to.GetCode() == v1.GetCode() || from.GetCode() == v1.GetCode())
		vu1BroadcastCacheReset();
	// Per-lane exponent extract: (raw >> 23) & 0xFF.
	armAsm->Ushr(v6.V4S(), to.V4S(), 23);
	armAsm->Ushr(v7.V4S(), from.V4S(), 23);
	armAsm->Movi(v8.V4S(), 0xFF);
	armAsm->And (v6.V16B(), v6.V16B(), v8.V16B());
	armAsm->And (v7.V16B(), v7.V16B(), v8.V16B());
	// v7 = exp(from) - exp(to) (signed). Matches upstream ecx variable.
	armAsm->Sub (v7.V4S(), v7.V4S(), v6.V4S());

	// Sign-only mask: 0x80000000 broadcast (8-bit imm 0x80 shifted left 24).
	armAsm->Movi(v8.V4S(), 0x80, vixl::aarch64::LSL, 24);

	// Case (exp_diff >= 25): zero 'to' except sign.
	{
		armAsm->Movi(v9.V4S(), 25);
		armAsm->Cmge(v9.V4S(), v7.V4S(), v9.V4S()); // mask: v7 >= 25
		armAsm->And (v6.V16B(), to.V16B(), v8.V16B()); // sign-only 'to'
		// Bsl(vd, vn, vm): vd = (vd & vn) | (~vd & vm). v9 selects: lanes
		// where v9 is all-1 take v6 (sign-only), else take 'to' (unchanged).
		armAsm->Bsl (v9.V16B(), v6.V16B(), to.V16B());
		armAsm->Mov (to.V16B(), v9.V16B());
	}

	// Case (exp_diff <= -25): zero 'from' except sign. NEON CMLE only takes
	// an immediate-zero RHS, so express via Cmge with operands swapped:
	// (v7 <= -25) <=> (-25 >= v7).
	{
		armAsm->Movi(v9.V4S(), 25);
		armAsm->Neg (v9.V4S(), v9.V4S()); // v9 = -25 (per lane)
		armAsm->Cmge(v9.V4S(), v9.V4S(), v7.V4S()); // mask: -25 >= v7  ==  v7 <= -25
		armAsm->And (v6.V16B(), from.V16B(), v8.V16B());
		armAsm->Bsl (v9.V16B(), v6.V16B(), from.V16B());
		armAsm->Mov (from.V16B(), v9.V16B());
	}
}

// --- Binary FMAC: dst = VF[fs] OP src2 ---
// op: 0=ADD, 1=SUB, 2=MUL
// isScalarAdd: caller is ADDi/ADDq (scalar broadcast ADD). Gates VuAddSubHack.
// same_operands: caller has already loaded v1 = VF[fs] (e.g., FULL variants
//   where fs == ft). Skips the redundant vfCacheLoadInto for v0 and instead
//   copies v1 → v0. Saves one Ldr/Mov per SQR-style op (MUL vfX,vfA,vfA;
//   ADD vfX,vfA,vfA). #3 from FMAC optimization deep-dive.
static void emitBinaryFmac(int op, u32 fs, int64_t dst_off, u32 xyzw,
                           bool isScalarAdd = false, bool same_operands = false)
{
	const bool needsFlags  = g_vu1NeedsFlags;
	const bool clampInputs = CHECK_VU_SIGN_OVERFLOW(0) || CHECK_VU_SIGN_OVERFLOW(1);
	// Output clamping is only required when we're skipping the C helper —
	// VU_MAC_UPDATE clamps inf/NaN→±MAX inside the C path. With flags
	// deferred, we replicate that behavior with FMINNM/FMAXNM in NEON.
	const bool clampOutput = !needsFlags && CHECK_VU_OVERFLOW(1);

	// VuAddSubHack gate (Tri-ace ADDi/ADDq with single-lane mask only).
	const bool singleLane = (xyzw == 0x1u || xyzw == 0x2u || xyzw == 0x4u || xyzw == 0x8u);
	const bool addSubHack = isScalarAdd && CHECK_VUADDSUBHACK && singleLane;

	// FMAC opt #18: fast-path eligibility. Two flavors:
	//   - FullSlot: read VF[ft] directly from its cache slot for FADD/FSUB/FMUL.
	//               Saves the Mov-into-v1 the FMAC_*_FULL macro used to emit.
	//   - Lane-FMUL (CacheLane/QIScalar): only for op=MUL; emits `Fmul vd, vn,
	//               slot.s[lane]` and skips the Dup-into-v1.
	// Both gates: !clampInputs (clamp needs v1 vector), !addSubHack (mutates v1),
	// !same_operands (we still use v0=Mov(v1) on the legacy path).
	const bool fastPathFullSlot = !same_operands && !clampInputs && !addSubHack
		&& (s_broadcastPlanKind == BroadcastPlanKind::FullSlot);
	const bool fastPathLaneFmul = !same_operands && !clampInputs && !addSubHack
		&& (op == 2)
		&& (s_broadcastPlanKind == BroadcastPlanKind::CacheLane
		    || s_broadcastPlanKind == BroadcastPlanKind::QIScalar);
	const bool fastPathAvailable = fastPathFullSlot || fastPathLaneFmul;
	const bool needsV1Materialized = !fastPathAvailable;
	if (needsV1Materialized)
		emitMaterializeBroadcast();

	// Load VF[fs] → v0 (via cache: hit → Mov from resident slot, miss → Ldr).
	if (same_operands)
		armAsm->Mov(v0.V16B(), v1.V16B());
	else
		vfCacheLoadInto(static_cast<int>(fs), v0);

	// vuDouble input clamping (matches interpreter's vuDouble on every operand)
	if (clampInputs || clampOutput)
		emitVuClampSetup();
	if (clampInputs)
	{
		emitVuClampVec(v0.V4S());
		emitVuClampVec(v1.V4S());
	}

	// VuAddSubHack: apply AFTER clamp (which uses v6/v7 for constants we
	// overwrite). Scalar single-lane ADDi/ADDq only — see helper comment.
	if (addSubHack)
		emitVuAddSubHack(v0, v1);

	// Perform NEON op: v5 = v0 OP src2 (slot-direct on fast path, v1 otherwise).
	bool opEmitted = false;
	if (fastPathFullSlot)
	{
		const a64::VRegister slotReg(s_broadcastPlanSlotReg, 128);
		switch (op) {
			case 0: armAsm->Fadd(v5.V4S(), v0.V4S(), slotReg.V4S()); break;
			case 1: armAsm->Fsub(v5.V4S(), v0.V4S(), slotReg.V4S()); break;
			case 2: armAsm->Fmul(v5.V4S(), v0.V4S(), slotReg.V4S()); break;
		}
		opEmitted = true;
	}
	else if (fastPathLaneFmul)
	{
		opEmitted = tryEmitFmulLaneBroadcast(v5, v0);
	}
	if (!opEmitted)
	{
		// V1Held/None/ConstZero/ConstOne hit the legacy v1 path. Materialize
		// is idempotent (no-op for V1Held/None, emits constant for Const*).
		emitMaterializeBroadcast();
		switch (op) {
			case 0: armAsm->Fadd(v5.V4S(), v0.V4S(), v1.V4S()); break;
			case 1: armAsm->Fsub(v5.V4S(), v0.V4S(), v1.V4S()); break;
			case 2: armAsm->Fmul(v5.V4S(), v0.V4S(), v1.V4S()); break;
		}
	}

	if (needsFlags)
	{
		emitFmacWriteback(dst_off, xyzw);
	}
	else
	{
		if (clampOutput)
			emitVuClampVec(v5.V4S());
		emitFmacStoreMasked(dst_off, xyzw);
	}
}

// --- Ternary FMAC: dst = ACC ± VF[fs] * src2 ---
// subtract: false=MADD (ACC + fs*src2), true=MSUB (ACC - fs*src2)
// same_operands: caller has already loaded v1 = VF[fs] (FULL variants where
//   fs == ft). Skips the redundant vfCacheLoadInto for v0. #3 from FMAC
//   optimization deep-dive.
// NOTE: The PS2 VU does NOT have fused multiply-add. It performs a separate
// multiply then add/sub with intermediate rounding. Using FMLA/FMLS would
// produce results differing by 1+ ULP, causing cascading precision errors
// in matrix transforms. We use separate FMUL + FADD/FSUB to match.
static void emitTernaryFmac(bool subtract, u32 fs, int64_t dst_off, u32 xyzw,
                            bool same_operands = false)
{
	const bool needsFlags  = g_vu1NeedsFlags;
	const bool clampInputs = CHECK_VU_SIGN_OVERFLOW(0) || CHECK_VU_SIGN_OVERFLOW(1);
	const bool clampOutput = !needsFlags && CHECK_VU_OVERFLOW(1);

	// FMAC opt #18: fast-path eligibility for the multiply step. Two flavors:
	//   - FullSlot: read VF[ft] directly from cache slot for `Fmul v4, v0, slot`.
	//   - Lane-FMUL (CacheLane/QIScalar): `Fmul v4, v0, slot.s[lane]`.
	// Subsequent FADD/FSUB still preserves PS2's two-rounding (FMUL + separate
	// FADD), since the slot-direct/lane-indexed FMUL is single-rounded only.
	const bool fastPathFullSlot = !same_operands && !clampInputs
		&& (s_broadcastPlanKind == BroadcastPlanKind::FullSlot);
	const bool fastPathLaneFmul = !same_operands && !clampInputs
		&& (s_broadcastPlanKind == BroadcastPlanKind::CacheLane
		    || s_broadcastPlanKind == BroadcastPlanKind::QIScalar);
	const bool fastPathAvailable = fastPathFullSlot || fastPathLaneFmul;
	const bool needsV1Materialized = !fastPathAvailable;
	if (needsV1Materialized)
		emitMaterializeBroadcast();

	// Load VF[fs] → v0 (via cache).
	if (same_operands)
		armAsm->Mov(v0.V16B(), v1.V16B());
	else
		vfCacheLoadInto(static_cast<int>(fs), v0);
	// v1 must already be loaded with src2.
	// Phase-8: ACC lives in pinned VU1_ACC_REG (q16). Copy into v5 for the
	// ±FMA pipeline below instead of reading from memory.
	armAsm->Mov(v5.V16B(), VU1_ACC_REG.V16B());

	// vuDouble input clamping (all 3 operands, matching interpreter)
	if (clampInputs || clampOutput)
		emitVuClampSetup();
	if (clampInputs)
	{
		emitVuClampVec(v0.V4S());
		emitVuClampVec(v1.V4S());
		emitVuClampVec(v5.V4S());
	}

	// Separate multiply: v4 = VF[fs] * src2 (with rounding). Slot-direct or
	// lane-indexed fast paths when the broadcast plan supports it.
	bool fmulEmitted = false;
	if (fastPathFullSlot)
	{
		const a64::VRegister slotReg(s_broadcastPlanSlotReg, 128);
		armAsm->Fmul(v4.V4S(), v0.V4S(), slotReg.V4S());
		fmulEmitted = true;
	}
	else if (fastPathLaneFmul)
	{
		fmulEmitted = tryEmitFmulLaneBroadcast(v4, v0);
	}
	if (!fmulEmitted)
	{
		// VF[0] const-broadcast (e.g., MADDw...vf00) and prior-pair-V1Held
		// hits land here and need v1 to be a real vector for the Fmul.
		// Idempotent on V1Held/None; emits Movi/Fmov for ConstZero/One.
		emitMaterializeBroadcast();
		armAsm->Fmul(v4.V4S(), v0.V4S(), v1.V4S());
	}

	// Separate add/sub: v5 = ACC ± product (with rounding)
	if (subtract)
		armAsm->Fsub(v5.V4S(), v5.V4S(), v4.V4S());
	else
		armAsm->Fadd(v5.V4S(), v5.V4S(), v4.V4S());

	if (needsFlags)
	{
		emitFmacWriteback(dst_off, xyzw);
	}
	else
	{
		if (clampOutput)
			emitVuClampVec(v5.V4S());
		emitFmacStoreMasked(dst_off, xyzw);
	}
}

// --- OPMULA / OPMSUB: cross-product outer-product FMAC ---
//
// OPMULA: ACC.xyz  = (fs.y*ft.z, fs.z*ft.x, fs.x*ft.y)     — ACC.w preserved
// OPMSUB: VF[fd].xyz = ACC.xyz - (fs.y*ft.z, fs.z*ft.x, fs.x*ft.y) — VF[fd].w preserved
//
// Only xyz lanes are meaningful; the instruction's xyzw field is ignored by
// the interpreter (MAC[w] is never touched either — that's why these need
// their own writeback helpers instead of vu1_fmac_writeback).
//
// Permutation layout:
//   v2 = {fs.y, fs.z, fs.x, _}   v3 = {ft.z, ft.x, ft.y, _}
// so that v4 = v2 * v3 produces lanes [rx, ry, rz, _] in natural ACC order.
//
// isSub: false=OPMULA (result→ACC), true=OPMSUB (result→VF[fd], fd_param used)
// fd_or_zero: for OPMSUB, the destination VF index (compile-time from VU1.code);
//             ignored for OPMULA (pass 0).
static void emitOpFmac(bool isSub, u32 fs, u32 ft, u32 fd_or_zero)
{
	const bool needsFlags  = g_vu1NeedsFlags;
	const bool clampInputs = CHECK_VU_SIGN_OVERFLOW(0) || CHECK_VU_SIGN_OVERFLOW(1);
	const bool clampOutput = !needsFlags && CHECK_VU_OVERFLOW(1);

	vfCacheLoadInto(static_cast<int>(fs), v0);
	vfCacheLoadInto(static_cast<int>(ft), v1);
	// Phase-8: OPMSUB reads ACC — pull from the pinned register, not memory.
	if (isSub)
		armAsm->Mov(v5.V16B(), VU1_ACC_REG.V16B());

	if (clampInputs || clampOutput)
		emitVuClampSetup();
	if (clampInputs)
	{
		emitVuClampVec(v0.V4S());
		emitVuClampVec(v1.V4S());
		if (isSub)
			emitVuClampVec(v5.V4S()); // ACC is a third input to OPMSUB
	}

	// Permute fs → v2 = {y, z, x, _}
	armAsm->Mov(v2.V4S(), 0, v0.V4S(), 1);
	armAsm->Mov(v2.V4S(), 1, v0.V4S(), 2);
	armAsm->Mov(v2.V4S(), 2, v0.V4S(), 0);
	// Permute ft → v3 = {z, x, y, _}
	armAsm->Mov(v3.V4S(), 0, v1.V4S(), 2);
	armAsm->Mov(v3.V4S(), 1, v1.V4S(), 0);
	armAsm->Mov(v3.V4S(), 2, v1.V4S(), 1);

	if (isSub)
	{
		// Separate FMUL then FSUB (no FMLS — PS2 has non-fused rounding).
		armAsm->Fmul(v4.V4S(), v2.V4S(), v3.V4S());
		armAsm->Fsub(v5.V4S(), v5.V4S(), v4.V4S());
	}
	else
	{
		// OPMULA: product goes straight into v5 (the writeback source).
		armAsm->Fmul(v5.V4S(), v2.V4S(), v3.V4S());
	}

	if (needsFlags)
	{
		// OpmXYZ mode inlines VU_MAC[x/y/z]_UPDATE + VU_STAT_UPDATE and
		// preserves macflag[w]'s existing bits (the interpreter never
		// touches MACw for OPMULA/OPMSUB). For OPMSUB fd==0 the inline
		// writeback skips the VF store via its vfOffStatic0 check.
		const int64_t dst_off = isSub ? vfOff(fd_or_zero) : accOff();
		emitFmacInlineWriteback(dst_off, 0xE, FmacWritebackMode::OpmXYZ);
	}
	else
	{
		if (clampOutput)
			emitVuClampVec(v5.V4S());
		// xyzw = 0xE → xyz write only, W lane preserved by merge-load-store.
		// For OPMSUB fd==0, emitFmacStoreMasked early-outs on dst_off==vfOff(0).
		const int64_t dst_off = isSub ? vfOff(fd_or_zero) : accOff();
		emitFmacStoreMasked(dst_off, 0xE);
	}
}

// --- MAX/MINI: bit-exact emulation of interpreter fp_max / fp_min ---
//
// PS2 VU MAX/MINI are NOT IEEE float compares: the interpreter does a
// sign-magnitude integer compare on the raw 32-bit bit patterns:
//
//     fp_max(a,b) = (both negative) ? signed_min(a,b) : signed_max(a,b)
//     fp_min(a,b) = (both negative) ? signed_max(a,b) : signed_min(a,b)
//
// FMAXNM/FMINNM cannot be substituted. Two divergences hit real game data:
//   1. NaN inputs — FMAXNM picks the non-NaN operand; fp_max preserves the
//      NaN bit pattern via integer compare.
//   2. Denormal inputs — under VU FPCR (FZ=1) FMAXNM flushes denormals to
//      ±0 before comparing; fp_max preserves the denormal bit pattern.
//
// Observed in San Andreas as shadow corruption: a full-vector `MAX vfX,vfY,vf0`
// (clamp-to-zero against vf0 = {0,0,0,1}) silently zeroed NaN/denormal lanes
// the rest of the VU program expected to flow through bit-perfectly.
//
// MUST use emitNoFlagWriteback — MAX/MINI on PS2 don't update MAC/Status.
static void emitMaxFmac(bool isMini, u32 fs, u32 fd, u32 xyzw)
{
	// FMAC opt #18: MAX/MINI consume v1 as a vector (Smax/Smin operands), so
	// we must materialize the broadcast plan first. ARM64 has no lane-indexed
	// Smax/Smin, and the consumer uses v1 throughout the comparison.
	emitMaterializeBroadcast();

	vfCacheLoadInto(static_cast<int>(fs), v0);
	// v1 must already be loaded with src2

	// Signed integer max/min on 32-bit lanes.
	armAsm->Smax(v2.V4S(), v0.V4S(), v1.V4S());
	armAsm->Smin(v3.V4S(), v0.V4S(), v1.V4S());

	// Build "both negative" mask in v6: arithmetic-shift right by 31 splats
	// the sign bit, AND of the two splats is 0xFFFFFFFF per lane iff both
	// inputs had the sign bit set.
	armAsm->Sshr(v4.V4S(), v0.V4S(), 31);
	armAsm->Sshr(v6.V4S(), v1.V4S(), 31);
	armAsm->And(v6.V16B(), v4.V16B(), v6.V16B());

	// Select per lane: result = both_neg ? swapped : signed_(max|min).
	// BIF Vd, Vn, Vm copies Vn into Vd where Vm bit is 0, else keeps Vd.
	if (isMini)
	{
		// fp_min: both_neg → signed_max ; otherwise → signed_min
		armAsm->Mov(v5.V16B(), v2.V16B());            // start = signed_max
		armAsm->Bif(v5.V16B(), v3.V16B(), v6.V16B()); // mask=0 → signed_min
	}
	else
	{
		// fp_max: both_neg → signed_min ; otherwise → signed_max
		armAsm->Mov(v5.V16B(), v3.V16B());            // start = signed_min
		armAsm->Bif(v5.V16B(), v2.V16B(), v6.V16B()); // mask=0 → signed_max
	}
	emitNoFlagWriteback(fd, xyzw);
}

// --- ABS: VF[ft] = fabs(VF[fs]) ---
// Interpreter (applyUnaryFunction<vuOpABS>) reads from _Fs_ and writes to _Ft_,
// returns early when _Ft_==0, and does not touch MAC/Status flags.
static void emitAbsFmac(u32 fs, u32 ft, u32 xyzw)
{
	// FMAC opt #4: ABS-of-known-positive elimination. Pass 1 proved the
	// source's lanes (in xyzw) are sign-bit-zero, so Fabs is a no-op for
	// those lanes. Lanes outside xyzw don't get stored by the masked
	// writeback either, so Fabs on them is irrelevant. Load the slot
	// straight into v5 and skip the Fabs entirely.
	if (g_vu1AbsSrcKnownNonNeg)
	{
		vfCacheLoadInto(static_cast<int>(fs), v5);
	}
	else
	{
		vfCacheLoadInto(static_cast<int>(fs), v0);
		armAsm->Fabs(v5.V4S(), v0.V4S());
	}
	emitNoFlagWriteback(ft, xyzw);
}

// ============================================================================
//  Macro: generate a binary FMAC rec function (ADD/SUB/MUL variants)
//
//  FMAC_BINARY_BC(name, op, comp)    — broadcast VF[ft].comp
//  FMAC_BINARY_Q(name, op)           — Q register broadcast
//  FMAC_BINARY_I(name, op)           — I register broadcast
//  FMAC_BINARY_FULL(name, op)        — full VF[ft] vector
//
//  op: 0=ADD, 1=SUB, 2=MUL
//  comp: 0=x, 1=y, 2=z, 3=w
//  toACC: false=VF[fd], true=ACC
// ============================================================================

#define FMAC_BINARY_BC(name, op, comp, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 ft = (VU1.code >> 16) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitLoadBroadcast(ft, comp); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitBinaryFmac(op, fs, dst, xyzw); \
	}

#define FMAC_BINARY_Q(name, op, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitLoadQI(viOff(REG_Q)); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitBinaryFmac(op, fs, dst, xyzw, /*isScalarAdd=*/((op) == 0)); \
	}

#define FMAC_BINARY_I(name, op, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitLoadQI(viOff(REG_I)); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitBinaryFmac(op, fs, dst, xyzw, /*isScalarAdd=*/((op) == 0)); \
	}

#define FMAC_BINARY_FULL(name, op, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 ft = (VU1.code >> 16) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		const int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		/* doSafeSub: SUB/SUBA (op=1) full-vector with fs==ft forces +0. \
		 * Mirrors x86 microVU_Upper.inl:173-187. Avoids INF-INF=NaN and  \
		 * NaN-NaN=NaN, either of which would corrupt the Z MAC flag.    \
		 * Broadcast variants (SUBx/y/z/w/q/i) are NOT affected per x86. */ \
		if ((op) == 1 && fs == ft) { \
			armAsm->Movi(v5.V4S(), 0); \
			if (g_vu1NeedsFlags) { \
				emitFmacWriteback(dst, xyzw); \
			} else { \
				emitFmacStoreMasked(dst, xyzw); \
			} \
			return; \
		} \
		/* FMAC opt #18: plan a slot-direct read of VF[ft] instead of pre- \
		 * loading v1. emitBinaryFmac's fast path emits the ALU op directly \
		 * against the slot; legacy materialize + Mov-into-v1 only fires for \
		 * clamp/hack/same_operands paths or VF[0]. */ \
		emitPlanFullSlot(ft); \
		/* SQR detection (#3 deep-dive): MUL/ADD with fs == ft still uses \
		 * the legacy v1-materialized path inside emitBinaryFmac. SUB+self \
		 * is handled above. */ \
		emitBinaryFmac(op, fs, dst, xyzw, /*isScalarAdd=*/false, \
		               /*same_operands=*/(fs == ft)); \
	}

// ============================================================================
//  Macro: generate a ternary FMAC rec function (MADD/MSUB variants)
// ============================================================================

#define FMAC_TERNARY_BC(name, isSub, comp, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 ft = (VU1.code >> 16) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitLoadBroadcast(ft, comp); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitTernaryFmac(isSub, fs, dst, xyzw); \
	}

#define FMAC_TERNARY_Q(name, isSub, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitLoadQI(viOff(REG_Q)); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitTernaryFmac(isSub, fs, dst, xyzw); \
	}

#define FMAC_TERNARY_I(name, isSub, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitLoadQI(viOff(REG_I)); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitTernaryFmac(isSub, fs, dst, xyzw); \
	}

#define FMAC_TERNARY_FULL(name, isSub, toACC) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 ft = (VU1.code >> 16) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		/* FMAC opt #18: plan a slot-direct read of VF[ft]. emitTernaryFmac's \
		 * multiply step emits Fmul against the slot directly; only clamp / \
		 * same_operands / VF[0] paths fall back to v1-materialize. */ \
		emitPlanFullSlot(ft); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		/* SQR detection (#3 deep-dive): when fs == ft, skip the redundant \
		 * v0 load and use v1 for both operands. */ \
		emitTernaryFmac(isSub, fs, dst, xyzw, /*same_operands=*/(fs == ft)); \
	}

// ============================================================================
//  Macro: generate MAX/MINI FMAC rec function
// ============================================================================

#define FMAC_MAXMINI_BC(name, isMini, comp) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 ft = (VU1.code >> 16) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitLoadBroadcast(ft, comp); \
		emitMaxFmac(isMini, fs, fd, xyzw); \
	}

#define FMAC_MAXMINI_I(name, isMini) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitLoadQI(viOff(REG_I)); \
		emitMaxFmac(isMini, fs, fd, xyzw); \
	}

#define FMAC_MAXMINI_FULL(name, isMini) \
	void recVU1_##name() { \
		const u32 fd = (VU1.code >> 6) & 0x1F; \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 ft = (VU1.code >> 16) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		/* FMAC opt #18: plan; emitMaxFmac materializes v1 at entry since \
		 * Smax/Smin have no slot-direct or lane-indexed variants. So this \
		 * doesn't actually save a Mov for MAX/MINI — but using the same \
		 * planning path keeps state consistent (broadcast cache reset, plan \
		 * tracked) instead of the bare vfCacheLoadInto bypass. */ \
		emitPlanFullSlot(ft); \
		emitMaxFmac(isMini, fs, fd, xyzw); \
	}

// ABS: Ft.xyzw = |Ft.xyzw|  (ft is both source and destination)
// ABS: VF[ft] = |VF[fs]|  (interpreter reads _Fs_, writes _Ft_).
// Most code uses fs==ft (in-place), but the JIT must follow the encoding.
#define FMAC_ABS(name) \
	void recVU1_##name() { \
		const u32 fs = (VU1.code >> 11) & 0x1F; \
		const u32 ft = (VU1.code >> 16) & 0x1F; \
		const u32 xyzw = (VU1.code >> 21) & 0xF; \
		emitAbsFmac(fs, ft, xyzw); \
	}

// ============================================================================
//  FTOI / ITOF — native NEON float/int conversions (mirrors iVU0Upper_arm64.cpp)
// ============================================================================

static void emitFTOI(int fbits)
{
	const u32 ft   = (VU1.code >> 16) & 0x1F;
	const u32 fs   = (VU1.code >> 11) & 0x1F;
	const u32 xyzw = (VU1.code >> 21) & 0xF;
	if (ft == 0) return;
	vfCacheLoadInto(static_cast<int>(fs), v0);
	const bool clampInputs = CHECK_VU_SIGN_OVERFLOW(1);
	if (clampInputs)
	{
		// FMINNM/FMAXNM against ±MAX_FLOAT: NaN treated as "missing" and
		// replaced, so NaN cannot reach FCVTZS below.
		emitVuClampSetup();
		emitVuClampVec(v0.V4S());
	}
	// FMAC opt #16: FTOI overwrites v1 with the converted integer vector.
	vu1BroadcastCacheReset();
	if (fbits > 0)
		armAsm->Fcvtzs(v1.V4S(), v0.V4S(), fbits);
	else
		armAsm->Fcvtzs(v1.V4S(), v0.V4S());

	// NaN fixup (only when input clamp is off).
	//
	// Interpreter's floatToInt (VUops.cpp:879) checks post-multiply exponent:
	//   if ((uvalue & 0x7f800000) >= 0x4f000000)
	//       return (sign) ? INT_MIN : INT_MAX;
	// which covers finite overflow (matches ARM64 FCVTZS saturation), ±Inf
	// (also matches FCVTZS: +Inf→INT_MAX, -Inf→INT_MIN), AND NaN (where
	// ARM64 FCVTZS diverges by returning 0).
	//
	// Replicate the NaN case: overwrite NaN lanes with sign-based
	// INT_MIN/INT_MAX. Finite + Inf already match, so we only need a
	// NaN-specific mask (IEEE self-equality: NaN != NaN).
	if (!clampInputs)
	{
		armAsm->Fcmeq(v2.V4S(), v0.V4S(), v0.V4S()); // 1s for non-NaN, 0s for NaN
		armAsm->Mvn(v2.V16B(), v2.V16B());            // invert → 1s for NaN
		armAsm->Sshr(v3.V4S(), v0.V4S(), 31);        // v3 = 0 (pos) or 0xFFFFFFFF (neg)
		armAsm->Mov(w0, 0x7FFFFFFFu);
		armAsm->Dup(v4.V4S(), w0);
		armAsm->Eor(v3.V16B(), v3.V16B(), v4.V16B()); // v3 = 0x7FFFFFFF (pos) or 0x80000000 (neg)
		armAsm->Bit(v1.V16B(), v3.V16B(), v2.V16B()); // blend: NaN lanes ← v3
	}

	// Phase 2: defer the write through the cache. vfCacheStore handles the
	// xyzw mask internally — full-mask path skips the load-merge, partial-
	// mask path force-loads the slot and Mov-merges. Replaces the prior
	// hand-rolled Ldr q5 + lane Movs + Str q5 sequence.
	//
	// vf_read_after_write deferred writeback: spill v1 to the stash and
	// record idx + xyzw. Commit lands in CompileBlock after lower's emit.
	if (g_vu1DeferVfWriteback)
	{
		armMoveAddressToReg(x4, &g_vu1DeferredVfStash[0]);
		armAsm->Str(v1.Q(), a64::MemOperand(x4));
		g_vu1DeferredVfIdx = static_cast<int>(ft);
		g_vu1DeferredXyzw  = static_cast<u8>(xyzw);
		return;
	}
	vfCacheStore(static_cast<int>(ft), v1, static_cast<u8>(xyzw));
}

static void emitITOF(int fbits)
{
	const u32 ft   = (VU1.code >> 16) & 0x1F;
	const u32 fs   = (VU1.code >> 11) & 0x1F;
	const u32 xyzw = (VU1.code >> 21) & 0xF;
	if (ft == 0) return;
	vfCacheLoadInto(static_cast<int>(fs), v0);
	// FMAC opt #16: ITOF overwrites v1 with the converted float vector.
	vu1BroadcastCacheReset();
	if (fbits > 0)
		armAsm->Scvtf(v1.V4S(), v0.V4S(), fbits);
	else
		armAsm->Scvtf(v1.V4S(), v0.V4S());
	if (g_vu1DeferVfWriteback)
	{
		armMoveAddressToReg(x4, &g_vu1DeferredVfStash[0]);
		armAsm->Str(v1.Q(), a64::MemOperand(x4));
		g_vu1DeferredVfIdx = static_cast<int>(ft);
		g_vu1DeferredXyzw  = static_cast<u8>(xyzw);
		return;
	}
	vfCacheStore(static_cast<int>(ft), v1, static_cast<u8>(xyzw));
}

// ============================================================================
//  C wrappers for OPMULA / OPMSUB (ISTUB fallback path only)
// ============================================================================

// OPMULA: ACC.xyz = cross-like multiply (fs.y*ft.z, fs.z*ft.x, fs.x*ft.y)
// x86 microVU (microVU_Upper.inl:422-445) emits raw SSE_MULPS on the shuffled
// operands with NO mVUclamp2 call. Matches that: read VF.f directly (VU FPCR
// handles denormal flush via FTZ mode), no vu1Double input clamping.
static void vu1_OPMULA(VURegs* VU)
{
	const u32 fs = (VU->code >> 11) & 0x1F;
	const u32 ft = (VU->code >> 16) & 0x1F;
	VU->ACC.i.x = VU_MACx_UPDATE(VU, VU->VF[fs].f.y * VU->VF[ft].f.z);
	VU->ACC.i.y = VU_MACy_UPDATE(VU, VU->VF[fs].f.z * VU->VF[ft].f.x);
	VU->ACC.i.z = VU_MACz_UPDATE(VU, VU->VF[fs].f.x * VU->VF[ft].f.y);
	VU_STAT_UPDATE(VU);
}

// OPMSUB: VF[fd].xyz = ACC.xyz - cross-like multiply (fs.y*ft.z, fs.z*ft.x, fs.x*ft.y)
// x86 microVU (microVU_Upper.inl:447-474) emits SSE_MULPS + SSE_SUBPS without
// mVUclamp2 on operands. Match: read VF.f/ACC.f directly.
// VF[0] writes are discarded (flags still updated).
static void vu1_OPMSUB(VURegs* VU)
{
	const u32 fs = (VU->code >> 11) & 0x1F;
	const u32 ft = (VU->code >> 16) & 0x1F;
	const u32 fd = (VU->code >>  6) & 0x1F;
	const float ftx = VU->VF[ft].f.x;
	const float fty = VU->VF[ft].f.y;
	const float ftz = VU->VF[ft].f.z;
	const float fsx = VU->VF[fs].f.x;
	const float fsy = VU->VF[fs].f.y;
	const float fsz = VU->VF[fs].f.z;
	u32 rx = VU_MACx_UPDATE(VU, VU->ACC.f.x - fsy * ftz);
	u32 ry = VU_MACy_UPDATE(VU, VU->ACC.f.y - fsz * ftx);
	u32 rz = VU_MACz_UPDATE(VU, VU->ACC.f.z - fsx * fty);
	if (fd != 0)
	{
		VU->VF[fd].i.x = rx;
		VU->VF[fd].i.y = ry;
		VU->VF[fd].i.z = rz;
	}
	VU_STAT_UPDATE(VU);
}

// ============================================================================
//  Per-instruction interp stub toggles (1 = interp, 0 = native)
// ============================================================================

#ifdef INTERP_VU_UPPER
// Group toggle: force all to interpreter
#define ISTUB_VU_ADDx    1
#define ISTUB_VU_ADDy    1
#define ISTUB_VU_ADDz    1
#define ISTUB_VU_ADDw    1
#define ISTUB_VU_ADDq    1
#define ISTUB_VU_ADDi    1
#define ISTUB_VU_ADD     1
#define ISTUB_VU_ADDAx   1
#define ISTUB_VU_ADDAy   1
#define ISTUB_VU_ADDAz   1
#define ISTUB_VU_ADDAw   1
#define ISTUB_VU_ADDAq   1
#define ISTUB_VU_ADDAi   1
#define ISTUB_VU_ADDA    1
#define ISTUB_VU_SUBx    1
#define ISTUB_VU_SUBy    1
#define ISTUB_VU_SUBz    1
#define ISTUB_VU_SUBw    1
#define ISTUB_VU_SUBq    1
#define ISTUB_VU_SUBi    1
#define ISTUB_VU_SUB     1
#define ISTUB_VU_SUBAx   1
#define ISTUB_VU_SUBAy   1
#define ISTUB_VU_SUBAz   1
#define ISTUB_VU_SUBAw   1
#define ISTUB_VU_SUBAq   1
#define ISTUB_VU_SUBAi   1
#define ISTUB_VU_SUBA    1
#define ISTUB_VU_MULx    1
#define ISTUB_VU_MULy    1
#define ISTUB_VU_MULz    1
#define ISTUB_VU_MULw    1
#define ISTUB_VU_MULq    1
#define ISTUB_VU_MULi    1
#define ISTUB_VU_MUL     1
#define ISTUB_VU_MULAx   1
#define ISTUB_VU_MULAy   1
#define ISTUB_VU_MULAz   1
#define ISTUB_VU_MULAw   1
#define ISTUB_VU_MULAq   1
#define ISTUB_VU_MULAi   1
#define ISTUB_VU_MULA    1
#define ISTUB_VU_MADDx   1
#define ISTUB_VU_MADDy   1
#define ISTUB_VU_MADDz   1
#define ISTUB_VU_MADDw   1
#define ISTUB_VU_MADDq   1
#define ISTUB_VU_MADDi   1
#define ISTUB_VU_MADD    1
#define ISTUB_VU_MADDAx  1
#define ISTUB_VU_MADDAy  1
#define ISTUB_VU_MADDAz  1
#define ISTUB_VU_MADDAw  1
#define ISTUB_VU_MADDAq  1
#define ISTUB_VU_MADDAi  1
#define ISTUB_VU_MADDA   1
#define ISTUB_VU_MSUBx   1
#define ISTUB_VU_MSUBy   1
#define ISTUB_VU_MSUBz   1
#define ISTUB_VU_MSUBw   1
#define ISTUB_VU_MSUBq   1
#define ISTUB_VU_MSUBi   1
#define ISTUB_VU_MSUB    1
#define ISTUB_VU_MSUBAx  1
#define ISTUB_VU_MSUBAy  1
#define ISTUB_VU_MSUBAz  1
#define ISTUB_VU_MSUBAw  1
#define ISTUB_VU_MSUBAq  1
#define ISTUB_VU_MSUBAi  1
#define ISTUB_VU_MSUBA   1
#define ISTUB_VU_MAXx    1
#define ISTUB_VU_MAXy    1
#define ISTUB_VU_MAXz    1
#define ISTUB_VU_MAXw    1
#define ISTUB_VU_MAXi    1
#define ISTUB_VU_MAX     1
#define ISTUB_VU_MINIx   1
#define ISTUB_VU_MINIy   1
#define ISTUB_VU_MINIz   1
#define ISTUB_VU_MINIw   1
#define ISTUB_VU_MINIi   1
#define ISTUB_VU_MINI    1
#define ISTUB_VU_ABS     1
#define ISTUB_VU_CLIP    1
#define ISTUB_VU_OPMULA  1
#define ISTUB_VU_OPMSUB  1
#define ISTUB_VU_NOP     1
#define ISTUB_VU_FTOI0   1
#define ISTUB_VU_FTOI4   1
#define ISTUB_VU_FTOI12  1
#define ISTUB_VU_FTOI15  1
#define ISTUB_VU_ITOF0   1
#define ISTUB_VU_ITOF4   1
#define ISTUB_VU_ITOF12  1
#define ISTUB_VU_ITOF15  1
#else
// Per-instruction control: set to 0 to enable native ARM64 codegen
#define ISTUB_VU_ADDx    0
#define ISTUB_VU_ADDy    0
#define ISTUB_VU_ADDz    0
#define ISTUB_VU_ADDw    0
#define ISTUB_VU_ADDq    0
#define ISTUB_VU_ADDi    0
#define ISTUB_VU_ADD     0
#define ISTUB_VU_ADDAx   0
#define ISTUB_VU_ADDAy   0
#define ISTUB_VU_ADDAz   0
#define ISTUB_VU_ADDAw   0
#define ISTUB_VU_ADDAq   0
#define ISTUB_VU_ADDAi   0
#define ISTUB_VU_ADDA    0
#define ISTUB_VU_SUBx    0
#define ISTUB_VU_SUBy    0
#define ISTUB_VU_SUBz    0
#define ISTUB_VU_SUBw    0
#define ISTUB_VU_SUBq    0
#define ISTUB_VU_SUBi    0
#define ISTUB_VU_SUB     0
#define ISTUB_VU_SUBAx   0
#define ISTUB_VU_SUBAy   0
#define ISTUB_VU_SUBAz   0
#define ISTUB_VU_SUBAw   0
#define ISTUB_VU_SUBAq   0
#define ISTUB_VU_SUBAi   0
#define ISTUB_VU_SUBA    0 //here
#define ISTUB_VU_MULx    0
#define ISTUB_VU_MULy    0
#define ISTUB_VU_MULz    0
#define ISTUB_VU_MULw    0
#define ISTUB_VU_MULq    0
#define ISTUB_VU_MULi    0
#define ISTUB_VU_MUL     0
#define ISTUB_VU_MULAx   0
#define ISTUB_VU_MULAy   0
#define ISTUB_VU_MULAz   0
#define ISTUB_VU_MULAw   0
#define ISTUB_VU_MULAq   0
#define ISTUB_VU_MULAi   0
#define ISTUB_VU_MULA    0
#define ISTUB_VU_MADDx   0
#define ISTUB_VU_MADDy   0
#define ISTUB_VU_MADDz   0
#define ISTUB_VU_MADDw   0
#define ISTUB_VU_MADDq   0
#define ISTUB_VU_MADDi   0
#define ISTUB_VU_MADD    0
#define ISTUB_VU_MADDAx  0
#define ISTUB_VU_MADDAy  0
#define ISTUB_VU_MADDAz  0
#define ISTUB_VU_MADDAw  0
#define ISTUB_VU_MADDAq  0
#define ISTUB_VU_MADDAi  0
#define ISTUB_VU_MADDA   0
#define ISTUB_VU_MSUBx   0
#define ISTUB_VU_MSUBy   0
#define ISTUB_VU_MSUBz   0
#define ISTUB_VU_MSUBw   0
#define ISTUB_VU_MSUBq   0
#define ISTUB_VU_MSUBi   0
#define ISTUB_VU_MSUB    0
#define ISTUB_VU_MSUBAx  0
#define ISTUB_VU_MSUBAy  0
#define ISTUB_VU_MSUBAz  0
#define ISTUB_VU_MSUBAw  0
#define ISTUB_VU_MSUBAq  0
#define ISTUB_VU_MSUBAi  0
#define ISTUB_VU_MSUBA   0
#define ISTUB_VU_MAXx    0
#define ISTUB_VU_MAXy    0
#define ISTUB_VU_MAXz    0
#define ISTUB_VU_MAXw    0
#define ISTUB_VU_MAXi    0
#define ISTUB_VU_MAX     0
#define ISTUB_VU_MINIx   0
#define ISTUB_VU_MINIy   0
#define ISTUB_VU_MINIz   0
#define ISTUB_VU_MINIw   0
#define ISTUB_VU_MINIi   0
#define ISTUB_VU_MINI    0
#define ISTUB_VU_ABS     0
#define ISTUB_VU_CLIP    0
#define ISTUB_VU_OPMULA  0
#define ISTUB_VU_OPMSUB  0
#define ISTUB_VU_NOP     0
#define ISTUB_VU_FTOI0   0
#define ISTUB_VU_FTOI4   0
#define ISTUB_VU_FTOI12  0
#define ISTUB_VU_FTOI15  0
#define ISTUB_VU_ITOF0   0
#define ISTUB_VU_ITOF4   0
#define ISTUB_VU_ITOF12  0
#define ISTUB_VU_ITOF15  0
#endif

// ============================================================================
//  Code-emitter macros: called at block-compile time.
//  VU1.code is set by CompileBlock before each of these is called.
// ============================================================================

// INTERP path (ISTUB=1): emit the full pinned-reg flush / BL-to-interp /
// reload dance so the C interpreter sees authoritative state and JIT
// pins pick up any writes the interp made. Without this, Phase-7/8/9
// pinned regs (macflag/statusflag/clipflag in w19/w20/w28, ACC in q16,
// VI[REG_Q] broadcast in q17) drift from memory across the call and
// the hybrid harness produces garbage output.
#define REC_VU1_UPPER_INTERP(name) \
	void recVU1_##name() { \
		emitVU1InterpBL(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f])); \
	}

// C-wrapper path (ISTUB=0): emit BL to a vu1_* C helper.
#define REC_VU1_UPPER_CALL(name) \
	void recVU1_##name() { \
		armAsm->Mov(x0, VU1_BASE_REG); \
		emitVu1Call(reinterpret_cast<const void*>(vu1_##name)); \
	}

// ============================================================================
//  ADD family — VF[fd] = VF[fs] + VF[ft] (broadcast variants)
// ============================================================================

#if ISTUB_VU_ADDx
REC_VU1_UPPER_INTERP(ADDx)
#else
FMAC_BINARY_BC(ADDx, 0, 0, false)
#endif

#if ISTUB_VU_ADDy
REC_VU1_UPPER_INTERP(ADDy)
#else
FMAC_BINARY_BC(ADDy, 0, 1, false)
#endif

#if ISTUB_VU_ADDz
REC_VU1_UPPER_INTERP(ADDz)
#else
FMAC_BINARY_BC(ADDz, 0, 2, false)
#endif

#if ISTUB_VU_ADDw
REC_VU1_UPPER_INTERP(ADDw)
#else
FMAC_BINARY_BC(ADDw, 0, 3, false)
#endif

#if ISTUB_VU_ADDq
REC_VU1_UPPER_INTERP(ADDq)
#else
FMAC_BINARY_Q(ADDq, 0, false)
#endif

#if ISTUB_VU_ADDi
REC_VU1_UPPER_INTERP(ADDi)
#else
FMAC_BINARY_I(ADDi, 0, false)
#endif

#if ISTUB_VU_ADD
REC_VU1_UPPER_INTERP(ADD)
#else
FMAC_BINARY_FULL(ADD, 0, false)
#endif

// ============================================================================
//  ADDA family — ACC = VF[fs] + VF[ft] (broadcast variants)
// ============================================================================

#if ISTUB_VU_ADDAx
REC_VU1_UPPER_INTERP(ADDAx)
#else
FMAC_BINARY_BC(ADDAx, 0, 0, true)
#endif

#if ISTUB_VU_ADDAy
REC_VU1_UPPER_INTERP(ADDAy)
#else
FMAC_BINARY_BC(ADDAy, 0, 1, true)
#endif

#if ISTUB_VU_ADDAz
REC_VU1_UPPER_INTERP(ADDAz)
#else
FMAC_BINARY_BC(ADDAz, 0, 2, true)
#endif

#if ISTUB_VU_ADDAw
REC_VU1_UPPER_INTERP(ADDAw)
#else
FMAC_BINARY_BC(ADDAw, 0, 3, true)
#endif

#if ISTUB_VU_ADDAq
REC_VU1_UPPER_INTERP(ADDAq)
#else
FMAC_BINARY_Q(ADDAq, 0, true)
#endif

#if ISTUB_VU_ADDAi
REC_VU1_UPPER_INTERP(ADDAi)
#else
FMAC_BINARY_I(ADDAi, 0, true)
#endif

#if ISTUB_VU_ADDA
REC_VU1_UPPER_INTERP(ADDA)
#else
FMAC_BINARY_FULL(ADDA, 0, true)
#endif

// ============================================================================
//  SUB family
// ============================================================================

#if ISTUB_VU_SUBx
REC_VU1_UPPER_INTERP(SUBx)
#else
FMAC_BINARY_BC(SUBx, 1, 0, false)
#endif

#if ISTUB_VU_SUBy
REC_VU1_UPPER_INTERP(SUBy)
#else
FMAC_BINARY_BC(SUBy, 1, 1, false)
#endif

#if ISTUB_VU_SUBz
REC_VU1_UPPER_INTERP(SUBz)
#else
FMAC_BINARY_BC(SUBz, 1, 2, false)
#endif

#if ISTUB_VU_SUBw
REC_VU1_UPPER_INTERP(SUBw)
#else
FMAC_BINARY_BC(SUBw, 1, 3, false)
#endif

#if ISTUB_VU_SUBq
REC_VU1_UPPER_INTERP(SUBq)
#else
FMAC_BINARY_Q(SUBq, 1, false)
#endif

#if ISTUB_VU_SUBi
REC_VU1_UPPER_INTERP(SUBi)
#else
FMAC_BINARY_I(SUBi, 1, false)
#endif

#if ISTUB_VU_SUB
REC_VU1_UPPER_INTERP(SUB)
#else
FMAC_BINARY_FULL(SUB, 1, false)
#endif

// ============================================================================
//  SUBA family
// ============================================================================

#if ISTUB_VU_SUBAx
REC_VU1_UPPER_INTERP(SUBAx)
#else
FMAC_BINARY_BC(SUBAx, 1, 0, true)
#endif

#if ISTUB_VU_SUBAy
REC_VU1_UPPER_INTERP(SUBAy)
#else
FMAC_BINARY_BC(SUBAy, 1, 1, true)
#endif

#if ISTUB_VU_SUBAz
REC_VU1_UPPER_INTERP(SUBAz)
#else
FMAC_BINARY_BC(SUBAz, 1, 2, true)
#endif

#if ISTUB_VU_SUBAw
REC_VU1_UPPER_INTERP(SUBAw)
#else
FMAC_BINARY_BC(SUBAw, 1, 3, true)
#endif

#if ISTUB_VU_SUBAq
REC_VU1_UPPER_INTERP(SUBAq)
#else
FMAC_BINARY_Q(SUBAq, 1, true)
#endif

#if ISTUB_VU_SUBAi
REC_VU1_UPPER_INTERP(SUBAi)
#else
FMAC_BINARY_I(SUBAi, 1, true)
#endif

#if ISTUB_VU_SUBA
REC_VU1_UPPER_INTERP(SUBA)
#else
FMAC_BINARY_FULL(SUBA, 1, true)
#endif

// ============================================================================
//  MUL family
// ============================================================================

#if ISTUB_VU_MULx
REC_VU1_UPPER_INTERP(MULx)
#else
FMAC_BINARY_BC(MULx, 2, 0, false)
#endif

#if ISTUB_VU_MULy
REC_VU1_UPPER_INTERP(MULy)
#else
FMAC_BINARY_BC(MULy, 2, 1, false)
#endif

#if ISTUB_VU_MULz
REC_VU1_UPPER_INTERP(MULz)
#else
FMAC_BINARY_BC(MULz, 2, 2, false)
#endif

#if ISTUB_VU_MULw
REC_VU1_UPPER_INTERP(MULw)
#else
FMAC_BINARY_BC(MULw, 2, 3, false)
#endif

#if ISTUB_VU_MULq
REC_VU1_UPPER_INTERP(MULq)
#else
FMAC_BINARY_Q(MULq, 2, false)
#endif

#if ISTUB_VU_MULi
REC_VU1_UPPER_INTERP(MULi)
#else
FMAC_BINARY_I(MULi, 2, false)
#endif

#if ISTUB_VU_MUL
REC_VU1_UPPER_INTERP(MUL)
#else
FMAC_BINARY_FULL(MUL, 2, false)
#endif

// ============================================================================
//  MULA family
// ============================================================================

#if ISTUB_VU_MULAx
REC_VU1_UPPER_INTERP(MULAx)
#else
// Hand-written instead of FMAC_BINARY_BC(MULAx, 2, 0, true) so we can
// intercept the MAC-cluster-fusion fast path. Baseline (non-cluster)
// codegen is byte-identical to what the macro would have produced.
void recVU1_MULAx() {
	if (g_vu1MacClusterLead) {
		// FMAC opt #19: full 4-pair MULAx→MADDAy→MADDAz→MADDw chain in
		// one NEON sequence. Reads pre-validated cluster fields set by
		// CompileBlock from the microIR.
		//
		// Layout (NEON regs):
		//   v0..v3 = matrix cols VF[fs0..fs3]
		//   v4     = broadcast vector VF[ft]
		//   v5     = scratch (intermediate per-pair product)
		//   v6.S[0] = saved ACC.w (xyz-only variant; preserves W lane)
		//   v16    = pinned VU1_ACC_REG (final ACC after MADDAz)
		//
		// Semantics preserved (vs per-pair):
		//   * Two-rounded math: each step uses separate Fmul + Fadd
		//     (NOT FMA) to match PS2's FMUL+FADD two-rounding behavior
		//     that the codebase explicitly preserves (see line 1242
		//     comment + memory armsx2-vu-interp-fp-contract).
		//   * Pinned ACC: VU1_ACC_REG carries the accumulator across
		//     pairs; per-pair Mov-from/to-ACC dance is collapsed.
		//   * MADDw semantics: VF[fd] = ACC + col3*vec.w; ACC keeps
		//     the value from MADDAz (NOT updated by MADDw).
		//   * xyz-only variant (mask 0xE): preserve ACC.w by saving to
		//     v6.S[0] before chain + restoring before MADDw; final
		//     writeback uses cache mask 0xE so VFd.w is untouched.
		const u32 fs0 = g_vu1MacClusterFs0;
		const u32 fs1 = g_vu1MacClusterFs1;
		const u32 fs2 = g_vu1MacClusterFs2;
		const u32 fs3 = g_vu1MacClusterFs3;
		const u32 ft  = g_vu1MacClusterFt;
		const u32 fd  = g_vu1MacClusterFd;
		const bool xyz_only = g_vu1MacClusterXyzOnly;

		// Step 1: load the four matrix columns and the broadcast vector
		// from the VF cache. vfCacheLoadInto hits the slot directly if
		// resident; otherwise emits one Ldr.
		vfCacheLoadInto(static_cast<int>(fs0), v0);
		vfCacheLoadInto(static_cast<int>(fs1), v1);
		vfCacheLoadInto(static_cast<int>(fs2), v2);
		vfCacheLoadInto(static_cast<int>(fs3), v3);
		vfCacheLoadInto(static_cast<int>(ft),  v4);

		// xyz-only chain: save VU1_ACC_REG.w into v6.S[0] so we can
		// restore it after the FMUL chain (which writes all 4 lanes).
		if (xyz_only)
			armAsm->Mov(v6.V4S(), 3, VU1_ACC_REG.V4S(), 3);

		// Step 2: MULAx → ACC = col0 * vec.x  (writes pinned VU1_ACC_REG).
		armAsm->Fmul(VU1_ACC_REG.V4S(), v0.V4S(), v4.S(), 0);

		// Step 3: MADDAy → ACC += col1 * vec.y  (separate FMUL+FADD).
		armAsm->Fmul(v5.V4S(), v1.V4S(), v4.S(), 1);
		armAsm->Fadd(VU1_ACC_REG.V4S(), VU1_ACC_REG.V4S(), v5.V4S());

		// Step 4: MADDAz → ACC += col2 * vec.z.
		armAsm->Fmul(v5.V4S(), v2.V4S(), v4.S(), 2);
		armAsm->Fadd(VU1_ACC_REG.V4S(), VU1_ACC_REG.V4S(), v5.V4S());

		// xyz-only: restore ACC.w from saved scratch so MADDw and any
		// downstream consumer sees the original VU1_ACC_REG.w.
		if (xyz_only)
			armAsm->Mov(VU1_ACC_REG.V4S(), 3, v6.V4S(), 3);

		// Step 5: MADDw → VFd = ACC + col3 * vec.w. ACC unchanged.
		armAsm->Fmul(v5.V4S(), v3.V4S(), v4.S(), 3);
		armAsm->Fadd(v5.V4S(), VU1_ACC_REG.V4S(), v5.V4S());

		// Step 6: writeback VFd via the cache (deferred Str fires at
		// block end / hazard boundary / BL via vfCacheFlushAndInvalidate).
		// fd != 0 is gated by Pass 1. Mask depends on the chain variant:
		// 0xF for full xyzw, 0xE for xyz-only (preserves VFd.w).
		vfCacheStore(static_cast<int>(fd), v5, xyz_only ? 0xEu : 0xFu);

		// Invalidate broadcast plan — we just bypassed the planning machinery.
		vu1BroadcastCacheReset();
		return;
	}

	// Baseline (mirrors FMAC_BINARY_BC(MULAx, 2, 0, true) expansion).
	const u32 fd = (VU1.code >> 6) & 0x1F;
	(void)fd;
	const u32 fs = (VU1.code >> 11) & 0x1F;
	const u32 ft = (VU1.code >> 16) & 0x1F;
	const u32 xyzw = (VU1.code >> 21) & 0xF;
	emitLoadBroadcast(ft, 0);
	int64_t dst = accOff();
	emitBinaryFmac(2, fs, dst, xyzw);
}
#endif

#if ISTUB_VU_MULAy
REC_VU1_UPPER_INTERP(MULAy)
#else
FMAC_BINARY_BC(MULAy, 2, 1, true)
#endif

#if ISTUB_VU_MULAz
REC_VU1_UPPER_INTERP(MULAz)
#else
FMAC_BINARY_BC(MULAz, 2, 2, true)
#endif

#if ISTUB_VU_MULAw
REC_VU1_UPPER_INTERP(MULAw)
#else
FMAC_BINARY_BC(MULAw, 2, 3, true)
#endif

#if ISTUB_VU_MULAq
REC_VU1_UPPER_INTERP(MULAq)
#else
FMAC_BINARY_Q(MULAq, 2, true)
#endif

#if ISTUB_VU_MULAi
REC_VU1_UPPER_INTERP(MULAi)
#else
FMAC_BINARY_I(MULAi, 2, true)
#endif

#if ISTUB_VU_MULA
REC_VU1_UPPER_INTERP(MULA)
#else
FMAC_BINARY_FULL(MULA, 2, true)
#endif

// ============================================================================
//  MADD family
// ============================================================================

#if ISTUB_VU_MADDx
REC_VU1_UPPER_INTERP(MADDx)
#else
FMAC_TERNARY_BC(MADDx, false, 0, false)
#endif

#if ISTUB_VU_MADDy
REC_VU1_UPPER_INTERP(MADDy)
#else
FMAC_TERNARY_BC(MADDy, false, 1, false)
#endif

#if ISTUB_VU_MADDz
REC_VU1_UPPER_INTERP(MADDz)
#else
FMAC_TERNARY_BC(MADDz, false, 2, false)
#endif

#if ISTUB_VU_MADDw
REC_VU1_UPPER_INTERP(MADDw)
#else
// Hand-written for MAC cluster member check (FMAC opt #19).
void recVU1_MADDw() {
	if (g_vu1MacClusterMember) return; // chain head already emitted everything
	const u32 fd = (VU1.code >> 6) & 0x1F;
	const u32 fs = (VU1.code >> 11) & 0x1F;
	const u32 ft = (VU1.code >> 16) & 0x1F;
	const u32 xyzw = (VU1.code >> 21) & 0xF;
	emitLoadBroadcast(ft, 3);
	int64_t dst = vfOff(fd);
	emitTernaryFmac(false, fs, dst, xyzw);
}
#endif

#if ISTUB_VU_MADDq
REC_VU1_UPPER_INTERP(MADDq)
#else
FMAC_TERNARY_Q(MADDq, false, false)
#endif

#if ISTUB_VU_MADDi
REC_VU1_UPPER_INTERP(MADDi)
#else
FMAC_TERNARY_I(MADDi, false, false)
#endif

#if ISTUB_VU_MADD
REC_VU1_UPPER_INTERP(MADD)
#else
FMAC_TERNARY_FULL(MADD, false, false)
#endif

// ============================================================================
//  MADDA family
// ============================================================================

#if ISTUB_VU_MADDAx
REC_VU1_UPPER_INTERP(MADDAx)
#else
FMAC_TERNARY_BC(MADDAx, false, 0, true)
#endif

#if ISTUB_VU_MADDAy
REC_VU1_UPPER_INTERP(MADDAy)
#else
// Hand-written for MAC cluster member check (FMAC opt #19).
void recVU1_MADDAy() {
	if (g_vu1MacClusterMember) return; // chain head already emitted everything
	const u32 fd = (VU1.code >> 6) & 0x1F;
	(void)fd;
	const u32 fs = (VU1.code >> 11) & 0x1F;
	const u32 ft = (VU1.code >> 16) & 0x1F;
	const u32 xyzw = (VU1.code >> 21) & 0xF;
	emitLoadBroadcast(ft, 1);
	int64_t dst = accOff();
	emitTernaryFmac(false, fs, dst, xyzw);
}
#endif

#if ISTUB_VU_MADDAz
REC_VU1_UPPER_INTERP(MADDAz)
#else
// Hand-written for MAC cluster member check (FMAC opt #19).
void recVU1_MADDAz() {
	if (g_vu1MacClusterMember) return; // chain head already emitted everything
	const u32 fd = (VU1.code >> 6) & 0x1F;
	(void)fd;
	const u32 fs = (VU1.code >> 11) & 0x1F;
	const u32 ft = (VU1.code >> 16) & 0x1F;
	const u32 xyzw = (VU1.code >> 21) & 0xF;
	emitLoadBroadcast(ft, 2);
	int64_t dst = accOff();
	emitTernaryFmac(false, fs, dst, xyzw);
}
#endif

#if ISTUB_VU_MADDAw
REC_VU1_UPPER_INTERP(MADDAw)
#else
FMAC_TERNARY_BC(MADDAw, false, 3, true)
#endif

#if ISTUB_VU_MADDAq
REC_VU1_UPPER_INTERP(MADDAq)
#else
FMAC_TERNARY_Q(MADDAq, false, true)
#endif

#if ISTUB_VU_MADDAi
REC_VU1_UPPER_INTERP(MADDAi)
#else
FMAC_TERNARY_I(MADDAi, false, true)
#endif

#if ISTUB_VU_MADDA
REC_VU1_UPPER_INTERP(MADDA)
#else
FMAC_TERNARY_FULL(MADDA, false, true)
#endif

// ============================================================================
//  MSUB family
// ============================================================================

#if ISTUB_VU_MSUBx
REC_VU1_UPPER_INTERP(MSUBx)
#else
FMAC_TERNARY_BC(MSUBx, true, 0, false)
#endif

#if ISTUB_VU_MSUBy
REC_VU1_UPPER_INTERP(MSUBy)
#else
FMAC_TERNARY_BC(MSUBy, true, 1, false)
#endif

#if ISTUB_VU_MSUBz
REC_VU1_UPPER_INTERP(MSUBz)
#else
FMAC_TERNARY_BC(MSUBz, true, 2, false)
#endif

#if ISTUB_VU_MSUBw
REC_VU1_UPPER_INTERP(MSUBw)
#else
FMAC_TERNARY_BC(MSUBw, true, 3, false)
#endif

#if ISTUB_VU_MSUBq
REC_VU1_UPPER_INTERP(MSUBq)
#else
FMAC_TERNARY_Q(MSUBq, true, false)
#endif

#if ISTUB_VU_MSUBi
REC_VU1_UPPER_INTERP(MSUBi)
#else
FMAC_TERNARY_I(MSUBi, true, false)
#endif

#if ISTUB_VU_MSUB
REC_VU1_UPPER_INTERP(MSUB)
#else
FMAC_TERNARY_FULL(MSUB, true, false)
#endif

// ============================================================================
//  MSUBA family
// ============================================================================

#if ISTUB_VU_MSUBAx
REC_VU1_UPPER_INTERP(MSUBAx)
#else
FMAC_TERNARY_BC(MSUBAx, true, 0, true)
#endif

#if ISTUB_VU_MSUBAy
REC_VU1_UPPER_INTERP(MSUBAy)
#else
FMAC_TERNARY_BC(MSUBAy, true, 1, true)
#endif

#if ISTUB_VU_MSUBAz
REC_VU1_UPPER_INTERP(MSUBAz)
#else
FMAC_TERNARY_BC(MSUBAz, true, 2, true)
#endif

#if ISTUB_VU_MSUBAw
REC_VU1_UPPER_INTERP(MSUBAw)
#else
FMAC_TERNARY_BC(MSUBAw, true, 3, true)
#endif

#if ISTUB_VU_MSUBAq
REC_VU1_UPPER_INTERP(MSUBAq)
#else
FMAC_TERNARY_Q(MSUBAq, true, true)
#endif

#if ISTUB_VU_MSUBAi
REC_VU1_UPPER_INTERP(MSUBAi)
#else
FMAC_TERNARY_I(MSUBAi, true, true)
#endif

#if ISTUB_VU_MSUBA
REC_VU1_UPPER_INTERP(MSUBA)
#else
FMAC_TERNARY_FULL(MSUBA, true, true)
#endif

// ============================================================================
//  MAX / MINI
// ============================================================================

#if ISTUB_VU_MAXx
REC_VU1_UPPER_INTERP(MAXx)
#else
FMAC_MAXMINI_BC(MAXx, false, 0)
#endif

#if ISTUB_VU_MAXy
REC_VU1_UPPER_INTERP(MAXy)
#else
FMAC_MAXMINI_BC(MAXy, false, 1)
#endif

#if ISTUB_VU_MAXz
REC_VU1_UPPER_INTERP(MAXz)
#else
FMAC_MAXMINI_BC(MAXz, false, 2)
#endif

#if ISTUB_VU_MAXw
REC_VU1_UPPER_INTERP(MAXw)
#else
FMAC_MAXMINI_BC(MAXw, false, 3)
#endif

#if ISTUB_VU_MAXi
REC_VU1_UPPER_INTERP(MAXi)
#else
FMAC_MAXMINI_I(MAXi, false)
#endif

#if ISTUB_VU_MAX
REC_VU1_UPPER_INTERP(MAX)
#else
FMAC_MAXMINI_FULL(MAX, false)
#endif

#if ISTUB_VU_MINIx
REC_VU1_UPPER_INTERP(MINIx)
#else
FMAC_MAXMINI_BC(MINIx, true, 0)
#endif

#if ISTUB_VU_MINIy
REC_VU1_UPPER_INTERP(MINIy)
#else
FMAC_MAXMINI_BC(MINIy, true, 1)
#endif

#if ISTUB_VU_MINIz
REC_VU1_UPPER_INTERP(MINIz)
#else
FMAC_MAXMINI_BC(MINIz, true, 2)
#endif

#if ISTUB_VU_MINIw
REC_VU1_UPPER_INTERP(MINIw)
#else
FMAC_MAXMINI_BC(MINIw, true, 3)
#endif

#if ISTUB_VU_MINIi
REC_VU1_UPPER_INTERP(MINIi)
#else
FMAC_MAXMINI_I(MINIi, true)
#endif

#if ISTUB_VU_MINI
REC_VU1_UPPER_INTERP(MINI)
#else
FMAC_MAXMINI_FULL(MINI, true)
#endif

// ============================================================================
//  ABS, CLIP, OPMULA, OPMSUB, NOP
// ============================================================================

#if ISTUB_VU_ABS
REC_VU1_UPPER_INTERP(ABS)
#else
FMAC_ABS(ABS)
#endif

#if ISTUB_VU_CLIP
REC_VU1_UPPER_INTERP(CLIP)
#else
// Native CLIP — mirrors _vuCLIP (VUops.cpp:911).
//
//   value = VF[ft].i.w raw bits
//   value = (value & 0x7f800000) ? (value & 0x7fffffff) : 0x007fffff
//   clipflag = (clipflag << 6) & 0xFFFFFF
//   for each component c in {x,y,z} at bit pairs 0/1, 2/3, 4/5:
//     clipflag |= ((s32)VF[fs].c         > value) ? bit_pos : 0
//     clipflag |= ((s32)VF[fs].c ^ 0x80..>> > value) ? bit_neg : 0
//   store VU->clipflag
//
// All-scalar: six signed-int compares are cheap, a NEON round-trip would
// save instructions but add lane-extract overhead. Registers used:
//   w0  = value (denormal-adjusted |ft.w|)
//   w1  = scratch for exponent mask
//   w2  = 0x007fffff denormal constant
//   w3/w4/w5 = fs.x / fs.y / fs.z raw bits
//   w6  = new clipflag accumulator
//   w7  = compare result (0/1)
//   w8  = sign-flipped scratch
void recVU1_CLIP() {
	const u32 fs = (VU1.code >> 11) & 0x1F;
	const u32 ft = (VU1.code >> 16) & 0x1F;

	// FMAC opt #6 (cache-read variant): if VF[ft] / VF[fs] are already
	// resident in NEON cache slots from a prior FMAC pair, extract the
	// scalar lanes via Umov (register-to-register) instead of going to
	// memory. With write-through, the slot value is always coherent with
	// memory, so cache reads are equivalent. Saves up to 4 Ldrs per CLIP
	// when both operands are hot in cache (typical right after a matrix
	// transform writes the projected vertex into the same VFs we now
	// CLIP-test). Falls back to direct Ldr for cache misses — allocating
	// fresh slots just for CLIP would cost more than 4 Ldr w (one Ldr q
	// + 4 Umov vs 4 Ldr w).
	//
	// vfreg == 0 (VF[0] hardwired (0,0,0,1)) takes the direct-Ldr path:
	// we don't allocate a cache slot for the constant.
	const bool ft_resident = (ft != 0) && vfCacheIsResident(static_cast<int>(ft));
	const bool fs_resident = (fs != 0) && vfCacheIsResident(static_cast<int>(fs));

	// value = raw ft.w
	if (ft_resident)
	{
		const auto ftSlot = vfCacheLoadResident(static_cast<int>(ft));
		armAsm->Umov(w0, ftSlot.V4S(), 3);
	}
	else
	{
		armAsm->Ldr(w0, MemOperand(VU1_BASE_REG, vfOff(ft) + 12));
	}
	// w1 = value & 0x7f800000 (exponent bits)
	armAsm->And(w1, w0, 0x7f800000u);
	// w0 = value & 0x7fffffff (abs value)
	armAsm->And(w0, w0, 0x7fffffffu);
	// w2 = 0x007fffff
	armAsm->Mov(w2, 0x007fffffu);
	// value = (exponent != 0) ? abs : 0x007fffff
	armAsm->Cmp(w1, 0);
	armAsm->Csel(w0, w0, w2, ne);

	// w6 = VU->clipflag << 6 (read from pinned VU1_CLIPFLAG_REG, not memory).
	armAsm->Lsl(w6, VU1_CLIPFLAG_REG, 6);

	// FMAC opt #6 (NEON-vectorized compare): pack the 6 signed-int compares
	// (3 components × {raw, sign-flipped}) into one parallel NEON pipeline.
	//
	// Layout per lane (X/Y/Z; lane W unused):
	//   vR1.lane[i] = (fs.lane[i] > value)             ? 0xFFFFFFFF : 0
	//   vR2.lane[i] = (fs.lane[i] ^ 0x80000000) > value ? 0xFFFFFFFF : 0
	//
	// Bit-pack masks (kept disjoint so Addv folds == OR):
	//   vM1 (positive) = {0x01, 0x04, 0x10, 0x00}  — bits 0/2/4
	//   vM2 (negative) = vM1 << 1  = {0x02, 0x08, 0x20, 0x00}  — bits 1/3/5
	//
	// After AND + OR + Addv, lane 0 holds the 6-bit clipflag delta to OR
	// into w6. Disjoint bits guarantee Addv (sum) == OR.
	//
	// Equivalent to the original 21-insn GPR compare chain (3 Ldr w + 6
	// Cmp/Cset/Orr × 2-or-3 each); NEON path is 14 insns and matches the
	// existing bit positions exactly. The chain ends with one Umov (NEON→
	// GPR) plus one Orr; the rest is parallelizable on a wide NEON pipe.
	{
		// Source vector for fs.xyzw. Resident slot is read-only — Cmgt/Eor
		// write their destinations (v6/v7), not the slot, so the cache
		// stays coherent.
		a64::VRegister vFs = v3;
		if (fs_resident)
		{
			vFs = vfCacheLoadResident(static_cast<int>(fs));
		}
		else
		{
			armAsm->Ldr(v3.Q(), MemOperand(VU1_BASE_REG, vfOff(fs)));
		}

		// vV.4S = broadcast(value) for parallel compare against all lanes.
		armAsm->Dup(v4.V4S(), w0);
		// vS.4S = 0x80000000 broadcast (sign-flip mask).
		armAsm->Movi(v5.V4S(), 0x80, LSL, 24);

		// Positive compares: vR1.lane[i] = (fs.lane[i] > value)
		armAsm->Cmgt(v6.V4S(), vFs.V4S(), v4.V4S());
		// Sign-flipped fs into v7: v7 = fs ^ vS.
		armAsm->Eor(v7.V16B(), vFs.V16B(), v5.V16B());
		// Negative compares: v7.lane[i] = ((fs.lane[i] ^ sign) > value)
		armAsm->Cmgt(v7.V4S(), v7.V4S(), v4.V4S());

		// Per-lane bit-pack masks. vM2 = vM1 << 1 (saves one literal pool entry).
		armLoadConstant128(v8.Q(), kClipMaskPos);
		armAsm->Shl(v2.V4S(), v8.V4S(), 1);

		// Mask compare results to their bit positions.
		armAsm->And(v6.V16B(), v6.V16B(), v8.V16B());
		armAsm->And(v7.V16B(), v7.V16B(), v2.V16B());
		// Combine pos + sign-flipped (disjoint bits).
		armAsm->Orr(v6.V16B(), v6.V16B(), v7.V16B());
		// Horizontal sum across all 4 lanes (bit-disjoint, so sum == OR).
		armAsm->Addv(s6, v6.V4S());
		// Extract the 6-bit packed result into w7.
		armAsm->Umov(w7, v6.V4S(), 0);
		// OR into the clipflag accumulator (w6 already holds clipflag << 6).
		armAsm->Orr(w6, w6, w7);
	}

	// Mask to 24 bits and write back to the pinned reg.
	armAsm->And(VU1_CLIPFLAG_REG, w6, 0xFFFFFFu);
}
#endif

#if ISTUB_VU_OPMULA
REC_VU1_UPPER_INTERP(OPMULA)
#else
void recVU1_OPMULA() {
	if (g_vu1OpmacClusterLead) {
		// FMAC opt #20: fused OPMULA + OPMSUB cross-product.
		//
		// cross(a, b) = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
		//
		// We emit:
		//   ACC.xyz = (a.y*b.z, a.z*b.x, a.x*b.y)  (OPMULA result)
		//   VFd.xyz = ACC.xyz - (a.z*b.y, a.x*b.z, a.y*b.x)
		//                     = (a.y*b.z - a.z*b.y, ...)  (OPMSUB result)
		//
		// Both .w lanes preserved (ACC.w and VFd.w untouched), matching
		// VU semantics.
		//
		// Per-pair codegen would do this same work twice (~26-30 NEON
		// insns). Fused: ~20 insns (skip one cache-load pair, ACC scratch
		// Mov, redundant clamp setup).
		const u32 fs = g_vu1OpmacClusterFs;
		const u32 ft = g_vu1OpmacClusterFt;
		const u32 fd = g_vu1OpmacClusterFd;

		vfCacheLoadInto(static_cast<int>(fs), v0); // v0 = a
		vfCacheLoadInto(static_cast<int>(ft), v1); // v1 = b

		// Build OPMULA permutations:
		//   v2 = {a.y, a.z, a.x, _}
		//   v3 = {b.z, b.x, b.y, _}
		armAsm->Mov(v2.V4S(), 0, v0.V4S(), 1);
		armAsm->Mov(v2.V4S(), 1, v0.V4S(), 2);
		armAsm->Mov(v2.V4S(), 2, v0.V4S(), 0);
		armAsm->Mov(v3.V4S(), 0, v1.V4S(), 2);
		armAsm->Mov(v3.V4S(), 1, v1.V4S(), 0);
		armAsm->Mov(v3.V4S(), 2, v1.V4S(), 1);

		// v5.xyz = OPMULA product (xyz lanes only — W untouched in scratch).
		armAsm->Fmul(v5.V4S(), v2.V4S(), v3.V4S());

		// Build OPMSUB permutations (operand-swap: OPMSUB sees (b, a)):
		//   v6 = {b.y, b.z, b.x, _}    perm of v1 (b)
		//   v7 = {a.z, a.x, a.y, _}    perm of v0 (a)
		armAsm->Mov(v6.V4S(), 0, v1.V4S(), 1);
		armAsm->Mov(v6.V4S(), 1, v1.V4S(), 2);
		armAsm->Mov(v6.V4S(), 2, v1.V4S(), 0);
		armAsm->Mov(v7.V4S(), 0, v0.V4S(), 2);
		armAsm->Mov(v7.V4S(), 1, v0.V4S(), 0);
		armAsm->Mov(v7.V4S(), 2, v0.V4S(), 1);

		// v4 = OPMSUB's second product.
		armAsm->Fmul(v4.V4S(), v6.V4S(), v7.V4S());

		// Writeback ACC.xyz from v5 (OPMULA result). Lane-by-lane Mov so
		// VU1_ACC_REG.w is preserved.
		armAsm->Mov(VU1_ACC_REG.V4S(), 0, v5.V4S(), 0);
		armAsm->Mov(VU1_ACC_REG.V4S(), 1, v5.V4S(), 1);
		armAsm->Mov(VU1_ACC_REG.V4S(), 2, v5.V4S(), 2);

		// VFd.xyz = OPMULA product - OPMSUB product = cross(a, b).
		armAsm->Fsub(v5.V4S(), v5.V4S(), v4.V4S());

		// Writeback VFd via cache (xyz mask; preserves VFd.w).
		vfCacheStore(static_cast<int>(fd), v5, 0xEu);

		vu1BroadcastCacheReset();
		return;
	}

	const u32 fs = (VU1.code >> 11) & 0x1F;
	const u32 ft = (VU1.code >> 16) & 0x1F;
	emitOpFmac(/*isSub=*/false, fs, ft, /*fd=*/0);
}
#endif

#if ISTUB_VU_OPMSUB
REC_VU1_UPPER_INTERP(OPMSUB)
#else
void recVU1_OPMSUB() {
	if (g_vu1OpmacClusterMember) return; // OPMULA lead already emitted everything
	const u32 fd = (VU1.code >>  6) & 0x1F;
	const u32 fs = (VU1.code >> 11) & 0x1F;
	const u32 ft = (VU1.code >> 16) & 0x1F;
	emitOpFmac(/*isSub=*/true, fs, ft, fd);
}
#endif

#if ISTUB_VU_NOP
REC_VU1_UPPER_INTERP(NOP)
#else
void recVU1_NOP() { } // VU NOP: nothing to emit
#endif

// ============================================================================
//  FTOI / ITOF — float/int conversion
// ============================================================================

#if ISTUB_VU_FTOI0
REC_VU1_UPPER_INTERP(FTOI0)
#else
void recVU1_FTOI0()  { emitFTOI(0);  }
#endif

#if ISTUB_VU_FTOI4
REC_VU1_UPPER_INTERP(FTOI4)
#else
void recVU1_FTOI4()  { emitFTOI(4);  }
#endif

#if ISTUB_VU_FTOI12
REC_VU1_UPPER_INTERP(FTOI12)
#else
void recVU1_FTOI12() { emitFTOI(12); }
#endif

#if ISTUB_VU_FTOI15
REC_VU1_UPPER_INTERP(FTOI15)
#else
void recVU1_FTOI15() { emitFTOI(15); }
#endif

#if ISTUB_VU_ITOF0
REC_VU1_UPPER_INTERP(ITOF0)
#else
void recVU1_ITOF0()  { emitITOF(0);  }
#endif

#if ISTUB_VU_ITOF4
REC_VU1_UPPER_INTERP(ITOF4)
#else
void recVU1_ITOF4()  { emitITOF(4);  }
#endif

#if ISTUB_VU_ITOF12
REC_VU1_UPPER_INTERP(ITOF12)
#else
void recVU1_ITOF12() { emitITOF(12); }
#endif

#if ISTUB_VU_ITOF15
REC_VU1_UPPER_INTERP(ITOF15)
#else
void recVU1_ITOF15() { emitITOF(15); }
#endif

// ============================================================================
//  FD sub-table dispatch (0x3C-0x3F).
//
//  VU1.code is set at JIT compile time before this is called. We resolve the
//  exact rec function using (VU1.code & 3) as the sub-type and
//  (VU1.code >> 6) & 0x1F as the index within that sub-table.
//  This calls the already-implemented recVU1_* emitters directly so all their
//  ISTUB guards and NEON paths apply normally.
//
//  Unknown/reserved slots (indices >= 12) and 0x30-0x3B fall back to the
//  interpreter via VU1_UPPER_OPCODE.
// ============================================================================
static void recVU1_Upper_FD()
{
	using FDFn = void (*)();
	const u32 fd_type = VU1.code & 3;
	const u32 idx = (VU1.code >> 6) & 0x1F;

	// clang-format off
	static const FDFn fd_00[] = { // 0x3C
		recVU1_ADDAx,  recVU1_SUBAx,  recVU1_MADDAx, recVU1_MSUBAx,
		recVU1_ITOF0,  recVU1_FTOI0,  recVU1_MULAx,  recVU1_MULAq,
		recVU1_ADDAq,  recVU1_SUBAq,  recVU1_ADDA,   recVU1_SUBA,
	};
	static const FDFn fd_01[] = { // 0x3D
		recVU1_ADDAy,  recVU1_SUBAy,  recVU1_MADDAy, recVU1_MSUBAy,
		recVU1_ITOF4,  recVU1_FTOI4,  recVU1_MULAy,  recVU1_ABS,
		recVU1_MADDAq, recVU1_MSUBAq, recVU1_MADDA,  recVU1_MSUBA,
	};
	static const FDFn fd_10[] = { // 0x3E
		recVU1_ADDAz,  recVU1_SUBAz,  recVU1_MADDAz, recVU1_MSUBAz,
		recVU1_ITOF12, recVU1_FTOI12, recVU1_MULAz,  recVU1_MULAi,
		recVU1_ADDAi,  recVU1_SUBAi,  recVU1_MULA,   recVU1_OPMULA,
	};
	static const FDFn fd_11[] = { // 0x3F
		recVU1_ADDAw,  recVU1_SUBAw,  recVU1_MADDAw, recVU1_MSUBAw,
		recVU1_ITOF15, recVU1_FTOI15, recVU1_MULAw,  recVU1_CLIP,
		recVU1_MADDAi, recVU1_MSUBAi, nullptr,       recVU1_NOP,
	};
	// clang-format on

	const FDFn* table;
	switch (fd_type) {
		case 0:  table = fd_00; break;
		case 1:  table = fd_01; break;
		case 2:  table = fd_10; break;
		default: table = fd_11; break;
	}

	if (idx < 12 && table[idx] != nullptr)
		table[idx]();
	else
		// Reserved/unknown slot — fall back to interp via the flush/reload
		// helper so pinned regs stay coherent (same rationale as
		// REC_VU1_UPPER_INTERP).
		emitVU1InterpBL(reinterpret_cast<const void*>(VU1_UPPER_OPCODE[VU1.code & 0x3f]));
}

// ============================================================================
//  emitVU1Upper — top-level upper-opcode dispatch.
//
//  Replaces the old recVU1_UpperTable[64] function-pointer array with a
//  compile-time switch on `upper & 0x3f`. Generates a jump table under most
//  optimisers (64 dense cases), but being in the same TU as the emitters
//  lets the compiler see straight through to each recVU1_* function instead
//  of stopping at an indirect call — so it can inline small emitters like
//  recVU1_NOP directly into the dispatcher.
//
//  Layout mirrors VU1_UPPER_OPCODE in VUops.cpp.
//    0x00-0x2F: direct FMAC / MAX / MINI variants.
//    0x30-0x3F: delegated to recVU1_Upper_FD (which does its own compile-time
//               FD sub-table dispatch; 0x30-0x3B fall back to the interpreter
//               via VU1_UPPER_OPCODE for reserved/unknown slots).
// ============================================================================
void emitVU1Upper(u32 upper)
{
	switch (upper & 0x3f)
	{
		// 0x00-0x03: ADD broadcast
		case 0x00: recVU1_ADDx();  break;
		case 0x01: recVU1_ADDy();  break;
		case 0x02: recVU1_ADDz();  break;
		case 0x03: recVU1_ADDw();  break;
		// 0x04-0x07: SUB broadcast
		case 0x04: recVU1_SUBx();  break;
		case 0x05: recVU1_SUBy();  break;
		case 0x06: recVU1_SUBz();  break;
		case 0x07: recVU1_SUBw();  break;
		// 0x08-0x0B: MADD broadcast
		case 0x08: recVU1_MADDx(); break;
		case 0x09: recVU1_MADDy(); break;
		case 0x0A: recVU1_MADDz(); break;
		case 0x0B: recVU1_MADDw(); break;
		// 0x0C-0x0F: MSUB broadcast
		case 0x0C: recVU1_MSUBx(); break;
		case 0x0D: recVU1_MSUBy(); break;
		case 0x0E: recVU1_MSUBz(); break;
		case 0x0F: recVU1_MSUBw(); break;
		// 0x10-0x13: MAX broadcast
		case 0x10: recVU1_MAXx();  break;
		case 0x11: recVU1_MAXy();  break;
		case 0x12: recVU1_MAXz();  break;
		case 0x13: recVU1_MAXw();  break;
		// 0x14-0x17: MINI broadcast
		case 0x14: recVU1_MINIx(); break;
		case 0x15: recVU1_MINIy(); break;
		case 0x16: recVU1_MINIz(); break;
		case 0x17: recVU1_MINIw(); break;
		// 0x18-0x1B: MUL broadcast
		case 0x18: recVU1_MULx();  break;
		case 0x19: recVU1_MULy();  break;
		case 0x1A: recVU1_MULz();  break;
		case 0x1B: recVU1_MULw();  break;
		// 0x1C-0x1F: MULq, MAXi, MULi, MINIi
		case 0x1C: recVU1_MULq();  break;
		case 0x1D: recVU1_MAXi();  break;
		case 0x1E: recVU1_MULi();  break;
		case 0x1F: recVU1_MINIi(); break;
		// 0x20-0x23: ADDq, MADDq, ADDi, MADDi
		case 0x20: recVU1_ADDq();  break;
		case 0x21: recVU1_MADDq(); break;
		case 0x22: recVU1_ADDi();  break;
		case 0x23: recVU1_MADDi(); break;
		// 0x24-0x27: SUBq, MSUBq, SUBi, MSUBi
		case 0x24: recVU1_SUBq();  break;
		case 0x25: recVU1_MSUBq(); break;
		case 0x26: recVU1_SUBi();  break;
		case 0x27: recVU1_MSUBi(); break;
		// 0x28-0x2B: ADD, MADD, MUL, MAX
		case 0x28: recVU1_ADD();   break;
		case 0x29: recVU1_MADD();  break;
		case 0x2A: recVU1_MUL();   break;
		case 0x2B: recVU1_MAX();   break;
		// 0x2C-0x2F: SUB, MSUB, OPMSUB, MINI
		case 0x2C: recVU1_SUB();    break;
		case 0x2D: recVU1_MSUB();   break;
		case 0x2E: recVU1_OPMSUB(); break;
		case 0x2F: recVU1_MINI();   break;
		// 0x30-0x3F: FD sub-table (reserved/unknown slots 0x30-0x3B fall
		// through inside recVU1_Upper_FD to the interpreter).
		default:   recVU1_Upper_FD(); break;
	}
}
