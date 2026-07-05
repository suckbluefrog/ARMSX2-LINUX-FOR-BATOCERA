// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU1 Recompiler — Class declaration.
// Initially wraps the VU1 interpreter; individual instructions are gradually
// replaced with native ARM64 codegen via per-instruction ISTUB toggles.

#pragma once

#include "VUmicro.h"

// ============================================================================
//  recArmVU1 — ARM64 VU1 recompiler
// ============================================================================

class recArmVU1 final : public BaseVUmicroCPU
{
public:
	recArmVU1();
	~recArmVU1() override { Shutdown(); }

	const char* GetShortName() const override { return "armVU1"; }
	const char* GetLongName() const override { return "ARM64 VU1 Recompiler"; }

	void Reserve();
	void Shutdown() override;
	void Reset() override;
	void SetStartPC(u32 startPC) override;
	void Step() override;
	void Execute(u32 cycles) override;
	void Clear(u32 addr, u32 size) override;
	void ResumeXGkick() override {}

	// VU1_PROFILE_BLOCKS — dump top-N hottest blocks. No-op when the macro
	// is undefined. Called from VMManager::Shutdown so per-game exits via
	// the overlay's Stop button surface the profile, not just full app exit.
	void DumpProfile();
};

extern recArmVU1 CpuArmVU1;

// ISTUB helper — emits flush of all VU1 pinned regs (ACC q16, flags
// w19/w20/w28, cycle x21, fmac/ialu wpos x24/x25), BL to interp_fn,
// and reload of the same regs. Called by the REC_VU1_*_INTERP macros in
// iVU1Upper_arm64.cpp / iVU1Lower_arm64.cpp when any INTERP_VU_* harness
// flag routes an op to the C interpreter. Keeping this as a single
// helper prevents the hybrid-harness pin-skew bug (ISTUB emits a bare BL
// and the interp sees stale state).
void emitVU1InterpBL(const void* interp_fn);

// ============================================================================
//  VF register cache (Phase 1: read-side, write-through with invalidation)
// ============================================================================
// Defined in iVU1micro_arm64.cpp; exposed for FMAC emitters in
// iVU1Upper_arm64.cpp / iVU1Lower_arm64.cpp. See the long comment at the
// definition site for the design rationale (mirrors the cross-pair VF
// residency from the old port-in-place's microRegAlloc).
namespace vixl::aarch64 { class VRegister; }

void vfCacheReset();
void vfCacheInvalidate(int vfreg);
void vfCacheInvalidateAll();
void vfCacheLoadInto(int vfreg, const vixl::aarch64::VRegister& scratch);
vixl::aarch64::VRegister vfCacheLoadResident(int vfreg);

// Cache residency check (FMAC opt #6 cache-read variant). Returns true if
// vfreg is currently held in a cache slot (no Ldr needed to consume it).
// Caller can then call vfCacheLoadResident to get the slot reg without
// triggering an allocation+Ldr — the function will hit the existing slot.
bool vfCacheIsResident(int vfreg);

// Phase 2: defer a write of `src_reg`'s `xyzw_mask` lanes into VF[vfreg].
// The slot is force-loaded from memory if not fully valid. Result lanes
// merge into the slot via per-lane Mov; the actual Str is deferred to
// vfCacheFlushDirty / vfCacheFlushAndInvalidate.
void vfCacheStore(int vfreg, const vixl::aarch64::VRegister& src_reg, u8 xyzw_mask);

// Phase 2: emit Strs for every dirty slot's dirty lanes, then clear dirty
// bits. Slots stay loaded so subsequent reads still hit. Used at sites
// where memory must be coherent but NEON state is preserved.
void vfCacheFlushDirty();

// Phase 2: flush dirty lanes of one VF and drop its slot. Use around op
// emitters that inline Ldr/Str VF memory directly (LQ/SQ family).
void vfCacheFlushOne(int vfreg);

// Phase 2: flush dirty lanes, then drop all tracker state. Used at BL sites
// (NEON caller-saved) and at block epilogue (linked successors don't share
// our compile-time slot map).
void vfCacheFlushAndInvalidate();

// Wrapper for armEmitCall that flushes deferred VF writes and drops the
// cache tracker. Every BL clobbers caller-saved NEON state, AND the helper
// may read VF memory directly.
void emitVu1Call(const void* fn);

// ============================================================================
//  VI register cache (write-through, mirrors VF cache architecture)
// ============================================================================
// 7-slot GPR cache mapped to w9..w15. Caches VI registers across pairs to
// eliminate redundant Ldrh/Strh memory traffic for IALU-heavy code (IADD,
// IBxx, ILW, IADDI, etc.). Write-through — every store updates the slot AND
// hits memory immediately. VI[0] is hardwired to 0; reads return wzr,
// writes are dropped.
//
// All callers and BL paths must invalidate the tracker around BLs (the pool
// is caller-saved). emitVu1Call already does this. Block start runs
// viCacheReset; block end / pre-link runs viCacheInvalidateAll.

namespace vixl::aarch64 { class Register; }

void viCacheReset();
void viCacheInvalidate(int vireg);
void viCacheInvalidateAll();
void viCacheFlushOne(int vireg);
void viCacheFlushAndInvalidate();
// Register parameters here are the universal vixl Register type configured
// 32-bit (w-form). Globals like `w0`, `wzr` are of type `Register`, not
// `WRegister` (vixl's WRegister/XRegister are constructor-only helpers; the
// real register-object class is Register). All callers pass `wN` directly.
void viCacheLoadInto(int vireg, const vixl::aarch64::Register& scratch);
vixl::aarch64::Register viCacheLoadResident(int vireg);
// Sign-extended read into `dest`. Use this when the consumer needs signed
// semantics — vixl's Sxth asserts on wzr-source, so callers must NOT do
// `Sxth(dest, viCacheLoadResident(reg))` directly when reg can be 0.
void viCacheLoadSignedInto(int vireg, const vixl::aarch64::Register& dest);
void viCacheStore(int vireg, const vixl::aarch64::Register& src_reg);

// ============================================================================
//  Broadcast operand cache (FMAC opt #16)
// ============================================================================
// emitLoadBroadcast (in iVU1Upper_arm64.cpp) caches the last (ft, comp) it
// loaded into v1 across pairs. Consecutive ADDx/MULx-style ops broadcasting
// the same VF lane reuse v1 — the second emit becomes a no-op (saves 1 Dup
// + 1 Ldr if cache miss). Reset hooks fire on BL invalidation, VF[ft] write,
// and any v1-writing helper. Defined in iVU1Upper_arm64.cpp.
void vu1BroadcastCacheReset();
void vu1BroadcastCacheNoteVfWritten(int vfreg);

// ============================================================================
//  Same-VF different-lane batching (FMAC opt #17)
// ============================================================================
// Resets the deferred batch state. Called at block start and after any
// site that could break our invariant about d10 holding K's deferred lane.
void vu1BatchCacheReset();
