// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0

#include "common/Console.h"
#include "common/Pcsx2Defs.h"
#include "MTVU.h"
#include "SaveState.h"
#include "vtlb.h"

#include "common/Assertions.h"

// vtlb_DynBackpatchLoadStore is now in arm64/recVTLB_arm64.cpp

bool SaveStateBase::vuJITFreeze()
{
	if(IsSaving())
		vu1Thread.WaitVU();

	Console.Warning("recompiler state is stubbed in arm64!");

	// HACK!!

	// size of microRegInfo structure
	std::array<u8,96> empty_data{};
	Freeze(empty_data);
	Freeze(empty_data);
	return true;
}

// microVU test harness — exercises the ARM64 VU JIT (CpuArmVU0 / CpuArmVU1).
// This is the path that validates JIT-side fixes in iVU0micro_arm64.cpp /
// iVU1micro_arm64.cpp (SAFE_SUB, FTOI NaN fixup, Stage A+B pipeline sim,
// pair-flag-deferral, etc.). Do NOT revert this to CpuIntVU* — the interpreter
// cannot catch JIT-specific regressions.

#include "VUmicro.h"
#include "VU.h"
#include "arm64/aVU0.h"
#include "arm64/aVU.h"

namespace
{
// Per-test cleanup of INTERNAL execution state only. The test runner
// (run_mvu_tests.cpp) already zeroes VF/VI/ACC/micro_flags and applies
// per-test presets (including pending_q/pending_p for REG_Q/I/R_PRESET)
// BEFORE calling TestExec, so anything the runner owns as a test input
// must NOT be touched here or we'd clobber the presets.
//
// This helper resets the subset that is purely internal to VU execution
// and can leak across tests: pipeline rings, branch machinery, cycle
// bookkeeping, VI-delay slot, XGKICK state. It is a superset of the
// original mVU*_TestExec reset — adds ring *contents* (not just their
// head/tail counters), branchpc/delaybranchpc/takedelaybranch,
// VIBackupCycles/VIOldValue/VIRegNumber, nextBlockCycles, and
// xgkick state.
void resetVUForTest(VURegs& VU)
{
	// FMAC / FDIV / EFU / IALU pipeline rings — zero the contents too,
	// not just the head/tail counters, so stale entries can't manufacture
	// false RAW/WAW stalls.
	std::memset(VU.fmac, 0, sizeof(VU.fmac));
	VU.fmacreadpos  = 0;
	VU.fmacwritepos = 0;
	VU.fmaccount    = 0;
	std::memset(&VU.fdiv, 0, sizeof(VU.fdiv));
	std::memset(&VU.efu,  0, sizeof(VU.efu));
	std::memset(VU.ialu, 0, sizeof(VU.ialu));
	VU.ialureadpos  = 0;
	VU.ialuwritepos = 0;
	VU.ialucount    = 0;

	// Cycle / flags / branch execution state
	VU.cycle            = 0;
	VU.nextBlockCycles  = 0;
	VU.flags            = 0;
	VU.branch           = 0;
	VU.branchpc         = 0;
	VU.delaybranchpc    = 0;
	VU.takedelaybranch  = false;
	VU.ebit             = 0;

	// VU-level flag registers. M-2: VU1 previously skipped these, causing
	// asymmetric cross-test leakage relative to VU0. Zero both uniformly.
	VU.clipflag   = 0;
	VU.statusflag = 0;
	VU.macflag    = 0;

	// VI-delay writeback slot — internal to the JIT, not a test input.
	VU.VIBackupCycles = 0;
	VU.VIOldValue     = 0;
	VU.VIRegNumber    = 0;

	// XGKICK state (VU1-relevant; VU0's fields dormant but zeroed anyway).
	VU.xgkickaddr           = 0;
	VU.xgkickdiff           = 0;
	VU.xgkicksizeremaining  = 0;
	VU.xgkicklastcycle      = 0;
	VU.xgkickcyclecount     = 0;
	VU.xgkickenable         = 0;
	VU.xgkickendpacket      = 0;

	// DO NOT reset VF, VI, ACC, q, p, pending_q, pending_p, micro_*flags,
	// code, or start_pc — these are either test-owned inputs the runner
	// has just set up (VF/VI/ACC/q/p/pending_*) or already cleared by the
	// runner (micro_*flags). Clobbering them here breaks every test that
	// relies on a register preset.
}
} // namespace

void mVU0_TestInit()
{
	// Reserve() re-initialises the code-buffer pointers + zeros the block
	// table; Reset() zeros VU0 pipeline counters + rewinds s_code_write.
	// Both are idempotent so TestInit can be called repeatedly.
	CpuArmVU0.Reserve();
	CpuArmVU0.Reset();
}

void mVU0_TestShutdown()
{
	CpuArmVU0.Shutdown();
}

void mVU0_TestWriteProg(const u32* words, u32 count)
{
	std::memcpy(VU0.Micro, words, count * sizeof(u32));
	// New program bytes → invalidate every JIT block and reclaim code-buffer
	// space (Reset() rewinds s_code_write). Without this, cached blocks from
	// the previous test program could dispatch stale code.
	CpuArmVU0.Reset();
}

void mVU0_TestExec(u32 startPC, u32 cycles)
{
	resetVUForTest(VU0);

	// TPC stored as instruction-index (recArmVU0::Execute shifts <<= 3 on entry).
	VU0.VI[REG_TPC].UL = startPC >> 3;
	// Mark VU0 as running
	VU0.VI[REG_VPU_STAT].UL |= 0x1;

	CpuArmVU0.Execute(cycles);
}

void mVU1_TestInit()
{
	CpuArmVU1.Reserve();
	CpuArmVU1.Reset();
}

void mVU1_TestShutdown()
{
	CpuArmVU1.Shutdown();
}

void mVU1_TestWriteProg(const u32* words, u32 count)
{
	std::memcpy(VU1.Micro, words, count * sizeof(u32));
	CpuArmVU1.Reset();
}

void mVU1_TestExec(u32 startPC, u32 cycles)
{
	resetVUForTest(VU1);

	VU1.VI[REG_TPC].UL = startPC >> 3;
	// Mark VU1 as running (bit 8 of VU0's VPU_STAT)
	VU0.VI[REG_VPU_STAT].UL |= 0x100;

	CpuArmVU1.Execute(cycles);
}
