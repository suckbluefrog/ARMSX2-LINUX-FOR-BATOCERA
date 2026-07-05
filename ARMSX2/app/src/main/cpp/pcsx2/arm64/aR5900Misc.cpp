// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — Misc / no-op / unknown opcode handlers.
// Counterpart of pcsx2/x86/iR5900Misc.cpp.
//
// Every other EE opcode now has a native implementation in its dedicated
// iR5900*_arm64.cpp file. This file is the home for the leftover misc
// instructions that emit nothing (or only a recompile-time log line):
//
//   - PREF/SYNC/CACHE              : emit nothing — PS2 cache/pipeline not modeled
//   - {,MMI_,COP0_,COP1_}Unknown   : log a recompile-time error, emit nothing

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"

using namespace R5900;

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ============================================================================
//  Misc no-ops
// ============================================================================
//
// CACHE/PREF/SYNC don't need any emitted instructions:
//   - SYNC : our model has no EE pipeline/cache state to flush
//   - PREF : prefetch hint, no observable effect
//   - CACHE: not emulated (Suikoden 3 hits this a lot per the x86 source)

void recCACHE() {}
void recPREF()  {}
void recSYNC()  {}

// ============================================================================
//  Unknown / unimplemented opcodes
// ============================================================================
//
// These fire at *recompile* time (when the block is built), not at execution,
// matching the behavior of the x86 recompiler. We deliberately do not emit a
// runtime trap — original PCSX2 logs the error once at compile time and lets
// the (empty) emitted code fall through.

void recUnknown()
{
	Console.Error("EE: Unrecognized op %x", cpuRegs.code);
}

void recMMI_Unknown()
{
	Console.Error("EE: Unrecognized MMI op %x", cpuRegs.code);
}

void recCOP0_Unknown()
{
	Console.Error("EE: Unrecognized COP0 op %x", cpuRegs.code);
}

void recCOP1_Unknown()
{
	Console.Error("EE: Unrecognized FPU/COP1 op %x", cpuRegs.code);
}

// ---- Traps — see iR5900Trap_arm64.cpp ----
// ---- MultDiv — see iR5900MultDiv_arm64.cpp ----
// ---- ALU — see iR5900ALU_arm64.cpp ----
// ---- Shifts — see iR5900Shift_arm64.cpp ----
// ---- Moves — see iR5900Move_arm64.cpp ----
// ---- Loads/Stores — see iR5900LoadStore_arm64.cpp ----
// ---- MMI — see iR5900MMI_arm64.cpp ----
// ---- COP0 — see iR5900COP0_arm64.cpp ----
// ---- COP1 — see iR5900COP1_arm64.cpp ----

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
