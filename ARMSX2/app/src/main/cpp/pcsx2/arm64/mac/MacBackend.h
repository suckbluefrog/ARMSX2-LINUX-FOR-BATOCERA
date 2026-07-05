// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#pragma once

// Bridging header for the macOS-port arm64 EE/IOP/VU recompiler that lives
// alongside our original Android-native arm64 backend. Everything inside
// pcsx2_macrec is the macOS port's translation units wrapped in a namespace
// so its internal helpers (armAsm, armEmitCall, dVifReset, etc.) don't
// collide at link time with the same-named symbols from arm64/*.cpp.
//
// The macOS port's microVU CPU classes are renamed recMacMVU0/recMacMVU1 (vs.
// VUmicro.h's recMicroVU0/recMicroVU1 which are final). The renamed classes
// are declared inside pcsx2_macrec — see arm64/mac/aVU.h.
//
// Android refresh installs this backend for all enabled ARM64 recompilers.

#include "R5900.h"
#include "R3000A.h"
#include "VUmicro.h"
#include "arm64/mac/aVU.h"

namespace pcsx2_macrec
{
	extern R5900cpu  recCpu;
	extern R3000Acpu psxRec;
}
