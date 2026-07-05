// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#pragma once
#include "Pcsx2Defs.h"
#include <cstdint>
#include <cstddef>

struct fastjmp_buf
{
#if defined(_WIN32)
	static constexpr std::size_t BUF_SIZE = 240;
#elif defined(ARCH_ARM64)
	// AArch64 fastjmp stores:
	// x16/x30, x19-x28, x29, and d8-d15. The final stp d14,d15 at offset
	// 160 writes through byte 175, so the buffer must be 176 bytes. 168
	// corrupts the following object and breaks recompiler exits on Android.
	static constexpr std::size_t BUF_SIZE = 176;
#else
	static constexpr std::size_t BUF_SIZE = 64;
#endif

	alignas(16) std::uint8_t buf[BUF_SIZE];
};

extern "C" {
int fastjmp_set(fastjmp_buf* buf);
__noreturn void fastjmp_jmp(const fastjmp_buf* buf, int ret);
}
