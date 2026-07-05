// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// TRACE_BLOCKS: per-block register state dump for differential testing.
// Uncomment the define below (or define TRACE_BLOCKS before including this header).
// Run the same ELF once with the recompiler and once with --interp, then diff
// /tmp/armsx2_reglog.txt (EE) and /tmp/armsx2_ioplog.txt (IOP) to find divergence.

#pragma once

//#define TRACE_BLOCKS 1

#ifdef TRACE_BLOCKS

#include <cstdio>
#include <zlib.h>
#include "R5900.h"
#include "R3000A.h"
#include "VUmicro.h"
#include "DebugTools/Debug.h"

static inline void eeTraceBlock(u32 blockpc)
{
	static FILE* fp = nullptr;
	static bool fp_opened = false;
	if (!fp_opened)
	{
		fp = std::fopen("/tmp/armsx2_reglog.txt", "wb");
		fp_opened = true;
	}
	if (!fp)
		return;

	u32 hash = crc32(0, (Bytef*)&cpuRegs, offsetof(cpuRegisters, pc));
	u32 hashf = crc32(0, (Bytef*)&fpuRegs, sizeof(fpuRegisters));
	u32 hashi = crc32(0, (Bytef*)&VU0, offsetof(VURegs, idx));

	std::fprintf(fp, "%08X (%u; %08X; %08X; %08X):", cpuRegs.pc, (u32)cpuRegs.cycle, hash, hashf, hashi);
	for (int i = 0; i < 34; i++)
		std::fprintf(fp, " %s: %08X%08X%08X%08X", R3000A::disRNameGPR[i],
			cpuRegs.GPR.r[i].UL[3], cpuRegs.GPR.r[i].UL[2],
			cpuRegs.GPR.r[i].UL[1], cpuRegs.GPR.r[i].UL[0]);

	std::fprintf(fp, "\nFPR: CR: %08X ACC: %08X", fpuRegs.fprc[31], fpuRegs.ACC.UL);
	for (int i = 0; i < 32; i++)
		std::fprintf(fp, " %08X", fpuRegs.fpr[i].UL);

	std::fprintf(fp, "\nVF: ");
	for (int i = 0; i < 32; i++)
		std::fprintf(fp, " %u: %08X %08X %08X %08X", i,
			VU0.VF[i].UL[0], VU0.VF[i].UL[1], VU0.VF[i].UL[2], VU0.VF[i].UL[3]);
	std::fprintf(fp, "\nVI: ");
	for (int i = 0; i < 32; i++)
		std::fprintf(fp, " %u: %08X", i, VU0.VI[i].UL);
	std::fprintf(fp, "\nACC: %08X %08X %08X %08X Q: %08X P: %08X",
		VU0.ACC.UL[0], VU0.ACC.UL[1], VU0.ACC.UL[2], VU0.ACC.UL[3], VU0.q.UL, VU0.p.UL);
	std::fprintf(fp, " MAC %08X %08X %08X %08X",
		VU0.micro_macflags[3], VU0.micro_macflags[2], VU0.micro_macflags[1], VU0.micro_macflags[0]);
	std::fprintf(fp, " CLIP %08X %08X %08X %08X",
		VU0.micro_clipflags[3], VU0.micro_clipflags[2], VU0.micro_clipflags[1], VU0.micro_clipflags[0]);
	std::fprintf(fp, " STATUS %08X %08X %08X %08X",
		VU0.micro_statusflags[3], VU0.micro_statusflags[2], VU0.micro_statusflags[1], VU0.micro_statusflags[0]);
	std::fprintf(fp, "\n");
}

static inline void iopTraceBlock(u32 blockpc)
{
	static FILE* fp = nullptr;
	static bool fp_opened = false;
	if (!fp_opened)
	{
		fp = std::fopen("/tmp/armsx2_ioplog.txt", "wb");
		fp_opened = true;
	}
	if (!fp)
		return;

	u32 hash = crc32(0, (Bytef*)&psxRegs, offsetof(psxRegisters, pc));

	std::fprintf(fp, "%08X (%u; %08X):", psxRegs.pc, (u32)psxRegs.cycle, hash);
	for (int i = 0; i < 34; i++)
		std::fprintf(fp, " %s: %08X", R3000A::disRNameGPR[i], psxRegs.GPR.r[i]);
	std::fprintf(fp, "\n");
}

#endif // TRACE_BLOCKS
