// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU0 Recompiler — Class declaration.
// Mirrors the VU1 ARM64 recompiler but for VU0 (4KB micro memory, no XGKICK).

#pragma once

#include "VUmicro.h"

class recArmVU0 final : public BaseVUmicroCPU
{
public:
	recArmVU0();
	~recArmVU0() override { Shutdown(); }

	const char* GetShortName() const override { return "armVU0"; }
	const char* GetLongName() const override { return "ARM64 VU0 Recompiler"; }

	void Reserve();
	void Shutdown() override;
	void Reset() override;
	void SetStartPC(u32 startPC) override;
	void Step() override;
	void Execute(u32 cycles) override;
	void Clear(u32 addr, u32 size) override;
	void ResumeXGkick() override {}
};

extern recArmVU0 CpuArmVU0;
