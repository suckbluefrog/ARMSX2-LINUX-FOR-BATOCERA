// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+
//
// ARM64 DMA Recompiler — Native DMA channel execution wrappers.
// Each channel can be individually toggled between interpreter fallback
// and native ARM64 codegen via INTERP_DMAC / ISTUB_DMAC_* defines
// in arm64Emitter.h.

#pragma once

// Per-channel DMA execution functions (replace dmaVIF0/dmaVIF1/etc.)
extern void recDMAC_VIF0();
extern void recDMAC_VIF1();
extern void recDMAC_GIF();
extern void recDMAC_IPU0();   // fromIPU
extern void recDMAC_IPU1();   // toIPU
extern void recDMAC_SIF0();
extern void recDMAC_SIF1();
extern void recDMAC_SIF2();
extern void recDMAC_SPR0();   // fromSPR
extern void recDMAC_SPR1();   // toSPR

// Per-channel DMA interrupt handlers
extern void recDMACInterrupt_VIF0();
extern void recDMACInterrupt_VIF1();
extern void recDMACInterrupt_GIF();
extern void recDMACInterrupt_IPU0();
extern void recDMACInterrupt_IPU1();
extern void recDMACInterrupt_SPR0();
extern void recDMACInterrupt_SPR1();
