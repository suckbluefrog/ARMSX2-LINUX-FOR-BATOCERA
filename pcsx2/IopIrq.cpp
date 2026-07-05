// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "DEV9/DEV9.h"
#include "USB/USB.h"
#include "IopHw.h"
#include "IopDma.h"
#include "Common.h"
#include "R3000A.h"
#include "PS1DrvTrace.h"

using namespace R3000A;

void dev9Interrupt()
{
	if (DEV9irqHandler() != 1) return;

	iopIntcIrq(13);
}

void dev9Irq(int cycles)
{
	PSX_INT(IopEvt_DEV9, cycles);
}

void usbInterrupt()
{
	iopIntcIrq(22);
}

void usbIrq(int cycles)
{
	PSX_INT(IopEvt_USB, cycles);
}

void fwIrq()
{
	iopIntcIrq(24);
}

void spu2Irq()
{
	#ifdef SPU2IRQTEST
		Console.Warning("spu2Irq");
	#endif
	iopIntcIrq(9);
}

// IRQ-name table for human-readable trace logs.
static const char* kIopIrqName(uint t)
{
	switch (t)
	{
		case 0:  return "VBLANK";
		case 1:  return "GPU";
		case 2:  return "CDROM";
		case 3:  return "DMA";
		case 4:  return "TMR0";
		case 5:  return "TMR1";
		case 6:  return "TMR2";
		case 7:  return "SIO0";
		case 8:  return "SIO1";
		case 9:  return "SPU";
		case 10: return "PIO";
		case 11: return "VBLANK_END";
		case 13: return "DEV9";
		case 22: return "USB";
		default: return "?";
	}
}

void iopIntcIrq(uint irqType)
{
	const u32 was_pending = psxHu32(HW_ISTAT) & (1u << irqType);
	psxHu32(HW_ISTAT) |= 1 << irqType;
	PS1DRV_LOG_IRQ("iopIntcIrq fire irq=%u (%s) was_pending=%d ISTAT=0x%08x IMASK=0x%08x",
		irqType, kIopIrqName(irqType), was_pending ? 1 : 0,
		psxHu32(HW_ISTAT), psxHu32(HW_IMASK));
	iopTestIntc();
}
