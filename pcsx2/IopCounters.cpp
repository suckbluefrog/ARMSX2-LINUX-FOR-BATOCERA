// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

// Note on INTC usage: All counters code is always called from inside the context of an
// event test, so instead of using the iopTestIntc we just set the 0x1070 flags directly.
// The EventText function will pick it up.

#include "IopCounters.h"
#include "R3000A.h"
#include "Common.h"
#include "SPU2/spu2.h"
#include "DEV9/DEV9.h"
#include "USB/USB.h"
#include "IopHw.h"
#include "IopDma.h"
#include "CDVD/CDVD.h"

#include <math.h>
#include <algorithm>

/* Config.PsxType == 1: PAL:
	 VBlank interlaced		50.00 Hz
	 VBlank non-interlaced	49.76 Hz
	 HBlank					15.625 KHz
   Config.PsxType == 0: NSTC
	 VBlank interlaced		59.94 Hz
	 VBlank non-interlaced	59.82 Hz
	 HBlank					15.73426573 KHz */

// Misc IOP Clocks
// FIXME: this divider is actually 2.73 (36864000 / 13500000), but not sure what uses it, so this'll do, we should maybe change things to float.
#define PSXPIXEL 3
#define PSXSOUNDCLK ((int)(48000))

psxCounter psxCounters[NUM_COUNTERS];
s32 psxNextDeltaCounter;
u64 psxNextStartCounter;

bool hBlanking = false;
bool vBlanking = false;

// flags when the gate is off or counter disabled. (do not count)
#define IOPCNT_STOPPED (0x10000000ul)

// used to disable targets until after an overflow
#define IOPCNT_FUTURE_TARGET (0x1000000000ULL)
#define IOPCNT_MODE_WRITE_MSK 0x63FF
#define IOPCNT_MODE_FLAG_MSK 0x1C00

#define IOPCNT_GATE_CNT_LOW 0           // Counts only when signal is low (V/H RENDER).
#define IOPCNT_GATE_CLR_END 1           // Counts continuously, clears at end of BLANK.
#define IOPCNT_GATE_CNT_HIGH_ZERO_OFF 2 // Starts at beginning of next BLANK, counts only during BLANK, returns zero any other time.
#define IOPCNT_GATE_START_AT_END 3      // Starts at end of next BLANK, continuous count, no clear.

// Use an arbitrary value to flag HBLANK counters.
// These counters will be counted by the hblank gates coming from the EE,
// which ensures they stay 100% in sync with the EE's hblank counters.
#define PSXHBLANK 0x2001

// =====================  IOP COUNTER RATE TUNING KNOBS  =====================
// Scale factor (in 1/256ths) applied to each IOP timer T0-T7 when its rate
// is set. 256 = 1.0x (upstream behavior).
//
// Counter rate determines how many IOP cycles per counter increment. Larger
// rate = SLOWER counter = game perceives MORE time per tick (so reading the
// counter, the game thinks more time passed → game advances FASTER).
//
// Counter 6 = SPU2 sample sync (rate=768, ticks at 44100Hz wall-clock in PS1
// mode). Counter 7 = RTC (rate=PSXCLK/1000, ticks at 1ms wall-clock).
//
// To find which counter the game uses for sub-frame timing: bump one knob
// to 4× (= 1024) or 16× (= 4096) and watch the game speed change.
//
// Note: PSXHBLANK and PSXPIXEL are sentinel values, not real rates — the
// scaling helper leaves those alone.
static u32 PSX_RCNT_SCALE_256[8] = {
	256, // T0 — pixel/dotclock-source available
	256, // T1 — HBLANK-source available
	256, // T2 — system clock /1 or /8
	256, // T3 — HBLANK-source available
	256, // T4 — system clock /1, /8, /16, /256
	256, // T5 — system clock /1, /8, /16, /256
	256, // T6 — SPU2 sample sync (rate=768, ticks at 44100Hz wall-clock)
	256, // T7 — RTC (rate=PSXCLK/1000, ticks at 1ms wall-clock)
};

static u32 psxRcntApplyRateScale(int idx, u32 base_rate)
{
	if (idx < 0 || idx >= 8) return base_rate;
	if (base_rate == PSXHBLANK) return base_rate;
	const u64 scaled = (static_cast<u64>(base_rate) * static_cast<u64>(PSX_RCNT_SCALE_256[idx])) >> 8;
	if (scaled < 1) return 1;
	if (scaled > 0xFFFFFFFFULL) return 0xFFFFFFFFu;
	return static_cast<u32>(scaled);
}

static bool psxRcntCanCount(int cntidx)
{
	if (psxCounters[cntidx].mode.stopped)
		return false;

	if (!(psxCounters[cntidx].mode.gateEnable))
		return true;

	const u32 gateMode = psxCounters[cntidx].mode.gateMode;

	if (cntidx == 2 || cntidx == 4 || cntidx == 5)
	{
		// Gates being enabled on these counters forces it to disable the counter if being on or off depends on a gate being on or off.
		return (gateMode & 1);
	}

	const bool blanking = cntidx == 0 ? hBlanking : vBlanking;

	// Stop counting if Gate mode 0 (only count when rendering) and blanking or Gate mode 2 (only count when blanking) when not blanking
	if ((gateMode == IOPCNT_GATE_CNT_LOW && blanking == true) || (gateMode == IOPCNT_GATE_CNT_HIGH_ZERO_OFF && blanking == false))
		return false;

	// All other cases allow counting.
	return true;
}

static void psxRcntSync(int cntidx)
{
	if ((psxCounters[cntidx].currentIrqMode.repeatInterrupt) && !(psxCounters[cntidx].currentIrqMode.toggleInterrupt))
	{ //Repeat IRQ mode Pulsed, resets a few cycles after the interrupt, this should do.
		psxCounters[cntidx].mode.intrEnable = true;
	}

	if (psxRcntCanCount(cntidx) && psxCounters[cntidx].rate != PSXHBLANK)
	{
		const u32 change = (psxRegs.cycle - psxCounters[cntidx].startCycle) / psxCounters[cntidx].rate;
		if (change > 0)
		{
			psxCounters[cntidx].count += change;
			psxCounters[cntidx].startCycle += change * psxCounters[cntidx].rate;

			psxCounters[cntidx].startCycle &= ~((u64)psxCounters[cntidx].rate - 1);
		}
	}
	else
	{
		if (psxCounters[cntidx].mode.gateEnable && psxCounters[cntidx].mode.gateMode == IOPCNT_GATE_CNT_HIGH_ZERO_OFF)
			psxCounters[cntidx].count = 0;

		psxCounters[cntidx].startCycle = psxRegs.cycle;
	}
}

static void _rcntSet(int cntidx)
{
	u64 overflowCap = (cntidx >= 3) ? 0x100000000ULL : 0x10000;
	u64 c;

	const psxCounter& counter = psxCounters[cntidx];

	// psxNextCounter is relative to the psxRegs.cycle when rcntUpdate() was last called.
	// However, the current _rcntSet could be called at any cycle count, so we need to take
	// that into account.  Adding the difference from that cycle count to the current one
	// will do the trick!

	if (counter.rate == PSXHBLANK || !psxRcntCanCount(cntidx))
		return;

	// check for special cases where the overflow or target has just passed
	// (we probably missed it because we're doing/checking other things)
	if (counter.count > overflowCap || counter.count > counter.target)
	{
		psxNextDeltaCounter = 4;
		return;
	}

	// FIXME: the u32 casts in this expression exist to match pre-64bit counter code, rewrite these
	c = (u64)((overflowCap - counter.count) * counter.rate) - ((u32)psxRegs.cycle - (u32)counter.startCycle);
	c += psxRegs.cycle - psxNextStartCounter; // adjust for time passed since last rcntUpdate();

	if (c < (u64)psxNextDeltaCounter)
	{
		psxNextDeltaCounter = (u32)c;
		psxSetNextBranch(psxNextStartCounter, psxNextDeltaCounter); //Need to update on counter resets/target changes
	}

	if (counter.target & IOPCNT_FUTURE_TARGET)
		return;

	c = (s64)((counter.target - counter.count) * counter.rate) - ((u32)psxRegs.cycle - (u32)counter.startCycle);
	c += psxRegs.cycle - psxNextStartCounter; // adjust for time passed since last rcntUpdate();

	if (c < (u64)psxNextDeltaCounter)
	{
		psxNextDeltaCounter = (u32)c;
		psxSetNextBranch(psxNextStartCounter, psxNextDeltaCounter); //Need to update on counter resets/target changes
	}
}


void psxRcntInit()
{
	int i;

	std::memset(psxCounters, 0, sizeof(psxCounters));

	for (i = 0; i < 3; i++)
	{
		psxCounters[i].rate = 1;
		psxCounters[i].mode.intrEnable = true;
		psxCounters[i].target = IOPCNT_FUTURE_TARGET;
		psxCounters[i].currentIrqMode.repeatInterrupt = false;
		psxCounters[i].currentIrqMode.toggleInterrupt = false;
	}
	for (i = 3; i < 6; i++)
	{
		psxCounters[i].rate = 1;
		psxCounters[i].mode.intrEnable = true;
		psxCounters[i].target = IOPCNT_FUTURE_TARGET;
		psxCounters[i].currentIrqMode.repeatInterrupt = false;
		psxCounters[i].currentIrqMode.toggleInterrupt = false;
	}

	psxCounters[0].interrupt = 0x10;
	psxCounters[1].interrupt = 0x20;
	psxCounters[2].interrupt = 0x40;

	psxCounters[3].interrupt = 0x04000;
	psxCounters[4].interrupt = 0x08000;
	psxCounters[5].interrupt = 0x10000;

	psxCounters[6].rate = psxRcntApplyRateScale(6, 768);
	psxCounters[6].deltaCycles = psxCounters[6].rate;
	psxCounters[6].mode.modeval = 0x8;

	psxCounters[7].rate = psxRcntApplyRateScale(7, PSXCLK / 1000);
	psxCounters[7].deltaCycles = psxCounters[7].rate;
	psxCounters[7].mode.modeval = 0x8;

	for (i = 0; i < 8; i++)
		psxCounters[i].startCycle = psxRegs.cycle;

	// Tell the IOP to branch ASAP, so that timers can get
	// configured properly.
	psxNextDeltaCounter = 1;
	psxNextStartCounter = psxRegs.cycle;
}

static void _rcntFireInterrupt(int i, bool isOverflow)
{
	bool updateIntr = psxCounters[i].currentIrqMode.repeatInterrupt;

	if (psxCounters[i].mode.intrEnable)
	{
		bool already_set = isOverflow ? psxCounters[i].mode.overflowFlag : psxCounters[i].mode.targetFlag;
		if (updateIntr || !already_set)
		{
			// IRQ fired
			//DevCon.Warning("Counter %d %s IRQ Fired count %x target %x psx Cycle %d", i, isOverflow ? "Overflow" : "Target", psxCounters[i].count, psxCounters[i].target, psxRegs.cycle);
			psxHu32(HW_ISTAT) |= psxCounters[i].interrupt;
			iopTestIntc();
		}

		updateIntr |= psxCounters[i].currentIrqMode.toggleInterrupt;
	}

	if (updateIntr)
	{
		if (psxCounters[i].currentIrqMode.toggleInterrupt)
		{
			// Toggle mode
			psxCounters[i].mode.intrEnable ^= true; // Interrupt flag inverted
		}
		else
		{
			psxCounters[i].mode.intrEnable = false; // Interrupt flag set low
		}
	}
}

static void _rcntTestTarget(int i)
{
	if (psxCounters[i].count < psxCounters[i].target)
		return;

	PSXCNT_LOG("IOP Counter[%d] target 0x%I64x >= 0x%I64x (mode: %x)",
			   i, psxCounters[i].count, psxCounters[i].target, psxCounters[i].mode.modeval);

	if (psxCounters[i].mode.targetIntr)
	{
		// Target interrupt
		_rcntFireInterrupt(i, false);
	}

	psxCounters[i].mode.targetFlag = true;

	if (psxCounters[i].mode.zeroReturn)
	{
		// Reset on target
		psxCounters[i].count -= psxCounters[i].target;
	}
	else
		psxCounters[i].target |= IOPCNT_FUTURE_TARGET;
}


static __fi void _rcntTestOverflow(int i)
{
	u64 maxTarget = (i < 3) ? 0xffff : 0xfffffffful;
	if (psxCounters[i].count <= maxTarget)
		return;

	PSXCNT_LOG("IOP Counter[%d] overflow 0x%I64x >= 0x%I64x (mode: %x)",
			   i, psxCounters[i].count, maxTarget, psxCounters[i].mode.modeval);

	if (psxCounters[i].mode.overflIntr)
	{
		// Overflow interrupt
		_rcntFireInterrupt(i, true);
	}

	psxCounters[i].mode.overflowFlag = true;

	// Update count.
	// Count wraps around back to zero, while the target is restored (if not in one shot mode).
	// (high bit of the target gets set by rcntWtarget when the target is behind
	// the counter value, and thus should not be flagged until after an overflow)
	psxCounters[i].count -= maxTarget + 1;
	psxCounters[i].target &= maxTarget;
}

/*
Gate:
   TM_NO_GATE                   000
   TM_GATE_ON_Count             001
   TM_GATE_ON_ClearStart        011
   TM_GATE_ON_Clear_OFF_Start   101
   TM_GATE_ON_Start             111

   = means counting
   - means not counting

   V-blank  ----+    +----------------------------+    +------
                |    |                            |    |
                |    |                            |    |
                +----+                            +----+
 TM_NO_GATE:

                0================================>============

 TM_GATE_ON_Count:

                <---->===========================><---->======

 TM_GATE_ON_ClearStart:

                =====>0================================>0=====

 TM_GATE_ON_Clear_OFF_Start:

                0====>0-------------------------->0====>0-----

 TM_GATE_ON_Start:

                <---->===========================>============
*/

static void _psxCheckStartGate(int i)
{
	if (!(psxCounters[i].mode.gateEnable))
		return; // Ignore Gate

	switch (psxCounters[i].mode.gateMode)
	{
		case 0x0: // GATE_ON_count - count while gate signal is low (RENDER)

			// get the current count at the time of stoppage:
			psxCounters[i].count = (i < 3) ?
									   psxRcntRcount16(i) :
									   psxRcntRcount32(i);

			// Not strictly necessary.
			psxCounters[i].startCycle = psxRegs.cycle & ~((u64)psxCounters[i].rate - 1);
			break;

		case 0x1: // GATE_ON_ClearStart - Counts constantly, clears on Blank END
			// do nothing!
			break;

		case 0x2: // GATE_ON_Clear_OFF_Start - Counts only when Blanking, clears on both ends, starts counting on next Blank Start.
			psxRcntSync(i);
			psxCounters[i].mode.stopped = false;
			psxCounters[i].count = 0;
			psxCounters[i].target &= ~IOPCNT_FUTURE_TARGET;
			break;

		case 0x3: //GATE_ON_Start - Starts counting when the next Blank Ends, no clear.
			// do nothing!
			break;
	}
}

static void _psxCheckEndGate(int i)
{
	if (!(psxCounters[i].mode.gateEnable))
		return; // Ignore Gate

	// NOTE: Starting and stopping of modes 0 and 2 are checked in psxRcntCanCount(), only need to update the start cycle and counts.
	switch (psxCounters[i].mode.gateMode)
	{
		case 0x0: // GATE_ON_count - count while gate signal is low (RENDER)
			psxCounters[i].startCycle = psxRegs.cycle & ~((u64)psxCounters[i].rate - 1);
			break;

		case 0x1: // GATE_ON_ClearStart - Counts constantly, clears on Blank END
			psxRcntSync(i);
			psxCounters[i].count = 0;
			psxCounters[i].target &= ~IOPCNT_FUTURE_TARGET;
			break;

		case 0x2: // GATE_ON_Clear_OFF_Start - Counts only when Blanking, clears on both ends, starts counting on next Blank Start.
			// No point in updating the count, since we're gonna clear it.
			psxRcntSync(i);
			psxCounters[i].count = 0;
			psxCounters[i].target &= ~IOPCNT_FUTURE_TARGET;
			break; // do not set the counter

		case 0x3: // GATE_ON_Start - Starts counting when the next Blank Ends, no clear.
			if (psxCounters[i].mode.stopped)
			{
				psxCounters[i].startCycle = psxRegs.cycle & ~((u64)psxCounters[i].rate - 1);
				psxCounters[i].mode.stopped = false;
			}
			break;
	}
}

void psxHBlankStart()
{
	// AlternateSource/scanline counters for Gates 1 and 3.
	// We count them here so that they stay nicely synced with the EE's hsync.
	if ((psxCounters[1].rate == PSXHBLANK) && psxRcntCanCount(1))
	{
		psxCounters[1].count++;
		_rcntTestOverflow(1);
		_rcntTestTarget(1);
	}

	if ((psxCounters[3].rate == PSXHBLANK) && psxRcntCanCount(3))
	{
		psxCounters[3].count++;
		_rcntTestOverflow(3);
		_rcntTestTarget(3);
	}

	_psxCheckStartGate(0);

	hBlanking = true;

	_rcntSet(0);
}

void psxHBlankEnd()
{
	_psxCheckEndGate(0);

	hBlanking = false;

	_rcntSet(0);
}

// =====================  IOP VSYNC IRQ DELIVERY KNOB  =====================
// Number of EE-driven vblanks per IOP-visible vsync IRQ.
//
// MANUAL knob (PSX_VSYNC_IRQ_EVERY_N): unconditional divider applied first.
// 1 = upstream (every emulator-vblank = 60Hz wall-clock NTSC).
// 2 = halve (= 30Hz). 4 = quarter.
//
// AUTO multiplier (PSX_VSYNC_IRQ_AUTO_FROM_GP1): when 1, doubles the divider
// whenever PS1 game has set GP1(08) interlace bit. The reasoning: PS1 GPU
// fires vsync at FIELD rate (60Hz NTSC) regardless of mode, but interlaced
// games typically want 30fps logic — they advance state every 2 fields. By
// auto-halving the IOP-visible vsync rate when the game requests interlaced,
// the game's per-vsync logic step lands on a real-time 30Hz cadence.
//
// Effective N = PSX_VSYNC_IRQ_EVERY_N * (GP1.interlace ? 2 : 1)
//
// NOTE: this also gates psxCheckStartGate / _rcntSet for counters 1 and 3
// (HBLANK gate counters) — those expect a vblank to "open" their gates.
// For diagnostic purposes that's fine; for production we'd split.
static u32 PSX_VSYNC_IRQ_EVERY_N = 2;
static u32 PSX_VSYNC_IRQ_AUTO_FROM_GP1 = 1;

#include "ps2/pgif.h"

void psxVBlankStart()
{
	cdvdVsync();

	// Apply the vsync divider ONLY when actually in PS1 mode (HW_ICFG bit
	// 3 set). Before the PS1 handoff and during PS2 mode generally, behave
	// exactly like upstream — otherwise the PS2 BIOS / PS1DRV setup path
	// corrupts state and crashes the EE (vtlb assert at unmapped
	// 0x1CB40400 during the SBUS_F240 handoff).
	const bool ps1_mode = (psxHu32(HW_ICFG) & 0x8) != 0;
	u32 effective_n = 1;
	if (ps1_mode)
	{
		effective_n = PSX_VSYNC_IRQ_EVERY_N;
		if (PSX_VSYNC_IRQ_AUTO_FROM_GP1 && pgpu.stat.bits.VILAC)
			effective_n *= 2;
		if (effective_n < 1) effective_n = 1;
	}

	// Periodic 1-Hz log so we can see the auto-detected mode while tuning.
	{
		static u32 s_logCnt = 0;
		if ((++s_logCnt % 60) == 1)
			Console.WriteLn(Color_StrongCyan,
				"[PSX-VSYNC] ps1=%d N=%u (manual=%u, auto=%u, GP1.VILAC=%u, GP1.VRES=%u)",
				ps1_mode ? 1 : 0, effective_n, PSX_VSYNC_IRQ_EVERY_N,
				PSX_VSYNC_IRQ_AUTO_FROM_GP1,
				(unsigned)pgpu.stat.bits.VILAC, (unsigned)pgpu.stat.bits.VRES);
	}

	// Deliver vsync IRQ + gate transitions to IOP only every Nth call.
	static u32 s_vbCount = 0;
	const bool deliver = ((++s_vbCount % effective_n) == 0);

	if (deliver)
	{
		iopIntcIrq(0);

		_psxCheckStartGate(1);
		_psxCheckStartGate(3);
	}

	vBlanking = true;

	if (deliver)
	{
		_rcntSet(1);
		_rcntSet(3);
	}
}

// Knob: throttle the vblank-END IRQ (IRQ 11) the same way as vblank-START.
// PS1 BIOS may chain a handler on the vblank-end vector for state-machine
// dispatch; if that's the speedup source, increasing this drags the BIOS.
static u32 PSX_VSYNC_END_IRQ_EVERY_N = 1;

void psxVBlankEnd()
{
	// Same PS1-mode gate as psxVBlankStart: don't throttle in PS2 mode or
	// during the PS1 handoff window before HW_ICFG bit 3 is set.
	const bool ps1_mode = (psxHu32(HW_ICFG) & 0x8) != 0;
	const u32 effective_n = ps1_mode ? std::max<u32>(1, PSX_VSYNC_END_IRQ_EVERY_N) : 1;

	static u32 s_veCount = 0;
	const bool deliver = ((++s_veCount % effective_n) == 0);

	if (deliver)
	{
		iopIntcIrq(11);

		_psxCheckEndGate(1);
		_psxCheckEndGate(3);
	}

	vBlanking = false;

	if (deliver)
	{
		_rcntSet(1);
		_rcntSet(3);
	}
}

void psxRcntUpdate()
{
	int i;

	psxNextDeltaCounter = 0x7fffffff;
	psxNextStartCounter = psxRegs.cycle;

	for (i = 0; i < 6; i++)
	{
		// don't count disabled or hblank counters...
		// We can't check the ALTSOURCE flag because the PSXCLOCK source *should*
		// be counted here.

		psxRcntSync(i);

		if (psxCounters[i].rate == PSXHBLANK)
			continue;

		if (!psxRcntCanCount(i))
			continue;

		_rcntTestOverflow(i);
		_rcntTestTarget(i);
	}

	const u32 spu2_delta = (psxRegs.cycle - lClocks) % 768;
	psxCounters[6].startCycle = psxRegs.cycle - spu2_delta;
	psxCounters[6].deltaCycles = psxCounters[6].rate;
	SPU2async();
	psxNextDeltaCounter = psxCounters[6].deltaCycles;

	DEV9async(1);    
	const s32 diffusb = psxRegs.cycle - psxCounters[7].startCycle;
	s32 cusb = psxCounters[7].deltaCycles;

	if (diffusb >= psxCounters[7].deltaCycles)
	{
		USBasync(diffusb);
		psxCounters[7].startCycle += psxCounters[7].rate * (diffusb / psxCounters[7].rate);
		psxCounters[7].deltaCycles = psxCounters[7].rate;
	}
	else
		cusb -= diffusb;

	if (cusb < psxNextDeltaCounter)
		psxNextDeltaCounter = cusb;

	for (i = 0; i < 6; i++)
		_rcntSet(i);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void psxRcntWcount16(int index, u16 value)
{
	pxAssert(index < 3);
	PSXCNT_LOG("16bit IOP Counter[%d] writeCount16 = %x", index, value);

	psxRcntSync(index);

	psxCounters[index].count = value & 0xffff;

	psxCounters[index].target &= 0xffff;

	if (psxCounters[index].count > psxCounters[index].target)
	{
		// Count already higher than Target
		//DevCon.Warning("32bit Count already higher than target");
		psxCounters[index].target |= IOPCNT_FUTURE_TARGET;
	}

	_rcntSet(index);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void psxRcntWcount32(int index, u32 value)
{
	pxAssert(index >= 3 && index < 6);
	PSXCNT_LOG("32bit IOP Counter[%d] writeCount32 = %x", index, value);

	psxRcntSync(index);

	psxCounters[index].count = value;

	psxCounters[index].target &= 0xffffffff;

	if (psxCounters[index].count > psxCounters[index].target)
	{
		// Count already higher than Target
		//DevCon.Warning("32bit Count already higher than target");
		psxCounters[index].target |= IOPCNT_FUTURE_TARGET;
	}

	_rcntSet(index);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
__fi void psxRcntWmode16(int index, u32 value)
{
	int irqmode = 0;
	PSXCNT_LOG("16bit IOP Counter[%d] writeMode = 0x%04X", index, value);

	pxAssume(index >= 0 && index < 3);
	psxCounter& counter = psxCounters[index];
	psxCounterMode oldMode = counter.mode;

	counter.mode.modeval = (value & IOPCNT_MODE_WRITE_MSK) | (counter.mode.modeval & IOPCNT_MODE_FLAG_MSK); // Write new value, preserve flags

	if (!((oldMode.targetFlag || oldMode.overflowFlag) && (oldMode.targetIntr || oldMode.overflIntr)))
		psxRcntSetNewIntrMode(index);

	if (counter.mode.repeatIntr != counter.currentIrqMode.repeatInterrupt || counter.mode.toggleIntr != counter.currentIrqMode.toggleInterrupt)
		DevCon.Warning("Write to psxCounter[%d] mode old repeat %d new %d old toggle %d new %d", index, counter.currentIrqMode.repeatInterrupt, counter.mode.repeatIntr, counter.currentIrqMode.toggleInterrupt, counter.mode.toggleIntr);
	
	if (value & (1 << 4))
	{
		irqmode += 1;
	}
	if (value & (1 << 5))
	{
		irqmode += 2;
	}
	if (value & (1 << 7))
	{
		PSXCNT_LOG("16 Counter %d Toggle IRQ on %s", index, (irqmode & 3) == 1 ? "Target" : ((irqmode & 3) == 2 ? "Overflow" : "Target and Overflow"));
	}
	else
	{
		PSXCNT_LOG("16 Counter %d Pulsed IRQ on %s", index, (irqmode & 3) == 1 ? "Target" : ((irqmode & 3) == 2 ? "Overflow" : "Target and Overflow"));
	}
	if (!(value & (1 << 6)))
	{
		PSXCNT_LOG("16 Counter %d One Shot", index);
	}
	else
	{
		PSXCNT_LOG("16 Counter %d Repeat", index);
	}
	if (index == 2)
	{
		if (counter.mode.t2Prescale)
			psxCounters[2].rate = psxRcntApplyRateScale(2, 8);
		else
			psxCounters[2].rate = psxRcntApplyRateScale(2, 1);
	}
	else
	{
		// Counters 0 and 1 can select PIXEL or HSYNC as an alternate source:
		counter.rate = psxRcntApplyRateScale(index, 1);

		if (counter.mode.extSignal)
			counter.rate = (index == 0) ? PSXPIXEL : PSXHBLANK;

		if (counter.rate == PSXPIXEL)
			Console.Warning("PSX Pixel clock set to time 0, sync may be incorrect");

		if (counter.mode.gateEnable)
		{
			// If set to gate mode 2 or 3, the counting starts at the start and end of the next blank respectively depending on which counter.
			if (counter.mode.gateMode >= IOPCNT_GATE_CNT_HIGH_ZERO_OFF)
				counter.mode.stopped = true;

			PSXCNT_LOG("IOP Counter[%d] Gate Check set, value = 0x%04X", index, value);
		}
	}

	// Current counter *always* resets on mode write.
	counter.count = 0;
	counter.startCycle = psxRegs.cycle & ~((u64)counter.rate - 1);
	counter.target &= 0xffff;

	_rcntSet(index);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
__fi void psxRcntWmode32(int index, u32 value)
{
	PSXCNT_LOG("32bit IOP Counter[%d] writeMode = 0x%04x", index, value);
	int irqmode = 0;
	pxAssume(index >= 3 && index < 6);
	psxCounter& counter = psxCounters[index];
	psxCounterMode oldMode = counter.mode;

	counter.mode.modeval = (value & IOPCNT_MODE_WRITE_MSK) | (counter.mode.modeval & IOPCNT_MODE_FLAG_MSK); // Write new value, preserve flags

	if (!((oldMode.targetFlag || oldMode.overflowFlag) && (oldMode.targetIntr || oldMode.overflIntr)))
		psxRcntSetNewIntrMode(index);

	if (counter.mode.repeatIntr != counter.currentIrqMode.repeatInterrupt || counter.mode.toggleIntr != counter.currentIrqMode.toggleInterrupt)
		DevCon.Warning("Write to psxCounter[%d] mode old repeat %d new %d old toggle %d new %d", index, counter.currentIrqMode.repeatInterrupt, counter.mode.repeatIntr, counter.currentIrqMode.toggleInterrupt, counter.mode.toggleIntr);
	
	if (value & (1 << 4))
	{
		irqmode += 1;
	}
	if (value & (1 << 5))
	{
		irqmode += 2;
	}
	if (value & (1 << 7))
	{
		PSXCNT_LOG("32 Counter %d Toggle IRQ on %s", index, (irqmode & 3) == 1 ? "Target" : ((irqmode & 3) == 2 ? "Overflow" : "Target and Overflow"));
	}
	else
	{
		PSXCNT_LOG("32 Counter %d Pulsed IRQ on %s", index, (irqmode & 3) == 1 ? "Target" : ((irqmode & 3) == 2 ? "Overflow" : "Target and Overflow"));
	}
	if (!(value & (1 << 6)))
	{
		PSXCNT_LOG("32 Counter %d One Shot", index);
	}
	else
	{
		PSXCNT_LOG("32 Counter %d Repeat", index);
	}
	if (index == 3)
	{
		// Counter 3 has the HBlank as an alternate source.
		counter.rate = psxRcntApplyRateScale(3, 1);

		if (counter.mode.extSignal)
			counter.rate = PSXHBLANK;

		if (counter.mode.gateEnable)
		{
			PSXCNT_LOG("IOP Counter[3] Gate Check set, value = %x", value);
			// If set to gate mode 2 or 3, the counting starts at the start and end of the next blank respectively depending on which counter.
			if (counter.mode.gateMode >= IOPCNT_GATE_CNT_HIGH_ZERO_OFF)
				counter.mode.stopped = true;
		}
	}
	else
	{
		switch (counter.mode.t4_5Prescale)
		{
			case 0x1:
				counter.rate = psxRcntApplyRateScale(index, 8);
				break;
			case 0x2:
				counter.rate = psxRcntApplyRateScale(index, 16);
				break;
			case 0x3:
				counter.rate = psxRcntApplyRateScale(index, 256);
				break;
			default:
				counter.rate = psxRcntApplyRateScale(index, 1);
				break;
		}
	}

	// Current counter *always* resets on mode write.
	counter.count = 0;
	counter.startCycle = psxRegs.cycle & ~(static_cast<u64>(counter.rate - 1));
	counter.target &= 0xffffffff;
	_rcntSet(index);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void psxRcntWtarget16(int index, u32 value)
{
	pxAssert(index < 3);
	PSXCNT_LOG("IOP Counter[%d] writeTarget16 = %lx", index, value);
	psxCounters[index].target = value & 0xffff;

	psxRcntSync(index);

	// protect the target from an early arrival.
	// if the target is behind the current count, then set the target overflow
	// flag, so that the target won't be active until after the next overflow.

	if (psxCounters[index].target <= psxCounters[index].count)
		psxCounters[index].target |= IOPCNT_FUTURE_TARGET;

	_rcntSet(index);
}

void psxRcntWtarget32(int index, u32 value)
{
	pxAssert(index >= 3 && index < 6);
	PSXCNT_LOG("IOP Counter[%d] writeTarget32 = %lx mode %x", index, value, psxCounters[index].mode.modeval);

	psxCounters[index].target = value;

	psxRcntSync(index);
	// protect the target from an early arrival.
	// if the target is behind the current count, then set the target overflow
	// flag, so that the target won't be active until after the next overflow.

	if (psxCounters[index].target <= psxCounters[index].count)
		psxCounters[index].target |= IOPCNT_FUTURE_TARGET;

	_rcntSet(index);
}

u16 psxRcntRcount16(int index)
{
	psxRcntSync(index);
	const u32 retval = (u32)psxCounters[index].count;

	pxAssert(index < 3);

	PSXCNT_LOG("IOP Counter[%d] readCount16 = %lx", index, (u16)retval);

	return (u16)retval;
}

u32 psxRcntRcount32(int index)
{
	psxRcntSync(index);
	
	const u32 retval = (u32)psxCounters[index].count;

	pxAssert(index >= 3 && index < 6);

	PSXCNT_LOG("IOP Counter[%d] readCount32 = %lx", index, retval);

	return retval;
}

void psxRcntSetNewIntrMode(int index)
{
	psxCounters[index].mode.targetFlag = false;
	psxCounters[index].mode.overflowFlag = false;
	psxCounters[index].mode.intrEnable = true;

	//if (psxCounters[index].mode.repeatIntr != psxCounters[index].currentIrqMode.repeatInterrupt || psxCounters[index].mode.toggleIntr != psxCounters[index].currentIrqMode.toggleInterrupt)
	//	DevCon.Warning("Updating psxCounter[%d] mode old repeat %d new %d old toggle %d new %d", index, psxCounters[index].mode.repeatIntr, psxCounters[index].currentIrqMode.repeatInterrupt, psxCounters[index].mode.toggleIntr, psxCounters[index].currentIrqMode.toggleInterrupt);
	
	psxCounters[index].currentIrqMode.repeatInterrupt = psxCounters[index].mode.repeatIntr;
	psxCounters[index].currentIrqMode.toggleInterrupt = psxCounters[index].mode.toggleIntr;
}

bool SaveStateBase::psxRcntFreeze()
{
	if (!FreezeTag("iopCounters"))
		return false;

	Freeze(psxCounters);
	Freeze(psxNextDeltaCounter);
	Freeze(psxNextStartCounter);
	Freeze(hBlanking);
	Freeze(vBlanking);

	if (!IsOkay())
		return false;

	if (IsLoading())
		psxRcntUpdate();

	return true;
}
