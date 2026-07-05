// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — COP2 (VU0 Macro Mode) Instructions
//
// Full dispatch for all VU0 macro-mode opcodes:
//   QMFC2 / CFC2 / QMTC2 / CTC2, BC2x,
//   SPECIAL1 (VADDx..w, VMADD, VMUL, VMAX, VMINI, VIADD, ..., VCALLMS),
//   SPECIAL2 (VADDAx..w, VITOF, VFTOI, VMULA, VSUBA, VOPMSUB, VMOVE, ..., VRXOR).
//
// QMFC2 and QMTC2 are implemented natively using 128-bit Q-register transfers.
// All VU0 math ops dispatch directly to their per-op interpreter functions,
// eliminating the double-dispatch overhead of the old single REC_INTERP(COP2) stub.
//
// Reference: app/src/main/cpp/pcsx2/x86/microVU_Macro.inl

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"
#include "VUmicro.h"
#include "VU.h"
#include "iRecAnalysis.h" // EEINST, EEINST_COP2_SYNC_VU0, EEINST_COP2_FINISH_VU0, g_pCurInstInfo

using namespace R5900;

extern void recompileNextInstruction(bool delayslot, bool swapped_delay_slot);

// Defined in iR5900_arm64.cpp; consumes & clears s_nBlockCycles, returns the
// scaled cycle count to bump cpuRegs.cycle by. Used by the native COP2 sync
// emit (mirrors x86 mVUSyncVU0's scaleblockcycles_clear() injection point).
extern u32 scaleblockcycles_clear();

// Forward declarations for VU0 per-op emit functions. These are defined at
// GLOBAL scope in iVU0Upper_arm64.cpp / iVU0Lower_arm64.cpp, so the externs
// must live outside the R5900::Dynarec::OpcodeImpl namespace. The link errors
// we hit (undefined R5900::Dynarec::OpcodeImpl::recVU0_*) came from declaring
// these inside the namespace.
extern "C++" {
// Phase 1 (broadcast FMAC, toACC=false): 20 ops
extern void recVU0_ADDx();   extern void recVU0_ADDy();   extern void recVU0_ADDz();   extern void recVU0_ADDw();
extern void recVU0_SUBx();   extern void recVU0_SUBy();   extern void recVU0_SUBz();   extern void recVU0_SUBw();
extern void recVU0_MULx();   extern void recVU0_MULy();   extern void recVU0_MULz();   extern void recVU0_MULw();
extern void recVU0_MADDx();  extern void recVU0_MADDy();  extern void recVU0_MADDz();  extern void recVU0_MADDw();
extern void recVU0_MSUBx();  extern void recVU0_MSUBy();  extern void recVU0_MSUBz();  extern void recVU0_MSUBw();

// Phase 2 — Accumulator broadcast FMAC (toACC=true): 20 ops
extern void recVU0_ADDAx();  extern void recVU0_ADDAy();  extern void recVU0_ADDAz();  extern void recVU0_ADDAw();
extern void recVU0_SUBAx();  extern void recVU0_SUBAy();  extern void recVU0_SUBAz();  extern void recVU0_SUBAw();
extern void recVU0_MULAx();  extern void recVU0_MULAy();  extern void recVU0_MULAz();  extern void recVU0_MULAw();
extern void recVU0_MADDAx(); extern void recVU0_MADDAy(); extern void recVU0_MADDAz(); extern void recVU0_MADDAw();
extern void recVU0_MSUBAx(); extern void recVU0_MSUBAy(); extern void recVU0_MSUBAz(); extern void recVU0_MSUBAw();

// Phase 2 — Full-vector FMAC: 10 ops
extern void recVU0_ADD();    extern void recVU0_SUB();    extern void recVU0_MUL();
extern void recVU0_MADD();   extern void recVU0_MSUB();
extern void recVU0_ADDA();   extern void recVU0_SUBA();   extern void recVU0_MULA();
extern void recVU0_MADDA();  extern void recVU0_MSUBA();

// Phase 2 — Q-register FMAC: 10 ops
extern void recVU0_ADDq();   extern void recVU0_SUBq();   extern void recVU0_MULq();
extern void recVU0_MADDq();  extern void recVU0_MSUBq();
extern void recVU0_ADDAq();  extern void recVU0_SUBAq();  extern void recVU0_MULAq();
extern void recVU0_MADDAq(); extern void recVU0_MSUBAq();

// Phase 2 — I-register FMAC: 10 ops
extern void recVU0_ADDi();   extern void recVU0_SUBi();   extern void recVU0_MULi();
extern void recVU0_MADDi();  extern void recVU0_MSUBi();
extern void recVU0_ADDAi();  extern void recVU0_SUBAi();  extern void recVU0_MULAi();
extern void recVU0_MADDAi(); extern void recVU0_MSUBAi();

// Phase 2 — OPMULA / OPMSUB (FMAC with SYNCMSFLAGS): 2 ops
extern void recVU0_OPMULA(); extern void recVU0_OPMSUB();

// Phase 2 — MAX / MINI (no flag updates): 12 ops
extern void recVU0_MAXx();   extern void recVU0_MAXy();   extern void recVU0_MAXz();   extern void recVU0_MAXw();
extern void recVU0_MAXi();   extern void recVU0_MAX();
extern void recVU0_MINIx();  extern void recVU0_MINIy();  extern void recVU0_MINIz();  extern void recVU0_MINIw();
extern void recVU0_MINIi();  extern void recVU0_MINI();

// Phase 2 — ABS (no flag updates): 1 op
extern void recVU0_ABS();

// Phase 2 — ITOF / FTOI (no flag updates): 8 ops
extern void recVU0_ITOF0();  extern void recVU0_ITOF4();  extern void recVU0_ITOF12(); extern void recVU0_ITOF15();
extern void recVU0_FTOI0();  extern void recVU0_FTOI4();  extern void recVU0_FTOI12(); extern void recVU0_FTOI15();

// Phase 2 — CLIP (SYNCCLIPFLAG): 1 op
extern void recVU0_CLIP();

// Phase 2 — Integer ALU LOWER (no flag sync): 5 ops
extern void recVU0_IADD();   extern void recVU0_ISUB();   extern void recVU0_IADDI();
extern void recVU0_IAND();   extern void recVU0_IOR();

// Phase 2 — Register transfer LOWER (no flag sync): 4 ops
extern void recVU0_MOVE();   extern void recVU0_MR32();
extern void recVU0_MFIR();   extern void recVU0_MTIR();

// Phase 2 — Memory LOWER (no flag sync, C-helper calls): 6 ops
extern void recVU0_LQI();    extern void recVU0_SQI();
extern void recVU0_LQD();    extern void recVU0_SQD();
extern void recVU0_ILWR();   extern void recVU0_ISWR();

// Phase 2 — RNG LOWER (no flag sync, C-helper calls): 4 ops
extern void recVU0_RINIT();  extern void recVU0_RGET();
extern void recVU0_RNEXT();  extern void recVU0_RXOR();

// Phase 2 — FDIV LOWER (SYNCFDIV sync): 3 ops
extern void recVU0_DIV();    extern void recVU0_SQRT();   extern void recVU0_RSQRT();
}

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ============================================================================
//  INTERP_COP2 master switch + per-op ISTUB toggles
//
//  Set INTERP_COP2 (or INTERP_EE) in arm64Emitter.h to force all COP2 ops to
//  interpreter stubs for bisection.  Per-op ISTUBs let you toggle individual
//  ops when the master switch is off.
// ============================================================================

#if defined(INTERP_COP2) || defined(INTERP_EE)
#define ISTUB_QMFC2       1
#define ISTUB_CFC2        1
#define ISTUB_QMTC2       1
#define ISTUB_CTC2        1
#define ISTUB_BC2F        1
#define ISTUB_BC2T        1
#define ISTUB_BC2FL       1
#define ISTUB_BC2TL       1
#define ISTUB_VU0_MACRO   1
#else
// Native: QMFC2/QMTC2 (128-bit Q-register LDR/STR), CFC2/CTC2 (simple VI
// reads/writes inline; interpreter fallback retained for I-bit interlock,
// STATUS_FLAG writes, CMSAR1, and FBRST — each has microVU-specific side
// effects that don't fit a pure memory op), BC2x (Csel-based branch on
// VPU_STAT bit 8, mirroring recBC1*).
#define ISTUB_QMFC2  0
#define ISTUB_CFC2   0
#define ISTUB_QMTC2  0
#define ISTUB_CTC2   0
#define ISTUB_BC2F   0
#define ISTUB_BC2T   0
#define ISTUB_BC2FL  0
#define ISTUB_BC2TL  0
// ISTUB_VU0_MACRO: master default for Phase 1 VU0 macro-mode native port
// (ADD/SUB/MUL/MADD/MSUB × x/y/z/w via arm64 VU0 microrec per-op emit).
// Default=1 (interp) — reported missing-geometry regression. The per-op
// ISTUB_VU0_<op> flags below all inherit from this; flip individual ones
// to 0 to re-enable a specific op natively and bisect which one regresses.
#define ISTUB_VU0_MACRO  0
#endif

// Per-op VU0 macro-mode ISTUBs — spelled out explicitly in both branches of
// the master #if/#else. Flip any individual flag to peel one op off to interp
// or native without touching the rest. See feedback_istub_style.md.
//   1 = interpreter fallback (safe)
//   0 = arm64 native VU0 microrec per-op emit
#if ISTUB_VU0_MACRO
// Broadcast FMAC (toACC=false)
#define ISTUB_VU0_ADDx    1
#define ISTUB_VU0_ADDy    1
#define ISTUB_VU0_ADDz    1
#define ISTUB_VU0_ADDw    1
#define ISTUB_VU0_SUBx    1
#define ISTUB_VU0_SUBy    1
#define ISTUB_VU0_SUBz    1
#define ISTUB_VU0_SUBw    1
#define ISTUB_VU0_MULx    1
#define ISTUB_VU0_MULy    1
#define ISTUB_VU0_MULz    1
#define ISTUB_VU0_MULw    1
#define ISTUB_VU0_MADDx   1
#define ISTUB_VU0_MADDy   1
#define ISTUB_VU0_MADDz   1
#define ISTUB_VU0_MADDw   1
#define ISTUB_VU0_MSUBx   1
#define ISTUB_VU0_MSUBy   1
#define ISTUB_VU0_MSUBz   1
#define ISTUB_VU0_MSUBw   1
// Accumulator broadcast FMAC (toACC=true)
#define ISTUB_VU0_ADDAx   1
#define ISTUB_VU0_ADDAy   1
#define ISTUB_VU0_ADDAz   1
#define ISTUB_VU0_ADDAw   1
#define ISTUB_VU0_SUBAx   1
#define ISTUB_VU0_SUBAy   1
#define ISTUB_VU0_SUBAz   1
#define ISTUB_VU0_SUBAw   1
#define ISTUB_VU0_MULAx   1
#define ISTUB_VU0_MULAy   1
#define ISTUB_VU0_MULAz   1
#define ISTUB_VU0_MULAw   1
#define ISTUB_VU0_MADDAx  1
#define ISTUB_VU0_MADDAy  1
#define ISTUB_VU0_MADDAz  1
#define ISTUB_VU0_MADDAw  1
#define ISTUB_VU0_MSUBAx  1
#define ISTUB_VU0_MSUBAy  1
#define ISTUB_VU0_MSUBAz  1
#define ISTUB_VU0_MSUBAw  1
// Full-vector FMAC
#define ISTUB_VU0_ADD     1
#define ISTUB_VU0_SUB     1
#define ISTUB_VU0_MUL     1
#define ISTUB_VU0_MADD    1
#define ISTUB_VU0_MSUB    1
#define ISTUB_VU0_ADDA    1
#define ISTUB_VU0_SUBA    1
#define ISTUB_VU0_MULA    1
#define ISTUB_VU0_MADDA   1
#define ISTUB_VU0_MSUBA   1
// Q-register FMAC
#define ISTUB_VU0_ADDq    1
#define ISTUB_VU0_SUBq    1
#define ISTUB_VU0_MULq    1
#define ISTUB_VU0_MADDq   1
#define ISTUB_VU0_MSUBq   1
#define ISTUB_VU0_ADDAq   1
#define ISTUB_VU0_SUBAq   1
#define ISTUB_VU0_MULAq   1
#define ISTUB_VU0_MADDAq  1
#define ISTUB_VU0_MSUBAq  1
// I-register FMAC
#define ISTUB_VU0_ADDi    1
#define ISTUB_VU0_SUBi    1
#define ISTUB_VU0_MULi    1
#define ISTUB_VU0_MADDi   1
#define ISTUB_VU0_MSUBi   1
#define ISTUB_VU0_ADDAi   1
#define ISTUB_VU0_SUBAi   1
#define ISTUB_VU0_MULAi   1
#define ISTUB_VU0_MADDAi  1
#define ISTUB_VU0_MSUBAi  1
// OPMULA / OPMSUB
#define ISTUB_VU0_OPMULA  1
#define ISTUB_VU0_OPMSUB  1
// MAX / MINI (no flag sync)
#define ISTUB_VU0_MAXx    1
#define ISTUB_VU0_MAXy    1
#define ISTUB_VU0_MAXz    1
#define ISTUB_VU0_MAXw    1
#define ISTUB_VU0_MAXi    1
#define ISTUB_VU0_MAX     1
#define ISTUB_VU0_MINIx   1
#define ISTUB_VU0_MINIy   1
#define ISTUB_VU0_MINIz   1
#define ISTUB_VU0_MINIw   1
#define ISTUB_VU0_MINIi   1
#define ISTUB_VU0_MINI    1
// ABS (no flag sync)
#define ISTUB_VU0_ABS     1
// ITOF / FTOI (no flag sync)
#define ISTUB_VU0_ITOF0   1
#define ISTUB_VU0_ITOF4   1
#define ISTUB_VU0_ITOF12  1
#define ISTUB_VU0_ITOF15  1
#define ISTUB_VU0_FTOI0   1
#define ISTUB_VU0_FTOI4   1
#define ISTUB_VU0_FTOI12  1
#define ISTUB_VU0_FTOI15  1
// CLIPw (clipflag sync)
#define ISTUB_VU0_CLIPw   1
// Integer ALU LOWER
#define ISTUB_VU0_IADD    1
#define ISTUB_VU0_ISUB    1
#define ISTUB_VU0_IADDI   1
#define ISTUB_VU0_IAND    1
#define ISTUB_VU0_IOR     1
// Register transfer LOWER
#define ISTUB_VU0_MOVE    1
#define ISTUB_VU0_MR32    1
#define ISTUB_VU0_MFIR    1
#define ISTUB_VU0_MTIR    1
// Memory LOWER
#define ISTUB_VU0_LQI     1
#define ISTUB_VU0_SQI     1
#define ISTUB_VU0_LQD     1
#define ISTUB_VU0_SQD     1
#define ISTUB_VU0_ILWR    1
#define ISTUB_VU0_ISWR    1
// RNG LOWER
#define ISTUB_VU0_RINIT   1
#define ISTUB_VU0_RGET    1
#define ISTUB_VU0_RNEXT   1
#define ISTUB_VU0_RXOR    1
// FDIV LOWER (SYNCFDIV)
#define ISTUB_VU0_DIV     1
#define ISTUB_VU0_SQRT    1
#define ISTUB_VU0_RSQRT   1
#else
// Broadcast FMAC (toACC=false)
#define ISTUB_VU0_ADDx    0
#define ISTUB_VU0_ADDy    0
#define ISTUB_VU0_ADDz    0
#define ISTUB_VU0_ADDw    0
#define ISTUB_VU0_SUBx    0
#define ISTUB_VU0_SUBy    0
#define ISTUB_VU0_SUBz    0
#define ISTUB_VU0_SUBw    0
#define ISTUB_VU0_MULx    0
#define ISTUB_VU0_MULy    0
#define ISTUB_VU0_MULz    0
#define ISTUB_VU0_MULw    0
#define ISTUB_VU0_MADDx   0
#define ISTUB_VU0_MADDy   0
#define ISTUB_VU0_MADDz   0
#define ISTUB_VU0_MADDw   0
#define ISTUB_VU0_MSUBx   0
#define ISTUB_VU0_MSUBy   0
#define ISTUB_VU0_MSUBz   0
#define ISTUB_VU0_MSUBw   0
// Accumulator broadcast FMAC (toACC=true)
#define ISTUB_VU0_ADDAx   0
#define ISTUB_VU0_ADDAy   0
#define ISTUB_VU0_ADDAz   0
#define ISTUB_VU0_ADDAw   0
#define ISTUB_VU0_SUBAx   0
#define ISTUB_VU0_SUBAy   0
#define ISTUB_VU0_SUBAz   0
#define ISTUB_VU0_SUBAw   0
#define ISTUB_VU0_MULAx   0
#define ISTUB_VU0_MULAy   0
#define ISTUB_VU0_MULAz   0
#define ISTUB_VU0_MULAw   0
#define ISTUB_VU0_MADDAx  0
#define ISTUB_VU0_MADDAy  0
#define ISTUB_VU0_MADDAz  0
#define ISTUB_VU0_MADDAw  0
#define ISTUB_VU0_MSUBAx  0
#define ISTUB_VU0_MSUBAy  0
#define ISTUB_VU0_MSUBAz  0
#define ISTUB_VU0_MSUBAw  0
// Full-vector FMAC
#define ISTUB_VU0_ADD     0
#define ISTUB_VU0_SUB     0
#define ISTUB_VU0_MUL     0
#define ISTUB_VU0_MADD    0
#define ISTUB_VU0_MSUB    0
#define ISTUB_VU0_ADDA    0
#define ISTUB_VU0_SUBA    0
#define ISTUB_VU0_MULA    0
#define ISTUB_VU0_MADDA   0
#define ISTUB_VU0_MSUBA   0
// Q-register FMAC
#define ISTUB_VU0_ADDq    0
#define ISTUB_VU0_SUBq    0
#define ISTUB_VU0_MULq    0
#define ISTUB_VU0_MADDq   0
#define ISTUB_VU0_MSUBq   0
#define ISTUB_VU0_ADDAq   0
#define ISTUB_VU0_SUBAq   0
#define ISTUB_VU0_MULAq   0
#define ISTUB_VU0_MADDAq  0
#define ISTUB_VU0_MSUBAq  0
// I-register FMAC
#define ISTUB_VU0_ADDi    0
#define ISTUB_VU0_SUBi    0
#define ISTUB_VU0_MULi    0
#define ISTUB_VU0_MADDi   0
#define ISTUB_VU0_MSUBi   0
#define ISTUB_VU0_ADDAi   0
#define ISTUB_VU0_SUBAi   0
#define ISTUB_VU0_MULAi   0
#define ISTUB_VU0_MADDAi  0
#define ISTUB_VU0_MSUBAi  0
// OPMULA / OPMSUB
#define ISTUB_VU0_OPMULA  0
#define ISTUB_VU0_OPMSUB  0
// MAX / MINI (no flag sync)
#define ISTUB_VU0_MAXx    0
#define ISTUB_VU0_MAXy    0
#define ISTUB_VU0_MAXz    0
#define ISTUB_VU0_MAXw    0
#define ISTUB_VU0_MAXi    0
#define ISTUB_VU0_MAX     0
#define ISTUB_VU0_MINIx   0
#define ISTUB_VU0_MINIy   0
#define ISTUB_VU0_MINIz   0
#define ISTUB_VU0_MINIw   0
#define ISTUB_VU0_MINIi   0
#define ISTUB_VU0_MINI    0
// ABS (no flag sync)
#define ISTUB_VU0_ABS     0
// ITOF / FTOI (no flag sync)
#define ISTUB_VU0_ITOF0   0
#define ISTUB_VU0_ITOF4   0
#define ISTUB_VU0_ITOF12  0
#define ISTUB_VU0_ITOF15  0
#define ISTUB_VU0_FTOI0   0
#define ISTUB_VU0_FTOI4   0
#define ISTUB_VU0_FTOI12  0
#define ISTUB_VU0_FTOI15  0
// CLIPw (clipflag sync)
#define ISTUB_VU0_CLIPw   0
// Integer ALU LOWER
#define ISTUB_VU0_IADD    0
#define ISTUB_VU0_ISUB    0
#define ISTUB_VU0_IADDI   0
#define ISTUB_VU0_IAND    0
#define ISTUB_VU0_IOR     0
// Register transfer LOWER
#define ISTUB_VU0_MOVE    0
#define ISTUB_VU0_MR32    0
#define ISTUB_VU0_MFIR    0
#define ISTUB_VU0_MTIR    0
// Memory LOWER
#define ISTUB_VU0_LQI     0
#define ISTUB_VU0_SQI     0
#define ISTUB_VU0_LQD     0
#define ISTUB_VU0_SQD     0
#define ISTUB_VU0_ILWR    0
#define ISTUB_VU0_ISWR    0
// RNG LOWER
#define ISTUB_VU0_RINIT   0
#define ISTUB_VU0_RGET    0
#define ISTUB_VU0_RNEXT   0
#define ISTUB_VU0_RXOR    0
// FDIV LOWER (SYNCFDIV)
#define ISTUB_VU0_DIV     0
#define ISTUB_VU0_SQRT    0
#define ISTUB_VU0_RSQRT   0
#endif

// ============================================================================
//  Helper macro — emit interpreter call for a global-scope VU0 interpreter
//  function.  The functions QMFC2/VADD/VMUL/etc. live in global scope in
//  R5900OpcodeTables.h, not in a sub-namespace.
// ============================================================================
#define REC_COP2_INTERP(name) \
    void recV##name() { armCallInterpreter(::V##name); }

// ============================================================================
//  VU0 macro-mode native codegen — reuses the per-op emit functions from the
//  arm64 VU0 microrecompiler (iVU0Upper_arm64.cpp / iVU0Lower_arm64.cpp).
//
//  The per-op functions (recVU0_ADDx, recVU0_MUL, etc.) emit NEON code that
//  references VU0 state through x23 (VU0_BASE_REG in the VU0 rec). In the EE
//  JIT, x23 is RFASTMEMBASE — a different global. We resolve the conflict by
//  saving RFASTMEMBASE to the stack, loading &VU0 into x23 for the duration
//  of the emitted op, and restoring RFASTMEMBASE afterwards. AAPCS callee-
//  saved discipline means any C helper the VU0 op calls (e.g.
//  vu0_fmac_writeback) preserves x23, so &VU0 survives through to the restore.
//
//  VU0.code (a C++ global, not a runtime memory location for our purposes
//  here) is set to cpuRegs.code at emit time so the per-op function's
//  compile-time bit extraction (`(VU0.code >> 6) & 0x1F` etc.) decodes the
//  COP2 instruction's register fields. COP2 SPECIAL1/SPECIAL2 use the same
//  fs/ft/fd/dest bit layout as VU native ops, so this just works.
//
//  Scope (Phase 1): broadcast FMAC ops (ADD/SUB/MUL/MADD/MSUB × xyzw × {Fd,
//  ACC}). Other op families follow the same pattern; extending to DIV/SQRT/
//  RSQRT, LQI/SQI/LQD/SQD, ITOF/FTOI, MOVE/MR32, RNG, and the full-vector
//  variants is Phase 2+.
// ============================================================================

// (Forward declarations for recVU0_* hoisted to global scope above the namespace.)

// Emit: save RFASTMEMBASE (x23), load &VU0 into it, then store the baked
// cpuRegs.code immediate into VU0.code. The FMAC_* macros decode VU0.code
// at compile time (baked as immediates into the emit), but C helpers like
// vu0_CLIP / vu0_OPMULA / vu0_OPMSUB read VU->code at RUNTIME — so without
// the runtime store, those would decode stale bits from the last compile.
// Must be paired with emitVU0MacroExit inside the same emitted sequence.
// [[maybe_unused]] because all per-op ISTUBs defaulting to 1 means no caller
// is instantiated — keep the symbol around so flipping any per-op flag picks
// it up without re-enabling a preprocessor gate.
//
// Reverted to minimal entry — no Q-load / status-denorm. Earlier work
// added mode-bit-driven entry pre-load (mirroring x86 setupMacroOp) to
// chase MGS2 collision physics, but that bug turned out to be the VU0
// linkEntry carryover-branch issue (see armsx2_vu0_carryover_branch_fix
// memory). The added per-FMAC-op emit (~4-7 ARM insns per call site)
// pushed EE blocks over the per-block code-buffer budget, tripping a
// vixl `managed_` assert in CodeBuffer::Grow. Re-enable only if we find
// a separate bug whose actual cause is missing entry pre-load.
[[maybe_unused]] static void emitVU0MacroEnter(int /*mode*/ = 0)
{
    armAsm->Str(RFASTMEMBASE, a64::MemOperand(a64::sp, -16, a64::PreIndex));
    armMoveAddressToReg(RFASTMEMBASE, &VU0);
    armAsm->Mov(RWSCRATCH, cpuRegs.code);
    armAsm->Str(RWSCRATCH, a64::MemOperand(RFASTMEMBASE,
        static_cast<int64_t>(offsetof(VURegs, code))));
}

// Emit: restore RFASTMEMBASE from the stack.
[[maybe_unused]] static void emitVU0MacroExit()
{
    armAsm->Ldr(RFASTMEMBASE, a64::MemOperand(a64::sp, 16, a64::PostIndex));
}

// Sync VU0.macflag / VU0.statusflag into the VI[REG_MAC_FLAG] / VI[REG_STATUS_FLAG]
// mirrors. Matches the interpreter's SYNCMSFLAGS (static in VUops.cpp) which runs
// at the end of every COP2 FMAC op — the VI mirrors are what CFC2 reads, and our
// vu0_fmac_writeback C helper only updates macflag/statusflag. Without this,
// `native MADDw; CFC2 mac_flag` reads stale bits from the last interpreter op,
// which broke vertex-transform output (missing geometry on the first test game).
//
// Bit layout of VI[REG_STATUS_FLAG]:
//   bits  3:0 — current flags (Z,S,U,O) from statusflag
//   bits  5:4 — cleared
//   bits  9:6 — sticky flags, old | new (statusflag & 0xF) << 6
//   bits 11:10 — D/I sticky, preserved from old VI_STATUS
[[maybe_unused]] static void vu0_macro_sync_flags_helper()
{
    VU0.VI[REG_STATUS_FLAG].UL = (VU0.VI[REG_STATUS_FLAG].UL & 0xFC0) |
                                  (VU0.statusflag & 0xF) |
                                  ((VU0.statusflag & 0xF) << 6);
    VU0.VI[REG_MAC_FLAG].UL = VU0.macflag;
}

// Emit: BL to the sync helper. Called after every native FMAC macro-mode op.
// The C helper reads VU0 via its C++ global — no register argument needed; the
// callee-saved x19-x28 (including x23=&VU0) are preserved across the call.
[[maybe_unused]] static void emitVU0MacroSyncFlags()
{
    armEmitCall(reinterpret_cast<const void*>(vu0_macro_sync_flags_helper));
}

// Sync VU0.clipflag into VI[REG_CLIP_FLAG]. Matches SYNCCLIPFLAG in VUops.cpp
// (called by VCLIPw after _vuCLIP). Native vu0_CLIP updates VU0.clipflag;
// the VI mirror is what CFC2 and the FCAND/FCEQ/FCOR/FCGET lower ops read.
[[maybe_unused]] static void vu0_macro_sync_clipflag_helper()
{
    VU0.VI[REG_CLIP_FLAG].UL = VU0.clipflag;
}

[[maybe_unused]] static void emitVU0MacroSyncClipFlag()
{
    armEmitCall(reinterpret_cast<const void*>(vu0_macro_sync_clipflag_helper));
}

// Sync the FDIV pipeline results (VU0.q + VU0.statusflag bits [5:4]) into
// their VI mirrors. Matches SYNCFDIV in VUops.cpp (called by VDIV/VSQRT/VRSQRT
// after _vuDIV/_vuSQRT/_vuRSQRT).
//
// VI[REG_Q].UL = VU0.q.UL   (the division result)
// VI[REG_STATUS_FLAG] bits [5:4] = current D/I (statusflag & 0x30)
// VI[REG_STATUS_FLAG] bits [11:10] sticky D/I = old | new (statusflag & 0x30) << 6
[[maybe_unused]] static void vu0_macro_sync_fdiv_helper()
{
    VU0.VI[REG_Q].UL = VU0.q.UL;
    VU0.VI[REG_STATUS_FLAG].UL = (VU0.VI[REG_STATUS_FLAG].UL & 0x3CF) |
                                  (VU0.statusflag & 0x30) |
                                  ((VU0.statusflag & 0x30) << 6);
}

[[maybe_unused]] static void emitVU0MacroSyncFdiv()
{
    armEmitCall(reinterpret_cast<const void*>(vu0_macro_sync_fdiv_helper));
}

// ============================================================================
//  Native VU0 sync emit — port of x86 mVUSyncVU0 / mVUFinishVU0
// ============================================================================
// Used in the analysis-flag-only sync path of recCFC2/recCTC2/recQMFC2/recQMTC2
// (NO i-bit). The i-bit case still falls back to the interpreter — the full
// COP2_Interlock heavy path includes _vu0WaitMicro / s_nBlockInterlocked
// bookkeeping that's tied to the recompile-time interlock state.
//
// Why native vs the previous armCallInterpreter fallback:
//   The interp's vu0Sync (VU0.cpp:88) calls _vu0run(0, 0, 1) which loops
//   `do { CpuVU0->Execute(runCycles); } while (...)` — running multiple VU0
//   blocks until the EE-VU0 cycle gap closes. x86 mVUSyncVU0 runs AT MOST
//   ONE block via BaseVUmicroCPU::ExecuteBlockJIT, only when gap >= 4. arm64
//   was over-syncing on every COP2 read/write, causing VF/macflag state to
//   advance ahead of what the COP2 reader expects (suspected source of MGS2
//   collision corruption — bullets through invisible walls, stair geometry).

// Mirror of x86 mVUSyncVU0 (microVU_Macro.inl:365-385):
//   1) Bump cpuRegs.cycle by scaleblockcycles_clear() (compile-time scale).
//   2) If VPU_STAT bit 0 clear (VU0 idle), skip.
//   3) If gap = cpuRegs.cycle - VU0.cycle (- nextBlockCycles if VUSyncHack)
//      >= 4: invoke ExecuteBlockJIT(CpuVU0, s_nBlockInterlocked) for ONE
//      block. Else skip.
// Pre: caller has done iFlushCall-equivalent (FlushPC/FlushCode/FlushConst).
// Post: RCYCLE has been bumped to match cpuRegs.cycle, then reloaded from
// memory (idempotent if no C call ran, refreshed if ExecuteBlockJIT ran).
[[maybe_unused]] static void emitVU0SyncVU0()
{
    armFlushPC();
    armFlushCode();
    armFlushConstRegs();

    // x86 stores the scaled bump into cpuRegs.cycle directly. arm64 keeps the
    // JIT-current cycle in RCYCLE; bump it first, then flush so the in-memory
    // cpuRegs.cycle that ExecuteBlockJIT reads is up-to-date.
    const u32 scale = scaleblockcycles_clear();
    if (scale != 0)
        armAsm->Add(RCYCLE, RCYCLE, scale);
    armEmitFlushCycleBeforeCall();

    a64::Label skipvuidle;
    armAsm->Mov(RSCRATCHGPR, reinterpret_cast<uintptr_t>(&VU0.VI[REG_VPU_STAT].UL));
    armAsm->Ldr(RWSCRATCH, a64::MemOperand(RSCRATCHGPR));
    armAsm->Tbz(RWSCRATCH, 0, &skipvuidle);

    // gap = cpuRegs.cycle - VU0.cycle
    a64::Label skip;
    armAsm->Ldr(a64::x9, a64::MemOperand(RCPUSTATE, CYCLE_OFFSET));
    armAsm->Mov(RSCRATCHGPR, reinterpret_cast<uintptr_t>(&VU0.cycle));
    armAsm->Ldr(a64::x10, a64::MemOperand(RSCRATCHGPR));
    armAsm->Sub(a64::x9, a64::x9, a64::x10);

    // VUSyncHack / FullVU0SyncHack tighten the gap by VU0.nextBlockCycles
    // (Ratchet flickering polygons, per microVU_Macro.inl:344-348 comment).
    if (EmuConfig.Gamefixes.VUSyncHack || EmuConfig.Gamefixes.FullVU0SyncHack)
    {
        armAsm->Mov(RSCRATCHGPR, reinterpret_cast<uintptr_t>(&VU0.nextBlockCycles));
        armAsm->Ldr(a64::x10, a64::MemOperand(RSCRATCHGPR));
        armAsm->Sub(a64::x9, a64::x9, a64::x10);
    }

    armAsm->Cmp(a64::x9, 4);
    armAsm->B(&skip, a64::lt);

    // ExecuteBlockJIT(CpuVU0, s_nBlockInterlocked)
    armAsm->Mov(a64::x0, reinterpret_cast<uintptr_t>(CpuVU0));
    armAsm->Mov(a64::w1, static_cast<int>(s_nBlockInterlocked));
    armEmitCall(reinterpret_cast<const void*>(&BaseVUmicroCPU::ExecuteBlockJIT));

    armAsm->Bind(&skip);
    armAsm->Bind(&skipvuidle);

    armEmitReloadCycleAfterCall();

    // Conservatively invalidate EE const-prop state — the C call (when taken)
    // may have changed cpuRegs state. Matches armCallInterpreter's post-call
    // bookkeeping (iR5900_arm64.cpp:920-921).
    g_cpuHasConstReg = 1;
    g_cpuFlushedConstReg = 1;
}

// Mirror of x86 mVUFinishVU0 (microVU_Macro.inl:387-394):
//   if VPU_STAT bit 0 set: call _vu0FinishMicro (runs VU0 until E-bit).
// Used when the analysis flagged EEINST_COP2_FINISH_VU0 (deeper sync than
// SYNC — needs VU0 to fully retire before the COP2 reader fires).
//
// Currently unused: the FINISH path falls back to the interpreter (see
// recQMFC2 / recQMTC2 / recCFC2 / recCTC2). Calling _vu0FinishMicro from
// the EE thread races with the GUI thread's pause-time vu1Thread.WaitVU()
// — the deep VU0 micro execution can hit a VU0 LQ/SQ on a VU1-aliased
// address (vu0MTVUSyncVU1Reg in iVU0Lower_arm64.cpp:281), which calls
// vu1Thread.WaitVU() concurrently and trips the WorkSema single-waiter
// assert (Semaphore.cpp:105). x86 has the same theoretical race; arm64
// hits it more reliably during touch-driven pauses on Android. Re-enable
// once the race is fixed at the source (e.g. proper EE-stop-then-drain
// ordering in VMManager::SetState).
[[maybe_unused]] static void emitVU0FinishVU0()
{
    armFlushPC();
    armFlushCode();
    armFlushConstRegs();
    armEmitFlushCycleBeforeCall();

    a64::Label skipvuidle;
    armAsm->Mov(RSCRATCHGPR, reinterpret_cast<uintptr_t>(&VU0.VI[REG_VPU_STAT].UL));
    armAsm->Ldr(RWSCRATCH, a64::MemOperand(RSCRATCHGPR));
    armAsm->Tbz(RWSCRATCH, 0, &skipvuidle);

    armEmitCall(reinterpret_cast<const void*>(&_vu0FinishMicro));

    armAsm->Bind(&skipvuidle);

    armEmitReloadCycleAfterCall();

    g_cpuHasConstReg = 1;
    g_cpuFlushedConstReg = 1;
}

// Dispatch a COP2 op through the VU0 microrecompiler's per-op emit function.
// At EMIT time: sets VU0.code = cpuRegs.code so the per-op function decodes
// the right fs/ft/fd/dest fields.
// At RUNTIME: the emitted code saves x23, loads &VU0, runs the VU0 NEON
// sequence, calls vu0_macro_sync_flags_helper to mirror macflag/statusflag
// into VI[REG_MAC_FLAG]/VI[REG_STATUS_FLAG] (what CFC2 reads), restores x23.
//
// Branch is resolved at compile time via `if constexpr` reading the per-op
// ISTUB_VU0_<name> flag. The dead branch is DCE'd — no runtime cost — and
// each of the 20 ops can be bisected independently.
//
// Only safe for ops that update VU0.macflag/statusflag (FMAC add/sub/mul/madd/
// msub families × {xyzw,ACC,Q,I,FULL} + OPMULA/OPMSUB). MAX/MINI/ABS/ITOF/FTOI/
// MOVE/MR32 don't touch those flags — use REC_COP2_MVU0_NOFLAGS. CLIP updates
// clipflag instead — use REC_COP2_MVU0_CLIP. DIV/SQRT/RSQRT update the FDIV
// status bits — not yet ported (stay on interp).
// FMAC arith (writes mac/status). Mode 0x10 mirrors x86 mode 0x110 mask
// for status — denormalize VI[REG_STATUS_FLAG] into VU0.statusflag at entry
// so the post-op flag-mirror sync composes correct sticky bits.
#define REC_COP2_MVU0(cop2_name, vu_rec_fn) \
    void recV##cop2_name() { \
        if constexpr (ISTUB_VU0_##cop2_name) { \
            armCallInterpreter(::V##cop2_name); \
        } else { \
            VU0.code = cpuRegs.code; \
            emitVU0MacroEnter(0x10); \
            vu_rec_fn(); \
            emitVU0MacroSyncFlags(); \
            emitVU0MacroExit(); \
        } \
    }

// FMAC arith *q variants (ADDq/SUBq/MULq/MADDq/MSUBq + A variants). Mode
// 0x11 = Q-read + status-write — mirrors x86 mode 0x111. Pre-loads
// VU0.q.UL ← VI[REG_Q].UL so the per-op emit reads the EE-canonical Q
// (the FDIV result that any prior CTC2/VDIV/VSQRT/VRSQRT placed in the VI
// mirror), not whatever the local FDIV pipe slot last held.
#define REC_COP2_MVU0_Q(cop2_name, vu_rec_fn) \
    void recV##cop2_name() { \
        if constexpr (ISTUB_VU0_##cop2_name) { \
            armCallInterpreter(::V##cop2_name); \
        } else { \
            VU0.code = cpuRegs.code; \
            emitVU0MacroEnter(0x11); \
            vu_rec_fn(); \
            emitVU0MacroSyncFlags(); \
            emitVU0MacroExit(); \
        } \
    }

// Same as REC_COP2_MVU0 but skips the MAC/STATUS flag sync. For MAX/MINI/ABS/
// ITOF/FTOI/MOVE/MR32 — the interpreter wrappers for these ops don't call
// SYNCMSFLAGS, so neither should we. Mode 0 = no entry pre-load (matches
// x86 mode 0x000).
#define REC_COP2_MVU0_NOFLAGS(cop2_name, vu_rec_fn) \
    void recV##cop2_name() { \
        if constexpr (ISTUB_VU0_##cop2_name) { \
            armCallInterpreter(::V##cop2_name); \
        } else { \
            VU0.code = cpuRegs.code; \
            emitVU0MacroEnter(0); \
            vu_rec_fn(); \
            emitVU0MacroExit(); \
        } \
    }

// Same as REC_COP2_MVU0 but syncs clipflag instead of macflag/statusflag.
// CLIPw's interpreter wrapper calls SYNCCLIPFLAG; the lower-op readers
// (FCAND/FCEQ/FCOR/FCGET) read VI[REG_CLIP_FLAG].UL. Mode 0x008 in x86
// touches no entry state.
#define REC_COP2_MVU0_CLIP(cop2_name, vu_rec_fn) \
    void recV##cop2_name() { \
        if constexpr (ISTUB_VU0_##cop2_name) { \
            armCallInterpreter(::V##cop2_name); \
        } else { \
            VU0.code = cpuRegs.code; \
            emitVU0MacroEnter(0); \
            vu_rec_fn(); \
            emitVU0MacroSyncClipFlag(); \
            emitVU0MacroExit(); \
        } \
    }

// For DIV/SQRT/RSQRT: writes Q + writes FDIV status bits (x86 mode 0x112).
// We don't need a Q-PRE-load (mode bit 0x01) because FDIV ops produce Q;
// mode 0x10 still applies for status denormalization. emitVU0MacroSyncFdiv
// at exit copies VU0.q → VI[REG_Q] and merges D/I status bits [5:4]/[11:10].
#define REC_COP2_MVU0_FDIV(cop2_name, vu_rec_fn) \
    void recV##cop2_name() { \
        if constexpr (ISTUB_VU0_##cop2_name) { \
            armCallInterpreter(::V##cop2_name); \
        } else { \
            VU0.code = cpuRegs.code; \
            emitVU0MacroEnter(0x10); \
            vu_rec_fn(); \
            emitVU0MacroSyncFdiv(); \
            emitVU0MacroExit(); \
        } \
    }

// ============================================================================
//  QMFC2 — Move 128-bit VF[rd] → GPR[rt]
// ============================================================================
void recQMFC2()
{
#if ISTUB_QMFC2
    armCallInterpreter(::QMFC2);
#else
    if (!_Rt_)
        return;
    // I-bit or analysis-flag SYNC/FINISH → fall back to interp. Native sync
    // emit was tried (port of x86 mVUSyncVU0) but added per-call-site emit
    // bloat that pushed EE blocks over the per-block code-buffer budget,
    // tripping a vixl `managed_` assert in CodeBuffer::Grow. Reverted in
    // favor of the simpler full-interp fallback.
    if ((cpuRegs.code & 1) ||
        (g_pCurInstInfo && (g_pCurInstInfo->info & (EEINST_COP2_SYNC_VU0 | EEINST_COP2_FINISH_VU0))))
    {
        armCallInterpreter(::QMFC2);
        return;
    }
    // Native 128-bit copy: VU0.VF[_Rd_] → cpuRegs.GPR[_Rt_]
    armAsm->Mov(RSCRATCHGPR, reinterpret_cast<uintptr_t>(&VU0.VF[_Rd_]));
    armAsm->Ldr(a64::q0, a64::MemOperand(RSCRATCHGPR));
    armAsm->Str(a64::q0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
    GPR_DEL_CONST(_Rt_);
#endif
}

// ============================================================================
//  QMTC2 — Move 128-bit GPR[rt] → VF[rd]
// ============================================================================
void recQMTC2()
{
#if ISTUB_QMTC2
    armCallInterpreter(::QMTC2);
#else
    if (!_Rd_)
        return;
    // I-bit or analysis-flag SYNC/FINISH → interp fallback (see recQMFC2
    // comment for rationale on the native-sync revert).
    if ((cpuRegs.code & 1) ||
        (g_pCurInstInfo && (g_pCurInstInfo->info & (EEINST_COP2_SYNC_VU0 | EEINST_COP2_FINISH_VU0))))
    {
        armCallInterpreter(::QMTC2);
        return;
    }
    // Native 128-bit copy: cpuRegs.GPR[_Rt_] → VU0.VF[_Rd_]
    // Commit any pending const-prop value before the direct 128-bit LDR.
    armFlushConstReg(_Rt_);
    armAsm->Ldr(a64::q0, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
    armAsm->Mov(RSCRATCHGPR, reinterpret_cast<uintptr_t>(&VU0.VF[_Rd_]));
    armAsm->Str(a64::q0, a64::MemOperand(RSCRATCHGPR));
#endif
}

// ============================================================================
//  CFC2 — Move from COP2 control register
//
//  Ports the simple read paths of x86 recCFC2 (microVU_Macro.inl:404-458).
//  Falls back to the interpreter for cases that need microVU interlock /
//  sync / finish handling (I-bit set, or EEINST_COP2_SYNC_VU0 /
//  EEINST_COP2_FINISH_VU0 flagged by the analysis pass).
//
//  Register-case matrix:
//    _Rd_ == 0              → GPR[_Rt_] = 0 (vi00 always reads zero)
//    _Rd_ < 16              → GPR[_Rt_] = u16 zero-extended to 64 bits
//    _Rd_ == REG_R (20)     → GPR[_Rt_] = sign_ext(VI[REG_R] & 0x7FFFFF)
//    _Rd_ >= 16 (other)     → GPR[_Rt_] = sign_ext(VI[_Rd_])
// ============================================================================
#if ISTUB_CFC2
void recCFC2() { armCallInterpreter(::CFC2); }
#else
void recCFC2()
{
    // I-bit or analysis-flag SYNC/FINISH → interp fallback (see recQMFC2
    // comment for rationale on the native-sync revert).
    if ((cpuRegs.code & 1) ||
        (g_pCurInstInfo && (g_pCurInstInfo->info & (EEINST_COP2_SYNC_VU0 | EEINST_COP2_FINISH_VU0))))
    {
        armCallInterpreter(::CFC2);
        return;
    }

    if (!_Rt_)
        return;

    armDelConstReg(_Rt_);

    // vi00 always reads as 0
    if (_Rd_ == 0)
    {
        armAsm->Str(a64::xzr, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
        return;
    }

    // Compute &VU0.VI[_Rd_].UL once
    armAsm->Mov(RSCRATCHGPR2, reinterpret_cast<uintptr_t>(&VU0.VI[_Rd_].UL));

    if (_Rd_ < 16)
    {
        // 16-bit VI register → zero-extend to 64 bits.
        // Ldrh into W-view implicitly zeroes the upper 32 bits of the X-view,
        // so a subsequent 64-bit Str produces the correct zero-extended GPR.
        armAsm->Ldrh(RWSCRATCH, a64::MemOperand(RSCRATCHGPR2));
        armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
    }
    else
    {
        // 32-bit VI register → sign-extend to 64 bits.
        armAsm->Ldr(RWSCRATCH, a64::MemOperand(RSCRATCHGPR2));
        if (_Rd_ == REG_R)
            armAsm->And(RWSCRATCH, RWSCRATCH, 0x7FFFFF);
        armAsm->Sxtw(RSCRATCHGPR, RWSCRATCH);
        armAsm->Str(RSCRATCHGPR, a64::MemOperand(RCPUSTATE, GPR_OFFSET(_Rt_)));
    }
}
#endif

// ============================================================================
//  CTC2 — Move to COP2 control register
//
//  Ports the simple write paths of x86 recCTC2 (microVU_Macro.inl:460-598).
//  Falls back to the interpreter for:
//    - I-bit / sync / finish interlock
//    - REG_STATUS_FLAG (sticky-flag maintenance + micro_statusflags broadcast)
//    - REG_CMSAR1     (triggers vu1Finish + vu1ExecMicro)
//    - REG_FBRST      (conditional vu0ResetRegs / vu1ResetRegs)
//
//  Native handles the common writes:
//    _Rd_ == REG_R         → VI[REG_R] = (GPR & 0x7FFFFF) | 0x3F800000
//    _Rd_ == MAC_FLAG/TPC/VPU_STAT → read-only, no-op
//    _Rd_ == 0             → ignore
//    _Rd_ < 16             → 16-bit store
//    _Rd_ >= 16 (other)    → 32-bit store
// ============================================================================
#if ISTUB_CTC2
void recCTC2() { armCallInterpreter(::CTC2); }
#else
void recCTC2()
{
    // I-bit or analysis-flag SYNC/FINISH → interp fallback.
    if ((cpuRegs.code & 1) ||
        (g_pCurInstInfo && (g_pCurInstInfo->info & (EEINST_COP2_SYNC_VU0 | EEINST_COP2_FINISH_VU0))))
    {
        armCallInterpreter(::CTC2);
        return;
    }

    if (!_Rd_)
        return;

    // Side-effect-heavy writes: interpreter owns them.
    if (_Rd_ == REG_STATUS_FLAG || _Rd_ == REG_CMSAR1 || _Rd_ == REG_FBRST)
    {
        armCallInterpreter(::CTC2);
        return;
    }

    // Read-only registers: the write is a no-op.
    if (_Rd_ == REG_MAC_FLAG || _Rd_ == REG_TPC || _Rd_ == REG_VPU_STAT)
        return;

    if (_Rd_ == REG_R)
    {
        // VI[REG_R] = (GPR[_Rt_] & 0x7FFFFF) | 0x3F800000
        armLoadGPR32(RWSCRATCH, _Rt_);
        armAsm->And(RWSCRATCH, RWSCRATCH, 0x7FFFFF);
        armAsm->Orr(RWSCRATCH, RWSCRATCH, 0x3F800000);
        armAsm->Mov(RSCRATCHGPR2, reinterpret_cast<uintptr_t>(&VU0.VI[REG_R].UL));
        armAsm->Str(RWSCRATCH, a64::MemOperand(RSCRATCHGPR2));
        return;
    }

    // Regular VI write: low 16 bits for VI[0..15], full 32 bits for VI[≥16].
    armLoadGPR32(RWSCRATCH, _Rt_);
    armAsm->Mov(RSCRATCHGPR2, reinterpret_cast<uintptr_t>(&VU0.VI[_Rd_].UL));

    if (_Rd_ < 16)
        armAsm->Strh(RWSCRATCH, a64::MemOperand(RSCRATCHGPR2));
    else
        armAsm->Str(RWSCRATCH, a64::MemOperand(RSCRATCHGPR2));
}
#endif

// ============================================================================
//  BC2x — branch on COP2 condition
//
//  The BC2 condition tests bit 8 (VBS0) of VPU_STAT — "is VU0 currently
//  running?". BC2T branches when the bit is set, BC2F when clear. Mirrors the
//  x86 pattern (microVU_Macro.inl:302-316): xTEST VPU_STAT & 0x100 + recDoBranchImm.
//
//  Arm64 uses the same Csel-based delay-slot approach as recBC1F/T and recBC0F/T
//  in the neighboring files:
//    - Non-likely: evaluate condition → store 0/1 in RDELAYSLOTGPR → compile
//      delay slot unconditionally → Csel target vs fallthrough.
//    - Likely: evaluate condition, Tbz/Tbnz to skip delay slot when not taken.
// ============================================================================

// Helper: non-likely BC2 branch.
static void recBC2_helper(bool branchIfSet)
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	// Load VPU_STAT (u32 at &VU0.VI[REG_VPU_STAT].UL) and test bit 8.
	armAsm->Mov(RSCRATCHGPR, reinterpret_cast<uintptr_t>(&VU0.VI[REG_VPU_STAT].UL));
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RSCRATCHGPR));
	armAsm->Tst(RWSCRATCH, 0x100);
	// Cset: 1 = taken. BC2F taken when bit clear (eq), BC2T when set (ne).
	armAsm->Cset(RDELAYSLOTGPR, branchIfSet ? a64::ne : a64::eq);

	armFlushConstRegs();
	recompileNextInstruction(true, false);
	armFlushConstRegs();

	armAsm->Mov(RWSCRATCH, branchTarget);
	armAsm->Mov(RWSCRATCH2, fallthrough);
	armAsm->Cmp(RDELAYSLOTGPR, 0);
	armAsm->Csel(RWSCRATCH, RWSCRATCH, RWSCRATCH2, a64::ne);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));

	g_branch = 1;
	g_cpuFlushedPC = true;
}

// Helper: likely BC2 branch (skip delay slot if not taken).
static void recBC2_Likely_helper(bool branchIfSet)
{
	u32 branchTarget = ((s32)_Imm_ * 4) + pc;
	u32 fallthrough = pc + 4;

	// Flush up front — both paths exit the block, so unflushed const-tracked
	// GPRs must hit memory regardless. Flushing before the Ldr also protects
	// RSCRATCHGPR/RWSCRATCH (used as scratch by armFlushConstRegs).
	armFlushConstRegs();

	armAsm->Mov(RSCRATCHGPR, reinterpret_cast<uintptr_t>(&VU0.VI[REG_VPU_STAT].UL));
	armAsm->Ldr(RWSCRATCH, a64::MemOperand(RSCRATCHGPR));

	a64::Label skipDelaySlot, done;
	// BC2FL not-taken when bit SET; BC2TL not-taken when bit CLEAR.
	// Tbz/Tbnz on bit 8 saves one instruction vs Tst + B.cc.
	if (branchIfSet)
		armAsm->Tbz(RWSCRATCH, 8, &skipDelaySlot);  // BC2TL: skip when clear
	else
		armAsm->Tbnz(RWSCRATCH, 8, &skipDelaySlot); // BC2FL: skip when set

	// Taken: execute delay slot, branch to target.
	recompileNextInstruction(true, false);
	armFlushConstRegs();
	armAsm->Mov(RWSCRATCH, branchTarget);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));
	armAsm->B(&done);

	// Not taken: skip delay slot, PC = fallthrough.
	armAsm->Bind(&skipDelaySlot);
	armAsm->Mov(RWSCRATCH, fallthrough);
	armAsm->Str(RWSCRATCH, a64::MemOperand(RCPUSTATE, PC_OFFSET));

	armAsm->Bind(&done);
	g_branch = 1;
	g_cpuFlushedPC = true;
}

#if ISTUB_BC2F
void recBC2F()  { armBranchInterpWithDSCycles(::BC2F); }
#else
void recBC2F()  { recBC2_helper(false); }
#endif

#if ISTUB_BC2T
void recBC2T()  { armBranchInterpWithDSCycles(::BC2T); }
#else
void recBC2T()  { recBC2_helper(true); }
#endif

#if ISTUB_BC2FL
void recBC2FL() { armBranchInterpWithDSCycles(::BC2FL); }
#else
void recBC2FL() { recBC2_Likely_helper(false); }
#endif

#if ISTUB_BC2TL
void recBC2TL() { armBranchInterpWithDSCycles(::BC2TL); }
#else
void recBC2TL() { recBC2_Likely_helper(true); }
#endif

// ============================================================================
//  VU0 SPECIAL1 math ops — direct per-op interpreter dispatch
//  (no double-dispatch overhead through the COP2 master interpreter switch)
// ============================================================================

// Broadcast scalar FMAC (toACC=false): ADD/SUB/MUL/MADD/MSUB × x/y/z/w
REC_COP2_MVU0(ADDx,  recVU0_ADDx)   REC_COP2_MVU0(ADDy,  recVU0_ADDy)   REC_COP2_MVU0(ADDz,  recVU0_ADDz)   REC_COP2_MVU0(ADDw,  recVU0_ADDw)
REC_COP2_MVU0(SUBx,  recVU0_SUBx)   REC_COP2_MVU0(SUBy,  recVU0_SUBy)   REC_COP2_MVU0(SUBz,  recVU0_SUBz)   REC_COP2_MVU0(SUBw,  recVU0_SUBw)
REC_COP2_MVU0(MADDx, recVU0_MADDx)  REC_COP2_MVU0(MADDy, recVU0_MADDy)  REC_COP2_MVU0(MADDz, recVU0_MADDz)  REC_COP2_MVU0(MADDw, recVU0_MADDw)
REC_COP2_MVU0(MSUBx, recVU0_MSUBx)  REC_COP2_MVU0(MSUBy, recVU0_MSUBy)  REC_COP2_MVU0(MSUBz, recVU0_MSUBz)  REC_COP2_MVU0(MSUBw, recVU0_MSUBw)
REC_COP2_MVU0(MULx,  recVU0_MULx)   REC_COP2_MVU0(MULy,  recVU0_MULy)   REC_COP2_MVU0(MULz,  recVU0_MULz)   REC_COP2_MVU0(MULw,  recVU0_MULw)

// MAX / MINI broadcast (no flag update)
REC_COP2_MVU0_NOFLAGS(MAXx,  recVU0_MAXx)   REC_COP2_MVU0_NOFLAGS(MAXy,  recVU0_MAXy)   REC_COP2_MVU0_NOFLAGS(MAXz,  recVU0_MAXz)   REC_COP2_MVU0_NOFLAGS(MAXw,  recVU0_MAXw)
REC_COP2_MVU0_NOFLAGS(MINIx, recVU0_MINIx)  REC_COP2_MVU0_NOFLAGS(MINIy, recVU0_MINIy)  REC_COP2_MVU0_NOFLAGS(MINIz, recVU0_MINIz)  REC_COP2_MVU0_NOFLAGS(MINIw, recVU0_MINIw)

// Q-register FMAC
REC_COP2_MVU0_Q(ADDq,  recVU0_ADDq)   REC_COP2_MVU0_Q(SUBq,  recVU0_SUBq)   REC_COP2_MVU0_Q(MULq,  recVU0_MULq)
REC_COP2_MVU0_Q(MADDq, recVU0_MADDq)  REC_COP2_MVU0_Q(MSUBq, recVU0_MSUBq)

// I-register FMAC
REC_COP2_MVU0(ADDi,  recVU0_ADDi)   REC_COP2_MVU0(SUBi,  recVU0_SUBi)   REC_COP2_MVU0(MULi,  recVU0_MULi)
REC_COP2_MVU0(MADDi, recVU0_MADDi)  REC_COP2_MVU0(MSUBi, recVU0_MSUBi)

// MAX/MINI Q/I (MAXi and MINIi broadcast VI[I]; no Q variant for MAX/MINI)
REC_COP2_MVU0_NOFLAGS(MAXi,  recVU0_MAXi)
REC_COP2_MVU0_NOFLAGS(MINIi, recVU0_MINIi)

// Full-vector FMAC
REC_COP2_MVU0(ADD,    recVU0_ADD)    REC_COP2_MVU0(SUB,    recVU0_SUB)    REC_COP2_MVU0(MUL,    recVU0_MUL)
REC_COP2_MVU0(MADD,   recVU0_MADD)   REC_COP2_MVU0(MSUB,   recVU0_MSUB)
REC_COP2_MVU0(OPMSUB, recVU0_OPMSUB)
REC_COP2_MVU0_NOFLAGS(MAX,  recVU0_MAX)
REC_COP2_MVU0_NOFLAGS(MINI, recVU0_MINI)

// Integer ALU ops (LOWER family, no flag sync) — native arm64 codegen in
// iVU0Lower_arm64.cpp reads/writes VU0.VI[] directly.
REC_COP2_MVU0_NOFLAGS(IADD,  recVU0_IADD)
REC_COP2_MVU0_NOFLAGS(ISUB,  recVU0_ISUB)
REC_COP2_MVU0_NOFLAGS(IADDI, recVU0_IADDI)
REC_COP2_MVU0_NOFLAGS(IAND,  recVU0_IAND)
REC_COP2_MVU0_NOFLAGS(IOR,   recVU0_IOR)

// Micro-subroutine calls — at parity with x86 INTERPRETATE_COP2_FUNC.
// x86 emits:   iFlushCall(FLUSH_FOR_POSSIBLE_MICRO_EXEC) + cycle-update + recCall
// arm64 emits: armFlushPC + armFlushCode + armFlushConstRegs +
//              armEmitFlushCycleBeforeCall + armEmitCall + armEmitReloadCycleAfterCall
//              + mark GPRs has-const/flushed (all wrapped in armCallInterpreter).
// The x86 FLUSH_FREE_XMM | FLUSH_FREE_VU0 flags have no arm64 analogue — neither
// the XMM-equivalent GPR cache nor a VU0-register cache exists in the EE JIT —
// so there's nothing extra to flush here.
REC_COP2_INTERP(CALLMS)  REC_COP2_INTERP(CALLMSR)

// ============================================================================
//  VU0 SPECIAL2 ops — accumulator, conversion, misc
// ============================================================================

// Accumulator broadcast FMAC (toACC=true): ADDA/SUBA/MULA/MADDA/MSUBA × x/y/z/w
REC_COP2_MVU0(ADDAx,  recVU0_ADDAx)   REC_COP2_MVU0(ADDAy,  recVU0_ADDAy)   REC_COP2_MVU0(ADDAz,  recVU0_ADDAz)   REC_COP2_MVU0(ADDAw,  recVU0_ADDAw)
REC_COP2_MVU0(SUBAx,  recVU0_SUBAx)   REC_COP2_MVU0(SUBAy,  recVU0_SUBAy)   REC_COP2_MVU0(SUBAz,  recVU0_SUBAz)   REC_COP2_MVU0(SUBAw,  recVU0_SUBAw)
REC_COP2_MVU0(MADDAx, recVU0_MADDAx)  REC_COP2_MVU0(MADDAy, recVU0_MADDAy)  REC_COP2_MVU0(MADDAz, recVU0_MADDAz)  REC_COP2_MVU0(MADDAw, recVU0_MADDAw)
REC_COP2_MVU0(MSUBAx, recVU0_MSUBAx)  REC_COP2_MVU0(MSUBAy, recVU0_MSUBAy)  REC_COP2_MVU0(MSUBAz, recVU0_MSUBAz)  REC_COP2_MVU0(MSUBAw, recVU0_MSUBAw)

// ITOF / FTOI — float/int conversions (no flag update)
REC_COP2_MVU0_NOFLAGS(ITOF0,  recVU0_ITOF0)   REC_COP2_MVU0_NOFLAGS(ITOF4,  recVU0_ITOF4)
REC_COP2_MVU0_NOFLAGS(ITOF12, recVU0_ITOF12)  REC_COP2_MVU0_NOFLAGS(ITOF15, recVU0_ITOF15)
REC_COP2_MVU0_NOFLAGS(FTOI0,  recVU0_FTOI0)   REC_COP2_MVU0_NOFLAGS(FTOI4,  recVU0_FTOI4)
REC_COP2_MVU0_NOFLAGS(FTOI12, recVU0_FTOI12)  REC_COP2_MVU0_NOFLAGS(FTOI15, recVU0_FTOI15)

// MULA broadcast FMAC (toACC=true)
REC_COP2_MVU0(MULAx,  recVU0_MULAx)   REC_COP2_MVU0(MULAy,  recVU0_MULAy)
REC_COP2_MVU0(MULAz,  recVU0_MULAz)   REC_COP2_MVU0(MULAw,  recVU0_MULAw)
// MULAq / MULAi (Q/I register broadcast, toACC=true)
REC_COP2_MVU0_Q(MULAq,  recVU0_MULAq)   REC_COP2_MVU0(MULAi,  recVU0_MULAi)
// ABS + CLIPw
REC_COP2_MVU0_NOFLAGS(ABS,   recVU0_ABS)
REC_COP2_MVU0_CLIP(   CLIPw, recVU0_CLIP)

// Accumulator Q/I FMAC (toACC=true)
REC_COP2_MVU0_Q(ADDAq,  recVU0_ADDAq)   REC_COP2_MVU0_Q(MADDAq, recVU0_MADDAq)
REC_COP2_MVU0(  ADDAi,  recVU0_ADDAi)   REC_COP2_MVU0(  MADDAi, recVU0_MADDAi)
REC_COP2_MVU0_Q(SUBAq,  recVU0_SUBAq)   REC_COP2_MVU0_Q(MSUBAq, recVU0_MSUBAq)
REC_COP2_MVU0(  SUBAi,  recVU0_SUBAi)   REC_COP2_MVU0(  MSUBAi, recVU0_MSUBAi)

// Full-vector accumulator FMAC
REC_COP2_MVU0(ADDA,   recVU0_ADDA)    REC_COP2_MVU0(MADDA,  recVU0_MADDA)   REC_COP2_MVU0(MULA,  recVU0_MULA)
REC_COP2_MVU0(SUBA,   recVU0_SUBA)    REC_COP2_MVU0(MSUBA,  recVU0_MSUBA)   REC_COP2_MVU0(OPMULA, recVU0_OPMULA)

// NOP — hardware no-op, emit nothing
void recVNOP() {}

// Data movement (LOWER family, no flag sync) — native in iVU0Lower_arm64.cpp.
REC_COP2_MVU0_NOFLAGS(MOVE, recVU0_MOVE)
REC_COP2_MVU0_NOFLAGS(MR32, recVU0_MR32)

// Load/store (LOWER family, C-helper calls, no flag sync) — vu0_LQI etc.
// read/write VF[] and VI[] directly; they're self-contained wrt flags.
REC_COP2_MVU0_NOFLAGS(LQI, recVU0_LQI)
REC_COP2_MVU0_NOFLAGS(SQI, recVU0_SQI)
REC_COP2_MVU0_NOFLAGS(LQD, recVU0_LQD)
REC_COP2_MVU0_NOFLAGS(SQD, recVU0_SQD)

// Division (LOWER family, SYNCFDIV sync) — native emits write VU0.q +
// statusflag D/I bits; the sync helper mirrors them into VI[REG_Q] /
// VI[REG_STATUS_FLAG] so CFC2 and VWAITQ see the result.
REC_COP2_MVU0_FDIV(DIV,   recVU0_DIV)
REC_COP2_MVU0_FDIV(SQRT,  recVU0_SQRT)
REC_COP2_MVU0_FDIV(RSQRT, recVU0_RSQRT)

// VWAITQ — stall until FDIV pipeline drains (interp handles the stall model)
void recVWAITQ() { armCallInterpreter(::VWAITQ); }

// Integer ↔ VF transfer (LOWER family, no flag sync) — native in
// iVU0Lower_arm64.cpp for MTIR/MFIR, C-helper for ILWR/ISWR.
REC_COP2_MVU0_NOFLAGS(MTIR, recVU0_MTIR)
REC_COP2_MVU0_NOFLAGS(MFIR, recVU0_MFIR)
REC_COP2_MVU0_NOFLAGS(ILWR, recVU0_ILWR)
REC_COP2_MVU0_NOFLAGS(ISWR, recVU0_ISWR)

// Random number generator (LOWER family, C-helper calls, no flag sync).
REC_COP2_MVU0_NOFLAGS(RNEXT, recVU0_RNEXT)
REC_COP2_MVU0_NOFLAGS(RGET,  recVU0_RGET)
REC_COP2_MVU0_NOFLAGS(RINIT, recVU0_RINIT)
REC_COP2_MVU0_NOFLAGS(RXOR,  recVU0_RXOR)

// ============================================================================
//  Unknown/invalid COP2 opcode
// ============================================================================
static void rec_C2UNK()
{
    Console.Error("COP2 unknown opcode: %08X", cpuRegs.code);
}

// ============================================================================
//  Sub-dispatchers (forward-declared because they reference each other via tables)
// ============================================================================
static void recCOP2_BC2();
static void recCOP2_SPEC1();
static void recCOP2_SPEC2();

// ============================================================================
//  Dispatch table: recCOP2t[32]
//  Indexed by _Rs_ (bits 25:21 of the instruction).
//  Mirrors x86 microVU_Macro.inl recCOP2t[].
// ============================================================================
static void (*recCOP2t[32])() = {
    rec_C2UNK,     recQMFC2,      recCFC2,       rec_C2UNK,
    rec_C2UNK,     recQMTC2,      recCTC2,       rec_C2UNK,
    recCOP2_BC2,   rec_C2UNK,     rec_C2UNK,     rec_C2UNK,
    rec_C2UNK,     rec_C2UNK,     rec_C2UNK,     rec_C2UNK,
    recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1,
    recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1,
    recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1,
    recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1, recCOP2_SPEC1,
};

// ============================================================================
//  Dispatch table: recCOP2_BC2t[32]
//  Indexed by _Rt_ (bits 20:16) when _Rs_ == 8.
// ============================================================================
static void (*recCOP2_BC2t[32])() = {
    recBC2F,   recBC2T,   recBC2FL,  recBC2TL,
    rec_C2UNK, rec_C2UNK, rec_C2UNK, rec_C2UNK,
    rec_C2UNK, rec_C2UNK, rec_C2UNK, rec_C2UNK,
    rec_C2UNK, rec_C2UNK, rec_C2UNK, rec_C2UNK,
    rec_C2UNK, rec_C2UNK, rec_C2UNK, rec_C2UNK,
    rec_C2UNK, rec_C2UNK, rec_C2UNK, rec_C2UNK,
    rec_C2UNK, rec_C2UNK, rec_C2UNK, rec_C2UNK,
    rec_C2UNK, rec_C2UNK, rec_C2UNK, rec_C2UNK,
};

// ============================================================================
//  Dispatch table: recCOP2SPECIAL1t[64]
//  Indexed by _Funct_ (bits 5:0) when _Rs_ >= 0x10.
//  Matches x86 recCOP2SPECIAL1t[] exactly.
// ============================================================================
static void (*recCOP2SPECIAL1t[64])() = {
    recVADDx,    recVADDy,    recVADDz,    recVADDw,
    recVSUBx,    recVSUBy,    recVSUBz,    recVSUBw,
    recVMADDx,   recVMADDy,   recVMADDz,   recVMADDw,
    recVMSUBx,   recVMSUBy,   recVMSUBz,   recVMSUBw,
    recVMAXx,    recVMAXy,    recVMAXz,    recVMAXw,
    recVMINIx,   recVMINIy,   recVMINIz,   recVMINIw,
    recVMULx,    recVMULy,    recVMULz,    recVMULw,
    recVMULq,    recVMAXi,    recVMULi,    recVMINIi,
    recVADDq,    recVMADDq,   recVADDi,    recVMADDi,
    recVSUBq,    recVMSUBq,   recVSUBi,    recVMSUBi,
    recVADD,     recVMADD,    recVMUL,     recVMAX,
    recVSUB,     recVMSUB,    recVOPMSUB,  recVMINI,
    recVIADD,    recVISUB,    recVIADDI,   rec_C2UNK,
    recVIAND,    recVIOR,     rec_C2UNK,   rec_C2UNK,
    recVCALLMS,  recVCALLMSR, rec_C2UNK,   rec_C2UNK,
    recCOP2_SPEC2, recCOP2_SPEC2, recCOP2_SPEC2, recCOP2_SPEC2,
};

// ============================================================================
//  Dispatch table: recCOP2SPECIAL2t[128]
//  Indexed by (code & 3) | ((code >> 4) & 0x7C) — 7-bit field from bits[10:6|1:0].
//  Matches x86 recCOP2SPECIAL2t[] exactly.
// ============================================================================
static void (*recCOP2SPECIAL2t[128])() = {
    recVADDAx,   recVADDAy,   recVADDAz,   recVADDAw,
    recVSUBAx,   recVSUBAy,   recVSUBAz,   recVSUBAw,
    recVMADDAx,  recVMADDAy,  recVMADDAz,  recVMADDAw,
    recVMSUBAx,  recVMSUBAy,  recVMSUBAz,  recVMSUBAw,
    recVITOF0,   recVITOF4,   recVITOF12,  recVITOF15,
    recVFTOI0,   recVFTOI4,   recVFTOI12,  recVFTOI15,
    recVMULAx,   recVMULAy,   recVMULAz,   recVMULAw,
    recVMULAq,   recVABS,     recVMULAi,   recVCLIPw,
    recVADDAq,   recVMADDAq,  recVADDAi,   recVMADDAi,
    recVSUBAq,   recVMSUBAq,  recVSUBAi,   recVMSUBAi,
    recVADDA,    recVMADDA,   recVMULA,    rec_C2UNK,
    recVSUBA,    recVMSUBA,   recVOPMULA,  recVNOP,
    recVMOVE,    recVMR32,    rec_C2UNK,   rec_C2UNK,
    recVLQI,     recVSQI,     recVLQD,     recVSQD,
    recVDIV,     recVSQRT,    recVRSQRT,   recVWAITQ,
    recVMTIR,    recVMFIR,    recVILWR,    recVISWR,
    recVRNEXT,   recVRGET,    recVRINIT,   recVRXOR,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
    rec_C2UNK,   rec_C2UNK,   rec_C2UNK,   rec_C2UNK,
};

// ============================================================================
//  Sub-dispatcher implementations
// ============================================================================

static void recCOP2_BC2()   { recCOP2_BC2t[_Rt_](); }
static void recCOP2_SPEC2() { recCOP2SPECIAL2t[(cpuRegs.code & 3) | ((cpuRegs.code >> 4) & 0x7C)](); }

static void recCOP2_SPEC1()
{
    // If analysis pass flagged that VU0 needs to be synced/finished before this
    // macro-mode instruction, emit a call to _vu0FinishMicro.  This matches the
    // x86 mVUFinishVU0() path in recCOP2_SPEC1().
    if (g_pCurInstInfo && (g_pCurInstInfo->info & (EEINST_COP2_SYNC_VU0 | EEINST_COP2_FINISH_VU0)))
        armCallInterpreter(_vu0FinishMicro);
    recCOP2SPECIAL1t[_Funct_]();
}

// ============================================================================
//  Master COP2 dispatch — called from the EE opcode dispatch table
// ============================================================================
void recCOP2() { recCOP2t[_Rs_](); }

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
