// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 VU0 Recompiler — Upper Instruction Stubs
// FMAC arithmetic (ADD/SUB/MUL/MADD/MSUB with xyzwqi broadcasts),
// accumulator variants (ADDA/SUBA/MULA/MADDA/MSUBA),
// MAX, MINI, ABS, CLIP, FTOI/ITOF, OPMULA, OPMSUB, NOP
//
// Mirrors iVU1Upper_arm64.cpp exactly; VU0_BASE_REG (x23) points to &VU0
// at runtime, same layout as VU1. ISTUB_VU0_* macros shadow ISTUB_VU_* so
// the two units bisect independently.

// Included from aVU0.cpp. All headers (Common.h / VUops.h / VU.h / VUflags.h /
// arm64/arm64Emitter.h / arm64/AsmHelpers.h) and `using namespace vixl::aarch64`
// are already in effect in the parent. VU0_BASE_REG / vfOff / viOff also come
// from the parent; accOff is used only here.

// ============================================================================
//  Native NEON codegen helpers
// ============================================================================
static constexpr int64_t accOff()
{
	return static_cast<int64_t>(offsetof(VURegs, ACC));
}

// C helper: MAC flag update + result clamping + writeback.
// Called from JIT after NEON math. Takes the 4 float results, applies
// VU_MACx_UPDATE (which clamps overflow/underflow and sets per-component flags),
// writes to destination with XYZW mask, and updates statusflag.
static void vu0_fmac_writeback(VURegs* VU, VECTOR* dst, u32 xyzw,
                                float rx, float ry, float rz, float rw)
{
	// VF[0] is hardwired to {0,0,0,1} — discard writes but still update flags
	const bool write = (dst != &VU->VF[0]);
	if (xyzw & 8) { u32 v = VU_MACx_UPDATE(VU, rx); if (write) dst->i.x = v; }
	else VU_MACx_CLEAR(VU);
	if (xyzw & 4) { u32 v = VU_MACy_UPDATE(VU, ry); if (write) dst->i.y = v; }
	else VU_MACy_CLEAR(VU);
	if (xyzw & 2) { u32 v = VU_MACz_UPDATE(VU, rz); if (write) dst->i.z = v; }
	else VU_MACz_CLEAR(VU);
	if (xyzw & 1) { u32 v = VU_MACw_UPDATE(VU, rw); if (write) dst->i.w = v; }
	else VU_MACw_CLEAR(VU);
	VU_STAT_UPDATE(VU);
}

// Emit the call to vu0_fmac_writeback.
// Result must be in v5.4S. dst_off = byte offset of destination from VU0_BASE_REG.
static void emitFmacWriteback(int64_t dst_off, u32 xyzw)
{
	// Extract v5 lanes to s0-s3 (ARM64 ABI: float args in s0-s3)
	armAsm->Mov(v3.V4S(), 0, v5.V4S(), 3); // s3 = W
	armAsm->Mov(v2.V4S(), 0, v5.V4S(), 2); // s2 = Z
	armAsm->Mov(v1.V4S(), 0, v5.V4S(), 1); // s1 = Y
	armAsm->Fmov(s0, s5);                   // s0 = X (lane 0)
	// Integer args: x0=VU, x1=dst, w2=xyzw
	armAsm->Mov(x0, VU0_BASE_REG);
	armAsm->Add(x1, VU0_BASE_REG, dst_off);
	armAsm->Mov(w2, xyzw);
	armEmitCall(reinterpret_cast<const void*>(vu0_fmac_writeback));
}

// Writeback for ops that DO NOT update MAC/Status flags (MAX, MINI, ABS).
// The interpreter's _vuMAX/_vuMINI*/_vuABS write directly to VF[dst] via
// applyMinMax/applyUnaryFunction, neither of which touches macflag/statusflag.
// Going through vu0_fmac_writeback would corrupt MAC flags that subsequent
// FMAND/FCAND/etc. read.
//
// dst_reg: destination register index (compile-time known via VU0.code).
//          Interpreter returns immediately when dst_reg==0, so no write *and*
//          no flag update — match that exactly.
// xyzw: write mask (compile-time known via VU0.code bits 21-24).
// Result must be in v5.4S.
static void emitNoFlagWriteback(u32 dst_reg, u32 xyzw)
{
	if (dst_reg == 0) return; // VF[0] hardwired; interpreter no-ops the whole insn
	const int64_t dst_off = vfOff(dst_reg);

	if (xyzw == 0xF)
	{
		armAsm->Str(q5, MemOperand(VU0_BASE_REG, dst_off));
		return;
	}

	// Partial write: load existing dst into v4, merge selected lanes from v5
	armAsm->Ldr(q4, MemOperand(VU0_BASE_REG, dst_off));
	if (xyzw & 8) armAsm->Mov(v4.V4S(), 0, v5.V4S(), 0); // x
	if (xyzw & 4) armAsm->Mov(v4.V4S(), 1, v5.V4S(), 1); // y
	if (xyzw & 2) armAsm->Mov(v4.V4S(), 2, v5.V4S(), 2); // z
	if (xyzw & 1) armAsm->Mov(v4.V4S(), 3, v5.V4S(), 3); // w
	armAsm->Str(q4, MemOperand(VU0_BASE_REG, dst_off));
}

// ============================================================================
//  NEON FMAC emit patterns
//
//  Binary (ADD/SUB/MUL):  dst = VF[fs] OP src2
//  Ternary (MADD/MSUB):   dst = ACC ± VF[fs] * src2
//
//  src2 is loaded into v1:
//    Broadcast x/y/z/w: ldr s1 + dup v1.4s
//    Q register:        ldr w0 from VI[REG_Q], dup v1.4s, w0
//    I register:        ldr w0 from VI[REG_I], dup v1.4s, w0
//    Full vector:       ldr q1 from VF[ft]
// ============================================================================

// Load second operand broadcast from VF[ft] component into v1.4S
static void emitLoadBroadcast(u32 ft, int comp) // comp: 0=x,1=y,2=z,3=w
{
	armAsm->Ldr(s1, MemOperand(VU0_BASE_REG, vfOff(ft) + comp * 4));
	armAsm->Dup(v1.V4S(), v1.V4S(), 0);
}

// Load Q or I register broadcast into v1.4S
static void emitLoadQI(int64_t vi_off)
{
	armAsm->Ldr(w0, MemOperand(VU0_BASE_REG, vi_off));
	armAsm->Dup(v1.V4S(), w0);
}

// --- vuDouble-style input clamping for NEON vectors ---
// The interpreter calls vuDouble() on every FMAC input, which:
//   exp=0 (denormal): flush to ±0 (ARM64 FZ mode handles this)
//   exp=0xFF (inf/NaN): clamp to ±MAX_FLOAT (gated on CHECK_VU_SIGN_OVERFLOW)
// Mirrors the x86 JIT's sign-preserving operand clamp (mVUclamp2 in
// microVU_Clamp.inl): SMIN against +MAX_FLOAT handles positive overflow,
// UMIN against -MAX_FLOAT (as unsigned bit pattern 0xFF7FFFFF) handles
// negative overflow, leaving the sign bit of -NaN / -INF intact.
// Setup: load +MAX_FLOAT into v6, -MAX_FLOAT into v7.

static void emitVuClampSetup()
{
	armAsm->Mov(w0, 0xFFFF);
	armAsm->Movk(w0, 0x7F7F, 16); // w0 = 0x7F7FFFFF = +MAX_FLOAT
	armAsm->Dup(v6.V4S(), w0);
	armAsm->Fneg(v7.V4S(), v6.V4S()); // v7 = 0xFF7FFFFF = -MAX_FLOAT
}

static void emitVuClampVec(const VRegister& vn)
{
	// Signed 32-bit min clamps +overflow (+NaN/+INF > +MAX as signed int).
	armAsm->Smin(vn, vn, v6.V4S());
	// Unsigned 32-bit min clamps -overflow (-NaN/-INF unsigned > -MAX's
	// unsigned bit pattern 0xFF7FFFFF). Positive values (<= 0x7F7FFFFF)
	// are smaller than 0xFF7FFFFF unsigned and pass through unchanged.
	armAsm->Umin(vn, vn, v7.V4S());
}

// VuAddSubHack (Tri-ace games): when |exp(to) - exp(from)| >= 25, the
// smaller-exponent operand contributes nothing to the sum at the bit-exact
// level the game's encryption check expects. Zero its mantissa+exponent
// (preserving sign) before the FADD. Mirrors x86 ADD_SS_TriAceHack at
// microVU_Misc.inl:509-535. Only applies to single-lane ADDi/ADDq per
// upstream (multi-lane ADDi/ADDq uses SSE_ADD2PS without the hack). Scratch:
// v6-v9. Caller must emit clamp setup BEFORE this if both are needed.
static void emitVuAddSubHack(const VRegister& to, const VRegister& from)
{
	armAsm->Ushr(v6.V4S(), to.V4S(), 23);
	armAsm->Ushr(v7.V4S(), from.V4S(), 23);
	armAsm->Movi(v8.V4S(), 0xFF);
	armAsm->And (v6.V16B(), v6.V16B(), v8.V16B());
	armAsm->And (v7.V16B(), v7.V16B(), v8.V16B());
	armAsm->Sub (v7.V4S(), v7.V4S(), v6.V4S()); // v7 = exp(from) - exp(to)
	armAsm->Movi(v8.V4S(), 0x80, vixl::aarch64::LSL, 24); // sign mask
	{
		armAsm->Movi(v9.V4S(), 25);
		armAsm->Cmge(v9.V4S(), v7.V4S(), v9.V4S());
		armAsm->And (v6.V16B(), to.V16B(), v8.V16B());
		armAsm->Bsl (v9.V16B(), v6.V16B(), to.V16B());
		armAsm->Mov (to.V16B(), v9.V16B());
	}
	{
		armAsm->Movi(v9.V4S(), 25);
		armAsm->Neg (v9.V4S(), v9.V4S());
		// CMLE only takes immediate-zero RHS; (v7 <= -25) <=> (-25 >= v7).
		armAsm->Cmge(v9.V4S(), v9.V4S(), v7.V4S());
		armAsm->And (v6.V16B(), from.V16B(), v8.V16B());
		armAsm->Bsl (v9.V16B(), v6.V16B(), from.V16B());
		armAsm->Mov (from.V16B(), v9.V16B());
	}
}

// --- Binary FMAC: dst = VF[fs] OP src2 ---
// op: 0=ADD, 1=SUB, 2=MUL
// isScalarAdd: caller is ADDi/ADDq (scalar broadcast ADD). Gates VuAddSubHack.
static void emitBinaryFmac(int op, u32 fs, int64_t dst_off, u32 xyzw,
                           bool isScalarAdd = false)
{
	// VuAddSubHack gate (Tri-ace ADDi/ADDq with single-lane mask only).
	const bool singleLane = (xyzw == 0x1u || xyzw == 0x2u || xyzw == 0x4u || xyzw == 0x8u);
	const bool addSubHack = isScalarAdd && CHECK_VUADDSUBHACK && singleLane;

	// Load VF[fs] → v0
	armAsm->Ldr(q0, MemOperand(VU0_BASE_REG, vfOff(fs)));
	// v1 must already be loaded with src2

	// vuDouble input clamping (matches interpreter's vuDouble on every operand)
	// INTENTIONAL DIVERGENCE FROM UPSTREAM x86 JIT: this port gates on
	// BOTH VU0 and VU1 sign-overflow flags as the signal for an event-test
	// optimization. The apparent VU1-flag-spillover is deliberate — do not
	// narrow to VU0-only or flip to CHECK_VU_OVERFLOW, either breaks event
	// testing on this port.
	if (CHECK_VU_SIGN_OVERFLOW(0) || CHECK_VU_SIGN_OVERFLOW(1))
	{
		emitVuClampSetup();
		emitVuClampVec(v0.V4S());
		emitVuClampVec(v1.V4S());
	}

	// VuAddSubHack: apply AFTER clamp (which uses v6/v7 — we overwrite).
	if (addSubHack)
		emitVuAddSubHack(v0, v1);

	// Perform NEON op: v5 = v0 OP v1
	switch (op) {
		case 0: armAsm->Fadd(v5.V4S(), v0.V4S(), v1.V4S()); break;
		case 1: armAsm->Fsub(v5.V4S(), v0.V4S(), v1.V4S()); break;
		case 2: armAsm->Fmul(v5.V4S(), v0.V4S(), v1.V4S()); break;
	}
	emitFmacWriteback(dst_off, xyzw);
}

// --- Ternary FMAC: dst = ACC ± VF[fs] * src2 ---
// subtract: false=MADD (ACC + fs*src2), true=MSUB (ACC - fs*src2)
// NOTE: The PS2 VU does NOT have fused multiply-add. It performs a separate
// multiply then add/sub with intermediate rounding. Using FMLA/FMLS would
// produce results differing by 1+ ULP, causing cascading precision errors
// in matrix transforms. We use separate FMUL + FADD/FSUB to match.
static void emitTernaryFmac(bool subtract, u32 fs, int64_t dst_off, u32 xyzw)
{
	// Load VF[fs] → v0
	armAsm->Ldr(q0, MemOperand(VU0_BASE_REG, vfOff(fs)));
	// v1 must already be loaded with src2
	// Load ACC → v5
	armAsm->Ldr(q5, MemOperand(VU0_BASE_REG, accOff()));

	// vuDouble input clamping (all 3 operands, matching interpreter)
	// INTENTIONAL DIVERGENCE FROM UPSTREAM x86 JIT: this port gates on
	// BOTH VU0 and VU1 sign-overflow flags as the signal for an event-test
	// optimization. The apparent VU1-flag-spillover is deliberate — do not
	// narrow to VU0-only or flip to CHECK_VU_OVERFLOW, either breaks event
	// testing on this port.
	if (CHECK_VU_SIGN_OVERFLOW(0) || CHECK_VU_SIGN_OVERFLOW(1))
	{
		emitVuClampSetup();
		emitVuClampVec(v0.V4S());
		emitVuClampVec(v1.V4S());
		emitVuClampVec(v5.V4S());
	}

	// Separate multiply: v4 = VF[fs] * src2 (with rounding)
	armAsm->Fmul(v4.V4S(), v0.V4S(), v1.V4S());
	// Separate add/sub: v5 = ACC ± product (with rounding)
	if (subtract)
		armAsm->Fsub(v5.V4S(), v5.V4S(), v4.V4S());
	else
		armAsm->Fadd(v5.V4S(), v5.V4S(), v4.V4S());
	emitFmacWriteback(dst_off, xyzw);
}

// --- MAX/MINI: VF[fd] = max/min(VF[fs], src2) using sign-aware integer compare ---
//
// Mirrors the VU1 implementation in iVU1Upper_arm64.cpp:1107 (and x86 microVU's
// MIN_MAX_PS at microVU_Misc.inl:380 plus interp fp_max/fp_min at VUops.cpp:813).
// All three use bit-pattern comparisons that PRESERVE denormal values rather
// than flushing them to ±0.
//
// Why not Fmaxnm/Fminnm: with FPCR FZ enabled (the default for VU emulation),
// NEON Fmaxnm flushes denormal operands to signed zero before comparison. PS2
// VU does NOT flush — denormal bit patterns pass through MAX/MIN unchanged.
// VU0 used to use Fmaxnm + FZ and that diverged from x86 + interp on
// denormal-vs-zero, surfaced by the VU0 shadow harness in Killzone (player
// model corruption at character select). VU1 was already fixed (the comment
// at the VU1 emit notes "San Andreas shadow corruption"); VU0 was missed.
//
// Algorithm: for both-positive and mixed-sign cases, signed-int max/min on
// the float bit patterns gives the correct FP ordering. For both-negative,
// the int order is reversed w.r.t. FP order, so we have to swap (use Smin
// when we want FP max, and vice versa). A "both-negative" mask drives a
// per-lane Bif select.
//
// MUST use emitNoFlagWriteback — MAX/MINI on PS2 don't update MAC/Status.
static void emitMaxFmac(bool isMini, u32 fs, u32 fd, u32 xyzw)
{
	armAsm->Ldr(q0, MemOperand(VU0_BASE_REG, vfOff(fs)));
	// v1 must already be loaded with src2.

	// Signed integer max/min on 32-bit lanes.
	armAsm->Smax(v2.V4S(), v0.V4S(), v1.V4S());
	armAsm->Smin(v3.V4S(), v0.V4S(), v1.V4S());

	// Build "both negative" mask in v6: arithmetic-shift right by 31 splats
	// the sign bit, AND of the two splats is 0xFFFFFFFF per lane iff both
	// inputs had the sign bit set.
	armAsm->Sshr(v4.V4S(), v0.V4S(), 31);
	armAsm->Sshr(v6.V4S(), v1.V4S(), 31);
	armAsm->And(v6.V16B(), v4.V16B(), v6.V16B());

	// Select per lane: result = both_neg ? swapped : signed_(max|min).
	// BIF Vd, Vn, Vm copies Vn into Vd where Vm bit is 0, else keeps Vd.
	if (isMini)
	{
		// fp_min: both_neg → signed_max ; otherwise → signed_min
		armAsm->Mov(v5.V16B(), v2.V16B());            // start = signed_max
		armAsm->Bif(v5.V16B(), v3.V16B(), v6.V16B()); // mask=0 → signed_min
	}
	else
	{
		// fp_max: both_neg → signed_min ; otherwise → signed_max
		armAsm->Mov(v5.V16B(), v3.V16B());            // start = signed_min
		armAsm->Bif(v5.V16B(), v2.V16B(), v6.V16B()); // mask=0 → signed_max
	}
	emitNoFlagWriteback(fd, xyzw);
}

// --- ABS: VF[ft] = fabs(VF[fs]) ---
// Interpreter (applyUnaryFunction<vuOpABS>) reads from _Fs_ and writes to _Ft_,
// returns early when _Ft_==0, and does not touch MAC/Status flags.
static void emitAbsFmac(u32 fs, u32 ft, u32 xyzw)
{
	armAsm->Ldr(q0, MemOperand(VU0_BASE_REG, vfOff(fs)));
	armAsm->Fabs(v5.V4S(), v0.V4S());
	emitNoFlagWriteback(ft, xyzw);
}

// ============================================================================
//  Macro: generate a binary FMAC rec function (ADD/SUB/MUL variants)
//
//  FMAC_BINARY_BC(name, op, comp)    — broadcast VF[ft].comp
//  FMAC_BINARY_Q(name, op)           — Q register broadcast
//  FMAC_BINARY_I(name, op)           — I register broadcast
//  FMAC_BINARY_FULL(name, op)        — full VF[ft] vector
//
//  op: 0=ADD, 1=SUB, 2=MUL
//  comp: 0=x, 1=y, 2=z, 3=w
//  toACC: false=VF[fd], true=ACC
// ============================================================================

#define FMAC_BINARY_BC(name, op, comp, toACC) \
	void recVU0_##name() { \
		const u32 fd = (VU0.code >> 6) & 0x1F; \
		const u32 fs = (VU0.code >> 11) & 0x1F; \
		const u32 ft = (VU0.code >> 16) & 0x1F; \
		const u32 xyzw = (VU0.code >> 21) & 0xF; \
		emitLoadBroadcast(ft, comp); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitBinaryFmac(op, fs, dst, xyzw); \
	}

#define FMAC_BINARY_Q(name, op, toACC) \
	void recVU0_##name() { \
		const u32 fd = (VU0.code >> 6) & 0x1F; \
		const u32 fs = (VU0.code >> 11) & 0x1F; \
		const u32 xyzw = (VU0.code >> 21) & 0xF; \
		emitLoadQI(viOff(REG_Q)); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitBinaryFmac(op, fs, dst, xyzw, /*isScalarAdd=*/((op) == 0)); \
	}

#define FMAC_BINARY_I(name, op, toACC) \
	void recVU0_##name() { \
		const u32 fd = (VU0.code >> 6) & 0x1F; \
		const u32 fs = (VU0.code >> 11) & 0x1F; \
		const u32 xyzw = (VU0.code >> 21) & 0xF; \
		emitLoadQI(viOff(REG_I)); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitBinaryFmac(op, fs, dst, xyzw, /*isScalarAdd=*/((op) == 0)); \
	}

#define FMAC_BINARY_FULL(name, op, toACC) \
	void recVU0_##name() { \
		const u32 fd = (VU0.code >> 6) & 0x1F; \
		const u32 fs = (VU0.code >> 11) & 0x1F; \
		const u32 ft = (VU0.code >> 16) & 0x1F; \
		const u32 xyzw = (VU0.code >> 21) & 0xF; \
		const int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		/* doSafeSub: SUB/SUBA (op=1) full-vector with fs==ft forces +0. \
		 * Mirrors x86 microVU_Upper.inl:173-187. Avoids INF-INF=NaN and  \
		 * NaN-NaN=NaN, either of which would corrupt the Z MAC flag.    \
		 * Broadcast variants (SUBx/y/z/w/q/i) are NOT affected per x86. */ \
		if ((op) == 1 && fs == ft) { \
			armAsm->Movi(v5.V4S(), 0); \
			emitFmacWriteback(dst, xyzw); \
			return; \
		} \
		armAsm->Ldr(q1, MemOperand(VU0_BASE_REG, vfOff(ft))); \
		emitBinaryFmac(op, fs, dst, xyzw); \
	}

// ============================================================================
//  Macro: generate a ternary FMAC rec function (MADD/MSUB variants)
// ============================================================================

#define FMAC_TERNARY_BC(name, isSub, comp, toACC) \
	void recVU0_##name() { \
		const u32 fd = (VU0.code >> 6) & 0x1F; \
		const u32 fs = (VU0.code >> 11) & 0x1F; \
		const u32 ft = (VU0.code >> 16) & 0x1F; \
		const u32 xyzw = (VU0.code >> 21) & 0xF; \
		emitLoadBroadcast(ft, comp); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitTernaryFmac(isSub, fs, dst, xyzw); \
	}

#define FMAC_TERNARY_Q(name, isSub, toACC) \
	void recVU0_##name() { \
		const u32 fd = (VU0.code >> 6) & 0x1F; \
		const u32 fs = (VU0.code >> 11) & 0x1F; \
		const u32 xyzw = (VU0.code >> 21) & 0xF; \
		emitLoadQI(viOff(REG_Q)); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitTernaryFmac(isSub, fs, dst, xyzw); \
	}

#define FMAC_TERNARY_I(name, isSub, toACC) \
	void recVU0_##name() { \
		const u32 fd = (VU0.code >> 6) & 0x1F; \
		const u32 fs = (VU0.code >> 11) & 0x1F; \
		const u32 xyzw = (VU0.code >> 21) & 0xF; \
		emitLoadQI(viOff(REG_I)); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitTernaryFmac(isSub, fs, dst, xyzw); \
	}

#define FMAC_TERNARY_FULL(name, isSub, toACC) \
	void recVU0_##name() { \
		const u32 fd = (VU0.code >> 6) & 0x1F; \
		const u32 fs = (VU0.code >> 11) & 0x1F; \
		const u32 ft = (VU0.code >> 16) & 0x1F; \
		const u32 xyzw = (VU0.code >> 21) & 0xF; \
		armAsm->Ldr(q1, MemOperand(VU0_BASE_REG, vfOff(ft))); \
		int64_t dst = (toACC) ? accOff() : vfOff(fd); \
		emitTernaryFmac(isSub, fs, dst, xyzw); \
	}

// ============================================================================
//  Macro: generate MAX/MINI FMAC rec function
// ============================================================================

#define FMAC_MAXMINI_BC(name, isMini, comp) \
	void recVU0_##name() { \
		const u32 fd = (VU0.code >> 6) & 0x1F; \
		const u32 fs = (VU0.code >> 11) & 0x1F; \
		const u32 ft = (VU0.code >> 16) & 0x1F; \
		const u32 xyzw = (VU0.code >> 21) & 0xF; \
		emitLoadBroadcast(ft, comp); \
		emitMaxFmac(isMini, fs, fd, xyzw); \
	}

#define FMAC_MAXMINI_I(name, isMini) \
	void recVU0_##name() { \
		const u32 fd = (VU0.code >> 6) & 0x1F; \
		const u32 fs = (VU0.code >> 11) & 0x1F; \
		const u32 xyzw = (VU0.code >> 21) & 0xF; \
		emitLoadQI(viOff(REG_I)); \
		emitMaxFmac(isMini, fs, fd, xyzw); \
	}

#define FMAC_MAXMINI_FULL(name, isMini) \
	void recVU0_##name() { \
		const u32 fd = (VU0.code >> 6) & 0x1F; \
		const u32 fs = (VU0.code >> 11) & 0x1F; \
		const u32 ft = (VU0.code >> 16) & 0x1F; \
		const u32 xyzw = (VU0.code >> 21) & 0xF; \
		armAsm->Ldr(q1, MemOperand(VU0_BASE_REG, vfOff(ft))); \
		emitMaxFmac(isMini, fs, fd, xyzw); \
	}

// ABS: VF[ft] = |VF[fs]|  (interpreter reads _Fs_, writes _Ft_).
// Most code uses fs==ft (in-place), but the JIT must follow the encoding.
#define FMAC_ABS(name) \
	void recVU0_##name() { \
		const u32 fs = (VU0.code >> 11) & 0x1F; \
		const u32 ft = (VU0.code >> 16) & 0x1F; \
		const u32 xyzw = (VU0.code >> 21) & 0xF; \
		emitAbsFmac(fs, ft, xyzw); \
	}

// ============================================================================
//  FTOI / ITOF — native NEON float/int conversions
//
//  FTOI<N>: dest.xyzw = (s32)(VF[fs].xyzw * 2^N)
//    ARM64: FCVTZS (vector, fixed-point) saturates at INT_MIN/INT_MAX — exact match.
//    For N=0 use the basic FCVTZS form; N>0 uses the fbits variant.
//
//  ITOF<N>: dest.xyzw = (float)(s32)(VF[fs].xyzw) / 2^N
//    ARM64: SCVTF (vector, fixed-point) — exact match.
//
//  Both use xyzw mask: only active components are written to VF[ft].
//  Full mask (0xF) uses a direct store; partial mask loads the existing reg
//  and overwrites only the active lanes.
// ============================================================================

static void emitFTOI(int fbits)
{
	const u32 ft   = (VU0.code >> 16) & 0x1F;
	const u32 fs   = (VU0.code >> 11) & 0x1F;
	const u32 xyzw = (VU0.code >> 21) & 0xF;
	if (ft == 0) return; // VF[0] is read-only (hardwired)
	armAsm->Ldr(q0, MemOperand(VU0_BASE_REG, vfOff(fs)));
	// Clamp inf/NaN to ±MAX_FLOAT before converting — keeps FCVTZS out of
	// NaN/INF territory. Gated on the port's event-test-opt signal (see
	// intentional divergence comment in emitBinaryFmac).
	const bool clampInputs = CHECK_VU_SIGN_OVERFLOW(0) || CHECK_VU_SIGN_OVERFLOW(1);
	if (clampInputs)
	{
		emitVuClampSetup();
		emitVuClampVec(v0.V4S());
	}
	// FCVTZS: float → s32 with fixed-point scale 2^fbits (truncate toward zero, saturates)
	if (fbits > 0)
		armAsm->Fcvtzs(v1.V4S(), v0.V4S(), fbits);
	else
		armAsm->Fcvtzs(v1.V4S(), v0.V4S());

	// NaN fixup (only when input clamp is off).
	//
	// Interpreter's floatToInt (VUops.cpp:879) checks post-multiply exponent:
	//   if ((uvalue & 0x7f800000) >= 0x4f000000)
	//       return (sign) ? INT_MIN : INT_MAX;
	// which covers finite overflow (matches ARM64 FCVTZS saturation), ±Inf
	// (also matches FCVTZS: +Inf→INT_MAX, -Inf→INT_MIN), AND NaN (where
	// ARM64 FCVTZS diverges by returning 0).
	//
	// Replicate the NaN case: overwrite NaN lanes with sign-based
	// INT_MIN/INT_MAX. Finite + Inf already match, so we only need a
	// NaN-specific mask (IEEE self-equality: NaN != NaN).
	if (!clampInputs)
	{
		armAsm->Fcmeq(v2.V4S(), v0.V4S(), v0.V4S()); // 1s for non-NaN, 0s for NaN
		armAsm->Mvn(v2.V16B(), v2.V16B());            // invert → 1s for NaN
		armAsm->Sshr(v3.V4S(), v0.V4S(), 31);        // v3 = 0 (pos) or 0xFFFFFFFF (neg)
		armAsm->Mov(w0, 0x7FFFFFFFu);
		armAsm->Dup(v4.V4S(), w0);
		armAsm->Eor(v3.V16B(), v3.V16B(), v4.V16B()); // v3 = 0x7FFFFFFF (pos) or 0x80000000 (neg)
		armAsm->Bit(v1.V16B(), v3.V16B(), v2.V16B()); // blend: NaN lanes ← v3
	}

	// Masked write to VF[ft]
	if (xyzw == 0xF)
	{
		armAsm->Str(q1, MemOperand(VU0_BASE_REG, vfOff(ft)));
	}
	else
	{
		armAsm->Ldr(q5, MemOperand(VU0_BASE_REG, vfOff(ft)));
		if (xyzw & 8) armAsm->Mov(v5.V4S(), 0, v1.V4S(), 0);
		if (xyzw & 4) armAsm->Mov(v5.V4S(), 1, v1.V4S(), 1);
		if (xyzw & 2) armAsm->Mov(v5.V4S(), 2, v1.V4S(), 2);
		if (xyzw & 1) armAsm->Mov(v5.V4S(), 3, v1.V4S(), 3);
		armAsm->Str(q5, MemOperand(VU0_BASE_REG, vfOff(ft)));
	}
}

static void emitITOF(int fbits)
{
	const u32 ft   = (VU0.code >> 16) & 0x1F;
	const u32 fs   = (VU0.code >> 11) & 0x1F;
	const u32 xyzw = (VU0.code >> 21) & 0xF;
	if (ft == 0) return;
	armAsm->Ldr(q0, MemOperand(VU0_BASE_REG, vfOff(fs)));
	// SCVTF: s32 → float with scale factor 2^-fbits (no input clamping needed)
	if (fbits > 0)
		armAsm->Scvtf(v1.V4S(), v0.V4S(), fbits);
	else
		armAsm->Scvtf(v1.V4S(), v0.V4S());
	// Masked write to VF[ft]
	if (xyzw == 0xF)
	{
		armAsm->Str(q1, MemOperand(VU0_BASE_REG, vfOff(ft)));
	}
	else
	{
		armAsm->Ldr(q5, MemOperand(VU0_BASE_REG, vfOff(ft)));
		if (xyzw & 8) armAsm->Mov(v5.V4S(), 0, v1.V4S(), 0);
		if (xyzw & 4) armAsm->Mov(v5.V4S(), 1, v1.V4S(), 1);
		if (xyzw & 2) armAsm->Mov(v5.V4S(), 2, v1.V4S(), 2);
		if (xyzw & 1) armAsm->Mov(v5.V4S(), 3, v1.V4S(), 3);
		armAsm->Str(q5, MemOperand(VU0_BASE_REG, vfOff(ft)));
	}
}

// ============================================================================
//  C wrapper helpers for upper ops that need flag-state writes
//  (CLIP, OPMULA, OPMSUB). Called via BL from JIT-compiled code.
//  x0 = VU0_BASE_REG at call time.
// ============================================================================

// CLIP: update clipflag by comparing |VF[ft].w| against VF[fs].xyz (±)
// Matches x86 microVU_Upper.inl:533-575 — denormal handling is on the Fs
// side (per-lane zero-out if exponent field is 0), NOT on Ft.w. The prior
// arm64 version substituted Ft.w with 0x007fffff on denormal Ft, which
// diverged from x86: with Ft.w=denormal and Fs=normal, x86 reads Ft.w
// threshold as a tiny positive value and clips Fs; prior arm64 reads it as
// 0x007fffff and also clips Fs — match for the normal-Fs case. With
// Fs=denormal, x86 zeros Fs before compare (no clip); prior arm64 compared
// denormal Fs bits against 0x007fffff (possibly clipped). Aligned to x86
// now.
static void vu0_CLIP(VURegs* VU)
{
	const u32 fs = (VU->code >> 11) & 0x1F;
	const u32 ft = (VU->code >> 16) & 0x1F;

	// |Ft.w| bit pattern. x86: xPAND Ft, absclip. No denormal substitution
	// here — denormal Ft.w becomes a tiny positive threshold.
	const s32 value = static_cast<s32>(VU->VF[ft].i.w & 0x7fffffffu);

	// Per-lane Fs: zero out if the exponent field is 0 (denormal or ±0).
	// Matches x86 xPAND(t1, Fs, exponent) + xPCMP.EQD(t1, 0) + xPANDN(t1, Fs):
	// t1[i] = (Fs[i] exponent == 0) ? 0 : Fs[i]. Keeps denormal Fs lanes
	// from appearing "above |Ft.w|" when |Ft.w| is tiny.
	auto fsLane = [](u32 raw) -> u32 { return (raw & 0x7f800000u) ? raw : 0u; };
	const u32 fsx = fsLane(VU->VF[fs].i.x);
	const u32 fsy = fsLane(VU->VF[fs].i.y);
	const u32 fsz = fsLane(VU->VF[fs].i.z);

	VU->clipflag <<= 6;
	if (static_cast<s32>(fsx)               > value) VU->clipflag |= 0x01;
	if (static_cast<s32>(fsx ^ 0x80000000u) > value) VU->clipflag |= 0x02;
	if (static_cast<s32>(fsy)               > value) VU->clipflag |= 0x04;
	if (static_cast<s32>(fsy ^ 0x80000000u) > value) VU->clipflag |= 0x08;
	if (static_cast<s32>(fsz)               > value) VU->clipflag |= 0x10;
	if (static_cast<s32>(fsz ^ 0x80000000u) > value) VU->clipflag |= 0x20;
	VU->clipflag &= 0xFFFFFF;
}

// Sign-preserving operand clamp. Mirrors interp vuDouble() in VUops.cpp:440-459
// and x86 microVU mVUclamp3 (sign-preserving variant of mVUclamp2):
//   exp=0x00 (denormal/±0): flush to ±0, preserve sign.
//   exp=0xFF (Inf/NaN):     if CHECK_VU_SIGN_OVERFLOW → ±MAX_FLOAT with sign.
//   otherwise: passthrough.
// FTZ mode at the hardware FPCR covers the denormal case on the arithmetic
// path, but Inf/NaN still need the software clamp — the hardware lets them
// propagate. OPMULA/OPMSUB without this clamp corrupt physics accumulators
// in games that feed NaN/Inf into the cross-product pipeline (MGS2).
static __fi float vu0OperandClamp(u32 f)
{
	const u32 exp = f & 0x7f800000u;
	if (exp == 0)
		f &= 0x80000000u;
	else if (exp == 0x7f800000u && (CHECK_VU_SIGN_OVERFLOW(0) || CHECK_VU_SIGN_OVERFLOW(1)))
		f = (f & 0x80000000u) | 0x7f7fffffu;
	return *reinterpret_cast<float*>(&f);
}

// OPMULA: ACC.xyz = cross-like multiply (fs.y*ft.z, fs.z*ft.x, fs.x*ft.y)
// x86 microVU (microVU_Upper.inl:422-445) emits SSE_MULPS on the shuffled
// operands. SSE_MULPS is wrapped by clampOp (microVU_Misc.inl:537-543) which
// calls mVUclamp3 on BOTH operands before the multiply — so x86 DOES apply
// operand clamping, prior comment here was wrong. Mirror that with the
// vu0OperandClamp helper.
// Note: VU_MACw_CLEAR is NOT called (see M-2 audit note).
static void vu0_OPMULA(VURegs* VU)
{
	const u32 fs = (VU->code >> 11) & 0x1F;
	const u32 ft = (VU->code >> 16) & 0x1F;
	VU->ACC.i.x = VU_MACx_UPDATE(VU, vu0OperandClamp(VU->VF[fs].i.y) * vu0OperandClamp(VU->VF[ft].i.z));
	VU->ACC.i.y = VU_MACy_UPDATE(VU, vu0OperandClamp(VU->VF[fs].i.z) * vu0OperandClamp(VU->VF[ft].i.x));
	VU->ACC.i.z = VU_MACz_UPDATE(VU, vu0OperandClamp(VU->VF[fs].i.x) * vu0OperandClamp(VU->VF[ft].i.y));
	VU_STAT_UPDATE(VU);
}

// OPMSUB: VF[fd].xyz = ACC.xyz - cross-like multiply (fs.y*ft.z, fs.z*ft.x, fs.x*ft.y)
// x86 microVU (microVU_Upper.inl:447-474) emits SSE_MULPS + SSE_SUBPS. Both go
// through clampOp (microVU_Misc.inl:537-543), which calls mVUclamp3 on every
// operand (Fs, Ft, and ACC) before the op. Prior comment claiming x86 skipped
// operand clamping was wrong.
// VF[0] writes are discarded (flags still updated).
static void vu0_OPMSUB(VURegs* VU)
{
	const u32 fs = (VU->code >> 11) & 0x1F;
	const u32 ft = (VU->code >> 16) & 0x1F;
	const u32 fd = (VU->code >>  6) & 0x1F;
	const float ftx = vu0OperandClamp(VU->VF[ft].i.x);
	const float fty = vu0OperandClamp(VU->VF[ft].i.y);
	const float ftz = vu0OperandClamp(VU->VF[ft].i.z);
	const float fsx = vu0OperandClamp(VU->VF[fs].i.x);
	const float fsy = vu0OperandClamp(VU->VF[fs].i.y);
	const float fsz = vu0OperandClamp(VU->VF[fs].i.z);
	const float accx = vu0OperandClamp(VU->ACC.i.x);
	const float accy = vu0OperandClamp(VU->ACC.i.y);
	const float accz = vu0OperandClamp(VU->ACC.i.z);
	u32 rx = VU_MACx_UPDATE(VU, accx - fsy * ftz);
	u32 ry = VU_MACy_UPDATE(VU, accy - fsz * ftx);
	u32 rz = VU_MACz_UPDATE(VU, accz - fsx * fty);
	if (fd != 0)
	{
		VU->VF[fd].i.x = rx;
		VU->VF[fd].i.y = ry;
		VU->VF[fd].i.z = rz;
	}
	VU_STAT_UPDATE(VU);
}

// ============================================================================
//  Per-instruction interp stub toggles (1 = interp, 0 = native)
// ============================================================================

#ifdef INTERP_VU0_UPPER
// Group toggle: force all to interpreter
#define ISTUB_VU0_ADDx    1
#define ISTUB_VU0_ADDy    1
#define ISTUB_VU0_ADDz    1
#define ISTUB_VU0_ADDw    1
#define ISTUB_VU0_ADDq    1
#define ISTUB_VU0_ADDi    1
#define ISTUB_VU0_ADD     1
#define ISTUB_VU0_ADDAx   1
#define ISTUB_VU0_ADDAy   1
#define ISTUB_VU0_ADDAz   1
#define ISTUB_VU0_ADDAw   1
#define ISTUB_VU0_ADDAq   1
#define ISTUB_VU0_ADDAi   1
#define ISTUB_VU0_ADDA    1
#define ISTUB_VU0_SUBx    1
#define ISTUB_VU0_SUBy    1
#define ISTUB_VU0_SUBz    1
#define ISTUB_VU0_SUBw    1
#define ISTUB_VU0_SUBq    1
#define ISTUB_VU0_SUBi    1
#define ISTUB_VU0_SUB     1
#define ISTUB_VU0_SUBAx   1
#define ISTUB_VU0_SUBAy   1
#define ISTUB_VU0_SUBAz   1
#define ISTUB_VU0_SUBAw   1
#define ISTUB_VU0_SUBAq   1
#define ISTUB_VU0_SUBAi   1
#define ISTUB_VU0_SUBA    1
#define ISTUB_VU0_MULx    1
#define ISTUB_VU0_MULy    1
#define ISTUB_VU0_MULz    1
#define ISTUB_VU0_MULw    1
#define ISTUB_VU0_MULq    1
#define ISTUB_VU0_MULi    1
#define ISTUB_VU0_MUL     1
#define ISTUB_VU0_MULAx   1
#define ISTUB_VU0_MULAy   1
#define ISTUB_VU0_MULAz   1
#define ISTUB_VU0_MULAw   1
#define ISTUB_VU0_MULAq   1
#define ISTUB_VU0_MULAi   1
#define ISTUB_VU0_MULA    1
#define ISTUB_VU0_MADDx   1
#define ISTUB_VU0_MADDy   1
#define ISTUB_VU0_MADDz   1
#define ISTUB_VU0_MADDw   1
#define ISTUB_VU0_MADDq   1
#define ISTUB_VU0_MADDi   1
#define ISTUB_VU0_MADD    1
#define ISTUB_VU0_MADDAx  1
#define ISTUB_VU0_MADDAy  1
#define ISTUB_VU0_MADDAz  1
#define ISTUB_VU0_MADDAw  1
#define ISTUB_VU0_MADDAq  1
#define ISTUB_VU0_MADDAi  1
#define ISTUB_VU0_MADDA   1
#define ISTUB_VU0_MSUBx   1
#define ISTUB_VU0_MSUBy   1
#define ISTUB_VU0_MSUBz   1
#define ISTUB_VU0_MSUBw   1
#define ISTUB_VU0_MSUBq   1
#define ISTUB_VU0_MSUBi   1
#define ISTUB_VU0_MSUB    1
#define ISTUB_VU0_MSUBAx  1
#define ISTUB_VU0_MSUBAy  1
#define ISTUB_VU0_MSUBAz  1
#define ISTUB_VU0_MSUBAw  1
#define ISTUB_VU0_MSUBAq  1
#define ISTUB_VU0_MSUBAi  1
#define ISTUB_VU0_MSUBA   1
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
#define ISTUB_VU0_ABS     1
#define ISTUB_VU0_CLIP    1
#define ISTUB_VU0_OPMULA  1
#define ISTUB_VU0_OPMSUB  1
#define ISTUB_VU0_NOP     1
#define ISTUB_VU0_FTOI0   1
#define ISTUB_VU0_FTOI4   1
#define ISTUB_VU0_FTOI12  1
#define ISTUB_VU0_FTOI15  1
#define ISTUB_VU0_ITOF0   1
#define ISTUB_VU0_ITOF4   1
#define ISTUB_VU0_ITOF12  1
#define ISTUB_VU0_ITOF15  1
#else
// Per-instruction control: set to 0 to enable native ARM64 codegen
#define ISTUB_VU0_ADDx    0
#define ISTUB_VU0_ADDy    0
#define ISTUB_VU0_ADDz    0
#define ISTUB_VU0_ADDw    0
#define ISTUB_VU0_ADDq    0
#define ISTUB_VU0_ADDi    0
#define ISTUB_VU0_ADD     0
#define ISTUB_VU0_ADDAx   0
#define ISTUB_VU0_ADDAy   0
#define ISTUB_VU0_ADDAz   0
#define ISTUB_VU0_ADDAw   0
#define ISTUB_VU0_ADDAq   0
#define ISTUB_VU0_ADDAi   0
#define ISTUB_VU0_ADDA    0
#define ISTUB_VU0_SUBx    0
#define ISTUB_VU0_SUBy    0
#define ISTUB_VU0_SUBz    0
#define ISTUB_VU0_SUBw    0
#define ISTUB_VU0_SUBq    0
#define ISTUB_VU0_SUBi    0
#define ISTUB_VU0_SUB     0
#define ISTUB_VU0_SUBAx   0
#define ISTUB_VU0_SUBAy   0
#define ISTUB_VU0_SUBAz   0
#define ISTUB_VU0_SUBAw   0
#define ISTUB_VU0_SUBAq   0
#define ISTUB_VU0_SUBAi   0
#define ISTUB_VU0_SUBA    0
#define ISTUB_VU0_MULx    0
#define ISTUB_VU0_MULy    0
#define ISTUB_VU0_MULz    0
#define ISTUB_VU0_MULw    0
#define ISTUB_VU0_MULq    0
#define ISTUB_VU0_MULi    0
#define ISTUB_VU0_MUL     0
#define ISTUB_VU0_MULAx   0
#define ISTUB_VU0_MULAy   0
#define ISTUB_VU0_MULAz   0
#define ISTUB_VU0_MULAw   0
#define ISTUB_VU0_MULAq   0
#define ISTUB_VU0_MULAi   0
#define ISTUB_VU0_MULA    0
#define ISTUB_VU0_MADDx   0
#define ISTUB_VU0_MADDy   0
#define ISTUB_VU0_MADDz   0
#define ISTUB_VU0_MADDw   0
#define ISTUB_VU0_MADDq   0
#define ISTUB_VU0_MADDi   0
#define ISTUB_VU0_MADD    0
#define ISTUB_VU0_MADDAx  0
#define ISTUB_VU0_MADDAy  0
#define ISTUB_VU0_MADDAz  0
#define ISTUB_VU0_MADDAw  0
#define ISTUB_VU0_MADDAq  0
#define ISTUB_VU0_MADDAi  0
#define ISTUB_VU0_MADDA   0
#define ISTUB_VU0_MSUBx   0
#define ISTUB_VU0_MSUBy   0
#define ISTUB_VU0_MSUBz   0
#define ISTUB_VU0_MSUBw   0
#define ISTUB_VU0_MSUBq   0
#define ISTUB_VU0_MSUBi   0
#define ISTUB_VU0_MSUB    0
#define ISTUB_VU0_MSUBAx  0
#define ISTUB_VU0_MSUBAy  0
#define ISTUB_VU0_MSUBAz  0
#define ISTUB_VU0_MSUBAw  0
#define ISTUB_VU0_MSUBAq  0
#define ISTUB_VU0_MSUBAi  0
#define ISTUB_VU0_MSUBA   0
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
#define ISTUB_VU0_ABS     0
#define ISTUB_VU0_CLIP    0
#define ISTUB_VU0_OPMULA  0
#define ISTUB_VU0_OPMSUB  0
#define ISTUB_VU0_NOP     0
#define ISTUB_VU0_FTOI0   0
#define ISTUB_VU0_FTOI4   0
#define ISTUB_VU0_FTOI12  0
#define ISTUB_VU0_FTOI15  0
#define ISTUB_VU0_ITOF0   0
#define ISTUB_VU0_ITOF4   0
#define ISTUB_VU0_ITOF12  0
#define ISTUB_VU0_ITOF15  0
#endif

// ============================================================================
//  Code-emitter macros: called at block-compile time.
//
//  VU0.code is set by CompileBlock before each of these is called.
// ============================================================================

// INTERP path (ISTUB=1): emit BL to the interpreter function via opcode table.
#define REC_VU0_UPPER_INTERP(name) \
	void recVU0_##name() { \
		armEmitCall(reinterpret_cast<const void*>(VU0_UPPER_OPCODE[VU0.code & 0x3f])); \
	}

// C-wrapper path (ISTUB=0): emit BL to a vu0_* C helper that bypasses the
// interpreter table and owns its own flag/state logic.
#define REC_VU0_UPPER_CALL(name) \
	void recVU0_##name() { \
		armAsm->Mov(x0, VU0_BASE_REG); \
		armEmitCall(reinterpret_cast<const void*>(vu0_##name)); \
	}

// ============================================================================
//  ADD family — VF[fd] = VF[fs] + VF[ft] (broadcast variants)
// ============================================================================

#if ISTUB_VU0_ADDx
REC_VU0_UPPER_INTERP(ADDx)
#else
FMAC_BINARY_BC(ADDx, 0, 0, false)
#endif

#if ISTUB_VU0_ADDy
REC_VU0_UPPER_INTERP(ADDy)
#else
FMAC_BINARY_BC(ADDy, 0, 1, false)
#endif

#if ISTUB_VU0_ADDz
REC_VU0_UPPER_INTERP(ADDz)
#else
FMAC_BINARY_BC(ADDz, 0, 2, false)
#endif

#if ISTUB_VU0_ADDw
REC_VU0_UPPER_INTERP(ADDw)
#else
FMAC_BINARY_BC(ADDw, 0, 3, false)
#endif

#if ISTUB_VU0_ADDq
REC_VU0_UPPER_INTERP(ADDq)
#else
FMAC_BINARY_Q(ADDq, 0, false)
#endif

#if ISTUB_VU0_ADDi
REC_VU0_UPPER_INTERP(ADDi)
#else
FMAC_BINARY_I(ADDi, 0, false)
#endif

#if ISTUB_VU0_ADD
REC_VU0_UPPER_INTERP(ADD)
#else
FMAC_BINARY_FULL(ADD, 0, false)
#endif

// ============================================================================
//  ADDA family — ACC = VF[fs] + VF[ft] (broadcast variants)
// ============================================================================

#if ISTUB_VU0_ADDAx
REC_VU0_UPPER_INTERP(ADDAx)
#else
FMAC_BINARY_BC(ADDAx, 0, 0, true)
#endif

#if ISTUB_VU0_ADDAy
REC_VU0_UPPER_INTERP(ADDAy)
#else
FMAC_BINARY_BC(ADDAy, 0, 1, true)
#endif

#if ISTUB_VU0_ADDAz
REC_VU0_UPPER_INTERP(ADDAz)
#else
FMAC_BINARY_BC(ADDAz, 0, 2, true)
#endif

#if ISTUB_VU0_ADDAw
REC_VU0_UPPER_INTERP(ADDAw)
#else
FMAC_BINARY_BC(ADDAw, 0, 3, true)
#endif

#if ISTUB_VU0_ADDAq
REC_VU0_UPPER_INTERP(ADDAq)
#else
FMAC_BINARY_Q(ADDAq, 0, true)
#endif

#if ISTUB_VU0_ADDAi
REC_VU0_UPPER_INTERP(ADDAi)
#else
FMAC_BINARY_I(ADDAi, 0, true)
#endif

#if ISTUB_VU0_ADDA
REC_VU0_UPPER_INTERP(ADDA)
#else
FMAC_BINARY_FULL(ADDA, 0, true)
#endif

// ============================================================================
//  SUB family
// ============================================================================

#if ISTUB_VU0_SUBx
REC_VU0_UPPER_INTERP(SUBx)
#else
FMAC_BINARY_BC(SUBx, 1, 0, false)
#endif

#if ISTUB_VU0_SUBy
REC_VU0_UPPER_INTERP(SUBy)
#else
FMAC_BINARY_BC(SUBy, 1, 1, false)
#endif

#if ISTUB_VU0_SUBz
REC_VU0_UPPER_INTERP(SUBz)
#else
FMAC_BINARY_BC(SUBz, 1, 2, false)
#endif

#if ISTUB_VU0_SUBw
REC_VU0_UPPER_INTERP(SUBw)
#else
FMAC_BINARY_BC(SUBw, 1, 3, false)
#endif

#if ISTUB_VU0_SUBq
REC_VU0_UPPER_INTERP(SUBq)
#else
FMAC_BINARY_Q(SUBq, 1, false)
#endif

#if ISTUB_VU0_SUBi
REC_VU0_UPPER_INTERP(SUBi)
#else
FMAC_BINARY_I(SUBi, 1, false)
#endif

#if ISTUB_VU0_SUB
REC_VU0_UPPER_INTERP(SUB)
#else
FMAC_BINARY_FULL(SUB, 1, false)
#endif

// ============================================================================
//  SUBA family
// ============================================================================

#if ISTUB_VU0_SUBAx
REC_VU0_UPPER_INTERP(SUBAx)
#else
FMAC_BINARY_BC(SUBAx, 1, 0, true)
#endif

#if ISTUB_VU0_SUBAy
REC_VU0_UPPER_INTERP(SUBAy)
#else
FMAC_BINARY_BC(SUBAy, 1, 1, true)
#endif

#if ISTUB_VU0_SUBAz
REC_VU0_UPPER_INTERP(SUBAz)
#else
FMAC_BINARY_BC(SUBAz, 1, 2, true)
#endif

#if ISTUB_VU0_SUBAw
REC_VU0_UPPER_INTERP(SUBAw)
#else
FMAC_BINARY_BC(SUBAw, 1, 3, true)
#endif

#if ISTUB_VU0_SUBAq
REC_VU0_UPPER_INTERP(SUBAq)
#else
FMAC_BINARY_Q(SUBAq, 1, true)
#endif

#if ISTUB_VU0_SUBAi
REC_VU0_UPPER_INTERP(SUBAi)
#else
FMAC_BINARY_I(SUBAi, 1, true)
#endif

#if ISTUB_VU0_SUBA
REC_VU0_UPPER_INTERP(SUBA)
#else
FMAC_BINARY_FULL(SUBA, 1, true)
#endif

// ============================================================================
//  MUL family
// ============================================================================

#if ISTUB_VU0_MULx
REC_VU0_UPPER_INTERP(MULx)
#else
FMAC_BINARY_BC(MULx, 2, 0, false)
#endif

#if ISTUB_VU0_MULy
REC_VU0_UPPER_INTERP(MULy)
#else
FMAC_BINARY_BC(MULy, 2, 1, false)
#endif

#if ISTUB_VU0_MULz
REC_VU0_UPPER_INTERP(MULz)
#else
FMAC_BINARY_BC(MULz, 2, 2, false)
#endif

#if ISTUB_VU0_MULw
REC_VU0_UPPER_INTERP(MULw)
#else
FMAC_BINARY_BC(MULw, 2, 3, false)
#endif

#if ISTUB_VU0_MULq
REC_VU0_UPPER_INTERP(MULq)
#else
FMAC_BINARY_Q(MULq, 2, false)
#endif

#if ISTUB_VU0_MULi
REC_VU0_UPPER_INTERP(MULi)
#else
FMAC_BINARY_I(MULi, 2, false)
#endif

#if ISTUB_VU0_MUL
REC_VU0_UPPER_INTERP(MUL)
#else
FMAC_BINARY_FULL(MUL, 2, false)
#endif

// ============================================================================
//  MULA family
// ============================================================================

#if ISTUB_VU0_MULAx
REC_VU0_UPPER_INTERP(MULAx)
#else
FMAC_BINARY_BC(MULAx, 2, 0, true)
#endif

#if ISTUB_VU0_MULAy
REC_VU0_UPPER_INTERP(MULAy)
#else
FMAC_BINARY_BC(MULAy, 2, 1, true)
#endif

#if ISTUB_VU0_MULAz
REC_VU0_UPPER_INTERP(MULAz)
#else
FMAC_BINARY_BC(MULAz, 2, 2, true)
#endif

#if ISTUB_VU0_MULAw
REC_VU0_UPPER_INTERP(MULAw)
#else
FMAC_BINARY_BC(MULAw, 2, 3, true)
#endif

#if ISTUB_VU0_MULAq
REC_VU0_UPPER_INTERP(MULAq)
#else
FMAC_BINARY_Q(MULAq, 2, true)
#endif

#if ISTUB_VU0_MULAi
REC_VU0_UPPER_INTERP(MULAi)
#else
FMAC_BINARY_I(MULAi, 2, true)
#endif

#if ISTUB_VU0_MULA
REC_VU0_UPPER_INTERP(MULA)
#else
FMAC_BINARY_FULL(MULA, 2, true)
#endif

// ============================================================================
//  MADD family
// ============================================================================

#if ISTUB_VU0_MADDx
REC_VU0_UPPER_INTERP(MADDx)
#else
FMAC_TERNARY_BC(MADDx, false, 0, false)
#endif

#if ISTUB_VU0_MADDy
REC_VU0_UPPER_INTERP(MADDy)
#else
FMAC_TERNARY_BC(MADDy, false, 1, false)
#endif

#if ISTUB_VU0_MADDz
REC_VU0_UPPER_INTERP(MADDz)
#else
FMAC_TERNARY_BC(MADDz, false, 2, false)
#endif

#if ISTUB_VU0_MADDw
REC_VU0_UPPER_INTERP(MADDw)
#else
FMAC_TERNARY_BC(MADDw, false, 3, false)
#endif

#if ISTUB_VU0_MADDq
REC_VU0_UPPER_INTERP(MADDq)
#else
FMAC_TERNARY_Q(MADDq, false, false)
#endif

#if ISTUB_VU0_MADDi
REC_VU0_UPPER_INTERP(MADDi)
#else
FMAC_TERNARY_I(MADDi, false, false)
#endif

#if ISTUB_VU0_MADD
REC_VU0_UPPER_INTERP(MADD)
#else
FMAC_TERNARY_FULL(MADD, false, false)
#endif

// ============================================================================
//  MADDA family
// ============================================================================

#if ISTUB_VU0_MADDAx
REC_VU0_UPPER_INTERP(MADDAx)
#else
FMAC_TERNARY_BC(MADDAx, false, 0, true)
#endif

#if ISTUB_VU0_MADDAy
REC_VU0_UPPER_INTERP(MADDAy)
#else
FMAC_TERNARY_BC(MADDAy, false, 1, true)
#endif

#if ISTUB_VU0_MADDAz
REC_VU0_UPPER_INTERP(MADDAz)
#else
FMAC_TERNARY_BC(MADDAz, false, 2, true)
#endif

#if ISTUB_VU0_MADDAw
REC_VU0_UPPER_INTERP(MADDAw)
#else
FMAC_TERNARY_BC(MADDAw, false, 3, true)
#endif

#if ISTUB_VU0_MADDAq
REC_VU0_UPPER_INTERP(MADDAq)
#else
FMAC_TERNARY_Q(MADDAq, false, true)
#endif

#if ISTUB_VU0_MADDAi
REC_VU0_UPPER_INTERP(MADDAi)
#else
FMAC_TERNARY_I(MADDAi, false, true)
#endif

#if ISTUB_VU0_MADDA
REC_VU0_UPPER_INTERP(MADDA)
#else
FMAC_TERNARY_FULL(MADDA, false, true)
#endif

// ============================================================================
//  MSUB family
// ============================================================================

#if ISTUB_VU0_MSUBx
REC_VU0_UPPER_INTERP(MSUBx)
#else
FMAC_TERNARY_BC(MSUBx, true, 0, false)
#endif

#if ISTUB_VU0_MSUBy
REC_VU0_UPPER_INTERP(MSUBy)
#else
FMAC_TERNARY_BC(MSUBy, true, 1, false)
#endif

#if ISTUB_VU0_MSUBz
REC_VU0_UPPER_INTERP(MSUBz)
#else
FMAC_TERNARY_BC(MSUBz, true, 2, false)
#endif

#if ISTUB_VU0_MSUBw
REC_VU0_UPPER_INTERP(MSUBw)
#else
FMAC_TERNARY_BC(MSUBw, true, 3, false)
#endif

#if ISTUB_VU0_MSUBq
REC_VU0_UPPER_INTERP(MSUBq)
#else
FMAC_TERNARY_Q(MSUBq, true, false)
#endif

#if ISTUB_VU0_MSUBi
REC_VU0_UPPER_INTERP(MSUBi)
#else
FMAC_TERNARY_I(MSUBi, true, false)
#endif

#if ISTUB_VU0_MSUB
REC_VU0_UPPER_INTERP(MSUB)
#else
FMAC_TERNARY_FULL(MSUB, true, false)
#endif

// ============================================================================
//  MSUBA family
// ============================================================================

#if ISTUB_VU0_MSUBAx
REC_VU0_UPPER_INTERP(MSUBAx)
#else
FMAC_TERNARY_BC(MSUBAx, true, 0, true)
#endif

#if ISTUB_VU0_MSUBAy
REC_VU0_UPPER_INTERP(MSUBAy)
#else
FMAC_TERNARY_BC(MSUBAy, true, 1, true)
#endif

#if ISTUB_VU0_MSUBAz
REC_VU0_UPPER_INTERP(MSUBAz)
#else
FMAC_TERNARY_BC(MSUBAz, true, 2, true)
#endif

#if ISTUB_VU0_MSUBAw
REC_VU0_UPPER_INTERP(MSUBAw)
#else
FMAC_TERNARY_BC(MSUBAw, true, 3, true)
#endif

#if ISTUB_VU0_MSUBAq
REC_VU0_UPPER_INTERP(MSUBAq)
#else
FMAC_TERNARY_Q(MSUBAq, true, true)
#endif

#if ISTUB_VU0_MSUBAi
REC_VU0_UPPER_INTERP(MSUBAi)
#else
FMAC_TERNARY_I(MSUBAi, true, true)
#endif

#if ISTUB_VU0_MSUBA
REC_VU0_UPPER_INTERP(MSUBA)
#else
FMAC_TERNARY_FULL(MSUBA, true, true)
#endif

// ============================================================================
//  MAX / MINI
// ============================================================================

#if ISTUB_VU0_MAXx
REC_VU0_UPPER_INTERP(MAXx)
#else
FMAC_MAXMINI_BC(MAXx, false, 0)
#endif

#if ISTUB_VU0_MAXy
REC_VU0_UPPER_INTERP(MAXy)
#else
FMAC_MAXMINI_BC(MAXy, false, 1)
#endif

#if ISTUB_VU0_MAXz
REC_VU0_UPPER_INTERP(MAXz)
#else
FMAC_MAXMINI_BC(MAXz, false, 2)
#endif

#if ISTUB_VU0_MAXw
REC_VU0_UPPER_INTERP(MAXw)
#else
FMAC_MAXMINI_BC(MAXw, false, 3)
#endif

#if ISTUB_VU0_MAXi
REC_VU0_UPPER_INTERP(MAXi)
#else
FMAC_MAXMINI_I(MAXi, false)
#endif

#if ISTUB_VU0_MAX
REC_VU0_UPPER_INTERP(MAX)
#else
FMAC_MAXMINI_FULL(MAX, false)
#endif

#if ISTUB_VU0_MINIx
REC_VU0_UPPER_INTERP(MINIx)
#else
FMAC_MAXMINI_BC(MINIx, true, 0)
#endif

#if ISTUB_VU0_MINIy
REC_VU0_UPPER_INTERP(MINIy)
#else
FMAC_MAXMINI_BC(MINIy, true, 1)
#endif

#if ISTUB_VU0_MINIz
REC_VU0_UPPER_INTERP(MINIz)
#else
FMAC_MAXMINI_BC(MINIz, true, 2)
#endif

#if ISTUB_VU0_MINIw
REC_VU0_UPPER_INTERP(MINIw)
#else
FMAC_MAXMINI_BC(MINIw, true, 3)
#endif

#if ISTUB_VU0_MINIi
REC_VU0_UPPER_INTERP(MINIi)
#else
FMAC_MAXMINI_I(MINIi, true)
#endif

#if ISTUB_VU0_MINI
REC_VU0_UPPER_INTERP(MINI)
#else
FMAC_MAXMINI_FULL(MINI, true)
#endif

// ============================================================================
//  ABS, CLIP, OPMULA, OPMSUB, NOP
// ============================================================================

#if ISTUB_VU0_ABS
REC_VU0_UPPER_INTERP(ABS)
#else
FMAC_ABS(ABS)
#endif

#if ISTUB_VU0_CLIP
REC_VU0_UPPER_INTERP(CLIP)
#else
REC_VU0_UPPER_CALL(CLIP)
#endif

#if ISTUB_VU0_OPMULA
REC_VU0_UPPER_INTERP(OPMULA)
#else
REC_VU0_UPPER_CALL(OPMULA)
#endif

#if ISTUB_VU0_OPMSUB
REC_VU0_UPPER_INTERP(OPMSUB)
#else
REC_VU0_UPPER_CALL(OPMSUB)
#endif

#if ISTUB_VU0_NOP
REC_VU0_UPPER_INTERP(NOP)
#else
void recVU0_NOP() { } // VU NOP: nothing to emit
#endif

// ============================================================================
//  FTOI / ITOF — float/int conversion
// ============================================================================

#if ISTUB_VU0_FTOI0
REC_VU0_UPPER_INTERP(FTOI0)
#else
void recVU0_FTOI0()  { emitFTOI(0);  }
#endif

#if ISTUB_VU0_FTOI4
REC_VU0_UPPER_INTERP(FTOI4)
#else
void recVU0_FTOI4()  { emitFTOI(4);  }
#endif

#if ISTUB_VU0_FTOI12
REC_VU0_UPPER_INTERP(FTOI12)
#else
void recVU0_FTOI12() { emitFTOI(12); }
#endif

#if ISTUB_VU0_FTOI15
REC_VU0_UPPER_INTERP(FTOI15)
#else
void recVU0_FTOI15() { emitFTOI(15); }
#endif

#if ISTUB_VU0_ITOF0
REC_VU0_UPPER_INTERP(ITOF0)
#else
void recVU0_ITOF0()  { emitITOF(0);  }
#endif

#if ISTUB_VU0_ITOF4
REC_VU0_UPPER_INTERP(ITOF4)
#else
void recVU0_ITOF4()  { emitITOF(4);  }
#endif

#if ISTUB_VU0_ITOF12
REC_VU0_UPPER_INTERP(ITOF12)
#else
void recVU0_ITOF12() { emitITOF(12); }
#endif

#if ISTUB_VU0_ITOF15
REC_VU0_UPPER_INTERP(ITOF15)
#else
void recVU0_ITOF15() { emitITOF(15); }
#endif

// ============================================================================
//  FD sub-table dispatch (0x3C-0x3F).
//
//  VU0.code is set at JIT compile time before this is called. We resolve the
//  exact rec function using (VU0.code & 3) as the sub-type and
//  (VU0.code >> 6) & 0x1F as the index within that sub-table.
//  This calls the already-implemented recVU0_* emitters directly so all their
//  ISTUB guards and NEON paths apply normally.
//
//  Unknown/reserved slots (indices >= 12) and 0x30-0x3B fall back to the
//  interpreter via VU0_UPPER_OPCODE.
// ============================================================================
static void recVU0_Upper_FD()
{
	using FDFn = void (*)();
	const u32 fd_type = VU0.code & 3;
	const u32 idx = (VU0.code >> 6) & 0x1F;

	// clang-format off
	static const FDFn fd_00[] = { // 0x3C
		recVU0_ADDAx,  recVU0_SUBAx,  recVU0_MADDAx, recVU0_MSUBAx,
		recVU0_ITOF0,  recVU0_FTOI0,  recVU0_MULAx,  recVU0_MULAq,
		recVU0_ADDAq,  recVU0_SUBAq,  recVU0_ADDA,   recVU0_SUBA,
	};
	static const FDFn fd_01[] = { // 0x3D
		recVU0_ADDAy,  recVU0_SUBAy,  recVU0_MADDAy, recVU0_MSUBAy,
		recVU0_ITOF4,  recVU0_FTOI4,  recVU0_MULAy,  recVU0_ABS,
		recVU0_MADDAq, recVU0_MSUBAq, recVU0_MADDA,  recVU0_MSUBA,
	};
	static const FDFn fd_10[] = { // 0x3E
		recVU0_ADDAz,  recVU0_SUBAz,  recVU0_MADDAz, recVU0_MSUBAz,
		recVU0_ITOF12, recVU0_FTOI12, recVU0_MULAz,  recVU0_MULAi,
		recVU0_ADDAi,  recVU0_SUBAi,  recVU0_MULA,   recVU0_OPMULA,
	};
	static const FDFn fd_11[] = { // 0x3F
		recVU0_ADDAw,  recVU0_SUBAw,  recVU0_MADDAw, recVU0_MSUBAw,
		recVU0_ITOF15, recVU0_FTOI15, recVU0_MULAw,  recVU0_CLIP,
		recVU0_MADDAi, recVU0_MSUBAi, nullptr,       recVU0_NOP,
	};
	// clang-format on

	const FDFn* table;
	switch (fd_type) {
		case 0:  table = fd_00; break;
		case 1:  table = fd_01; break;
		case 2:  table = fd_10; break;
		default: table = fd_11; break;
	}

	if (idx < 12 && table[idx] != nullptr)
		table[idx]();
	else
		armEmitCall(reinterpret_cast<const void*>(VU0_UPPER_OPCODE[VU0.code & 0x3f]));
}

// ============================================================================
//  recVU0_UpperTable[64]
//
//  Maps upper opcode index (upper_word & 0x3f) to a code-emitter function.
//  Layout mirrors VU0_UPPER_OPCODE in VUops.cpp.
//
//  Indices 0x30-0x3B are reserved/unknown in the VU ISA; recVU0_Upper_FD
//  falls back to VU0_UPPER_OPCODE for those (includes unknown-insn logging).
//  Indices 0x3C-0x3F are the FD sub-table; recVU0_Upper_FD does compile-time
//  dispatch into the appropriate recVU0_* emitter based on VU0.code.
// ============================================================================
using VU0RecFn = void (*)();

VU0RecFn recVU0_UpperTable[64] = {
	// 0x00-0x03: ADD broadcast
	recVU0_ADDx, recVU0_ADDy, recVU0_ADDz, recVU0_ADDw,
	// 0x04-0x07: SUB broadcast
	recVU0_SUBx, recVU0_SUBy, recVU0_SUBz, recVU0_SUBw,
	// 0x08-0x0B: MADD broadcast
	recVU0_MADDx, recVU0_MADDy, recVU0_MADDz, recVU0_MADDw,
	// 0x0C-0x0F: MSUB broadcast
	recVU0_MSUBx, recVU0_MSUBy, recVU0_MSUBz, recVU0_MSUBw,
	// 0x10-0x13: MAX broadcast
	recVU0_MAXx, recVU0_MAXy, recVU0_MAXz, recVU0_MAXw,
	// 0x14-0x17: MINI broadcast
	recVU0_MINIx, recVU0_MINIy, recVU0_MINIz, recVU0_MINIw,
	// 0x18-0x1B: MUL broadcast
	recVU0_MULx, recVU0_MULy, recVU0_MULz, recVU0_MULw,
	// 0x1C-0x1F: MULq, MAXi, MULi, MINIi
	recVU0_MULq, recVU0_MAXi, recVU0_MULi, recVU0_MINIi,
	// 0x20-0x23: ADDq, MADDq, ADDi, MADDi
	recVU0_ADDq, recVU0_MADDq, recVU0_ADDi, recVU0_MADDi,
	// 0x24-0x27: SUBq, MSUBq, SUBi, MSUBi
	recVU0_SUBq, recVU0_MSUBq, recVU0_SUBi, recVU0_MSUBi,
	// 0x28-0x2B: ADD, MADD, MUL, MAX
	recVU0_ADD, recVU0_MADD, recVU0_MUL, recVU0_MAX,
	// 0x2C-0x2F: SUB, MSUB, OPMSUB, MINI
	recVU0_SUB, recVU0_MSUB, recVU0_OPMSUB, recVU0_MINI,
	// 0x30-0x3B: reserved/unknown — delegate via FD dispatcher (falls through to interp)
	recVU0_Upper_FD, recVU0_Upper_FD, recVU0_Upper_FD, recVU0_Upper_FD,
	recVU0_Upper_FD, recVU0_Upper_FD, recVU0_Upper_FD, recVU0_Upper_FD,
	recVU0_Upper_FD, recVU0_Upper_FD, recVU0_Upper_FD, recVU0_Upper_FD,
	// 0x3C-0x3F: FD sub-table dispatch
	recVU0_Upper_FD, recVU0_Upper_FD, recVU0_Upper_FD, recVU0_Upper_FD,
};
