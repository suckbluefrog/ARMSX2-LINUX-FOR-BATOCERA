// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0
//
// ARM64 EE Recompiler — Trap Instructions
// TGE, TGEU, TLT, TLTU, TEQ, TNE, TGEI, TGEIU, TLTI, TLTIU, TEQI, TNEI

#include "Common.h"
#include "R5900OpcodeTables.h"
#include "arm64/arm64Emitter.h"

using namespace R5900;

// Per-instruction interp stub toggle. Set to 1 = interp, 0 = native.
// Traps are rarely executed by games, so interpreter stubs are fine.
#if defined(INTERP_TRAP) || defined(INTERP_EE)
#define ISTUB_TGEI     1
#define ISTUB_TGEIU    1
#define ISTUB_TLTI     1
#define ISTUB_TLTIU    1
#define ISTUB_TEQI     1
#define ISTUB_TNEI     1
#define ISTUB_TGE      1
#define ISTUB_TGEU     1
#define ISTUB_TLT      1
#define ISTUB_TLTU     1
#define ISTUB_TEQ      1
#define ISTUB_TNE      1
#else
#define ISTUB_TGEI     1   // traps are rare — keep as interp by default
#define ISTUB_TGEIU    1
#define ISTUB_TLTI     1
#define ISTUB_TLTIU    1
#define ISTUB_TEQI     1
#define ISTUB_TNEI     1
#define ISTUB_TGE      1
#define ISTUB_TGEU     1
#define ISTUB_TLT      1
#define ISTUB_TLTU     1
#define ISTUB_TEQ      1
#define ISTUB_TNE      1
#endif

namespace R5900 {
namespace Dynarec {
namespace OpcodeImpl {

// ============================================================================
//  Trap with immediate operand
//  Format: OP rs, imm
//  Compare GPR[rs] against sign-extended immediate, trap if condition met.
// ============================================================================

#if ISTUB_TGEI
void recTGEI()  { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TGEI); }
#else
void recTGEI()  { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TGEI); }
#endif

#if ISTUB_TGEIU
void recTGEIU() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TGEIU); }
#else
void recTGEIU() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TGEIU); }
#endif

#if ISTUB_TLTI
void recTLTI()  { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TLTI); }
#else
void recTLTI()  { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TLTI); }
#endif

#if ISTUB_TLTIU
void recTLTIU() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TLTIU); }
#else
void recTLTIU() { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TLTIU); }
#endif

#if ISTUB_TEQI
void recTEQI()  { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TEQI); }
#else
void recTEQI()  { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TEQI); }
#endif

#if ISTUB_TNEI
void recTNEI()  { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TNEI); }
#else
void recTNEI()  { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TNEI); }
#endif

// ============================================================================
//  Register traps
//  Format: OP rs, rt
//  Compare GPR[rs] against GPR[rt], trap if condition met.
// ============================================================================

#if ISTUB_TGE
void recTGE()   { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TGE); }
#else
void recTGE()   { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TGE); }
#endif

#if ISTUB_TGEU
void recTGEU()  { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TGEU); }
#else
void recTGEU()  { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TGEU); }
#endif

#if ISTUB_TLT
void recTLT()   { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TLT); }
#else
void recTLT()   { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TLT); }
#endif

#if ISTUB_TLTU
void recTLTU()  { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TLTU); }
#else
void recTLTU()  { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TLTU); }
#endif

#if ISTUB_TEQ
void recTEQ()   { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TEQ); }
#else
void recTEQ()   { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TEQ); }
#endif

#if ISTUB_TNE
void recTNE()   { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TNE); }
#else
void recTNE()   { armBranchCallInterpreter(R5900::Interpreter::OpcodeImpl::TNE); }
#endif

} // namespace OpcodeImpl
} // namespace Dynarec
} // namespace R5900
