// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "Common.h"

#include "IPU/IPU.h"
#include "IPU/IPUdma.h"
#include "IPU/yuv2rgb.h"
#include "IPU/IPU_MultiISA.h"

MULTI_ISA_UNSHARED_START

void ipu_dither_reference(const macroblock_rgb32 &rgb32, macroblock_rgb16 &rgb16, int dte);

#if defined(_M_X86) || defined(ARCH_X86)
void ipu_dither_sse2(const macroblock_rgb32 &rgb32, macroblock_rgb16 &rgb16, int dte);
#endif

#if defined(ARCH_ARM64)
void ipu_dither_neon(const macroblock_rgb32 &rgb32, macroblock_rgb16 &rgb16, int dte);
#endif

__ri void ipu_dither(const macroblock_rgb32 &rgb32, macroblock_rgb16 &rgb16, int dte)
{
#if defined(_M_X86) || defined(ARCH_X86)
    ipu_dither_sse2(rgb32, rgb16, dte);
#elif defined(ARCH_ARM64)
    ipu_dither_neon(rgb32, rgb16, dte);
#else
    ipu_dither_reference(rgb32, rgb16, dte);
#endif
}

__ri void ipu_dither_reference(const macroblock_rgb32 &rgb32, macroblock_rgb16 &rgb16, int dte)
{
    if (dte) {
        // I'm guessing values are rounded down when clamping.
        const int dither_coefficient[4][4] = {
            {-4, 0, -3, 1},
            {2, -2, 3, -1},
            {-3, 1, -4, 0},
            {3, -1, 2, -2},
        };
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                const int dither = dither_coefficient[i & 3][j & 3];
                const int r = std::max(0, std::min(rgb32.c[i][j].r + dither, 255));
                const int g = std::max(0, std::min(rgb32.c[i][j].g + dither, 255));
                const int b = std::max(0, std::min(rgb32.c[i][j].b + dither, 255));

                rgb16.c[i][j].r = r >> 3;
                rgb16.c[i][j].g = g >> 3;
                rgb16.c[i][j].b = b >> 3;
                rgb16.c[i][j].a = rgb32.c[i][j].a == 0x40;
            }
        }
    } else {
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                rgb16.c[i][j].r = rgb32.c[i][j].r >> 3;
                rgb16.c[i][j].g = rgb32.c[i][j].g >> 3;
                rgb16.c[i][j].b = rgb32.c[i][j].b >> 3;
                rgb16.c[i][j].a = rgb32.c[i][j].a == 0x40;
            }
        }
    }
}

#if defined(_M_X86)

__ri void ipu_dither_sse2(const macroblock_rgb32 &rgb32, macroblock_rgb16 &rgb16, int dte)
{
    const __m128i alpha_test = _mm_set1_epi16(0x40);
    const __m128i dither_add_matrix[] = {
        _mm_setr_epi32(0x00000000, 0x00000000, 0x00000000, 0x00010101),
        _mm_setr_epi32(0x00020202, 0x00000000, 0x00030303, 0x00000000),
        _mm_setr_epi32(0x00000000, 0x00010101, 0x00000000, 0x00000000),
        _mm_setr_epi32(0x00030303, 0x00000000, 0x00020202, 0x00000000),
    };
    const __m128i dither_sub_matrix[] = {
        _mm_setr_epi32(0x00040404, 0x00000000, 0x00030303, 0x00000000),
        _mm_setr_epi32(0x00000000, 0x00020202, 0x00000000, 0x00010101),
        _mm_setr_epi32(0x00030303, 0x00000000, 0x00040404, 0x00000000),
        _mm_setr_epi32(0x00000000, 0x00010101, 0x00000000, 0x00020202),
    };
    for (int i = 0; i < 16; ++i) {
        const __m128i dither_add = dither_add_matrix[i & 3];
        const __m128i dither_sub = dither_sub_matrix[i & 3];
        for (int n = 0; n < 2; ++n) {
            __m128i rgba_8_0123 = _mm_load_si128(reinterpret_cast<const __m128i *>(&rgb32.c[i][n * 8]));
            __m128i rgba_8_4567 = _mm_load_si128(reinterpret_cast<const __m128i *>(&rgb32.c[i][n * 8 + 4]));

            // Dither and clamp
            if (dte) {
                rgba_8_0123 = _mm_adds_epu8(rgba_8_0123, dither_add);
                rgba_8_0123 = _mm_subs_epu8(rgba_8_0123, dither_sub);
                rgba_8_4567 = _mm_adds_epu8(rgba_8_4567, dither_add);
                rgba_8_4567 = _mm_subs_epu8(rgba_8_4567, dither_sub);
            }

            // Split into channel components and extend to 16 bits
            const __m128i rgba_16_0415 = _mm_unpacklo_epi8(rgba_8_0123, rgba_8_4567);
            const __m128i rgba_16_2637 = _mm_unpackhi_epi8(rgba_8_0123, rgba_8_4567);
            const __m128i rgba_32_0246 = _mm_unpacklo_epi8(rgba_16_0415, rgba_16_2637);
            const __m128i rgba_32_1357 = _mm_unpackhi_epi8(rgba_16_0415, rgba_16_2637);
            const __m128i rg_64_01234567 = _mm_unpacklo_epi8(rgba_32_0246, rgba_32_1357);
            const __m128i ba_64_01234567 = _mm_unpackhi_epi8(rgba_32_0246, rgba_32_1357);

            const __m128i zero = _mm_setzero_si128();
            __m128i r = _mm_unpacklo_epi8(rg_64_01234567, zero);
            __m128i g = _mm_unpackhi_epi8(rg_64_01234567, zero);
            __m128i b = _mm_unpacklo_epi8(ba_64_01234567, zero);
            __m128i a = _mm_unpackhi_epi8(ba_64_01234567, zero);

            // Create RGBA
            r = _mm_srli_epi16(r, 3);
            g = _mm_slli_epi16(_mm_srli_epi16(g, 3), 5);
            b = _mm_slli_epi16(_mm_srli_epi16(b, 3), 10);
            a = _mm_slli_epi16(_mm_cmpeq_epi16(a, alpha_test), 15);

            const __m128i rgba16 = _mm_or_si128(_mm_or_si128(r, g), _mm_or_si128(b, a));

            _mm_store_si128(reinterpret_cast<__m128i *>(&rgb16.c[i][n * 8]), rgba16);
        }
    }
}

#endif

#if defined(ARCH_ARM64)

// Dither coefficient table (matches ipu_dither_reference):
//   Row 0: -4, 0, -3, 1     Row 1:  2,-2, 3,-1
//   Row 2: -3, 1, -4, 0     Row 3:  3,-1, 2,-2
// Split into saturating-add (positive part) and saturating-sub (negative part),
// repeated twice for the 8-pixel column group [0..7] = [0..3, 0..3].
alignas(8) static const uint8_t s_dith_add[4][8] = {
    {0, 0, 0, 1,  0, 0, 0, 1},  // row mod 4 == 0
    {2, 0, 3, 0,  2, 0, 3, 0},  // row mod 4 == 1
    {0, 1, 0, 0,  0, 1, 0, 0},  // row mod 4 == 2
    {3, 0, 2, 0,  3, 0, 2, 0},  // row mod 4 == 3
};
alignas(8) static const uint8_t s_dith_sub[4][8] = {
    {4, 0, 3, 0,  4, 0, 3, 0},  // row mod 4 == 0
    {0, 2, 0, 1,  0, 2, 0, 1},  // row mod 4 == 1
    {3, 0, 4, 0,  3, 0, 4, 0},  // row mod 4 == 2
    {0, 1, 0, 2,  0, 1, 0, 2},  // row mod 4 == 3
};

// Pack separate u8 R/G/B/A lanes into rgb16_t (r:5, g:5, b:5, a:1) words.
// Input: r/g/b already shifted >> 3; a_bit is 0 or 1.
static __fi uint16x8_t pack_rgb16(uint8x8_t r, uint8x8_t g, uint8x8_t b, uint8x8_t a_bit)
{
    uint16x8_t r16 = vmovl_u8(r);
    uint16x8_t g16 = vshlq_n_u16(vmovl_u8(g), 5);
    uint16x8_t b16 = vshlq_n_u16(vmovl_u8(b), 10);
    uint16x8_t a16 = vshlq_n_u16(vmovl_u8(a_bit), 15);
    return vorrq_u16(vorrq_u16(r16, g16), vorrq_u16(b16, a16));
}

__ri void ipu_dither_neon(const macroblock_rgb32 &rgb32, macroblock_rgb16 &rgb16, int dte)
{
    const uint8x8_t alpha_ref = vdup_n_u8(0x40);

    for (int i = 0; i < 16; ++i)
    {
        for (int n = 0; n < 2; ++n)
        {
            // Load 8 RGBA pixels deinterleaved: val[0]=R, val[1]=G, val[2]=B, val[3]=A
            uint8x8x4_t rgba = vld4_u8(reinterpret_cast<const uint8_t*>(&rgb32.c[i][n * 8]));

            if (dte)
            {
                // Apply dither to R, G, B (same coefficient for all channels).
                // Saturating add/sub clamps to [0,255] naturally.
                uint8x8_t dadd = vld1_u8(s_dith_add[i & 3]);
                uint8x8_t dsub = vld1_u8(s_dith_sub[i & 3]);
                rgba.val[0] = vqadd_u8(vqsub_u8(rgba.val[0], dsub), dadd);
                rgba.val[1] = vqadd_u8(vqsub_u8(rgba.val[1], dsub), dadd);
                rgba.val[2] = vqadd_u8(vqsub_u8(rgba.val[2], dsub), dadd);
            }

            // Shift channels to 5-bit precision
            uint8x8_t r = vshr_n_u8(rgba.val[0], 3);
            uint8x8_t g = vshr_n_u8(rgba.val[1], 3);
            uint8x8_t b = vshr_n_u8(rgba.val[2], 3);

            // Alpha bit: 1 where A == 0x40, else 0
            uint8x8_t a_bit = vshr_n_u8(vceq_u8(rgba.val[3], alpha_ref), 7);

            uint16x8_t out = pack_rgb16(r, g, b, a_bit);
            vst1q_u16(reinterpret_cast<uint16_t*>(&rgb16.c[i][n * 8]), out);
        }
    }
}

#endif // ARCH_ARM64

MULTI_ISA_UNSHARED_END
