#pragma once

// Poted from here 
// https://github.com/zama-ai/tfhe-rs/blob/main/tfhe/src/core_crypto/fft_impl/fft64/math/fft/x86.rs

//! For documentation on the various intrinsics used here, refer to Intel's intrinsics guide.
//! <https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html>
//!
//! currently we dispatch based on the availability of
//!  - avx+avx2(advanced vector extensions for 256 intrinsics)+fma(fused multiply add for complex
//!    multiplication, usually comes with avx+avx2),
//!  - or the availability of avx512f[+avx512dq(doubleword/quadword intrinsics for conversion of f64
//!    to/from i64. usually comes with avx512f on modern cpus)]
//!
//! more dispatch options may be added in the future


#include <immintrin.h>

namespace SPQLIOS{

// Convert a vector of f64 values to a vector of i64 values.
// Uses the magic-constant trick: adding 3*2^51 to a double in (-2^51, 2^51)
// places the integer value directly in the mantissa bits.
// This is ~3 ops vs ~13 ops for the bit-twiddling version.
// Requires |x| < 2^51 (satisfied for all practical TFHE parameters).
inline __m256i mm256_cvtpd_epi64(const __m256d x) {
    // 3 * 2^51 = 6755399441055744.0, bit pattern 0x4338000000000000
    const __m256d magic_d = _mm256_set1_pd(6755399441055744.0);
    const __m256i magic_i = _mm256_set1_epi64x(0x4338000000000000LL);
    // Add magic constant: shifts integer value into mantissa bits
    const __m256d adjusted = _mm256_add_pd(x, magic_d);
    // Subtract magic as integer to recover the signed integer value
    return _mm256_sub_epi64(_mm256_castpd_si256(adjusted), magic_i);
}

// Convert a vector of i64 values to a vector of f64 values.
// Uses the magic-constant trick (reverse direction): adding the magic integer
// then reinterpreting as double and subtracting the magic double.
// Requires |x| < 2^51 (satisfied for all practical TFHE parameters).
inline __m256d mm256_cvtepi64_pd(const __m256i x) {
    const __m256i magic_i = _mm256_set1_epi64x(0x4338000000000000LL);
    const __m256d magic_d = _mm256_set1_pd(6755399441055744.0); // 3 * 2^51
    // Add magic as integer, reinterpret as double, subtract magic as double
    return _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_add_epi64(x, magic_i)),
        magic_d
    );
}

// Convert a vector of i64 values to a vector of i32 values without AVX512.
// https://stackoverflow.com/questions/69408063/how-to-convert-int-64-to-int-32-with-avx-but-without-avx-512
// Slower
// __m128 mm256_cvtepi64_epi32_avx(const __m256i v)
// {
//    const __m256 vf = _mm256_castsi256_ps( v );      // free
//    const __m128 hi = _mm256_extractf128_ps(vf, 1);  // vextractf128
//    const __m128 lo = _mm256_castps256_ps128( vf );  // also free
//    // take the bottom 32 bits of each 64-bit chunk in lo and hi
//    const __m128 packed = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));  // shufps
//    //return _mm_castps_si128(packed);  // if you want
//    return packed;
// }

// 2x 256 -> 1x 256-bit result
__m256i pack64to32(__m256i a, __m256i b)
{
    // grab the 32-bit low halves of 64-bit elements into one vector
   __m256 combined = _mm256_shuffle_ps(_mm256_castsi256_ps(a),
                                       _mm256_castsi256_ps(b), _MM_SHUFFLE(2,0,2,0));
    // {b3,b2, a3,a2 | b1,b0, a1,a0}  from high to low

    // re-arrange pairs of 32-bit elements with vpermpd (or vpermq if you want)
    __m256d ordered = _mm256_permute4x64_pd(_mm256_castps_pd(combined), _MM_SHUFFLE(3,1,2,0));
    return _mm256_castpd_si256(ordered);
}


inline void convert_f64_to_u32(uint32_t* const res, const double* const real_inout_direct, const int32_t N) {
#ifdef USE_AVX512
    for (int32_t i = 0; i < N; i += 8) {
        const __m512d vals = _mm512_loadu_pd(&real_inout_direct[i]);
        const __m512i i64 = _mm512_cvtpd_epi64(vals);
        const __m256i i32 = _mm512_cvtepi64_epi32(i64);
        _mm256_storeu_si256((__m256i*)&res[i], i32);
    }
#else
    for (int32_t i = 0; i < N; i += 8) {
        // Load 4 double values
        const __m256d real_vals1 = _mm256_loadu_pd(&real_inout_direct[i]);
        const __m256d real_vals2 = _mm256_loadu_pd(&real_inout_direct[i + 4]);

        // Convert double to int64
        const __m256i int64_vals1 = mm256_cvtpd_epi64(real_vals1);
        const __m256i int64_vals2 = mm256_cvtpd_epi64(real_vals2);

        const __m256i packed32 = pack64to32(int64_vals1,int64_vals2);

        // Store the result
        _mm256_storeu_si256((__m256i*)&res[i], packed32);
    }
#endif
}
}