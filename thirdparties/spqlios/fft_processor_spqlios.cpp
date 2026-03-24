#include <cassert>
#include <cmath>
#include <cstring>
#include<cstdint>

#include<params.hpp>

#include "x86.h"
#include "fft_processor_spqlios.h"

// Constructor, destructor, thread_local instances, and C interface (fft/ifft/new_*_table)
// are in stockham_fft_avx2.cpp. This file contains only the execute_* methods.

using namespace std;

void FFT_Processor_Spqlios::execute_reverse_uint(double *res, const uint32_t *a) {
    #ifdef USE_AVX512
    {
        double *dst = res;
        const uint32_t *ait = a;
        const uint32_t *aend = a + N;
        // __asm__ __volatile__ (
        // "0:\n"
        //         "vmovupd (%1),%%xmm0\n"
        //         "vcvtudq2pd %%xmm0,%%ymm1\n"
        //         "vmovapd %%ymm1,(%0)\n"
        //         "addq $16,%1\n"
        //         "addq $32,%0\n"
        //         "cmpq %2,%1\n"
        //         "jb 0b\n"
        // : "=r"(dst), "=r"(ait), "=r"(aend)
        // : "0"(dst), "1"(ait), "2"(aend)
        // : "%xmm0", "%ymm1", "memory"
        // );
        __asm__ __volatile__ (
        "0:\n"
                "vmovdqu (%1),%%ymm0\n"
                "vcvtudq2pd %%ymm0,%%zmm1\n"
                "vmovapd %%zmm1,(%0)\n"
                "addq $32,%1\n"
                "addq $64,%0\n"
                "cmpq %2,%1\n"
                "jb 0b\n"
        : "=r"(dst), "=r"(ait), "=r"(aend)
        : "0"(dst), "1"(ait), "2"(aend)
        : "%ymm0", "%zmm1", "memory"
        );
    }
    #else
    for (int32_t i=0; i<N; i++) res[i]=(double)a[i];
    #endif
    ifft(tables_reverse, res);
}

void FFT_Processor_Spqlios::execute_reverse_int(double *res, const int32_t *a) {
    //for (int32_t i=0; i<N; i++) real_inout_rev[i]=(double)a[i];
    {
        double *dst = res;
        const int32_t *ait = a;
        const int32_t *aend = a + N;
        #ifdef USE_AVX512
        __asm__ __volatile__ (
        "0:\n"
            "vmovdqu (%1),%%ymm0\n"         // Load 8 int32_t values from `ait` into ymm0
            "vcvtdq2pd %%ymm0,%%zmm1\n"     // Convert 8 int32_t values to 8 double-precision values
            "vmovapd %%zmm1,(%0)\n"         // Store the result (8 doubles) in `dst`
            "addq $32,%1\n"                 // Increment `ait` by 32 bytes (8 int32_t values)
            "addq $64,%0\n"                 // Increment `dst` by 64 bytes (8 double-precision values)
            "cmpq %2,%1\n"                  // Compare `ait` with `aend`
            "jb 0b\n"                       // Jump back if `ait < aend`
            : "=r"(dst), "=r"(ait), "=r"(aend)
            : "0"(dst), "1"(ait), "2"(aend)
            : "%ymm0", "%zmm1", "memory"
        );
        #else
        __asm__ __volatile__ (
        "0:\n"
                "vmovupd (%1),%%xmm0\n"
                "vcvtdq2pd %%xmm0,%%ymm1\n"
                "vmovapd %%ymm1,(%0)\n"
                "addq $16,%1\n"
                "addq $32,%0\n"
                "cmpq %2,%1\n"
                "jb 0b\n"
        : "=r"(dst), "=r"(ait), "=r"(aend)
        : "0"(dst), "1"(ait), "2"(aend)
        : "%xmm0", "%ymm1", "memory"
        );
        #endif
    }
    ifft(tables_reverse, res);
}

extern "C" void ifft_from_i32(const void *tables, double *out, const int32_t *in_re, const int32_t *in_im);

void FFT_Processor_Spqlios::execute_reverse_torus32(double *res, const uint32_t *a) {
    // Fused: convert int32→double + inverse twist + IFFT in one pipeline
    const int32_t *aa = (const int32_t *)a;
    ifft_from_i32(tables_reverse, res, aa, aa + Ns2);
}

void FFT_Processor_Spqlios::execute_reverse_torus64(double* res, const uint64_t* a) {
    const int64_t *aa = (const int64_t *)a;
    for (int i=0; i<N; i+=4) {
        __m256i vi = _mm256_loadu_si256((const __m256i*)(aa + i));
        __m256d vd = SPQLIOS::mm256_cvtepi64_pd(vi);
        _mm256_storeu_pd(res + i, vd);
    }
    ifft(tables_reverse,res);
}

void FFT_Processor_Spqlios::execute_reverse_torus64_uint(
    double *res, const uint64_t *a)
{
    for (int i = 0; i < N; i++) res[i] = static_cast<double>(a[i]);
    ifft(tables_reverse, res);
}

void FFT_Processor_Spqlios::execute_direct_torus32(uint32_t *res, const double *a) {
    // 2/N scaling is baked into the Stockham forward twist twiddles.
    double *ap = const_cast<double *>(a);
    fft(tables_direct, ap);
    SPQLIOS::convert_f64_to_u32(res, ap, N);
}

void FFT_Processor_Spqlios::execute_direct_torus32_add(uint32_t *res, const double *a) {
    // 2/N scaling is baked into the Stockham forward twist twiddles.
    double *ap = const_cast<double *>(a);
    fft(tables_direct, ap);
    SPQLIOS::convert_f64_add_u32(res, ap, N);
}

void FFT_Processor_Spqlios::execute_direct_torus32_q(uint32_t *res, const double *a, const uint32_t q) {
    memcpy(real_inout_direct, a, N * sizeof(double));
    fft(tables_direct, real_inout_direct);
    for (int32_t i = 0; i < N; i++) res[i] = uint32_t((int64_t(real_inout_direct[i])%q+q)%q);
}

void FFT_Processor_Spqlios::execute_direct_torus32_rescale(uint32_t *res, const double *a, const double D) {
    memcpy(real_inout_direct, a, N * sizeof(double));
    fft(tables_direct, real_inout_direct);
    for (int32_t i = 0; i < N; i++) res[i] = static_cast<uint32_t>(int64_t(real_inout_direct[i]/D));
}

void FFT_Processor_Spqlios::execute_direct_torus32_rescale_clpx(
    uint32_t *res, const double *a, const double q, const uint32_t plain_modulus) {
    memcpy(real_inout_direct, a, N * sizeof(double));
    fft(tables_direct, real_inout_direct);
    for (int32_t i = 0; i < N; i++) {
        if (i == 0)
            res[i] = static_cast<uint32_t>(std::llround(-real_inout_direct[N-1]/q - real_inout_direct[0]*plain_modulus/q));
        else
            res[i] = static_cast<uint32_t>(std::llround(real_inout_direct[i-1]/q - real_inout_direct[i]*plain_modulus/q));
    }
}

// AVX2 vectorized IEEE754 double→int64 conversion
static inline void f64_to_i64_avx2(uint64_t* res, const double* a, int N) {
    const __m256i vmask0 = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFll);
    const __m256i vmask1 = _mm256_set1_epi64x(0x0010000000000000ll);
    const __m256i vexpmask = _mm256_set1_epi64x(0x7FFll);
    const __m256i voffset = _mm256_set1_epi64x(1075);
    const __m256i vzero = _mm256_setzero_si256();
    const uint64_t* vals = (const uint64_t*)a;
    for (int i = 0; i < N; i += 4) {
        __m256i raw = _mm256_loadu_si256((const __m256i*)(vals + i));
        __m256i mantissa = _mm256_or_si256(_mm256_and_si256(raw, vmask0), vmask1);
        __m256i expo = _mm256_and_si256(_mm256_srli_epi64(raw, 52), vexpmask);
        __m256i shift = _mm256_sub_epi64(expo, voffset);
        __m256i neg_shift = _mm256_sub_epi64(voffset, expo);
        __m256i val2 = _mm256_or_si256(
            _mm256_sllv_epi64(mantissa, shift),
            _mm256_srlv_epi64(mantissa, neg_shift));
        __m256i sign = _mm256_srli_epi64(raw, 63);
        __m256i sign_mask = _mm256_sub_epi64(vzero, sign);
        __m256i result = _mm256_sub_epi64(_mm256_xor_si256(val2, sign_mask), sign_mask);
        _mm256_storeu_si256((__m256i*)(res + i), result);
    }
}

static inline void f64_to_i64_add_avx2(uint64_t* res, const double* a, int N) {
    const __m256i vmask0 = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFll);
    const __m256i vmask1 = _mm256_set1_epi64x(0x0010000000000000ll);
    const __m256i vexpmask = _mm256_set1_epi64x(0x7FFll);
    const __m256i voffset = _mm256_set1_epi64x(1075);
    const __m256i vzero = _mm256_setzero_si256();
    const uint64_t* vals = (const uint64_t*)a;
    for (int i = 0; i < N; i += 4) {
        __m256i raw = _mm256_loadu_si256((const __m256i*)(vals + i));
        __m256i mantissa = _mm256_or_si256(_mm256_and_si256(raw, vmask0), vmask1);
        __m256i expo = _mm256_and_si256(_mm256_srli_epi64(raw, 52), vexpmask);
        __m256i shift = _mm256_sub_epi64(expo, voffset);
        __m256i neg_shift = _mm256_sub_epi64(voffset, expo);
        __m256i val2 = _mm256_or_si256(
            _mm256_sllv_epi64(mantissa, shift),
            _mm256_srlv_epi64(mantissa, neg_shift));
        __m256i sign = _mm256_srli_epi64(raw, 63);
        __m256i sign_mask = _mm256_sub_epi64(vzero, sign);
        __m256i result = _mm256_sub_epi64(_mm256_xor_si256(val2, sign_mask), sign_mask);
        __m256i prev = _mm256_loadu_si256((const __m256i*)(res + i));
        _mm256_storeu_si256((__m256i*)(res + i), _mm256_add_epi64(prev, result));
    }
}

void FFT_Processor_Spqlios::execute_direct_torus64(uint64_t* res, double* a) {
    // 2/N scaling baked into Stockham forward twist twiddles.
    fft(tables_direct, a);
    f64_to_i64_avx2(res, a, N);
}

void FFT_Processor_Spqlios::execute_direct_torus64_add(uint64_t* res, double* a) {
    // 2/N scaling baked into Stockham forward twist twiddles.
    fft(tables_direct, a);
    f64_to_i64_add_avx2(res, a, N);
}

void FFT_Processor_Spqlios::execute_direct_torus64_rescale(uint64_t* res, const double* a, const double D) {
    memcpy(real_inout_direct, a, N * sizeof(double));
    fft(tables_direct, real_inout_direct);
    for (int i=0; i<N; i++) res[i] = uint64_t(std::round(real_inout_direct[i]/D));
}

void FFT_Processor_Spqlios::execute_direct_torus64_rescale_clpx(
    uint64_t *res, const double *a, const uint32_t plain_modulus) {
    memcpy(real_inout_direct, a, N * sizeof(double));
    fft(tables_direct, real_inout_direct);
    f64_to_i64_avx2(res, real_inout_direct, N);
}

// Destructor and thread_local instances are in stockham_fft_avx2.cpp
