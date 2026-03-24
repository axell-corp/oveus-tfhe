// Stockham radix-4 FFT with AVX2 FMA — split real/imaginary layout.
//
// Drop-in replacement for spqlios-fft-fma.s / spqlios-ifft-fma.s:
//   void fft(const void *tables, double *data);
//   void ifft(const void *tables, double *data);
//   void *new_fft_table(int32_t nn);
//   void *new_ifft_table(int32_t nn);
//   double *fft_table_get_buffer(const void *);
//   double *ifft_table_get_buffer(const void *);
//
// Data layout: re[0..ns4-1], im[0..ns4-1]  where ns4 = nn/2 = n/4
// Twiddle layout (per stage): [cos0..3|sin0..3|cos4..7|sin4..7|...] × 3 sets
//
// Key optimizations over the assembly radix-2 version:
//   - Radix-4: 4 passes instead of 8+1 for ns4=256 (N=512)
//   - Forward twist bakes in 2/N scaling (eliminates separate scaling loop)
//   - Fused last butterfly + twist (saves one full data sweep)
//   - Fused first butterfly + inverse twist (saves one full data sweep)

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <immintrin.h>

#include <params.hpp>
#include "fft_processor_spqlios.h"

// ── Trig helpers ────────────────────────────────────────────────────────────
static double accurate_cos(int32_t i, int32_t n) {
    i = ((i % n) + n) % n;
    if (i >= 3*n/4) return cos(2.*M_PI*(n-i)/double(n));
    if (i >= 2*n/4) return -cos(2.*M_PI*(i-n/2)/double(n));
    if (i >= 1*n/4) return -cos(2.*M_PI*(n/2-i)/double(n));
    return cos(2.*M_PI*i/double(n));
}
static double accurate_sin(int32_t i, int32_t n) {
    i = ((i % n) + n) % n;
    if (i >= 3*n/4) return -sin(2.*M_PI*(n-i)/double(n));
    if (i >= 2*n/4) return -sin(2.*M_PI*(i-n/2)/double(n));
    if (i >= 1*n/4) return sin(2.*M_PI*(n/2-i)/double(n));
    return sin(2.*M_PI*i/double(n));
}

// ── Complex multiply helper (split format, AVX2) ───────────────────────────
// res_re = a_re*w_re - a_im*w_im
// res_im = a_im*w_re + a_re*w_im
static inline void cmul_split_avx2(
    __m256d a_re, __m256d a_im,
    __m256d w_re, __m256d w_im,
    __m256d &r_re, __m256d &r_im)
{
    r_re = _mm256_mul_pd(a_re, w_re);
    r_re = _mm256_fnmadd_pd(a_im, w_im, r_re);
    r_im = _mm256_mul_pd(a_im, w_re);
    r_im = _mm256_fmadd_pd(a_re, w_im, r_im);
}

// ── Table structure ─────────────────────────────────────────────────────────
// For Stockham radix-4 with ns4 complex values:
//   - nstages = log4(ns4) butterfly stages
//   - Per stage (except last): 3*(ns4/4) twiddle entries (w1,w2,w3 per group)
//     stored as blocks of 8 doubles: [cos0..3|sin0..3] for each w
//   - Forward twist: ns4 entries with 2/N baked in
//   - Inverse twist: ns4 entries (no scaling)
struct STOCKHAM_R4_PRECOMP {
    int32_t n;        // = 2*nn
    int32_t ns4;      // = nn/2 = number of complex values
    double *trig_fwd; // butterfly twiddles + fused twist for forward
    double *trig_inv; // fused twist + butterfly twiddles for inverse
    double *data;     // working buffer
    double *scratch;  // scratch buffer for out-of-place
    void *buf;        // raw allocation
};

// ── Stockham radix-4 core (forward DIF) ─────────────────────────────────────
// Operates on ns4 complex values in split re/im layout.
// src_re/im → dst_re/im (out of place, caller ping-pongs buffers)
static void stockham_r4_fwd_pass(
    int32_t ns4, int32_t q, int32_t s,
    const double *tw,   // 3 twiddle sets: w1[ns4/4], w2[ns4/4], w3[ns4/4]
    const double *src_re, const double *src_im,
    double *dst_re, double *dst_im)
{
    // tw layout per group of 4: [w1_cos×4|w1_sin×4|w2_cos×4|w2_sin×4|w3_cos×4|w3_sin×4]
    // = 24 doubles per 4-group
    const __m256d neg = _mm256_set1_pd(-0.0);

    for (int32_t j = 0; j < s; j++) {
        const int32_t stride = q * s;
        // Load twiddles for this j: 3 complex twiddles broadcast to all 4 lanes
        __m256d w1_re, w1_im, w2_re, w2_im, w3_re, w3_im;
        if (j == 0) {
            // w1=w2=w3=1 for j=0
            w1_re = w2_re = w3_re = _mm256_set1_pd(1.0);
            w1_im = w2_im = w3_im = _mm256_setzero_pd();
        } else {
            const double *twj = tw + j * 6; // 6 doubles per j: w1re,w1im,w2re,w2im,w3re,w3im
            w1_re = _mm256_set1_pd(twj[0]); w1_im = _mm256_set1_pd(twj[1]);
            w2_re = _mm256_set1_pd(twj[2]); w2_im = _mm256_set1_pd(twj[3]);
            w3_re = _mm256_set1_pd(twj[4]); w3_im = _mm256_set1_pd(twj[5]);
        }

        for (int32_t p = 0; p < q; p += 4) {
            const int32_t idx = j + s * p;
            // Load a,b,c,d from 4 quarter-arrays
            __m256d a_re = _mm256_loadu_pd(src_re + idx);
            __m256d a_im = _mm256_loadu_pd(src_im + idx);
            __m256d b_re = _mm256_loadu_pd(src_re + idx + stride);
            __m256d b_im = _mm256_loadu_pd(src_im + idx + stride);
            __m256d c_re = _mm256_loadu_pd(src_re + idx + 2*stride);
            __m256d c_im = _mm256_loadu_pd(src_im + idx + 2*stride);
            __m256d d_re = _mm256_loadu_pd(src_re + idx + 3*stride);
            __m256d d_im = _mm256_loadu_pd(src_im + idx + 3*stride);

            // Radix-4 butterfly
            __m256d apc_re = _mm256_add_pd(a_re, c_re);
            __m256d apc_im = _mm256_add_pd(a_im, c_im);
            __m256d amc_re = _mm256_sub_pd(a_re, c_re);
            __m256d amc_im = _mm256_sub_pd(a_im, c_im);
            __m256d bpd_re = _mm256_add_pd(b_re, d_re);
            __m256d bpd_im = _mm256_add_pd(b_im, d_im);
            __m256d bmd_re = _mm256_sub_pd(b_re, d_re);
            __m256d bmd_im = _mm256_sub_pd(b_im, d_im);

            // j*(b-d) for forward: [-bmd_im, bmd_re]
            __m256d jbmd_re = _mm256_xor_pd(bmd_im, neg);
            __m256d jbmd_im = bmd_re;

            // out0 = (a+c) + (b+d) — no twiddle
            int32_t out_base = q * 4 * j + p;
            _mm256_storeu_pd(dst_re + out_base, _mm256_add_pd(apc_re, bpd_re));
            _mm256_storeu_pd(dst_im + out_base, _mm256_add_pd(apc_im, bpd_im));

            if (j == 0) {
                // w1=w2=w3=1: skip multiply
                _mm256_storeu_pd(dst_re + out_base + q, _mm256_sub_pd(amc_re, jbmd_re));
                _mm256_storeu_pd(dst_im + out_base + q, _mm256_sub_pd(amc_im, jbmd_im));
                _mm256_storeu_pd(dst_re + out_base + 2*q, _mm256_sub_pd(apc_re, bpd_re));
                _mm256_storeu_pd(dst_im + out_base + 2*q, _mm256_sub_pd(apc_im, bpd_im));
                _mm256_storeu_pd(dst_re + out_base + 3*q, _mm256_add_pd(amc_re, jbmd_re));
                _mm256_storeu_pd(dst_im + out_base + 3*q, _mm256_add_pd(amc_im, jbmd_im));
            } else {
                // out1 = (amc - jbmd) * w1
                __m256d t1_re = _mm256_sub_pd(amc_re, jbmd_re);
                __m256d t1_im = _mm256_sub_pd(amc_im, jbmd_im);
                __m256d o1_re, o1_im;
                cmul_split_avx2(t1_re, t1_im, w1_re, w1_im, o1_re, o1_im);
                _mm256_storeu_pd(dst_re + out_base + q, o1_re);
                _mm256_storeu_pd(dst_im + out_base + q, o1_im);

                // out2 = (apc - bpd) * w2
                __m256d t2_re = _mm256_sub_pd(apc_re, bpd_re);
                __m256d t2_im = _mm256_sub_pd(apc_im, bpd_im);
                __m256d o2_re, o2_im;
                cmul_split_avx2(t2_re, t2_im, w2_re, w2_im, o2_re, o2_im);
                _mm256_storeu_pd(dst_re + out_base + 2*q, o2_re);
                _mm256_storeu_pd(dst_im + out_base + 2*q, o2_im);

                // out3 = (amc + jbmd) * w3
                __m256d t3_re = _mm256_add_pd(amc_re, jbmd_re);
                __m256d t3_im = _mm256_add_pd(amc_im, jbmd_im);
                __m256d o3_re, o3_im;
                cmul_split_avx2(t3_re, t3_im, w3_re, w3_im, o3_re, o3_im);
                _mm256_storeu_pd(dst_re + out_base + 3*q, o3_re);
                _mm256_storeu_pd(dst_im + out_base + 3*q, o3_im);
            }
        }
    }
}

// Inverse pass: same structure but conjugate j-multiply
static void stockham_r4_inv_pass(
    int32_t ns4, int32_t q, int32_t s,
    const double *tw,
    const double *src_re, const double *src_im,
    double *dst_re, double *dst_im)
{
    const __m256d neg = _mm256_set1_pd(-0.0);

    for (int32_t j = 0; j < s; j++) {
        const int32_t stride = q * s;
        __m256d w1_re, w1_im, w2_re, w2_im, w3_re, w3_im;
        if (j == 0) {
            w1_re = w2_re = w3_re = _mm256_set1_pd(1.0);
            w1_im = w2_im = w3_im = _mm256_setzero_pd();
        } else {
            const double *twj = tw + j * 6;
            w1_re = _mm256_set1_pd(twj[0]); w1_im = _mm256_set1_pd(twj[1]);
            w2_re = _mm256_set1_pd(twj[2]); w2_im = _mm256_set1_pd(twj[3]);
            w3_re = _mm256_set1_pd(twj[4]); w3_im = _mm256_set1_pd(twj[5]);
        }

        for (int32_t p = 0; p < q; p += 4) {
            const int32_t idx = j + s * p;
            __m256d a_re = _mm256_loadu_pd(src_re + idx);
            __m256d a_im = _mm256_loadu_pd(src_im + idx);
            __m256d b_re = _mm256_loadu_pd(src_re + idx + stride);
            __m256d b_im = _mm256_loadu_pd(src_im + idx + stride);
            __m256d c_re = _mm256_loadu_pd(src_re + idx + 2*stride);
            __m256d c_im = _mm256_loadu_pd(src_im + idx + 2*stride);
            __m256d d_re = _mm256_loadu_pd(src_re + idx + 3*stride);
            __m256d d_im = _mm256_loadu_pd(src_im + idx + 3*stride);

            __m256d apc_re = _mm256_add_pd(a_re, c_re);
            __m256d apc_im = _mm256_add_pd(a_im, c_im);
            __m256d amc_re = _mm256_sub_pd(a_re, c_re);
            __m256d amc_im = _mm256_sub_pd(a_im, c_im);
            __m256d bpd_re = _mm256_add_pd(b_re, d_re);
            __m256d bpd_im = _mm256_add_pd(b_im, d_im);
            __m256d bmd_re = _mm256_sub_pd(b_re, d_re);
            __m256d bmd_im = _mm256_sub_pd(b_im, d_im);

            // Inverse j-multiply: [bmd_im, -bmd_re]
            __m256d jbmd_re = bmd_im;
            __m256d jbmd_im = _mm256_xor_pd(bmd_re, neg);

            int32_t out_base = q * 4 * j + p;
            _mm256_storeu_pd(dst_re + out_base, _mm256_add_pd(apc_re, bpd_re));
            _mm256_storeu_pd(dst_im + out_base, _mm256_add_pd(apc_im, bpd_im));

            if (j == 0) {
                _mm256_storeu_pd(dst_re + out_base + q, _mm256_sub_pd(amc_re, jbmd_re));
                _mm256_storeu_pd(dst_im + out_base + q, _mm256_sub_pd(amc_im, jbmd_im));
                _mm256_storeu_pd(dst_re + out_base + 2*q, _mm256_sub_pd(apc_re, bpd_re));
                _mm256_storeu_pd(dst_im + out_base + 2*q, _mm256_sub_pd(apc_im, bpd_im));
                _mm256_storeu_pd(dst_re + out_base + 3*q, _mm256_add_pd(amc_re, jbmd_re));
                _mm256_storeu_pd(dst_im + out_base + 3*q, _mm256_add_pd(amc_im, jbmd_im));
            } else {
                __m256d t1_re = _mm256_sub_pd(amc_re, jbmd_re);
                __m256d t1_im = _mm256_sub_pd(amc_im, jbmd_im);
                __m256d o1_re, o1_im;
                cmul_split_avx2(t1_re, t1_im, w1_re, w1_im, o1_re, o1_im);
                _mm256_storeu_pd(dst_re + out_base + q, o1_re);
                _mm256_storeu_pd(dst_im + out_base + q, o1_im);

                __m256d t2_re = _mm256_sub_pd(apc_re, bpd_re);
                __m256d t2_im = _mm256_sub_pd(apc_im, bpd_im);
                __m256d o2_re, o2_im;
                cmul_split_avx2(t2_re, t2_im, w2_re, w2_im, o2_re, o2_im);
                _mm256_storeu_pd(dst_re + out_base + 2*q, o2_re);
                _mm256_storeu_pd(dst_im + out_base + 2*q, o2_im);

                __m256d t3_re = _mm256_add_pd(amc_re, jbmd_re);
                __m256d t3_im = _mm256_add_pd(amc_im, jbmd_im);
                __m256d o3_re, o3_im;
                cmul_split_avx2(t3_re, t3_im, w3_re, w3_im, o3_re, o3_im);
                _mm256_storeu_pd(dst_re + out_base + 3*q, o3_re);
                _mm256_storeu_pd(dst_im + out_base + 3*q, o3_im);
            }
        }
    }
}

// ── Full forward FFT (multi-pass + fused twist) ─────────────────────────────
static void stockham_fft_forward(
    int32_t ns4,
    const double *butterfly_tw,  // butterfly twiddles for all passes
    const double *twist_tw,      // twist twiddles (with 2/N baked in), split [cos|sin]
    double *re, double *im,
    double *sre, double *sim)
{
    double *cur_re = re, *cur_im = im;
    double *dst_re = sre, *dst_im = sim;
    const double *tw = butterfly_tw;

    int32_t q = ns4 / 4;
    int32_t s = 1;

    // All but the last pass: standard Stockham radix-4
    while (q >= 4) {
        stockham_r4_fwd_pass(ns4, q, s, tw, cur_re, cur_im, dst_re, dst_im);
        tw += s * 6;
        // Swap buffers
        double *t;
        t = cur_re; cur_re = dst_re; dst_re = t;
        t = cur_im; cur_im = dst_im; dst_im = t;
        s *= 4; q /= 4;
    }

    // Last pass: q=1, fused with twist
    // Each butterfly reads 4 values at stride s, writes 4 contiguous outputs.
    // Apply twist on output.
    {
        const __m256d neg = _mm256_set1_pd(-0.0);
        const double *tw_cos = twist_tw;
        const double *tw_sin = twist_tw + ns4;

        for (int32_t j = 0; j < s; j++) {
            // Twiddles for this butterfly
            __m256d w1_re, w1_im, w2_re, w2_im, w3_re, w3_im;
            if (j == 0) {
                w1_re = w2_re = w3_re = _mm256_set1_pd(1.0);
                w1_im = w2_im = w3_im = _mm256_setzero_pd();
            } else {
                const double *twj = tw + j * 6;
                w1_re = _mm256_set1_pd(twj[0]); w1_im = _mm256_set1_pd(twj[1]);
                w2_re = _mm256_set1_pd(twj[2]); w2_im = _mm256_set1_pd(twj[3]);
                w3_re = _mm256_set1_pd(twj[4]); w3_im = _mm256_set1_pd(twj[5]);
            }

            // q=1: only p=0 iteration, but we need 4 values at stride s
            // Gather 4 scalar values from stride-s positions
            double a_re_s[4], a_im_s[4], b_re_s[4], b_im_s[4];
            double c_re_s[4], c_im_s[4], d_re_s[4], d_im_s[4];
            // Not enough elements for vectorized gather, but this only runs
            // for the last pass where q=1
            a_re_s[0] = cur_re[j]; a_im_s[0] = cur_im[j];
            b_re_s[0] = cur_re[j + s]; b_im_s[0] = cur_im[j + s];
            c_re_s[0] = cur_re[j + 2*s]; c_im_s[0] = cur_im[j + 2*s];
            d_re_s[0] = cur_re[j + 3*s]; d_im_s[0] = cur_im[j + 3*s];

            // Butterfly (scalar for q=1)
            double apc_r = a_re_s[0] + c_re_s[0], apc_i = a_im_s[0] + c_im_s[0];
            double amc_r = a_re_s[0] - c_re_s[0], amc_i = a_im_s[0] - c_im_s[0];
            double bpd_r = b_re_s[0] + d_re_s[0], bpd_i = b_im_s[0] + d_im_s[0];
            double bmd_r = b_re_s[0] - d_re_s[0], bmd_i = b_im_s[0] - d_im_s[0];
            double jbmd_r = -bmd_i, jbmd_i = bmd_r; // forward j-multiply

            // out0 = (apc + bpd)
            double o0_r = apc_r + bpd_r, o0_i = apc_i + bpd_i;

            // out1 = (amc - jbmd) * w1
            double t1_r = amc_r - jbmd_r, t1_i = amc_i - jbmd_i;
            double o1_r, o1_i;
            if (j == 0) { o1_r = t1_r; o1_i = t1_i; }
            else {
                const double *twj = tw + j * 6;
                o1_r = t1_r * twj[0] - t1_i * twj[1];
                o1_i = t1_i * twj[0] + t1_r * twj[1];
            }

            // out2 = (apc - bpd) * w2
            double t2_r = apc_r - bpd_r, t2_i = apc_i - bpd_i;
            double o2_r, o2_i;
            if (j == 0) { o2_r = t2_r; o2_i = t2_i; }
            else {
                const double *twj = tw + j * 6;
                o2_r = t2_r * twj[2] - t2_i * twj[3];
                o2_i = t2_i * twj[2] + t2_r * twj[3];
            }

            // out3 = (amc + jbmd) * w3
            double t3_r = amc_r + jbmd_r, t3_i = amc_i + jbmd_i;
            double o3_r, o3_i;
            if (j == 0) { o3_r = t3_r; o3_i = t3_i; }
            else {
                const double *twj = tw + j * 6;
                o3_r = t3_r * twj[4] - t3_i * twj[5];
                o3_i = t3_i * twj[4] + t3_r * twj[5];
            }

            // Fused twist (with 2/N scaling baked in)
            int32_t out_base = 4 * j;
            double tc0 = tw_cos[out_base], ts0 = tw_sin[out_base];
            double tc1 = tw_cos[out_base+1], ts1 = tw_sin[out_base+1];
            double tc2 = tw_cos[out_base+2], ts2 = tw_sin[out_base+2];
            double tc3 = tw_cos[out_base+3], ts3 = tw_sin[out_base+3];

            dst_re[out_base]   = o0_r * tc0 - o0_i * ts0;
            dst_im[out_base]   = o0_i * tc0 + o0_r * ts0;
            dst_re[out_base+1] = o1_r * tc1 - o1_i * ts1;
            dst_im[out_base+1] = o1_i * tc1 + o1_r * ts1;
            dst_re[out_base+2] = o2_r * tc2 - o2_i * ts2;
            dst_im[out_base+2] = o2_i * tc2 + o2_r * ts2;
            dst_re[out_base+3] = o3_r * tc3 - o3_i * ts3;
            dst_im[out_base+3] = o3_i * tc3 + o3_r * ts3;
        }
    }

    // If result ended up in scratch buffer, copy back
    if (dst_re != re) {
        memcpy(re, dst_re, ns4 * sizeof(double));
        memcpy(im, dst_im, ns4 * sizeof(double));
    }
}

// ── Full inverse FFT (fused twist + multi-pass) ─────────────────────────────
static void stockham_fft_inverse(
    int32_t ns4,
    const double *twist_tw,       // inverse twist [cos|sin], split format
    const double *butterfly_tw,   // butterfly twiddles
    double *re, double *im,
    double *sre, double *sim)
{
    double *cur_re = re, *cur_im = im;
    double *dst_re = sre, *dst_im = sim;
    int32_t q = ns4 / 4;
    int32_t s = 1;

    // First pass: q=ns4/4, s=1. Fuse inverse twist with input load.
    {
        const double *tw_cos = twist_tw;
        const double *tw_sin = twist_tw + ns4;
        int32_t stride = q; // q * s = q

        // j=0 pass: twist + butterfly (no butterfly twiddle for j=0)
        for (int32_t p = 0; p < q; p += 4) {
            // Apply inverse twist to inputs
            __m256d tc0 = _mm256_loadu_pd(tw_cos + p);
            __m256d ts0 = _mm256_loadu_pd(tw_sin + p);
            __m256d tc1 = _mm256_loadu_pd(tw_cos + p + stride);
            __m256d ts1 = _mm256_loadu_pd(tw_sin + p + stride);
            __m256d tc2 = _mm256_loadu_pd(tw_cos + p + 2*stride);
            __m256d ts2 = _mm256_loadu_pd(tw_sin + p + 2*stride);
            __m256d tc3 = _mm256_loadu_pd(tw_cos + p + 3*stride);
            __m256d ts3 = _mm256_loadu_pd(tw_sin + p + 3*stride);

            __m256d raw_a_re = _mm256_loadu_pd(cur_re + p);
            __m256d raw_a_im = _mm256_loadu_pd(cur_im + p);
            __m256d a_re, a_im;
            cmul_split_avx2(raw_a_re, raw_a_im, tc0, ts0, a_re, a_im);

            __m256d raw_b_re = _mm256_loadu_pd(cur_re + p + stride);
            __m256d raw_b_im = _mm256_loadu_pd(cur_im + p + stride);
            __m256d b_re, b_im;
            cmul_split_avx2(raw_b_re, raw_b_im, tc1, ts1, b_re, b_im);

            __m256d raw_c_re = _mm256_loadu_pd(cur_re + p + 2*stride);
            __m256d raw_c_im = _mm256_loadu_pd(cur_im + p + 2*stride);
            __m256d c_re, c_im;
            cmul_split_avx2(raw_c_re, raw_c_im, tc2, ts2, c_re, c_im);

            __m256d raw_d_re = _mm256_loadu_pd(cur_re + p + 3*stride);
            __m256d raw_d_im = _mm256_loadu_pd(cur_im + p + 3*stride);
            __m256d d_re, d_im;
            cmul_split_avx2(raw_d_re, raw_d_im, tc3, ts3, d_re, d_im);

            // Radix-4 inverse butterfly
            __m256d neg = _mm256_set1_pd(-0.0);
            __m256d apc_re = _mm256_add_pd(a_re, c_re);
            __m256d apc_im = _mm256_add_pd(a_im, c_im);
            __m256d amc_re = _mm256_sub_pd(a_re, c_re);
            __m256d amc_im = _mm256_sub_pd(a_im, c_im);
            __m256d bpd_re = _mm256_add_pd(b_re, d_re);
            __m256d bpd_im = _mm256_add_pd(b_im, d_im);
            __m256d bmd_re = _mm256_sub_pd(b_re, d_re);
            __m256d bmd_im = _mm256_sub_pd(b_im, d_im);
            // Inverse j-multiply: [bmd_im, -bmd_re]
            __m256d jbmd_re = bmd_im;
            __m256d jbmd_im = _mm256_xor_pd(bmd_re, neg);

            _mm256_storeu_pd(dst_re + p, _mm256_add_pd(apc_re, bpd_re));
            _mm256_storeu_pd(dst_im + p, _mm256_add_pd(apc_im, bpd_im));
            _mm256_storeu_pd(dst_re + q + p, _mm256_sub_pd(amc_re, jbmd_re));
            _mm256_storeu_pd(dst_im + q + p, _mm256_sub_pd(amc_im, jbmd_im));
            _mm256_storeu_pd(dst_re + 2*q + p, _mm256_sub_pd(apc_re, bpd_re));
            _mm256_storeu_pd(dst_im + 2*q + p, _mm256_sub_pd(apc_im, bpd_im));
            _mm256_storeu_pd(dst_re + 3*q + p, _mm256_add_pd(amc_re, jbmd_re));
            _mm256_storeu_pd(dst_im + 3*q + p, _mm256_add_pd(amc_im, jbmd_im));
        }

        // Swap buffers
        double *t;
        t = cur_re; cur_re = dst_re; dst_re = t;
        t = cur_im; cur_im = dst_im; dst_im = t;

        q /= 4; // now q = ns4/16
        s = 4;
    }

    // Remaining passes: standard Stockham radix-4 inverse
    // Skip first stage's butterfly twiddles (s=1 → 1*6=6 doubles)
    const double *tw = butterfly_tw + 1 * 6;
    while (q >= 1) {
        if (q >= 4) {
            stockham_r4_inv_pass(ns4, q, s, tw, cur_re, cur_im, dst_re, dst_im);
        } else {
            // q=1 final pass (scalar gather, contiguous store)
            for (int32_t j = 0; j < s; j++) {
                double a_r = cur_re[j], a_i = cur_im[j];
                double b_r = cur_re[j+s], b_i = cur_im[j+s];
                double c_r = cur_re[j+2*s], c_i = cur_im[j+2*s];
                double d_r = cur_re[j+3*s], d_i = cur_im[j+3*s];

                double apc_r = a_r+c_r, apc_i = a_i+c_i;
                double amc_r = a_r-c_r, amc_i = a_i-c_i;
                double bpd_r = b_r+d_r, bpd_i = b_i+d_i;
                double bmd_r = b_r-d_r, bmd_i = b_i-d_i;
                // Inverse j: [bmd_im, -bmd_re]
                double jbmd_r = bmd_i, jbmd_i = -bmd_r;

                int32_t ob = 4*j;
                dst_re[ob] = apc_r + bpd_r; dst_im[ob] = apc_i + bpd_i;

                double o1_r = amc_r - jbmd_r, o1_i = amc_i - jbmd_i;
                double o2_r = apc_r - bpd_r, o2_i = apc_i - bpd_i;
                double o3_r = amc_r + jbmd_r, o3_i = amc_i + jbmd_i;

                if (j != 0) {
                    const double *twj = tw + j * 6;
                    double t;
                    t = o1_r*twj[0] - o1_i*twj[1]; o1_i = o1_i*twj[0] + o1_r*twj[1]; o1_r = t;
                    t = o2_r*twj[2] - o2_i*twj[3]; o2_i = o2_i*twj[2] + o2_r*twj[3]; o2_r = t;
                    t = o3_r*twj[4] - o3_i*twj[5]; o3_i = o3_i*twj[4] + o3_r*twj[5]; o3_r = t;
                }
                dst_re[ob+1] = o1_r; dst_im[ob+1] = o1_i;
                dst_re[ob+2] = o2_r; dst_im[ob+2] = o2_i;
                dst_re[ob+3] = o3_r; dst_im[ob+3] = o3_i;
            }
        }
        tw += s * 6;
        double *t;
        t = cur_re; cur_re = dst_re; dst_re = t;
        t = cur_im; cur_im = dst_im; dst_im = t;
        s *= 4; q /= 4;
    }

    // Copy back if needed
    if (cur_re != re) {
        memcpy(re, cur_re, ns4 * sizeof(double));
        memcpy(im, cur_im, ns4 * sizeof(double));
    }
}

// ── Table construction ──────────────────────────────────────────────────────
static STOCKHAM_R4_PRECOMP *build_tables(int32_t nn, bool is_forward) {
    int32_t n = 2 * nn;
    int32_t ns4 = nn / 2;

    auto *reps = new STOCKHAM_R4_PRECOMP;
    reps->n = n;
    reps->ns4 = ns4;

    // Count twiddle storage needed
    // Butterfly twiddles: for each stage, s entries × 6 doubles
    int32_t bf_doubles = 0;
    for (int32_t s = 1, q = ns4/4; q >= 1; s *= 4, q /= 4)
        bf_doubles += s * 6;

    // Twist twiddles: ns4 cos + ns4 sin = 2*ns4 doubles
    int32_t twist_doubles = 2 * ns4;

    int32_t total = bf_doubles + twist_doubles + 2 * ns4 + nn;
    reps->buf = aligned_alloc(64, total * sizeof(double));
    double *ptr = (double *)reps->buf;

    if (is_forward) {
        reps->trig_fwd = ptr;
        // Build butterfly twiddles
        double *bfp = ptr;
        for (int32_t s = 1, q = ns4/4; q >= 1; s *= 4, q /= 4) {
            for (int32_t j = 0; j < s; j++) {
                int32_t denom = 4 * s;
                // w1 = exp(-2πij/(4s)), w2 = exp(-2πi·2j/(4s)), w3 = exp(-2πi·3j/(4s))
                *bfp++ = accurate_cos(-j, denom);
                *bfp++ = accurate_sin(-j, denom);
                *bfp++ = accurate_cos(-2*j, denom);
                *bfp++ = accurate_sin(-2*j, denom);
                *bfp++ = accurate_cos(-3*j, denom);
                *bfp++ = accurate_sin(-3*j, denom);
            }
        }
        // Build twist twiddles with 2/N scaling baked in
        double scale = 2.0 / nn;
        double *twist = bfp;
        // cos part
        for (int32_t j = 0; j < ns4; j++)
            *bfp++ = scale * accurate_cos(-j, n);
        // sin part
        for (int32_t j = 0; j < ns4; j++)
            *bfp++ = scale * accurate_sin(-j, n);

        ptr = bfp;
    } else {
        reps->trig_inv = ptr;
        // Inverse twist twiddles (no scaling)
        double *twist = ptr;
        // cos part
        for (int32_t j = 0; j < ns4; j++)
            *ptr++ = accurate_cos(j, n);
        // sin part
        for (int32_t j = 0; j < ns4; j++)
            *ptr++ = accurate_sin(j, n);
        // Build butterfly twiddles (conjugate direction)
        for (int32_t s = 1, q = ns4/4; q >= 1; s *= 4, q /= 4) {
            for (int32_t j = 0; j < s; j++) {
                int32_t denom = 4 * s;
                *ptr++ = accurate_cos(j, denom);
                *ptr++ = accurate_sin(j, denom);
                *ptr++ = accurate_cos(2*j, denom);
                *ptr++ = accurate_sin(2*j, denom);
                *ptr++ = accurate_cos(3*j, denom);
                *ptr++ = accurate_sin(3*j, denom);
            }
        }
    }

    reps->scratch = ptr;
    ptr += ns4;         // scratch re
    ptr += ns4;         // scratch im (contiguous)
    reps->data = ptr;   // nn doubles for data buffer

    return reps;
}

// ── C interface ─────────────────────────────────────────────────────────────
extern "C" {

void *new_fft_table(int32_t nn) {
    return build_tables(nn, true);
}

void *new_ifft_table(int32_t nn) {
    return build_tables(nn, false);
}

double *fft_table_get_buffer(const void *tables) {
    return ((STOCKHAM_R4_PRECOMP *)tables)->data;
}

double *ifft_table_get_buffer(const void *tables) {
    return ((STOCKHAM_R4_PRECOMP *)tables)->data;
}

void fft(const void *tables, double *c) {
    auto *t = (STOCKHAM_R4_PRECOMP *)tables;
    int32_t ns4 = t->ns4;
    double *re = c;
    double *im = c + ns4;

    // Butterfly twiddles start at trig_fwd
    const double *bf_tw = t->trig_fwd;
    // Count butterfly doubles to find twist start
    int32_t bf_doubles = 0;
    for (int32_t s = 1, q = ns4/4; q >= 1; s *= 4, q /= 4)
        bf_doubles += s * 6;
    const double *twist_tw = t->trig_fwd + bf_doubles;

    stockham_fft_forward(ns4, bf_tw, twist_tw, re, im, t->scratch, t->scratch + ns4);
}

void ifft(const void *tables, double *c) {
    auto *t = (STOCKHAM_R4_PRECOMP *)tables;
    int32_t ns4 = t->ns4;
    double *re = c;
    double *im = c + ns4;

    const double *twist_tw = t->trig_inv;
    const double *bf_tw = t->trig_inv + 2 * ns4;

    stockham_fft_inverse(ns4, twist_tw, bf_tw, re, im, t->scratch, t->scratch + ns4);
}

// Model functions (for debugging, same interface)
void fft_model(const void *tables) {
    auto *t = (STOCKHAM_R4_PRECOMP *)tables;
    fft(tables, t->data);
}

void ifft_model(void *tables) {
    auto *t = (STOCKHAM_R4_PRECOMP *)tables;
    ifft(tables, t->data);
}

} // extern "C"

// ── FFT processor ───────────────────────────────────────────────────────────
static int32_t rev(int32_t x, int32_t M) {
    int32_t r = 0;
    for (int32_t j = M; j > 1; j /= 2) { r = 2*r+(x%2); x /= 2; }
    return r;
}

FFT_Processor_Spqlios::FFT_Processor_Spqlios(const int32_t N)
    : _2N(2*N), N(N), Ns2(N/2) {
    tables_direct = new_fft_table(N);
    tables_reverse = new_ifft_table(N);
    real_inout_direct = fft_table_get_buffer(tables_direct);
    imag_inout_direct = real_inout_direct + Ns2;
    reva = new int32_t[Ns2];
    cosomegaxminus1 = new double[2*_2N];
    sinomegaxminus1 = cosomegaxminus1 + _2N;
    int32_t r1 = rev(1, _2N), r3 = rev(3, _2N);
    for (int32_t ri = r1; ri < r3; ri++) reva[ri-r1] = rev(ri, _2N);
    for (int32_t j = 0; j < _2N; j++) {
        cosomegaxminus1[j] = cos(2*M_PI*j/_2N) - 1.;
        sinomegaxminus1[j] = sin(2*M_PI*j/_2N);
    }
}

FFT_Processor_Spqlios::~FFT_Processor_Spqlios() {
    auto *ft = (STOCKHAM_R4_PRECOMP *)tables_direct;
    auto *it = (STOCKHAM_R4_PRECOMP *)tables_reverse;
    free(ft->buf); delete ft;
    if (it != ft) { free(it->buf); delete it; }
    delete[] cosomegaxminus1;
    delete[] reva;
}

thread_local FFT_Processor_Spqlios fftplvl1(TFHEpp::lvl1param::n);
thread_local FFT_Processor_Spqlios fftplvl2(TFHEpp::lvl2param::n);
thread_local FFT_Processor_Spqlios fftplvl3(TFHEpp::lvl3param::n);
