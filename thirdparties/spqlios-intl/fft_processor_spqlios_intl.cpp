// Interleaved-format FFT using Stockham radix-4 algorithm.
//
// Data layout: [re0, im0, re1, im1, ...] (N doubles = N/2 complex values)
// Each YMM register holds 2 complex values (c64x2).
//
// Based on the Stockham auto-sort algorithm (same approach as OTFFT/tfhe-rs):
// - Out-of-place: alternates between data and scratch buffers
// - Radix-4: processes 4 inputs per butterfly, halving the number of passes
// - Mixed radix: uses radix-2 final pass when n is not a power of 4
// - mul_j optimization: multiplication by ±j is free (swap+negate)

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <immintrin.h>

#include <params.hpp>
#include "fft_processor_spqlios_intl.h"

// ── Trig helpers ─────────────────────────────────────────────────────────────

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

#ifdef USE_AVX512

// ── Complex arithmetic with AVX512 ──────────────────────────────────────────
// Each ZMM holds 4 complex values: [re0,im0, re1,im1, re2,im2, re3,im3]

static inline __m512d cmul512(__m512d a, __m512d w) {
    __m512d w_swap = _mm512_permute_pd(w, 0x55);        // swap re/im pairs
    __m512d a_re = _mm512_unpacklo_pd(a, a);             // broadcast re
    __m512d a_im = _mm512_unpackhi_pd(a, a);             // broadcast im
    return _mm512_fmaddsub_pd(a_re, w, _mm512_mul_pd(a_im, w_swap));
}

// Multiply by j: [re,im] → [-im,re]
static inline __m512d mul_j_fwd512(__m512d x) {
    __m512d swapped = _mm512_permute_pd(x, 0x55);
    return _mm512_xor_pd(swapped,
        _mm512_set_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0));
}

// Multiply by -j: [re,im] → [im,-re]
static inline __m512d mul_j_inv512(__m512d x) {
    __m512d swapped = _mm512_permute_pd(x, 0x55);
    return _mm512_xor_pd(swapped,
        _mm512_set_pd(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0));
}

// Radix-4 butterfly with twiddle (4 complex per ZMM)
static inline void radix4_dit_butterfly512(
    __m512d a, __m512d b, __m512d c, __m512d d,
    __m512d w1, __m512d w2, __m512d w3, bool fwd,
    __m512d &out0, __m512d &out1, __m512d &out2, __m512d &out3)
{
    __m512d apc = _mm512_add_pd(a, c);
    __m512d amc = _mm512_sub_pd(a, c);
    __m512d bpd = _mm512_add_pd(b, d);
    __m512d bmd = _mm512_sub_pd(b, d);
    __m512d jbmd = fwd ? mul_j_fwd512(bmd) : mul_j_inv512(bmd);

    out0 = _mm512_add_pd(apc, bpd);
    out1 = cmul512(_mm512_sub_pd(amc, jbmd), w1);
    out2 = cmul512(_mm512_sub_pd(apc, bpd), w2);
    out3 = cmul512(_mm512_add_pd(amc, jbmd), w3);
}

// Last radix-4 butterfly (no twiddle)
static inline void radix4_last_butterfly512(
    __m512d a, __m512d b, __m512d c, __m512d d, bool fwd,
    __m512d &out0, __m512d &out1, __m512d &out2, __m512d &out3)
{
    __m512d apc = _mm512_add_pd(a, c);
    __m512d amc = _mm512_sub_pd(a, c);
    __m512d bpd = _mm512_add_pd(b, d);
    __m512d bmd = _mm512_sub_pd(b, d);
    __m512d jbmd = fwd ? mul_j_fwd512(bmd) : mul_j_inv512(bmd);

    out0 = _mm512_add_pd(apc, bpd);
    out1 = _mm512_sub_pd(amc, jbmd);
    out2 = _mm512_sub_pd(apc, bpd);
    out3 = _mm512_add_pd(amc, jbmd);
}

#else  // AVX2

// ── Complex arithmetic with AVX2 ────────────────────────────────────────────

// Complex multiply: a * w
static inline __m256d cmul(__m256d a, __m256d w) {
    __m256d w_swap = _mm256_permute_pd(w, 0b0101);
    __m256d a_re = _mm256_unpacklo_pd(a, a);
    __m256d a_im = _mm256_unpackhi_pd(a, a);
    return _mm256_fmaddsub_pd(a_re, w, _mm256_mul_pd(a_im, w_swap));
}

// Multiply by j (imaginary unit): [re,im] → [-im,re]
static inline __m256d mul_j_fwd(__m256d x) {
    __m256d swapped = _mm256_permute_pd(x, 0b0101);
    return _mm256_xor_pd(swapped, _mm256_set_pd(0.0, -0.0, 0.0, -0.0));
}

// Multiply by -j: [re,im] → [im,-re]
static inline __m256d mul_j_inv(__m256d x) {
    __m256d swapped = _mm256_permute_pd(x, 0b0101);
    return _mm256_xor_pd(swapped, _mm256_set_pd(-0.0, 0.0, -0.0, 0.0));
}

// ── Stockham radix-4 DIF butterfly ──────────────────────────────────────────
// Processes 2 complex values per YMM.
// Input: a,b,c,d from 4 quarters of the source array
// Output: 4 results with twiddle factors applied
// w1,w2,w3 are broadcast twiddles (same for both complex values in YMM)

static inline void radix4_dit_butterfly(
    __m256d a, __m256d b, __m256d c, __m256d d,
    __m256d w1, __m256d w2, __m256d w3, bool fwd,
    __m256d &out0, __m256d &out1, __m256d &out2, __m256d &out3)
{
    __m256d apc = _mm256_add_pd(a, c);
    __m256d amc = _mm256_sub_pd(a, c);
    __m256d bpd = _mm256_add_pd(b, d);
    __m256d bmd = _mm256_sub_pd(b, d);
    __m256d jbmd = fwd ? mul_j_fwd(bmd) : mul_j_inv(bmd);

    out0 = _mm256_add_pd(apc, bpd);
    out1 = cmul(_mm256_sub_pd(amc, jbmd), w1);
    out2 = cmul(_mm256_sub_pd(apc, bpd), w2);
    out3 = cmul(_mm256_add_pd(amc, jbmd), w3);
}

// Last radix-4 butterfly (no twiddle)
static inline void radix4_last_butterfly(
    __m256d a, __m256d b, __m256d c, __m256d d, bool fwd,
    __m256d &out0, __m256d &out1, __m256d &out2, __m256d &out3)
{
    __m256d apc = _mm256_add_pd(a, c);
    __m256d amc = _mm256_sub_pd(a, c);
    __m256d bpd = _mm256_add_pd(b, d);
    __m256d bmd = _mm256_sub_pd(b, d);
    __m256d jbmd = fwd ? mul_j_fwd(bmd) : mul_j_inv(bmd);

    out0 = _mm256_add_pd(apc, bpd);
    out1 = _mm256_sub_pd(amc, jbmd);
    out2 = _mm256_sub_pd(apc, bpd);
    out3 = _mm256_add_pd(amc, jbmd);
}

#endif  // USE_AVX512

// ── Table structures ────────────────────────────────────────────────────────

struct INTL_FFT_PRECOMP {
    int32_t n;             // = 2*N
    int32_t ns2;           // = N/2 = number of complex points
    double *trig_fwd;      // forward twiddles + twist
    double *trig_inv;      // inverse twiddles + twist
    double *scratch;       // scratch buffer for out-of-place FFT
    double *data;          // data buffer for execute_direct
    void *buf;             // single allocation
};

#ifdef USE_AVX512

// ── AVX512 interleave/de-interleave index vectors ──────────────────────────
// Used by conversion loops: 8 re + 8 im → [re0,im0,...,re7,im7] (2 ZMMs)
static const __m512i idx_intl_lo = _mm512_set_epi64(11, 3, 10, 2,  9, 1,  8, 0);
static const __m512i idx_intl_hi = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
// Reverse: [re0,im0,...] (2 ZMMs) → 8 re, 8 im
static const __m512i idx_deinl_re = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
static const __m512i idx_deinl_im = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);

// ── Stockham radix-4 FFT (AVX512) ──────────────────────────────────────────
// Each ZMM processes 4 complex values at a time.
// src_ro: read-only initial source (may differ from x for zero-copy FFT).
// x, y: writable work/scratch buffers.

static void stockham_r4_from(int32_t ns2, bool fwd, const double *trig,
                             const double *src_ro, double *x, double *y) {
    const double *src = src_ro;
    double *dst = y;
    const double *tw = trig;
    int32_t q = ns2 / 4;
    int32_t s = 1;

    while (q >= 4) {
        int32_t stride = q * s;
        if (s == 1) {
            for (int32_t p = 0; p < q; p += 4) {
                __m512d a = _mm512_loadu_pd(src + p * 2);
                __m512d b = _mm512_loadu_pd(src + (p + stride) * 2);
                __m512d c = _mm512_loadu_pd(src + (p + 2*stride) * 2);
                __m512d d = _mm512_loadu_pd(src + (p + 3*stride) * 2);
                __m512d r0, r1, r2, r3;
                radix4_last_butterfly512(a, b, c, d, fwd, r0, r1, r2, r3);
                _mm512_storeu_pd(dst + p * 2, r0);
                _mm512_storeu_pd(dst + (p + q) * 2, r1);
                _mm512_storeu_pd(dst + (p + 2*q) * 2, r2);
                _mm512_storeu_pd(dst + (p + 3*q) * 2, r3);
            }
        } else {
            for (int32_t j = 0; j < s; j += 4) {
                __m512d w1 = _mm512_loadu_pd(tw); tw += 8;
                __m512d w2 = _mm512_loadu_pd(tw); tw += 8;
                __m512d w3 = _mm512_loadu_pd(tw); tw += 8;
                for (int32_t p = 0; p < q; p++) {
                    int32_t idx = j + s * p;
                    __m512d a = _mm512_loadu_pd(src + idx * 2);
                    __m512d b = _mm512_loadu_pd(src + (idx + stride) * 2);
                    __m512d c = _mm512_loadu_pd(src + (idx + 2*stride) * 2);
                    __m512d d = _mm512_loadu_pd(src + (idx + 3*stride) * 2);
                    int32_t o = p + q * (4 * j);
                    __m512d r0, r1, r2, r3;
                    radix4_dit_butterfly512(a, b, c, d, w1, w2, w3, fwd, r0, r1, r2, r3);
                    _mm512_storeu_pd(dst + o * 2, r0);
                    _mm512_storeu_pd(dst + (o + q) * 2, r1);
                    _mm512_storeu_pd(dst + (o + 2*q) * 2, r2);
                    _mm512_storeu_pd(dst + (o + 3*q) * 2, r3);
                }
            }
        }
        // After first pass, switch to mutable buffers
        if (s == 1) { src = dst; dst = x; }
        else { const double *tmp = src; src = dst; dst = const_cast<double*>(tmp); }
        s *= 4; q /= 4;
    }
    if (q == 1) {
        int32_t stride = s;
        for (int32_t j = 0; j < s; j += 4) {
            __m512d a = _mm512_loadu_pd(src + j * 2);
            __m512d b = _mm512_loadu_pd(src + (j + stride) * 2);
            __m512d c = _mm512_loadu_pd(src + (j + 2*stride) * 2);
            __m512d d = _mm512_loadu_pd(src + (j + 3*stride) * 2);
            __m512d r0, r1, r2, r3;
            radix4_last_butterfly512(a, b, c, d, fwd, r0, r1, r2, r3);
            _mm512_storeu_pd(dst + (4*j) * 2, r0);
            _mm512_storeu_pd(dst + (4*j + 4) * 2, r1);
            _mm512_storeu_pd(dst + (4*j + 8) * 2, r2);
            _mm512_storeu_pd(dst + (4*j + 12) * 2, r3);
        }
    } else if (q == 2) {
        int32_t half = ns2 / 2;
        for (int32_t j = 0; j < half; j += 4) {
            __m512d a = _mm512_loadu_pd(src + j * 2);
            __m512d b = _mm512_loadu_pd(src + (j + half) * 2);
            __m512d sum = _mm512_add_pd(a, b);
            __m512d diff = _mm512_sub_pd(a, b);
            __m512d out0 = _mm512_shuffle_f64x2(sum, diff, 0x44);
            __m512d out1 = _mm512_shuffle_f64x2(sum, diff, 0xEE);
            _mm512_storeu_pd(dst + (2*j) * 2, out0);
            _mm512_storeu_pd(dst + (2*j + 4) * 2, out1);
        }
    }
    // Ensure result ends up in x
    if (dst != x) memcpy(x, dst, ns2 * 2 * sizeof(double));
}

// Convenience wrapper: src and work buffer are the same (in-place)
static void stockham_r4(int32_t ns2, bool fwd, const double *trig, double *x, double *y) {
    stockham_r4_from(ns2, fwd, trig, x, x, y);
}

#else  // AVX2

// ── Stockham radix-4 FFT (AVX2) ────────────────────────────────────────────
// Reads from 4 quarters of src, writes interleaved to dst.
// Each pass reduces the quarter size by 4x.

static void stockham_r4(int32_t ns2, bool fwd, const double *trig, double *x, double *y) {
    double *src = x, *dst = y;
    const double *tw = trig;
    int32_t q = ns2 / 4;  // quarter size (shrinks each pass)
    int32_t s = 1;         // stride within each quarter (grows each pass)

    // Radix-4 DIF passes: read from 4 quarters of src, write interleaved to dst
    while (q >= 4) {
        int32_t stride = q * s;  // = ns2/4 (distance between quarters in src)
        for (int32_t j = 0; j < s; j += 2) {
            __m256d w1, w2, w3;
            if (s >= 2) {
                w1 = _mm256_loadu_pd(tw); tw += 4;
                w2 = _mm256_loadu_pd(tw); tw += 4;
                w3 = _mm256_loadu_pd(tw); tw += 4;
            }
            for (int32_t p = 0; p < q; p++) {
                int32_t idx = j + s * p;
                __m256d a = _mm256_loadu_pd(src + idx * 2);
                __m256d b = _mm256_loadu_pd(src + (idx + stride) * 2);
                __m256d c = _mm256_loadu_pd(src + (idx + 2*stride) * 2);
                __m256d d = _mm256_loadu_pd(src + (idx + 3*stride) * 2);
                int32_t o = p + q * (4 * j);
                __m256d r0, r1, r2, r3;
                if (s >= 2)
                    radix4_dit_butterfly(a, b, c, d, w1, w2, w3, fwd, r0, r1, r2, r3);
                else
                    radix4_last_butterfly(a, b, c, d, fwd, r0, r1, r2, r3);
                _mm256_storeu_pd(dst + o * 2, r0);
                _mm256_storeu_pd(dst + (o + q) * 2, r1);
                _mm256_storeu_pd(dst + (o + 2*q) * 2, r2);
                _mm256_storeu_pd(dst + (o + 3*q) * 2, r3);
            }
        }
        double *tmp = src; src = dst; dst = tmp;
        s *= 4; q /= 4;
    }
    if (q == 1) {
        int32_t stride = s;
        for (int32_t j = 0; j < s; j += 2) {
            __m256d a = _mm256_loadu_pd(src + j * 2);
            __m256d b = _mm256_loadu_pd(src + (j + stride) * 2);
            __m256d c = _mm256_loadu_pd(src + (j + 2*stride) * 2);
            __m256d d = _mm256_loadu_pd(src + (j + 3*stride) * 2);
            __m256d r0, r1, r2, r3;
            radix4_last_butterfly(a, b, c, d, fwd, r0, r1, r2, r3);
            _mm256_storeu_pd(dst + (4*j) * 2, r0);
            _mm256_storeu_pd(dst + (4*j+1) * 2, r1);
            _mm256_storeu_pd(dst + (4*j+2) * 2, r2);
            _mm256_storeu_pd(dst + (4*j+3) * 2, r3);
        }
    } else if (q == 2) {
        int32_t half = ns2 / 2;
        for (int32_t j = 0; j < half; j += 2) {
            __m256d a = _mm256_loadu_pd(src + j * 2);
            __m256d b = _mm256_loadu_pd(src + (j + half) * 2);
            _mm256_storeu_pd(dst + (2*j) * 2, _mm256_add_pd(a, b));
            _mm256_storeu_pd(dst + (2*j+1) * 2, _mm256_sub_pd(a, b));
        }
    }
    if (dst != x) memcpy(x, dst, ns2 * 2 * sizeof(double));
}

#endif  // USE_AVX512

// ── Forward/Inverse FFT wrappers ────────────────────────────────────────────

// Forward FFT: Stockham + twist.  Reads from src_in (may be != c).
// Scale factor (2/N) is pre-baked into the forward twist twiddles.
static void intl_fft_from(const INTL_FFT_PRECOMP *tables, const double *src_in,
                          double *c) {
    const int32_t ns2 = tables->ns2;
    const double *trig = tables->trig_fwd;

#ifdef USE_AVX512
    stockham_r4_from(ns2, true, trig, src_in, c, tables->scratch);

    const double *tw_twist = trig;
    for (int32_t s = 4; 4*s < ns2; s *= 4)
        tw_twist += (s / 4) * 24;

    for (int32_t j = 0; j < ns2; j += 4) {
        double *p = c + j * 2;
        __m512d a = _mm512_load_pd(p);
        __m512d w = _mm512_load_pd(tw_twist + j * 2);
        _mm512_store_pd(p, cmul512(a, w));
    }
#else
    // AVX2: must copy first since stockham_r4 needs mutable src
    if (src_in != c) memcpy(c, src_in, ns2 * 2 * sizeof(double));
    stockham_r4(ns2, true, trig, c, tables->scratch);

    const double *tw_twist = trig;
    for (int32_t s = 1; 4*s < ns2; s *= 4)
        tw_twist += (s / 2) * 12;

    for (int32_t j = 0; j < ns2; j += 2) {
        double *p = c + j * 2;
        __m256d a = _mm256_load_pd(p);
        __m256d w = _mm256_load_pd(tw_twist + j * 2);
        _mm256_store_pd(p, cmul(a, w));
    }
#endif
}

static void intl_fft(const INTL_FFT_PRECOMP *tables, double *c) {
    intl_fft_from(tables, c, c);
}

static void intl_ifft(const INTL_FFT_PRECOMP *tables, double *c) {
    const int32_t ns2 = tables->ns2;
    const double *trig = tables->trig_inv;

    // Apply twist first (inverse direction)
    const double *tw_twist = trig;
#ifdef USE_AVX512
    for (int32_t j = 0; j < ns2; j += 4) {
        double *p = c + j * 2;
        __m512d a = _mm512_load_pd(p);
        __m512d w = _mm512_load_pd(tw_twist + j * 2);
        _mm512_store_pd(p, cmul512(a, w));
    }
#else
    for (int32_t j = 0; j < ns2; j += 2) {
        double *p = c + j * 2;
        __m256d a = _mm256_load_pd(p);
        __m256d w = _mm256_load_pd(tw_twist + j * 2);
        _mm256_store_pd(p, cmul(a, w));
    }
#endif

    // Butterfly twiddles start after twist
    const double *tw_bf = trig + ns2 * 2;

    // Run Stockham radix-4 (inverse)
    stockham_r4(ns2, false, tw_bf, c, tables->scratch);
}

// ── Table construction ──────────────────────────────────────────────────────

static void build_tables(int32_t nn, INTL_FFT_PRECOMP *reps) {
    int32_t n = 2 * nn;
    int32_t ns2 = nn / 2;
    reps->n = n;
    reps->ns2 = ns2;

#ifdef USE_AVX512
    // AVX512: twiddles packed as 4 complex (8 doubles) per ZMM.
    // First pass (s=1) uses no twiddles.
    // Subsequent passes (s=4,16,...): j steps by 4, s/4 groups per pass, 3 ZMMs each = 24 doubles.
    int32_t bf_twiddle_doubles = 0;
    for (int32_t s = 4; 4*s < ns2; s *= 4) {
        bf_twiddle_doubles += (s / 4) * 24;
    }
#else
    int32_t bf_twiddle_doubles = 0;
    for (int32_t s = 1; 4*s < ns2; s *= 4) {
        int32_t groups = (s < 2) ? 1 : s / 2;
        bf_twiddle_doubles += groups * 12;
    }
#endif

    int32_t twist_doubles = ns2 * 2;
    int32_t total_per_dir = bf_twiddle_doubles + twist_doubles;

    int32_t total_doubles = 2 * total_per_dir + nn + nn;
    reps->buf = aligned_alloc(64, total_doubles * sizeof(double));
    double *ptr = (double *)reps->buf;

    reps->trig_fwd = ptr; ptr += total_per_dir;
    reps->trig_inv = ptr; ptr += total_per_dir;
    reps->scratch = ptr; ptr += nn;
    reps->data = ptr;

#ifdef USE_AVX512
    // Build forward butterfly twiddles (AVX512: 4 complex per ZMM)
    double *fwd = reps->trig_fwd;
    for (int32_t s = 4; 4*s < ns2; s *= 4) {
        int32_t denom = 4 * s;
        for (int32_t j = 0; j < s; j += 4) {
            // w1: exp(-2πi*(j+jj)/(4s)) for jj=0..3
            for (int32_t jj = 0; jj < 4; jj++) {
                fwd[jj*2]   = accurate_cos(-(j+jj), denom);
                fwd[jj*2+1] = accurate_sin(-(j+jj), denom);
            }
            fwd += 8;
            // w2: exp(-2πi*2*(j+jj)/(4s))
            for (int32_t jj = 0; jj < 4; jj++) {
                fwd[jj*2]   = accurate_cos(-2*(j+jj), denom);
                fwd[jj*2+1] = accurate_sin(-2*(j+jj), denom);
            }
            fwd += 8;
            // w3: exp(-2πi*3*(j+jj)/(4s))
            for (int32_t jj = 0; jj < 4; jj++) {
                fwd[jj*2]   = accurate_cos(-3*(j+jj), denom);
                fwd[jj*2+1] = accurate_sin(-3*(j+jj), denom);
            }
            fwd += 8;
        }
    }
    // Forward twist: bake in scale factor 2/N so execute_direct can skip scaling
    {
        const double scale = 2.0 / nn;
        for (int32_t k = 0; k < ns2; k++) {
            *fwd++ = scale * accurate_cos(-k, n);
            *fwd++ = scale * accurate_sin(-k, n);
        }
    }

    // Build inverse twiddles
    double *inv = reps->trig_inv;
    // Inverse twist first
    for (int32_t k = 0; k < ns2; k++) {
        *inv++ = accurate_cos(k, n);
        *inv++ = accurate_sin(k, n);
    }
    // Inverse butterfly twiddles (positive angles)
    for (int32_t s = 4; 4*s < ns2; s *= 4) {
        int32_t denom = 4 * s;
        for (int32_t j = 0; j < s; j += 4) {
            for (int32_t jj = 0; jj < 4; jj++) {
                inv[jj*2]   = accurate_cos(j+jj, denom);
                inv[jj*2+1] = accurate_sin(j+jj, denom);
            }
            inv += 8;
            for (int32_t jj = 0; jj < 4; jj++) {
                inv[jj*2]   = accurate_cos(2*(j+jj), denom);
                inv[jj*2+1] = accurate_sin(2*(j+jj), denom);
            }
            inv += 8;
            for (int32_t jj = 0; jj < 4; jj++) {
                inv[jj*2]   = accurate_cos(3*(j+jj), denom);
                inv[jj*2+1] = accurate_sin(3*(j+jj), denom);
            }
            inv += 8;
        }
    }

#else  // AVX2
    // Build forward butterfly twiddles
    double *fwd = reps->trig_fwd;
    for (int32_t s = 1; 4*s < ns2; s *= 4) {
        for (int32_t j = 0; j < s; j += 2) {
            int32_t denom = 4 * s;
            for (int32_t jj = 0; jj < 2 && (j + jj) < s; jj++) {
                fwd[0 + jj*2] = accurate_cos(-(j+jj), denom);
                fwd[1 + jj*2] = accurate_sin(-(j+jj), denom);
            }
            if (s == 1) { fwd[2] = fwd[0]; fwd[3] = fwd[1]; }
            fwd += 4;
            for (int32_t jj = 0; jj < 2 && (j + jj) < s; jj++) {
                fwd[0 + jj*2] = accurate_cos(-2*(j+jj), denom);
                fwd[1 + jj*2] = accurate_sin(-2*(j+jj), denom);
            }
            if (s == 1) { fwd[2] = fwd[0]; fwd[3] = fwd[1]; }
            fwd += 4;
            for (int32_t jj = 0; jj < 2 && (j + jj) < s; jj++) {
                fwd[0 + jj*2] = accurate_cos(-3*(j+jj), denom);
                fwd[1 + jj*2] = accurate_sin(-3*(j+jj), denom);
            }
            if (s == 1) { fwd[2] = fwd[0]; fwd[3] = fwd[1]; }
            fwd += 4;
        }
    }
    for (int32_t k = 0; k < ns2; k++) {
        *fwd++ = accurate_cos(-k, n);
        *fwd++ = accurate_sin(-k, n);
    }

    double *inv = reps->trig_inv;
    for (int32_t k = 0; k < ns2; k++) {
        *inv++ = accurate_cos(k, n);
        *inv++ = accurate_sin(k, n);
    }
    for (int32_t s = 1; 4*s < ns2; s *= 4) {
        for (int32_t j = 0; j < s; j += 2) {
            int32_t denom = 4 * s;
            for (int32_t jj = 0; jj < 2 && (j + jj) < s; jj++) {
                inv[0 + jj*2] = accurate_cos(j+jj, denom);
                inv[1 + jj*2] = accurate_sin(j+jj, denom);
            }
            if (s == 1) { inv[2] = inv[0]; inv[3] = inv[1]; }
            inv += 4;
            for (int32_t jj = 0; jj < 2 && (j + jj) < s; jj++) {
                inv[0 + jj*2] = accurate_cos(2*(j+jj), denom);
                inv[1 + jj*2] = accurate_sin(2*(j+jj), denom);
            }
            if (s == 1) { inv[2] = inv[0]; inv[3] = inv[1]; }
            inv += 4;
            for (int32_t jj = 0; jj < 2 && (j + jj) < s; jj++) {
                inv[0 + jj*2] = accurate_cos(3*(j+jj), denom);
                inv[1 + jj*2] = accurate_sin(3*(j+jj), denom);
            }
            if (s == 1) { inv[2] = inv[0]; inv[3] = inv[1]; }
            inv += 4;
        }
    }
#endif  // USE_AVX512
}

// ── Conversion helpers ──────────────────────────────────────────────────────

static inline __m256i magic_cvtpd_epi64(__m256d x) {
    const __m256d m = _mm256_set1_pd(6755399441055744.0);
    const __m256i mi = _mm256_set1_epi64x(0x4338000000000000LL);
    return _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(x, m)), mi);
}

// ── FFT_Processor implementation ────────────────────────────────────────────

static int32_t rev(int32_t x, int32_t M) {
    int32_t r = 0;
    for (int32_t j = M; j > 1; j /= 2) { r = 2*r+(x%2); x /= 2; }
    return r;
}

FFT_Processor_Spqlios_Intl::FFT_Processor_Spqlios_Intl(const int32_t N)
    : _2N(2*N), N(N), Ns2(N/2) {
    auto *tables = new INTL_FFT_PRECOMP;
    build_tables(N, tables);
    tables_direct = tables;
    tables_reverse = tables;  // same struct, different trig pointers
    real_inout_direct = tables->data;
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

FFT_Processor_Spqlios_Intl::~FFT_Processor_Spqlios_Intl() {
    auto *tables = (INTL_FFT_PRECOMP *)tables_direct;
    free(tables->buf);
    delete tables;
    delete[] cosomegaxminus1;
    delete[] reva;
}

void FFT_Processor_Spqlios_Intl::execute_reverse_torus32(double *res, const uint32_t *a) {
    const int32_t *aa = (const int32_t*)a;
#ifdef USE_AVX512
    // Vectorized int32→double with interleave
    for (int32_t i = 0; i < Ns2; i += 8) {
        __m256i re_i32 = _mm256_loadu_si256((const __m256i*)(aa + i));
        __m256i im_i32 = _mm256_loadu_si256((const __m256i*)(aa + i + Ns2));
        __m512d re = _mm512_cvtepi32_pd(re_i32);
        __m512d im = _mm512_cvtepi32_pd(im_i32);
        _mm512_storeu_pd(res + 2*i,     _mm512_permutex2var_pd(re, idx_intl_lo, im));
        _mm512_storeu_pd(res + 2*i + 8, _mm512_permutex2var_pd(re, idx_intl_hi, im));
    }
#else
    // AVX2: convert 4 int32 → 4 double, interleave re/im with unpack
    for (int32_t i = 0; i < Ns2; i += 4) {
        __m128i re_i32 = _mm_loadu_si128((const __m128i*)(aa + i));
        __m128i im_i32 = _mm_loadu_si128((const __m128i*)(aa + i + Ns2));
        __m256d re = _mm256_cvtepi32_pd(re_i32);
        __m256d im = _mm256_cvtepi32_pd(im_i32);
        // Interleave: [re0,re1,re2,re3] + [im0,im1,im2,im3] → [re0,im0,re1,im1], [re2,im2,re3,im3]
        __m256d lo = _mm256_unpacklo_pd(re, im);  // [re0,im0,re2,im2]
        __m256d hi = _mm256_unpackhi_pd(re, im);  // [re1,im1,re3,im3]
        _mm256_storeu_pd(res + 2*i,     _mm256_permute2f128_pd(lo, hi, 0x20)); // [re0,im0,re1,im1]
        _mm256_storeu_pd(res + 2*i + 4, _mm256_permute2f128_pd(lo, hi, 0x31)); // [re2,im2,re3,im3]
    }
#endif
    intl_ifft((const INTL_FFT_PRECOMP*)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_reverse_int(double *res, const int32_t *a) {
#ifdef USE_AVX512
    for (int32_t i = 0; i < Ns2; i += 8) {
        __m256i re_i32 = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i im_i32 = _mm256_loadu_si256((const __m256i*)(a + i + Ns2));
        __m512d re = _mm512_cvtepi32_pd(re_i32);
        __m512d im = _mm512_cvtepi32_pd(im_i32);
        _mm512_storeu_pd(res + 2*i,     _mm512_permutex2var_pd(re, idx_intl_lo, im));
        _mm512_storeu_pd(res + 2*i + 8, _mm512_permutex2var_pd(re, idx_intl_hi, im));
    }
#else
    for (int32_t i = 0; i < Ns2; i += 4) {
        __m128i re_i32 = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i im_i32 = _mm_loadu_si128((const __m128i*)(a + i + Ns2));
        __m256d re = _mm256_cvtepi32_pd(re_i32);
        __m256d im = _mm256_cvtepi32_pd(im_i32);
        __m256d lo = _mm256_unpacklo_pd(re, im);
        __m256d hi = _mm256_unpackhi_pd(re, im);
        _mm256_storeu_pd(res + 4*i,     _mm256_permute2f128_pd(lo, hi, 0x20));
        _mm256_storeu_pd(res + 4*i + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
    }
#endif
    intl_ifft((const INTL_FFT_PRECOMP*)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_reverse_uint(double *res, const uint32_t *a) {
#ifdef USE_AVX512
    for (int32_t i = 0; i < Ns2; i += 8) {
        __m256i re_i32 = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i im_i32 = _mm256_loadu_si256((const __m256i*)(a + i + Ns2));
        __m512d re = _mm512_cvtepu32_pd(re_i32);
        __m512d im = _mm512_cvtepu32_pd(im_i32);
        _mm512_storeu_pd(res + 2*i,     _mm512_permutex2var_pd(re, idx_intl_lo, im));
        _mm512_storeu_pd(res + 2*i + 8, _mm512_permutex2var_pd(re, idx_intl_hi, im));
    }
#else
    for (int32_t i = 0; i < Ns2; i += 4) {
        __m128i re_i32 = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i im_i32 = _mm_loadu_si128((const __m128i*)(a + i + Ns2));
        __m256d re = _mm256_cvtepi32_pd(re_i32);  // unsigned but small values, cvt signed is fine
        __m256d im = _mm256_cvtepi32_pd(im_i32);
        __m256d lo = _mm256_unpacklo_pd(re, im);
        __m256d hi = _mm256_unpackhi_pd(re, im);
        _mm256_storeu_pd(res + 4*i,     _mm256_permute2f128_pd(lo, hi, 0x20));
        _mm256_storeu_pd(res + 4*i + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
    }
#endif
    intl_ifft((const INTL_FFT_PRECOMP*)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_reverse_torus64(double *res, const uint64_t *a) {
    const int64_t *aa = (const int64_t*)a;
    for (int32_t i = 0; i < Ns2; i++) {
        res[2*i]     = (double)aa[i];
        res[2*i + 1] = (double)aa[i + Ns2];
    }
    intl_ifft((const INTL_FFT_PRECOMP*)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_reverse_torus64_uint(double *res, const uint64_t *a) {
    for (int32_t i = 0; i < Ns2; i++) {
        res[2*i]     = (double)a[i];
        res[2*i + 1] = (double)a[i + Ns2];
    }
    intl_ifft((const INTL_FFT_PRECOMP*)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus32(uint32_t *res, const double *a) {
    auto *tables = (const INTL_FFT_PRECOMP*)tables_direct;
    // Scale is baked into forward twist — read input directly, no copy needed
    intl_fft_from(tables, a, real_inout_direct);
#ifdef USE_AVX512
    // Vectorized de-interleave + double→int32 extraction
    for (int32_t i = 0; i < Ns2; i += 8) {
        __m512d in0 = _mm512_load_pd(real_inout_direct + 2*i);
        __m512d in1 = _mm512_load_pd(real_inout_direct + 2*i + 8);
        __m512d re = _mm512_permutex2var_pd(in0, idx_deinl_re, in1);
        __m512d im = _mm512_permutex2var_pd(in0, idx_deinl_im, in1);
        __m256i re_i32 = _mm512_cvtepi64_epi32(_mm512_cvttpd_epi64(re));
        __m256i im_i32 = _mm512_cvtepi64_epi32(_mm512_cvttpd_epi64(im));
        _mm256_storeu_si256((__m256i*)(res + i), re_i32);
        _mm256_storeu_si256((__m256i*)(res + i + Ns2), im_i32);
    }
#else
    // AVX2: de-interleave [re0,im0,re1,im1,...] → split re[]+im[] with magic f64→i32
    {
        const __m256d magic_d = _mm256_set1_pd(6755399441055744.0);
        const __m256i magic_i = _mm256_set1_epi64x(0x4338000000000000LL);
        for (int32_t i = 0; i < Ns2; i += 4) {
            __m256d v0 = _mm256_loadu_pd(real_inout_direct + 2*i);       // [re0,im0,re1,im1]
            __m256d v1 = _mm256_loadu_pd(real_inout_direct + 2*i + 4);   // [re2,im2,re3,im3]
            // De-interleave: extract re and im
            __m256d re = _mm256_permute2f128_pd(
                _mm256_unpacklo_pd(v0, v1),   // [re0,re2,re1,re3]
                _mm256_unpacklo_pd(v0, v1), 0x20);
            // Actually simpler: use shuffle_pd for within-lane, permute2f128 for cross-lane
            __m256d re_raw = _mm256_shuffle_pd(v0, v1, 0b0000); // [re0,re2,re1,re3]
            __m256d im_raw = _mm256_shuffle_pd(v0, v1, 0b1111); // [im0,im2,im1,im3]
            re_raw = _mm256_permute4x64_pd(re_raw, 0b11011000);  // [re0,re1,re2,re3]
            im_raw = _mm256_permute4x64_pd(im_raw, 0b11011000);  // [im0,im1,im2,im3]
            // Magic f64→i64→i32
            __m256i re_i64 = _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(re_raw, magic_d)), magic_i);
            __m256i im_i64 = _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(im_raw, magic_d)), magic_i);
            // Pack i64→i32: shuffle + permute
            __m256 combined = _mm256_shuffle_ps(_mm256_castsi256_ps(re_i64),
                                                _mm256_castsi256_ps(im_i64), _MM_SHUFFLE(2,0,2,0));
            __m256d ordered = _mm256_permute4x64_pd(_mm256_castps_pd(combined), _MM_SHUFFLE(3,1,2,0));
            __m128i re_i32 = _mm256_castsi256_si128(_mm256_castpd_si256(ordered));
            __m128i im_i32 = _mm256_extracti128_si256(_mm256_castpd_si256(ordered), 1);
            _mm_storeu_si128((__m128i*)(res + i), re_i32);
            _mm_storeu_si128((__m128i*)(res + i + Ns2), im_i32);
        }
    }
#endif
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus32_add(uint32_t *res, const double *a) {
    auto *tables = (const INTL_FFT_PRECOMP*)tables_direct;
    intl_fft_from(tables, a, real_inout_direct);
#ifdef USE_AVX512
    for (int32_t i = 0; i < Ns2; i += 8) {
        __m512d in0 = _mm512_load_pd(real_inout_direct + 2*i);
        __m512d in1 = _mm512_load_pd(real_inout_direct + 2*i + 8);
        __m512d re = _mm512_permutex2var_pd(in0, idx_deinl_re, in1);
        __m512d im = _mm512_permutex2var_pd(in0, idx_deinl_im, in1);
        __m256i re_i32 = _mm512_cvtepi64_epi32(_mm512_cvttpd_epi64(re));
        __m256i im_i32 = _mm512_cvtepi64_epi32(_mm512_cvttpd_epi64(im));
        __m256i old_re = _mm256_loadu_si256((__m256i*)(res + i));
        __m256i old_im = _mm256_loadu_si256((__m256i*)(res + i + Ns2));
        _mm256_storeu_si256((__m256i*)(res + i), _mm256_add_epi32(old_re, re_i32));
        _mm256_storeu_si256((__m256i*)(res + i + Ns2), _mm256_add_epi32(old_im, im_i32));
    }
#else
    {
        const __m256d magic_d = _mm256_set1_pd(6755399441055744.0);
        const __m256i magic_i = _mm256_set1_epi64x(0x4338000000000000LL);
        for (int32_t i = 0; i < Ns2; i += 4) {
            __m256d v0 = _mm256_loadu_pd(real_inout_direct + 4*i);
            __m256d v1 = _mm256_loadu_pd(real_inout_direct + 4*i + 4);
            __m256d re_raw = _mm256_permute4x64_pd(_mm256_shuffle_pd(v0, v1, 0b0000), 0b11011000);
            __m256d im_raw = _mm256_permute4x64_pd(_mm256_shuffle_pd(v0, v1, 0b1111), 0b11011000);
            __m256i re_i64 = _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(re_raw, magic_d)), magic_i);
            __m256i im_i64 = _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(im_raw, magic_d)), magic_i);
            __m256 combined = _mm256_shuffle_ps(_mm256_castsi256_ps(re_i64),
                                                _mm256_castsi256_ps(im_i64), _MM_SHUFFLE(2,0,2,0));
            __m256d ordered = _mm256_permute4x64_pd(_mm256_castps_pd(combined), _MM_SHUFFLE(3,1,2,0));
            __m128i re_i32 = _mm256_castsi256_si128(_mm256_castpd_si256(ordered));
            __m128i im_i32 = _mm256_extracti128_si256(_mm256_castpd_si256(ordered), 1);
            __m128i old_re = _mm_loadu_si128((__m128i*)(res + i));
            __m128i old_im = _mm_loadu_si128((__m128i*)(res + i + Ns2));
            _mm_storeu_si128((__m128i*)(res + i), _mm_add_epi32(old_re, re_i32));
            _mm_storeu_si128((__m128i*)(res + i + Ns2), _mm_add_epi32(old_im, im_i32));
        }
    }
#endif
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus64(uint64_t *res, double *a) {
    auto *tables = (const INTL_FFT_PRECOMP*)tables_direct;
    intl_fft_from(tables, a, real_inout_direct);
    const double magic = 6755399441055744.0;
    const int64_t magic_i = 0x4338000000000000LL;
    for (int32_t i = 0; i < Ns2; i++) {
        union { double d; int64_t l; } u;
        u.d = real_inout_direct[2*i] + magic; res[i] = (uint64_t)(u.l - magic_i);
        u.d = real_inout_direct[2*i+1] + magic; res[i+Ns2] = (uint64_t)(u.l - magic_i);
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus64_add(uint64_t *res, double *a) {
    auto *tables = (const INTL_FFT_PRECOMP*)tables_direct;
    intl_fft_from(tables, a, real_inout_direct);
    const double magic = 6755399441055744.0;
    const int64_t magic_i = 0x4338000000000000LL;
    for (int32_t i = 0; i < Ns2; i++) {
        union { double d; int64_t l; } u;
        u.d = real_inout_direct[2*i] + magic; res[i] += (uint64_t)(u.l - magic_i);
        u.d = real_inout_direct[2*i+1] + magic; res[i+Ns2] += (uint64_t)(u.l - magic_i);
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus32_q(uint32_t *res, const double *a, const uint32_t q) {
    auto *tables = (const INTL_FFT_PRECOMP*)tables_direct;
    intl_fft_from(tables, a, real_inout_direct);
    for (int32_t i = 0; i < Ns2; i++) {
        res[i] = uint32_t((int64_t(real_inout_direct[2*i])%q+q)%q);
        res[i+Ns2] = uint32_t((int64_t(real_inout_direct[2*i+1])%q+q)%q);
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus32_rescale(uint32_t *res, const double *a, const double D) {
    auto *tables = (const INTL_FFT_PRECOMP*)tables_direct;
    intl_fft_from(tables, a, real_inout_direct);
    for (int32_t i = 0; i < Ns2; i++) {
        res[i] = (uint32_t)(int64_t)(real_inout_direct[2*i]/D);
        res[i+Ns2] = (uint32_t)(int64_t)(real_inout_direct[2*i+1]/D);
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus32_rescale_clpx(
    uint32_t *res, const double *a, const double q, const uint32_t plain_modulus) {
    execute_direct_torus32(res, a);
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus64_rescale(uint64_t *res, const double *a, const double D) {
    auto *tables = (const INTL_FFT_PRECOMP*)tables_direct;
    alignas(64) double tmp[N];
    intl_fft_from(tables, a, tmp);
    for (int32_t i = 0; i < Ns2; i++) {
        res[i] = (uint64_t)std::round(tmp[2*i]/D);
        res[i+Ns2] = (uint64_t)std::round(tmp[2*i+1]/D);
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus64_rescale_clpx(
    uint64_t *res, const double *a, const uint32_t plain_modulus) {
    execute_direct_torus64(res, const_cast<double*>(a));
}

thread_local FFT_Processor_Spqlios_Intl fftplvl1(TFHEpp::lvl1param::n);
thread_local FFT_Processor_Spqlios_Intl fftplvl2(TFHEpp::lvl2param::n);
thread_local FFT_Processor_Spqlios_Intl fftplvl3(TFHEpp::lvl3param::n);
