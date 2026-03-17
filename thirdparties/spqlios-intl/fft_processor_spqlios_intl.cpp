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

// ── Stockham radix-4 FFT ────────────────────────────────────────────────────
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

// ── Forward/Inverse FFT wrappers ────────────────────────────────────────────

static void intl_fft(const INTL_FFT_PRECOMP *tables, double *c) {
    const int32_t ns2 = tables->ns2;
    const double *trig = tables->trig_fwd;

    // Compute total twiddle size to find twist offset
    int32_t tw_count = 0;
    for (int32_t s = 1; 4*s < ns2; s *= 4)
        tw_count += s * 3;  // s/2 groups × 6 doubles each... actually s groups × 3 complex × 2 doubles / 2 per YMM

    // Run Stockham radix-4
    stockham_r4(ns2, true, trig, c, tables->scratch);

    // Apply final twist
    const double *tw_twist = trig;
    // Skip past butterfly twiddles
    for (int32_t s = 1; 4*s < ns2; s *= 4)
        tw_twist += (s / 2) * 12;  // s/2 groups × 12 doubles per group
    // Twist: multiply each complex by exp(-2πi*k/(2N))
    for (int32_t j = 0; j < ns2; j += 2) {
        double *p = c + j * 2;
        __m256d a = _mm256_load_pd(p);
        __m256d w = _mm256_load_pd(tw_twist + j * 2);
        _mm256_store_pd(p, cmul(a, w));
    }
}

static void intl_ifft(const INTL_FFT_PRECOMP *tables, double *c) {
    const int32_t ns2 = tables->ns2;
    const double *trig = tables->trig_inv;

    // Apply twist first (inverse direction)
    const double *tw_twist = trig;
    for (int32_t j = 0; j < ns2; j += 2) {
        double *p = c + j * 2;
        __m256d a = _mm256_load_pd(p);
        __m256d w = _mm256_load_pd(tw_twist + j * 2);
        _mm256_store_pd(p, cmul(a, w));
    }

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

    // Count twiddle table size
    // Butterfly twiddles: for each pass, s/2 groups of 3 complex (6 doubles each)
    int32_t bf_twiddle_doubles = 0;
    for (int32_t s = 1; 4*s < ns2; s *= 4)
        bf_twiddle_doubles += (s / 2) * 12;  // s/2 groups × 12 doubles
    // Handle s=1 specially (1 group instead of 0.5)
    // Actually for s=1: j goes 0..0 step 2, that's 1 iteration (j=0 only), producing 12 doubles
    // For s=4: j goes 0,2, that's 2 iterations, producing 24 doubles
    // For s=16: j goes 0,2,...,14, that's 8 iterations, producing 96 doubles
    // Let me recompute: for each pass, j goes from 0 to s-1 in steps of 2: s/2 iterations (but at least 1)
    // Actually s starts at 1, so s/2 = 0 for first pass. Need to handle s=1 as 1 group.
    bf_twiddle_doubles = 0;
    for (int32_t s = 1; 4*s < ns2; s *= 4) {
        int32_t groups = (s < 2) ? 1 : s / 2;
        bf_twiddle_doubles += groups * 12;
    }

    int32_t twist_doubles = ns2 * 2;  // ns2 complex values
    int32_t total_per_dir = bf_twiddle_doubles + twist_doubles;

    // Allocate: 2 twiddle tables (fwd+inv) + scratch + data
    int32_t total_doubles = 2 * total_per_dir + nn + nn;  // 2 trig + scratch + data
    reps->buf = aligned_alloc(64, total_doubles * sizeof(double));
    double *ptr = (double *)reps->buf;

    reps->trig_fwd = ptr; ptr += total_per_dir;
    reps->trig_inv = ptr; ptr += total_per_dir;
    reps->scratch = ptr; ptr += nn;
    reps->data = ptr;

    // Build forward butterfly twiddles
    double *fwd = reps->trig_fwd;
    for (int32_t s = 1; 4*s < ns2; s *= 4) {
        for (int32_t j = 0; j < s; j += 2) {
            int32_t denom = 4 * s;
            for (int32_t jj = 0; jj < 2 && (j + jj) < s; jj++) {
                // w1 = exp(-2πi * (j+jj) / (4s))
                fwd[0 + jj*2] = accurate_cos(-(j+jj), denom);
                fwd[1 + jj*2] = accurate_sin(-(j+jj), denom);
            }
            if (s == 1) { fwd[2] = fwd[0]; fwd[3] = fwd[1]; }  // pad for YMM
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
    // Forward twist: exp(-2πi*k/(2N)) = exp(-2πi*k/n)
    for (int32_t k = 0; k < ns2; k++) {
        *fwd++ = accurate_cos(-k, n);
        *fwd++ = accurate_sin(-k, n);
    }

    // Build inverse twiddles
    double *inv = reps->trig_inv;
    // Inverse twist first: exp(+2πi*k/n)
    for (int32_t k = 0; k < ns2; k++) {
        *inv++ = accurate_cos(k, n);
        *inv++ = accurate_sin(k, n);
    }
    // Inverse butterfly twiddles: same structure, positive angles
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
    for (int32_t i = 0; i < Ns2; i++) {
        res[2*i]     = (double)aa[i];
        res[2*i + 1] = (double)aa[i + Ns2];
    }
    intl_ifft((const INTL_FFT_PRECOMP*)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_reverse_int(double *res, const int32_t *a) {
    for (int32_t i = 0; i < Ns2; i++) {
        res[2*i]     = (double)a[i];
        res[2*i + 1] = (double)a[i + Ns2];
    }
    intl_ifft((const INTL_FFT_PRECOMP*)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_reverse_uint(double *res, const uint32_t *a) {
    for (int32_t i = 0; i < Ns2; i++) {
        res[2*i]     = (double)a[i];
        res[2*i + 1] = (double)a[i + Ns2];
    }
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
    const double s = 2.0 / N;
    for (int32_t i = 0; i < N; i++) real_inout_direct[i] = a[i] * s;
    intl_fft((const INTL_FFT_PRECOMP*)tables_direct, real_inout_direct);
    for (int32_t i = 0; i < Ns2; i++) {
        res[i]       = (uint32_t)(int64_t)real_inout_direct[2*i];
        res[i + Ns2] = (uint32_t)(int64_t)real_inout_direct[2*i + 1];
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus32_add(uint32_t *res, const double *a) {
    const double s = 2.0 / N;
    for (int32_t i = 0; i < N; i++) real_inout_direct[i] = a[i] * s;
    intl_fft((const INTL_FFT_PRECOMP*)tables_direct, real_inout_direct);
    for (int32_t i = 0; i < Ns2; i++) {
        res[i]       += (uint32_t)(int64_t)real_inout_direct[2*i];
        res[i + Ns2] += (uint32_t)(int64_t)real_inout_direct[2*i + 1];
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus64(uint64_t *res, double *a) {
    const double s = 2.0 / N;
    for (int32_t i = 0; i < N; i++) a[i] *= s;
    intl_fft((const INTL_FFT_PRECOMP*)tables_direct, a);
    const double magic = 6755399441055744.0;
    const int64_t magic_i = 0x4338000000000000LL;
    for (int32_t i = 0; i < Ns2; i++) {
        union { double d; int64_t l; } u;
        u.d = a[2*i] + magic; res[i] = (uint64_t)(u.l - magic_i);
        u.d = a[2*i+1] + magic; res[i+Ns2] = (uint64_t)(u.l - magic_i);
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus64_add(uint64_t *res, double *a) {
    const double s = 2.0 / N;
    for (int32_t i = 0; i < N; i++) a[i] *= s;
    intl_fft((const INTL_FFT_PRECOMP*)tables_direct, a);
    const double magic = 6755399441055744.0;
    const int64_t magic_i = 0x4338000000000000LL;
    for (int32_t i = 0; i < Ns2; i++) {
        union { double d; int64_t l; } u;
        u.d = a[2*i] + magic; res[i] += (uint64_t)(u.l - magic_i);
        u.d = a[2*i+1] + magic; res[i+Ns2] += (uint64_t)(u.l - magic_i);
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus32_q(uint32_t *res, const double *a, const uint32_t q) {
    const double s = 2.0 / N;
    for (int32_t i = 0; i < N; i++) real_inout_direct[i] = a[i] * s;
    intl_fft((const INTL_FFT_PRECOMP*)tables_direct, real_inout_direct);
    for (int32_t i = 0; i < Ns2; i++) {
        res[i] = uint32_t((int64_t(real_inout_direct[2*i])%q+q)%q);
        res[i+Ns2] = uint32_t((int64_t(real_inout_direct[2*i+1])%q+q)%q);
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus32_rescale(uint32_t *res, const double *a, const double D) {
    const double s = 2.0 / N;
    for (int32_t i = 0; i < N; i++) real_inout_direct[i] = a[i] * s;
    intl_fft((const INTL_FFT_PRECOMP*)tables_direct, real_inout_direct);
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
    const double s = 2.0 / N;
    auto *tables = (INTL_FFT_PRECOMP*)tables_direct;
    alignas(64) double tmp[N];
    for (int32_t i = 0; i < N; i++) tmp[i] = a[i] * s;
    intl_fft(tables, tmp);
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
