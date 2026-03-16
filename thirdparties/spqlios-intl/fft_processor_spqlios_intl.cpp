// Interleaved-format SPQLIOS FFT processor.
//
// Data layout: [re0, im0, re1, im1, ...] (N doubles = N/2 complex values)
//
// The FFT operates on N/2 complex points. Twiddle tables store interleaved
// complex values [cos, sin, cos, sin, ...]. Butterflies use AVX2 complex
// multiply via unpacklo/unpackhi + vfmaddsub (3 ops per complex mul).
//
// This is a C++ implementation with intrinsics. It follows the same
// Cooley-Tukey DIT / Gentleman-Sande DIF structure as SPQLIOS, but with
// interleaved data so each YMM register holds 2 complete complex numbers.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <immintrin.h>

#include <params.hpp>
#include "fft_processor_spqlios_intl.h"

// ── Trig table helpers ──────────────────────────────────────────────────────

static double accurate_cos(int32_t i, int32_t n) {
    i = ((i % n) + n) % n;
    if (i >= 3 * n / 4) return cos(2. * M_PI * (n - i) / double(n));
    if (i >= 2 * n / 4) return -cos(2. * M_PI * (i - n / 2) / double(n));
    if (i >= 1 * n / 4) return -cos(2. * M_PI * (n / 2 - i) / double(n));
    return cos(2. * M_PI * i / double(n));
}

static double accurate_sin(int32_t i, int32_t n) {
    i = ((i % n) + n) % n;
    if (i >= 3 * n / 4) return -sin(2. * M_PI * (n - i) / double(n));
    if (i >= 2 * n / 4) return -sin(2. * M_PI * (i - n / 2) / double(n));
    if (i >= 1 * n / 4) return sin(2. * M_PI * (n / 2 - i) / double(n));
    return sin(2. * M_PI * i / double(n));
}

// ── Interleaved trig table layout ───────────────────────────────────────────
// For each halfnn stage, store twiddles as interleaved complex values:
//   [cos0, sin0, cos1, sin1, cos2, sin2, cos3, sin3]
// Each group of 4 twiddles = 8 doubles = 1 YMM pair (or 2 c64x2).
//
// FFT trig tables (DIT, -omega twiddles):
//   For halfnn = 2, 4, 8, ..., ns4:
//     for off = 0; off < halfnn; off += 2:
//       [cos(-j*(off)), sin(-j*(off)), cos(-j*(off+1)), sin(-j*(off+1))]
//   Then final twist twiddles (also interleaved).
//
// IFFT trig tables (DIF, +omega twiddles):
//   First: twist twiddles (interleaved)
//   Then for nn = ns4, ns4/2, ..., 4:
//     halfnn = nn/2, j = n/nn
//     for off = 0; off < halfnn; off += 2:
//       [cos(j*off), sin(j*off), cos(j*(off+1)), sin(j*(off+1))]

struct INTL_FFT_PRECOMP {
    int32_t n;
    double *trig_tables;
    double *data;
    void *buf;
};

struct INTL_IFFT_PRECOMP {
    int32_t n;
    double *trig_tables;
    double *data;
    void *buf;
};

// ── Complex multiply with AVX2 ──────────────────────────────────────────────
// a = [a_re0, a_im0, a_re1, a_im1]
// w = [w_re0, w_im0, w_re1, w_im1]
// result = a * w (complex product for each pair)
static inline __m256d cmul(__m256d a, __m256d w) {
    // w = [x0, y0, x1, y1], swap to get [y0, x0, y1, x1]
    __m256d w_swap = _mm256_permute_pd(w, 0b0101);
    // a_re = [a_re0, a_re0, a_re1, a_re1]
    __m256d a_re = _mm256_unpacklo_pd(a, a);
    // a_im = [a_im0, a_im0, a_im1, a_im1]
    __m256d a_im = _mm256_unpackhi_pd(a, a);
    // a_re * w = [a_re0*x0, a_re0*y0, a_re1*x1, a_re1*y1]
    // a_im * w_swap = [a_im0*y0, a_im0*x0, a_im1*y1, a_im1*x1]
    // result = a_re*w -/+ a_im*w_swap = [a_re*x - a_im*y, a_re*y + a_im*x, ...]
    return _mm256_fmaddsub_pd(a_re, w, _mm256_mul_pd(a_im, w_swap));
}

// ── DIT FFT (forward) ───────────────────────────────────────────────────────
// Interleaved butterfly: processes 2 complex values per YMM.
// Data: c[ns2] complex values stored as [re, im, re, im, ...]
// ns2 = N/2 = number of complex points.
static void intl_fft(const INTL_FFT_PRECOMP *tables, double *c) {
    const int32_t n = tables->n;       // = 2*N
    const int32_t ns2 = n / 4;        // = N/2 = number of complex points
    const double *trig = tables->trig_tables;
    // c has ns2 complex values = ns2*2 doubles

    // ── Size-2 DIT butterfly (no twiddle) ──
    // Pairs: (c[k], c[k + ns2/2]) for k = 0..ns2/2-1
    // But in DIT split-radix, the first pass pairs (0, 1), (2, 3), ...
    // For Cooley-Tukey DIT with bit-reversal input, we do pairs at distance 1.
    // In SPQLIOS style (assumes bit-reversed input): pairs at stride halfnn=1.
    //
    // Actually, SPQLIOS uses a specific ordering. The interleaved version
    // operates on the same indexing but with interleaved storage.
    // For the DIT butterfly at halfnn:
    //   for block in 0..ns2 step 2*halfnn:
    //     for off in 0..halfnn:
    //       a = c[block + off]
    //       b = c[block + halfnn + off] * twiddle[off]
    //       c[block + off] = a + b
    //       c[block + halfnn + off] = a - b
    //
    // Each complex value is 2 doubles. So c[k] starts at c_data[2*k].
    // With YMM, we process 2 complex values at a time (off, off+1).

    // Size-2 butterfly (halfnn=1): no twiddle (W=1)
    for (int32_t block = 0; block < ns2; block += 2) {
        double *p0 = c + block * 2;      // complex c[block]
        double *p1 = c + (block + 1) * 2; // complex c[block+1]
        // Load 2 complex from p0 and p1 (but they're adjacent, so load 4 complex)
        __m256d v0 = _mm256_load_pd(p0);  // [re0, im0, re1, im1]
        // v0 contains c[block] in low 128 and c[block+1] in high 128
        // For size-2: new_lo = lo + hi, new_hi = lo - hi
        // But lo = c[block], hi = c[block+1]
        __m128d lo = _mm256_castpd256_pd128(v0);
        __m128d hi = _mm256_extractf128_pd(v0, 1);
        __m128d sum = _mm_add_pd(lo, hi);
        __m128d dif = _mm_sub_pd(lo, hi);
        _mm_store_pd(p0, sum);
        _mm_store_pd(p1, dif);
    }

    // General DIT butterfly for halfnn = 2, 4, 8, ...
    const double *cur_trig = trig;
    for (int32_t halfnn = 2; halfnn < ns2; halfnn *= 2) {
        int32_t nn = halfnn * 2;
        for (int32_t block = 0; block < ns2; block += nn) {
            const double *tw = cur_trig;
            for (int32_t off = 0; off < halfnn; off += 2) {
                double *p0 = c + (block + off) * 2;
                double *p1 = c + (block + halfnn + off) * 2;
                __m256d a = _mm256_load_pd(p0);   // 2 complex: c[block+off], c[block+off+1]
                __m256d b = _mm256_load_pd(p1);   // 2 complex: c[block+halfnn+off], ...
                __m256d w = _mm256_load_pd(tw);   // 2 twiddles
                __m256d bw = cmul(b, w);
                _mm256_store_pd(p0, _mm256_add_pd(a, bw));
                _mm256_store_pd(p1, _mm256_sub_pd(a, bw));
                tw += 4;
            }
        }
        cur_trig += halfnn * 2;  // advance past this stage's twiddles
    }

    // Final twist multiply
    for (int32_t j = 0; j < ns2; j += 2) {
        double *p = c + j * 2;
        __m256d a = _mm256_load_pd(p);
        __m256d w = _mm256_load_pd(cur_trig);
        _mm256_store_pd(p, cmul(a, w));
        cur_trig += 4;
    }
}

// ── DIF IFFT (inverse) ──────────────────────────────────────────────────────
static void intl_ifft(const INTL_IFFT_PRECOMP *tables, double *c) {
    const int32_t n = tables->n;
    const int32_t ns2 = n / 4;
    const double *trig = tables->trig_tables;

    // First: twist multiply
    const double *cur_trig = trig;
    for (int32_t j = 0; j < ns2; j += 2) {
        double *p = c + j * 2;
        __m256d a = _mm256_load_pd(p);
        __m256d w = _mm256_load_pd(cur_trig);
        _mm256_store_pd(p, cmul(a, w));
        cur_trig += 4;
    }

    // General DIF butterfly for nn = ns2, ns2/2, ..., 4
    for (int32_t nn = ns2; nn >= 4; nn /= 2) {
        int32_t halfnn = nn / 2;
        for (int32_t block = 0; block < ns2; block += nn) {
            const double *tw = cur_trig;
            for (int32_t off = 0; off < halfnn; off += 2) {
                double *p0 = c + (block + off) * 2;
                double *p1 = c + (block + halfnn + off) * 2;
                __m256d a = _mm256_load_pd(p0);
                __m256d b = _mm256_load_pd(p1);
                __m256d sum = _mm256_add_pd(a, b);
                __m256d dif = _mm256_sub_pd(a, b);
                __m256d w = _mm256_load_pd(tw);
                _mm256_store_pd(p0, sum);
                _mm256_store_pd(p1, cmul(dif, w));
                tw += 4;
            }
        }
        cur_trig += halfnn * 2;
    }

    // Size-2 DIF butterfly (halfnn=1): no twiddle
    for (int32_t block = 0; block < ns2; block += 2) {
        double *p0 = c + block * 2;
        double *p1 = c + (block + 1) * 2;
        __m128d lo = _mm_load_pd(p0);
        __m128d hi = _mm_load_pd(p1);
        _mm_store_pd(p0, _mm_add_pd(lo, hi));
        _mm_store_pd(p1, _mm_sub_pd(lo, hi));
    }
}

// ── Table construction ──────────────────────────────────────────────────────

static void *new_intl_fft_table(int32_t nn) {
    assert(nn >= 16 && (nn & (nn - 1)) == 0);
    int32_t n = 2 * nn;
    int32_t ns2 = nn / 2;  // number of complex points

    // Count trig table size:
    // General stages: halfnn = 2, 4, ..., ns2/2 → Σ halfnn = ns2 - 2 complex values
    // Final twist: ns2 complex values
    // Total: (ns2 - 2 + ns2) = 2*ns2 - 2 complex values = (2*ns2 - 2) * 2 doubles
    int32_t trig_count = 0;
    for (int32_t halfnn = 2; halfnn < ns2; halfnn *= 2)
        trig_count += halfnn;
    trig_count += ns2;  // final twist

    INTL_FFT_PRECOMP *reps = new INTL_FFT_PRECOMP;
    void *buf = aligned_alloc(64, (trig_count * 2 + nn) * sizeof(double));
    reps->n = n;
    reps->trig_tables = (double *)buf;
    reps->data = reps->trig_tables + trig_count * 2;
    reps->buf = buf;

    double *ptr = reps->trig_tables;

    // General stages: DIT twiddles
    for (int32_t halfnn = 2; halfnn < ns2; halfnn *= 2) {
        int32_t nn_stage = 2 * halfnn;
        int32_t j = ns2 / nn_stage;  // frequency step
        for (int32_t off = 0; off < halfnn; off += 2) {
            for (int32_t k = 0; k < 2; k++) {
                *(ptr++) = accurate_cos(-j * (off + k), ns2);
                *(ptr++) = accurate_sin(-j * (off + k), ns2);
            }
        }
    }

    // Final twist: multiply by omega^j where omega = e^(-2pi*i / n)
    for (int32_t j = 0; j < ns2; j += 2) {
        for (int32_t k = 0; k < 2; k++) {
            *(ptr++) = accurate_cos(-(j + k), n);
            *(ptr++) = accurate_sin(-(j + k), n);
        }
    }

    return reps;
}

static void *new_intl_ifft_table(int32_t nn) {
    assert(nn >= 16 && (nn & (nn - 1)) == 0);
    int32_t n = 2 * nn;
    int32_t ns2 = nn / 2;

    int32_t trig_count = ns2;  // twist
    for (int32_t nn_stage = ns2; nn_stage >= 4; nn_stage /= 2)
        trig_count += nn_stage / 2;

    INTL_IFFT_PRECOMP *reps = new INTL_IFFT_PRECOMP;
    void *buf = aligned_alloc(64, (trig_count * 2 + nn) * sizeof(double));
    reps->n = n;
    reps->trig_tables = (double *)buf;
    reps->data = reps->trig_tables + trig_count * 2;
    reps->buf = buf;

    double *ptr = reps->trig_tables;

    // Twist: multiply by omega^j where omega = e^(+2pi*i / n)
    for (int32_t j = 0; j < ns2; j += 2) {
        for (int32_t k = 0; k < 2; k++) {
            *(ptr++) = accurate_cos(j + k, n);
            *(ptr++) = accurate_sin(j + k, n);
        }
    }

    // DIF twiddles
    for (int32_t nn_stage = ns2; nn_stage >= 4; nn_stage /= 2) {
        int32_t halfnn = nn_stage / 2;
        int32_t j = ns2 / nn_stage;
        for (int32_t off = 0; off < halfnn; off += 2) {
            for (int32_t k = 0; k < 2; k++) {
                *(ptr++) = accurate_cos(j * (off + k), ns2);
                *(ptr++) = accurate_sin(j * (off + k), ns2);
            }
        }
    }

    return reps;
}

// ── Conversion helpers ──────────────────────────────────────────────────────
// Convert split [re0,re1,...,reN/2-1] to interleaved [re0,im0,re1,im1,...]
// The SPQLIOS convention: re[0..ns2-1] are the even-indexed frequency bins,
// and im[0..ns2-1] are the odd-indexed. With interleaved format, we pair them.

// Magic-constant i64↔f64 conversion
static inline __m256i mm256_cvtpd_epi64(const __m256d x) {
    const __m256d magic_d = _mm256_set1_pd(6755399441055744.0);
    const __m256i magic_i = _mm256_set1_epi64x(0x4338000000000000LL);
    return _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(x, magic_d)), magic_i);
}

static inline __m256d mm256_cvtepi64_pd(const __m256i x) {
    const __m256i magic_i = _mm256_set1_epi64x(0x4338000000000000LL);
    const __m256d magic_d = _mm256_set1_pd(6755399441055744.0);
    return _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_add_epi64(x, magic_i)), magic_d);
}

// ── FFT_Processor_Spqlios_Intl implementation ───────────────────────────────

static int32_t rev(int32_t x, int32_t M) {
    int32_t reps = 0;
    for (int32_t j = M; j > 1; j /= 2) {
        reps = 2 * reps + (x % 2);
        x /= 2;
    }
    return reps;
}

FFT_Processor_Spqlios_Intl::FFT_Processor_Spqlios_Intl(const int32_t N)
    : _2N(2 * N), N(N), Ns2(N / 2)
{
    tables_direct = new_intl_fft_table(N);
    tables_reverse = new_intl_ifft_table(N);
    real_inout_direct = ((INTL_FFT_PRECOMP *)tables_direct)->data;
    reva = new int32_t[Ns2];
    cosomegaxminus1 = new double[2 * _2N];
    sinomegaxminus1 = cosomegaxminus1 + _2N;
    int32_t rev1 = rev(1, _2N);
    int32_t rev3 = rev(3, _2N);
    for (int32_t revi = rev1; revi < rev3; revi++)
        reva[revi - rev1] = rev(revi, _2N);
    for (int32_t j = 0; j < _2N; j++) {
        cosomegaxminus1[j] = cos(2 * M_PI * j / _2N) - 1.;
        sinomegaxminus1[j] = sin(2 * M_PI * j / _2N);
    }
}

FFT_Processor_Spqlios_Intl::~FFT_Processor_Spqlios_Intl() {
    free(((INTL_FFT_PRECOMP *)tables_direct)->buf);
    free(((INTL_IFFT_PRECOMP *)tables_reverse)->buf);
    delete (INTL_FFT_PRECOMP *)tables_direct;
    delete (INTL_IFFT_PRECOMP *)tables_reverse;
    delete[] cosomegaxminus1;
    delete[] reva;
}

// IFFT: torus32 → interleaved frequency domain
void FFT_Processor_Spqlios_Intl::execute_reverse_torus32(double *res, const uint32_t *a) {
    const int32_t *aa = (const int32_t *)a;
    // Convert int32 to double, then arrange as interleaved complex
    // In SPQLIOS convention, the IFFT input is N real values.
    // We need to pack them into ns2 complex values for the IFFT.
    // The mapping: complex[k] = (a[reva[k]], 0) ... actually no.
    //
    // In SPQLIOS, the IFFT converts N real torus values into ns2 complex
    // frequency-domain values. The real values are first converted to doubles,
    // then the raw IFFT is called on the ns2 complex points.
    //
    // For interleaved format: the output res[] is [re0,im0,re1,im1,...] with
    // N doubles total = ns2 complex values.
    //
    // The input is N int32 values. In SPQLIOS, these map to the real and
    // imaginary parts of ns2 complex inputs: re[0..ns2-1] = a[0..ns2-1],
    // im[0..ns2-1] = a[ns2..N-1].
    //
    // For interleaved: we interleave a[0..ns2-1] (re) with a[ns2..N-1] (im).

    for (int32_t i = 0; i < Ns2; i++) {
        res[2 * i]     = (double)aa[i];        // real part
        res[2 * i + 1] = (double)aa[i + Ns2];  // imaginary part
    }
    intl_ifft((const INTL_IFFT_PRECOMP *)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_reverse_int(double *res, const int32_t *a) {
    for (int32_t i = 0; i < Ns2; i++) {
        res[2 * i]     = (double)a[i];
        res[2 * i + 1] = (double)a[i + Ns2];
    }
    intl_ifft((const INTL_IFFT_PRECOMP *)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_reverse_uint(double *res, const uint32_t *a) {
    for (int32_t i = 0; i < Ns2; i++) {
        res[2 * i]     = (double)a[i];
        res[2 * i + 1] = (double)a[i + Ns2];
    }
    intl_ifft((const INTL_IFFT_PRECOMP *)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_reverse_torus64(double *res, const uint64_t *a) {
    const int64_t *aa = (const int64_t *)a;
    for (int32_t i = 0; i < Ns2; i += 2) {
        // Convert 2 re + 2 im using magic constants, interleave
        __m256i vi_re = _mm256_loadu_si256((const __m256i *)(aa + i));
        __m256i vi_im = _mm256_loadu_si256((const __m256i *)(aa + i + Ns2));
        __m256d vd_re = mm256_cvtepi64_pd(vi_re);
        __m256d vd_im = mm256_cvtepi64_pd(vi_im);
        // Interleave: [re0, im0, re1, im1]
        __m256d lo = _mm256_unpacklo_pd(vd_re, vd_im);  // [re0, im0, re2, im2] - wrong!
        __m256d hi = _mm256_unpackhi_pd(vd_re, vd_im);  // [re1, im1, re3, im3] - wrong!
        // Actually for 256-bit: unpacklo gives [re0,im0] in low lane, [re2,im2] in high
        // We need [re0,im0,re1,im1]. So permute:
        __m256d out0 = _mm256_permute2f128_pd(lo, hi, 0x20); // [re0,im0,re1,im1]
        __m256d out1 = _mm256_permute2f128_pd(lo, hi, 0x31); // [re2,im2,re3,im3]
        _mm256_store_pd(res + 4 * i / 2, out0);  // wait, index wrong
        // i goes 0,2. For i=0: store at res[0..3] and res[4..7]
        // But we process 2 complex per iteration (i and i+1).
        // Actually let me redo this more carefully.
    }
    // Simpler approach for correctness first:
    for (int32_t i = 0; i < Ns2; i++) {
        res[2 * i]     = (double)((const int64_t *)a)[i];
        res[2 * i + 1] = (double)((const int64_t *)a)[i + Ns2];
    }
    intl_ifft((const INTL_IFFT_PRECOMP *)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_reverse_torus64_uint(double *res, const uint64_t *a) {
    for (int32_t i = 0; i < Ns2; i++) {
        res[2 * i]     = (double)a[i];
        res[2 * i + 1] = (double)a[i + Ns2];
    }
    intl_ifft((const INTL_IFFT_PRECOMP *)tables_reverse, res);
}

// FFT: interleaved frequency domain → torus32
void FFT_Processor_Spqlios_Intl::execute_direct_torus32(uint32_t *res, const double *a) {
    static const double _2sN = double(2) / double(N);
    // Copy and scale
    for (int32_t i = 0; i < N; i++)
        real_inout_direct[i] = a[i] * _2sN;
    intl_fft((const INTL_FFT_PRECOMP *)tables_direct, real_inout_direct);
    // De-interleave: extract re[0..ns2-1] and im[0..ns2-1]
    for (int32_t i = 0; i < Ns2; i++) {
        res[i]       = (uint32_t)(int64_t)real_inout_direct[2 * i];
        res[i + Ns2] = (uint32_t)(int64_t)real_inout_direct[2 * i + 1];
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus32_add(uint32_t *res, const double *a) {
    static const double _2sN = double(2) / double(N);
    for (int32_t i = 0; i < N; i++)
        real_inout_direct[i] = a[i] * _2sN;
    intl_fft((const INTL_FFT_PRECOMP *)tables_direct, real_inout_direct);
    for (int32_t i = 0; i < Ns2; i++) {
        res[i]       += (uint32_t)(int64_t)real_inout_direct[2 * i];
        res[i + Ns2] += (uint32_t)(int64_t)real_inout_direct[2 * i + 1];
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus64(uint64_t *res, double *a) {
    static const double _2sN = double(2) / double(N);
    for (int32_t i = 0; i < N; i++)
        a[i] *= _2sN;
    intl_fft((const INTL_FFT_PRECOMP *)tables_direct, a);
    for (int32_t i = 0; i < Ns2; i++) {
        // De-interleave and convert
        double re = a[2 * i];
        double im = a[2 * i + 1];
        // Use magic constant trick for f64→i64
        union { double d; int64_t i; } u;
        const double magic = 6755399441055744.0;
        u.d = re + magic; res[i] = (uint64_t)(u.i - 0x4338000000000000LL);
        u.d = im + magic; res[i + Ns2] = (uint64_t)(u.i - 0x4338000000000000LL);
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus64_add(uint64_t *res, double *a) {
    static const double _2sN = double(2) / double(N);
    for (int32_t i = 0; i < N; i++)
        a[i] *= _2sN;
    intl_fft((const INTL_FFT_PRECOMP *)tables_direct, a);
    for (int32_t i = 0; i < Ns2; i++) {
        double re = a[2 * i];
        double im = a[2 * i + 1];
        union { double d; int64_t i; } u;
        const double magic = 6755399441055744.0;
        u.d = re + magic; res[i] += (uint64_t)(u.i - 0x4338000000000000LL);
        u.d = im + magic; res[i + Ns2] += (uint64_t)(u.i - 0x4338000000000000LL);
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus32_q(uint32_t *res, const double *a, const uint32_t q) {
    static const double _2sN = double(2) / double(N);
    for (int32_t i = 0; i < N; i++)
        real_inout_direct[i] = a[i] * _2sN;
    intl_fft((const INTL_FFT_PRECOMP *)tables_direct, real_inout_direct);
    for (int32_t i = 0; i < Ns2; i++) {
        res[i]       = uint32_t((int64_t(real_inout_direct[2*i])%q+q)%q);
        res[i + Ns2] = uint32_t((int64_t(real_inout_direct[2*i+1])%q+q)%q);
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus32_rescale(uint32_t *res, const double *a, const double Δ) {
    static const double _2sN = double(2) / double(N);
    for (int32_t i = 0; i < N; i++)
        real_inout_direct[i] = a[i] * _2sN;
    intl_fft((const INTL_FFT_PRECOMP *)tables_direct, real_inout_direct);
    for (int32_t i = 0; i < Ns2; i++) {
        res[i]       = (uint32_t)(int64_t)(real_inout_direct[2*i]/Δ);
        res[i + Ns2] = (uint32_t)(int64_t)(real_inout_direct[2*i+1]/Δ);
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus32_rescale_clpx(
    uint32_t *res, const double *a, const double q, const uint32_t plain_modulus)
{
    // TODO: implement properly
    static const double _2sN = double(2) / double(N);
    for (int32_t i = 0; i < N; i++)
        real_inout_direct[i] = a[i] * _2sN;
    intl_fft((const INTL_FFT_PRECOMP *)tables_direct, real_inout_direct);
    // De-interleave to split format for CLPX formula
    double *split_re = (double *)alloca(Ns2 * sizeof(double));
    double *split_im = (double *)alloca(Ns2 * sizeof(double));
    for (int32_t i = 0; i < Ns2; i++) {
        split_re[i] = real_inout_direct[2*i];
        split_im[i] = real_inout_direct[2*i+1];
    }
    for (int32_t i = 0; i < N; i++) {
        double val_re = (i < Ns2) ? split_re[i] : split_im[i - Ns2];
        if (i == 0)
            res[i] = (uint32_t)std::llround(-split_re[N-1]/q - split_re[0]*plain_modulus/q);
        else if (i < Ns2)
            res[i] = (uint32_t)std::llround(split_re[i-1]/q - split_re[i]*plain_modulus/q);
        else if (i == Ns2)
            res[i] = (uint32_t)std::llround(-split_im[Ns2-1]/q - split_im[0]*plain_modulus/q);
        else
            res[i] = (uint32_t)std::llround(split_im[i-Ns2-1]/q - split_im[i-Ns2]*plain_modulus/q);
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus64_rescale(uint64_t *res, const double *a, const double Δ) {
    static const double _2sN = double(2) / double(N);
    alignas(64) double tmp[N];
    for (int32_t i = 0; i < N; i++)
        tmp[i] = a[i] * _2sN;
    intl_fft((const INTL_FFT_PRECOMP *)tables_direct, tmp);
    for (int32_t i = 0; i < Ns2; i++) {
        res[i]       = (uint64_t)std::round(tmp[2*i]/Δ);
        res[i + Ns2] = (uint64_t)std::round(tmp[2*i+1]/Δ);
    }
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus64_rescale_clpx(
    uint64_t *res, const double *a, const uint32_t plain_modulus)
{
    // TODO: implement properly
    execute_direct_torus64(res, const_cast<double*>(a));
}

thread_local FFT_Processor_Spqlios_Intl fftplvl1(TFHEpp::lvl1param::n);
thread_local FFT_Processor_Spqlios_Intl fftplvl2(TFHEpp::lvl2param::n);
thread_local FFT_Processor_Spqlios_Intl fftplvl3(TFHEpp::lvl3param::n);
