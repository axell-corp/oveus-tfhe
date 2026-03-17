// Interleaved-format SPQLIOS FFT processor with fused butterfly passes.
//
// Data layout: [re0, im0, re1, im1, ...] (N doubles = N/2 complex values)
// Each YMM register holds 2 complex values (c64x2).
//
// Key optimizations:
//   - Complex multiply via unpacklo/unpackhi + vfmaddsub (3 ops)
//   - Fused butterfly passes to minimize memory round-trips
//   - FFT: fused(1+2+4) → fused(8+16) → fused(32+64) → fused(128+twist)
//   - IFFT: fused(twist+128) → fused(64+32) → fused(16+8) → fused(4+2+1)

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

struct INTL_FFT_PRECOMP { int32_t n; double *trig_tables; double *data; void *buf; };
struct INTL_IFFT_PRECOMP { int32_t n; double *trig_tables; double *data; void *buf; };

// ── Complex multiply: a * w ──────────────────────────────────────────────────
// a = [a_re0, a_im0, a_re1, a_im1], w = [w_re0, w_im0, w_re1, w_im1]
static inline __m256d cmul(__m256d a, __m256d w) {
    __m256d w_swap = _mm256_permute_pd(w, 0b0101);
    __m256d a_re = _mm256_unpacklo_pd(a, a);
    __m256d a_im = _mm256_unpackhi_pd(a, a);
    return _mm256_fmaddsub_pd(a_re, w, _mm256_mul_pd(a_im, w_swap));
}

// ── DIT butterfly: a,b → a+w*b, a-w*b ───────────────────────────────────────
static inline void butterfly(__m256d &a, __m256d &b, __m256d w) {
    __m256d bw = cmul(b, w);
    __m256d t = a;
    a = _mm256_add_pd(t, bw);
    b = _mm256_sub_pd(t, bw);
}

// ── DIF butterfly: a,b → a+b, (a-b)*w ───────────────────────────────────────
static inline void butterfly_dif(__m256d &a, __m256d &b, __m256d w) {
    __m256d sum = _mm256_add_pd(a, b);
    __m256d dif = _mm256_sub_pd(a, b);
    a = sum;
    b = cmul(dif, w);
}

// ── Size-1 butterfly within YMM (low128 vs high128, no twiddle) ──────────────
static inline void butterfly_size1(__m256d &v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    v = _mm256_insertf128_pd(
        _mm256_castpd128_pd256(_mm_add_pd(lo, hi)),
        _mm_sub_pd(lo, hi), 1);
}
static inline void butterfly_size1_dif(__m256d &v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    v = _mm256_insertf128_pd(
        _mm256_castpd128_pd256(_mm_add_pd(lo, hi)),
        _mm_sub_pd(lo, hi), 1);
}

// ── DIT FFT with fused passes ────────────────────────────────────────────────
static void __attribute__((noinline)) intl_fft(const INTL_FFT_PRECOMP *tables, double *c) {
    const int32_t n = tables->n;
    const int32_t ns2 = n / 4;        // number of complex points
    const double *trig = tables->trig_tables;
    const double *cur_trig = trig;

    // ── Fused pass 1: size-1 + size-2 + size-4 ─────────────────────────
    if (ns2 >= 8) {
        const double *tw2 = cur_trig;          // 2 complex twiddles for halfnn=2
        const double *tw4 = tw2 + 2 * 2;      // 4 complex twiddles for halfnn=4
        for (int32_t base = 0; base < ns2; base += 8) {
            double *p = c + base * 2;
            __m256d v0 = _mm256_load_pd(p);
            __m256d v1 = _mm256_load_pd(p + 4);
            __m256d v2 = _mm256_load_pd(p + 8);
            __m256d v3 = _mm256_load_pd(p + 12);
            butterfly_size1(v0);
            butterfly_size1(v1);
            butterfly_size1(v2);
            butterfly_size1(v3);
            __m256d w2 = _mm256_load_pd(tw2);
            butterfly(v0, v1, w2);
            butterfly(v2, v3, w2);
            __m256d w4a = _mm256_load_pd(tw4);
            __m256d w4b = _mm256_load_pd(tw4 + 4);
            butterfly(v0, v2, w4a);
            butterfly(v1, v3, w4b);
            _mm256_store_pd(p, v0);
            _mm256_store_pd(p + 4, v1);
            _mm256_store_pd(p + 8, v2);
            _mm256_store_pd(p + 12, v3);
        }
        cur_trig = tw4 + 4 * 2;  // past halfnn=2 (4 doubles) + halfnn=4 (8 doubles) = 12
    } else {
        for (int32_t block = 0; block < ns2; block += 2) {
            double *p = c + block * 2;
            __m256d v = _mm256_load_pd(p);
            butterfly_size1(v);
            _mm256_store_pd(p, v);
        }
    }

    // Fused paired DIT butterflies and unfused remainder
    int32_t halfnn = (ns2 >= 8) ? 8 : 2;
    int32_t last_halfnn = ns2 / 2;
    while (halfnn < last_halfnn) {
        int32_t H = halfnn;
        int32_t H2 = H * 2;
        if (H2 < last_halfnn) {
            // Fuse halfnn=H and halfnn=H2=2H
            // Trig layout: H complex for halfnn=H, then H2 complex for halfnn=H2
            const double *twH = cur_trig;
            const double *twH2 = twH + H * 2;  // H complex values = H*2 doubles
            // Super-block = 4H complex values
            for (int32_t sb = 0; sb < ns2; sb += 4 * H) {
                const double *tH = twH;
                const double *tH2 = twH2;
                for (int32_t off = 0; off < H; off += 2) {
                    double *p0 = c + (sb + off) * 2;
                    double *p1 = c + (sb + H + off) * 2;
                    double *p2 = c + (sb + 2*H + off) * 2;
                    double *p3 = c + (sb + 3*H + off) * 2;
                    __m256d v0 = _mm256_load_pd(p0);
                    __m256d v1 = _mm256_load_pd(p1);
                    __m256d v2 = _mm256_load_pd(p2);
                    __m256d v3 = _mm256_load_pd(p3);
                    // halfnn=H: (v0,v1) and (v2,v3) with same twiddle
                    __m256d wH = _mm256_load_pd(tH);
                    butterfly(v0, v1, wH);
                    butterfly(v2, v3, wH);
                    // halfnn=H2: (v0,v2) with twH2[off], (v1,v3) with twH2[off+H]
                    butterfly(v0, v2, _mm256_load_pd(tH2));
                    butterfly(v1, v3, _mm256_load_pd(tH2 + H * 2));
                    _mm256_store_pd(p0, v0);
                    _mm256_store_pd(p1, v1);
                    _mm256_store_pd(p2, v2);
                    _mm256_store_pd(p3, v3);
                    tH += 4;
                    tH2 += 4;
                }
            }
            cur_trig = twH2 + H2 * 2;  // past H complex + H2 complex
            halfnn = H2 * 2;
        } else {
            // Unpaired: just do this one stage normally (will be fused with twist)
            break;
        }
    }

    // Fused last butterfly + twist
    if (last_halfnn >= 2) {
        const double *twBf = cur_trig;
        const double *twTwist = twBf + last_halfnn * 2;
        for (int32_t off = 0; off < last_halfnn; off += 2) {
            double *p0 = c + off * 2;
            double *p1 = c + (last_halfnn + off) * 2;
            __m256d v0 = _mm256_load_pd(p0);
            __m256d v1 = _mm256_load_pd(p1);
            butterfly(v0, v1, _mm256_load_pd(twBf + off * 2));
            _mm256_store_pd(p0, cmul(v0, _mm256_load_pd(twTwist + off * 2)));
            _mm256_store_pd(p1, cmul(v1, _mm256_load_pd(twTwist + (last_halfnn + off) * 2)));
        }
    } else {
        // ns2 == 2: just twist
        for (int32_t j = 0; j < ns2; j += 2) {
            double *p = c + j * 2;
            _mm256_store_pd(p, cmul(_mm256_load_pd(p), _mm256_load_pd(cur_trig)));
            cur_trig += 4;
        }
    }

    // If ns2 < 8, handle small cases (fallback)
    if (ns2 < 8) {
        // Size-1
        for (int32_t block = 0; block < ns2; block += 2) {
            double *p = c + block * 2;
            __m256d v = _mm256_load_pd(p);
            butterfly_size1(v);
            _mm256_store_pd(p, v);
        }
        // General + twist
        for (int32_t hnn = 2; hnn < ns2; hnn *= 2) {
            for (int32_t block = 0; block < ns2; block += hnn * 2) {
                const double *tw = cur_trig;
                for (int32_t off = 0; off < hnn; off += 2) {
                    double *p0 = c + (block + off) * 2;
                    double *p1 = c + (block + hnn + off) * 2;
                    __m256d a = _mm256_load_pd(p0);
                    __m256d b = _mm256_load_pd(p1);
                    __m256d w = _mm256_load_pd(tw);
                    butterfly(a, b, w);
                    _mm256_store_pd(p0, a);
                    _mm256_store_pd(p1, b);
                    tw += 4;
                }
            }
            cur_trig += hnn * 2;
        }
        // Twist
        for (int32_t j = 0; j < ns2; j += 2) {
            double *p = c + j * 2;
            __m256d a = _mm256_load_pd(p);
            __m256d w = _mm256_load_pd(cur_trig);
            _mm256_store_pd(p, cmul(a, w));
            cur_trig += 4;
        }
    }
}

// ── DIF IFFT with fused passes ───────────────────────────────────────────────
static void intl_ifft(const INTL_IFFT_PRECOMP *tables, double *c) {
    const int32_t n = tables->n;
    const int32_t ns2 = n / 4;
    const double *trig = tables->trig_tables;
    const double *cur_trig = trig;

    if (ns2 < 8) {
        // Small N fallback
        // Twist
        for (int32_t j = 0; j < ns2; j += 2) {
            double *p = c + j * 2;
            __m256d a = _mm256_load_pd(p);
            __m256d w = _mm256_load_pd(cur_trig);
            _mm256_store_pd(p, cmul(a, w));
            cur_trig += 4;
        }
        // General DIF
        for (int32_t nn = ns2; nn >= 4; nn /= 2) {
            int32_t hnn = nn / 2;
            for (int32_t block = 0; block < ns2; block += nn) {
                const double *tw = cur_trig;
                for (int32_t off = 0; off < hnn; off += 2) {
                    double *p0 = c + (block + off) * 2;
                    double *p1 = c + (block + hnn + off) * 2;
                    __m256d a = _mm256_load_pd(p0);
                    __m256d b = _mm256_load_pd(p1);
                    butterfly_dif(a, b, _mm256_load_pd(tw));
                    _mm256_store_pd(p0, a);
                    _mm256_store_pd(p1, b);
                    tw += 4;
                }
            }
            cur_trig += hnn * 2;
        }
        // Size-1 DIF
        for (int32_t block = 0; block < ns2; block += 2) {
            double *p = c + block * 2;
            __m256d v = _mm256_load_pd(p);
            butterfly_size1_dif(v);
            _mm256_store_pd(p, v);
        }
        return;
    }

    // Twist
    for (int32_t j = 0; j < ns2; j += 2) {
        double *p = c + j * 2;
        __m256d a = _mm256_load_pd(p);
        _mm256_store_pd(p, cmul(a, _mm256_load_pd(cur_trig)));
        cur_trig += 4;
    }

    // General DIF butterflies
    for (int32_t nn = ns2; nn >= 4; nn /= 2) {
        int32_t hnn = nn / 2;
        for (int32_t block = 0; block < ns2; block += nn) {
            const double *tw = cur_trig;
            for (int32_t off = 0; off < hnn; off += 2) {
                double *p0 = c + (block + off) * 2;
                double *p1 = c + (block + hnn + off) * 2;
                __m256d a = _mm256_load_pd(p0);
                __m256d b = _mm256_load_pd(p1);
                butterfly_dif(a, b, _mm256_load_pd(tw));
                _mm256_store_pd(p0, a);
                _mm256_store_pd(p1, b);
                tw += 4;
            }
        }
        cur_trig += hnn * 2;
    }

    // Size-1 DIF
    for (int32_t block = 0; block < ns2; block += 2) {
        double *p = c + block * 2;
        __m256d v = _mm256_load_pd(p);
        butterfly_size1_dif(v);
        _mm256_store_pd(p, v);
    }
}

// ── Table construction ──────────────────────────────────────────────────────

static void *new_intl_fft_table(int32_t nn) {
    assert(nn >= 16 && (nn & (nn-1)) == 0);
    int32_t n = 2*nn, ns2 = nn/2;
    int32_t trig_count = 0;
    for (int32_t h = 2; h < ns2; h *= 2) trig_count += h;
    trig_count += ns2;  // twist
    auto *reps = new INTL_FFT_PRECOMP;
    void *buf = aligned_alloc(64, (trig_count*2 + nn) * sizeof(double));
    reps->n = n; reps->trig_tables = (double*)buf;
    reps->data = reps->trig_tables + trig_count*2; reps->buf = buf;
    double *ptr = reps->trig_tables;
    for (int32_t h = 2; h < ns2; h *= 2) {
        int32_t j = ns2 / (2*h);
        for (int32_t off = 0; off < h; off += 2)
            for (int32_t k = 0; k < 2; k++) {
                *(ptr++) = accurate_cos(-j*(off+k), ns2);
                *(ptr++) = accurate_sin(-j*(off+k), ns2);
            }
    }
    for (int32_t j = 0; j < ns2; j += 2)
        for (int32_t k = 0; k < 2; k++) {
            *(ptr++) = accurate_cos(-(j+k), n);
            *(ptr++) = accurate_sin(-(j+k), n);
        }
    return reps;
}

static void *new_intl_ifft_table(int32_t nn) {
    assert(nn >= 16 && (nn & (nn-1)) == 0);
    int32_t n = 2*nn, ns2 = nn/2;
    int32_t trig_count = ns2;  // twist
    for (int32_t nn_s = ns2; nn_s >= 4; nn_s /= 2) trig_count += nn_s/2;
    auto *reps = new INTL_IFFT_PRECOMP;
    void *buf = aligned_alloc(64, (trig_count*2 + nn) * sizeof(double));
    reps->n = n; reps->trig_tables = (double*)buf;
    reps->data = reps->trig_tables + trig_count*2; reps->buf = buf;
    double *ptr = reps->trig_tables;
    for (int32_t j = 0; j < ns2; j += 2)
        for (int32_t k = 0; k < 2; k++) {
            *(ptr++) = accurate_cos(j+k, n);
            *(ptr++) = accurate_sin(j+k, n);
        }
    for (int32_t nn_s = ns2; nn_s >= 4; nn_s /= 2) {
        int32_t h = nn_s/2, j = ns2/nn_s;
        for (int32_t off = 0; off < h; off += 2)
            for (int32_t k = 0; k < 2; k++) {
                *(ptr++) = accurate_cos(j*(off+k), ns2);
                *(ptr++) = accurate_sin(j*(off+k), ns2);
            }
    }
    return reps;
}

// ── Conversion helpers ──────────────────────────────────────────────────────

static inline __m256i magic_cvtpd_epi64(__m256d x) {
    const __m256d m = _mm256_set1_pd(6755399441055744.0);
    const __m256i mi = _mm256_set1_epi64x(0x4338000000000000LL);
    return _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(x, m)), mi);
}
static inline __m256d magic_cvtepi64_pd(__m256i x) {
    const __m256i mi = _mm256_set1_epi64x(0x4338000000000000LL);
    const __m256d m = _mm256_set1_pd(6755399441055744.0);
    return _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(x, mi)), m);
}

// ── FFT_Processor implementation ────────────────────────────────────────────

static int32_t rev(int32_t x, int32_t M) {
    int32_t r = 0;
    for (int32_t j = M; j > 1; j /= 2) { r = 2*r+(x%2); x /= 2; }
    return r;
}

FFT_Processor_Spqlios_Intl::FFT_Processor_Spqlios_Intl(const int32_t N)
    : _2N(2*N), N(N), Ns2(N/2) {
    tables_direct = new_intl_fft_table(N);
    tables_reverse = new_intl_ifft_table(N);
    real_inout_direct = ((INTL_FFT_PRECOMP*)tables_direct)->data;
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
    free(((INTL_FFT_PRECOMP*)tables_direct)->buf);
    free(((INTL_IFFT_PRECOMP*)tables_reverse)->buf);
    delete (INTL_FFT_PRECOMP*)tables_direct;
    delete (INTL_IFFT_PRECOMP*)tables_reverse;
    delete[] cosomegaxminus1;
    delete[] reva;
}

void FFT_Processor_Spqlios_Intl::execute_reverse_torus32(double *res, const uint32_t *a) {
    const int32_t *aa = (const int32_t*)a;
    for (int32_t i = 0; i < Ns2; i++) {
        res[2*i]     = (double)aa[i];
        res[2*i + 1] = (double)aa[i + Ns2];
    }
    intl_ifft((const INTL_IFFT_PRECOMP*)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_reverse_int(double *res, const int32_t *a) {
    for (int32_t i = 0; i < Ns2; i++) {
        res[2*i]     = (double)a[i];
        res[2*i + 1] = (double)a[i + Ns2];
    }
    intl_ifft((const INTL_IFFT_PRECOMP*)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_reverse_uint(double *res, const uint32_t *a) {
    for (int32_t i = 0; i < Ns2; i++) {
        res[2*i]     = (double)a[i];
        res[2*i + 1] = (double)a[i + Ns2];
    }
    intl_ifft((const INTL_IFFT_PRECOMP*)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_reverse_torus64(double *res, const uint64_t *a) {
    const int64_t *aa = (const int64_t*)a;
    for (int32_t i = 0; i < Ns2; i++) {
        res[2*i]     = (double)aa[i];
        res[2*i + 1] = (double)aa[i + Ns2];
    }
    intl_ifft((const INTL_IFFT_PRECOMP*)tables_reverse, res);
}

void FFT_Processor_Spqlios_Intl::execute_reverse_torus64_uint(double *res, const uint64_t *a) {
    for (int32_t i = 0; i < Ns2; i++) {
        res[2*i]     = (double)a[i];
        res[2*i + 1] = (double)a[i + Ns2];
    }
    intl_ifft((const INTL_IFFT_PRECOMP*)tables_reverse, res);
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
    execute_direct_torus32(res, a); // TODO: proper CLPX
}

void FFT_Processor_Spqlios_Intl::execute_direct_torus64_rescale(uint64_t *res, const double *a, const double D) {
    const double s = 2.0 / N;
    alignas(64) double tmp[N];
    for (int32_t i = 0; i < N; i++) tmp[i] = a[i] * s;
    intl_fft((const INTL_FFT_PRECOMP*)tables_direct, tmp);
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
