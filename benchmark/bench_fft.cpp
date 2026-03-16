// Pin benchmark to a specific CPU core for stable measurements
#include <sched.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

#include "../include/tfhe++.hpp"
#include <immintrin.h>
#include "google-benchmark/include/benchmark/benchmark.h"

// Forward-declare the SPQLIOS conversion functions from x86.h
namespace SPQLIOS {
inline __m256i mm256_cvtpd_epi64(const __m256d x) {
    const __m256d magic_d = _mm256_set1_pd(6755399441055744.0);
    const __m256i magic_i = _mm256_set1_epi64x(0x4338000000000000LL);
    const __m256d adjusted = _mm256_add_pd(x, magic_d);
    return _mm256_sub_epi64(_mm256_castpd_si256(adjusted), magic_i);
}
inline __m256d mm256_cvtepi64_pd(const __m256i x) {
    const __m256i magic_i = _mm256_set1_epi64x(0x4338000000000000LL);
    const __m256d magic_d = _mm256_set1_pd(6755399441055744.0);
    return _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_add_epi64(x, magic_i)), magic_d);
}
} // namespace SPQLIOS

// Pin the process to CPU core 0 (3D V-Cache CCD on Ryzen 9950X3D)
static void pin_to_core(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0)
        fprintf(stderr, "Warning: sched_setaffinity failed\n");
}

// Global initialization: pin to core 0 before any benchmarks run
struct PinInit {
    PinInit() { pin_to_core(0); }
} pin_init;

// ============================================================
// Benchmark: IFFT (execute_reverse_torus32) for lvl1
// ============================================================
static void BM_IFFT_lvl1(benchmark::State& state)
{
    constexpr int N = TFHEpp::lvl1param::n;
    alignas(64) std::array<uint32_t, N> input;
    alignas(64) TFHEpp::PolynomialInFD<TFHEpp::lvl1param> output;

    std::mt19937 rng(42);
    for (auto& v : input) v = rng();

    for (auto _ : state) {
        fftplvl1.execute_reverse_torus32(output.data(), input.data());
        benchmark::DoNotOptimize(output.data());
    }
}

// ============================================================
// Benchmark: FFT (execute_direct_torus32) for lvl1
// ============================================================
static void BM_FFT_lvl1(benchmark::State& state)
{
    constexpr int N = TFHEpp::lvl1param::n;
    alignas(64) TFHEpp::PolynomialInFD<TFHEpp::lvl1param> input;
    alignas(64) std::array<uint32_t, N> output;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1e9, 1e9);
    for (auto& v : input) v = dist(rng);

    for (auto _ : state) {
        fftplvl1.execute_direct_torus32(output.data(), input.data());
        benchmark::DoNotOptimize(output.data());
    }
}

// ============================================================
// Benchmark: MulInFD (pointwise complex multiplication)
// ============================================================
static void BM_MulInFD_lvl1(benchmark::State& state)
{
    constexpr int N = TFHEpp::lvl1param::n;
    alignas(64) TFHEpp::PolynomialInFD<TFHEpp::lvl1param> a, b, res;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    for (auto _ : state) {
        TFHEpp::MulInFD<N>(res, a, b);
        benchmark::DoNotOptimize(res.data());
    }
}

// ============================================================
// Benchmark: FMAInFD (pointwise complex FMA - hot path)
// ============================================================
static void BM_FMAInFD_lvl1(benchmark::State& state)
{
    constexpr int N = TFHEpp::lvl1param::n;
    alignas(64) TFHEpp::PolynomialInFD<TFHEpp::lvl1param> a, b, res;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    for (auto _ : state) {
        // Reset accumulator each iteration to avoid FP overflow
        memset(res.data(), 0, sizeof(res));
        TFHEpp::FMAInFD<N>(res, a, b);
        benchmark::DoNotOptimize(res.data());
    }
}

// ============================================================
// Benchmark: Full PolyMul pipeline for lvl1
// ============================================================
static void BM_PolyMul_lvl1(benchmark::State& state)
{
    constexpr int N = TFHEpp::lvl1param::n;
    alignas(64) TFHEpp::Polynomial<TFHEpp::lvl1param> a, b, res;

    std::mt19937 rng(42);
    for (auto& v : a) v = rng();
    for (auto& v : b) v = rng();

    for (auto _ : state) {
        TFHEpp::PolyMul<TFHEpp::lvl1param>(res, a, b);
        benchmark::DoNotOptimize(res.data());
    }
}

// ============================================================
// Benchmark: TwistIFFT + MulInFD + TwistFFT (simulates ExternalProduct core)
// ============================================================
static void BM_IFFTMulFFT_lvl1(benchmark::State& state)
{
    constexpr int N = TFHEpp::lvl1param::n;
    alignas(64) TFHEpp::Polynomial<TFHEpp::lvl1param> a, res;
    alignas(64) TFHEpp::PolynomialInFD<TFHEpp::lvl1param> fftb;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : a) v = rng();
    for (auto& v : fftb) v = dist(rng);

    for (auto _ : state) {
        alignas(64) TFHEpp::PolynomialInFD<TFHEpp::lvl1param> ffta;
        TFHEpp::TwistIFFT<TFHEpp::lvl1param>(ffta, a);
        TFHEpp::MulInFD<N>(ffta, fftb);
        TFHEpp::TwistFFT<TFHEpp::lvl1param>(res, ffta);
        benchmark::DoNotOptimize(res.data());
    }
}

// ============================================================
// Raw FFT/IFFT benchmarks for multiple N values (256, 512, 1024)
// Uses FFT_Processor_Spqlios directly with explicit N
// ============================================================
template <int N>
static void BM_RawIFFT(benchmark::State& state)
{
    static thread_local FFT_Processor_Spqlios proc(N);
    alignas(64) std::array<uint32_t, N> input;
    alignas(64) std::array<double, N> output;

    std::mt19937 rng(42);
    for (auto& v : input) v = rng();

    for (auto _ : state) {
        proc.execute_reverse_torus32(output.data(), input.data());
        benchmark::DoNotOptimize(output.data());
    }
}

template <int N>
static void BM_RawFFT(benchmark::State& state)
{
    static thread_local FFT_Processor_Spqlios proc(N);
    alignas(64) std::array<double, N> input;
    alignas(64) std::array<uint32_t, N> output;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1e9, 1e9);
    for (auto& v : input) v = dist(rng);

    for (auto _ : state) {
        proc.execute_direct_torus32(output.data(), input.data());
        benchmark::DoNotOptimize(output.data());
    }
}

template <int N>
static void BM_RawIFFTMulFFT(benchmark::State& state)
{
    static thread_local FFT_Processor_Spqlios proc(N);
    alignas(64) std::array<uint32_t, N> a;
    alignas(64) std::array<double, N> ffta, fftb;
    alignas(64) std::array<uint32_t, N> res;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : a) v = rng();
    for (auto& v : fftb) v = dist(rng);

    for (auto _ : state) {
        proc.execute_reverse_torus32(ffta.data(), a.data());
        // pointwise complex multiply (split re/im layout)
        const int ns2 = N / 2;
        for (int i = 0; i < ns2; i++) {
            double are = ffta[i], aim = ffta[i + ns2];
            double bre = fftb[i], bim = fftb[i + ns2];
            ffta[i]       = are * bre - aim * bim;
            ffta[i + ns2] = are * bim + aim * bre;
        }
        proc.execute_direct_torus32(res.data(), ffta.data());
        benchmark::DoNotOptimize(res.data());
    }
}

// ============================================================
// Benchmark: isolated i64→f64 conversion (mm256_cvtepi64_pd)
// ============================================================
template <int N>
static void BM_ConvI64toF64(benchmark::State& state)
{
    alignas(64) std::array<int64_t, N> input;
    alignas(64) std::array<double, N> output;
    std::mt19937_64 rng(42);
    for (auto& v : input) v = static_cast<int64_t>(rng()) >> 16;  // fits in 2^48
    for (auto _ : state) {
        for (int i = 0; i < N; i += 4) {
            __m256i vi = _mm256_load_si256((const __m256i*)(input.data() + i));
            __m256d vd = SPQLIOS::mm256_cvtepi64_pd(vi);
            _mm256_store_pd(output.data() + i, vd);
        }
        benchmark::DoNotOptimize(output.data());
    }
}

// ============================================================
// Benchmark: isolated f64→i64 conversion (mm256_cvtpd_epi64)
// ============================================================
template <int N>
static void BM_ConvF64toI64(benchmark::State& state)
{
    alignas(64) std::array<double, N> input;
    alignas(64) std::array<int64_t, N> output;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1e15, 1e15);
    for (auto& v : input) v = dist(rng);
    for (auto _ : state) {
        for (int i = 0; i < N; i += 4) {
            __m256d vd = _mm256_load_pd(input.data() + i);
            __m256i vi = SPQLIOS::mm256_cvtpd_epi64(vd);
            _mm256_store_si256((__m256i*)(output.data() + i), vi);
        }
        benchmark::DoNotOptimize(output.data());
    }
}

// ============================================================
// Benchmark: torus64 IFFT (i64→f64 conversion + raw ifft)
// ============================================================
template <int N>
static void BM_RawIFFT64(benchmark::State& state)
{
    static thread_local FFT_Processor_Spqlios proc(N);
    alignas(64) std::array<uint64_t, N> input;
    alignas(64) std::array<double, N> output;
    std::mt19937_64 rng(42);
    for (auto& v : input) v = rng();
    for (auto _ : state) {
        proc.execute_reverse_torus64(output.data(), input.data());
        benchmark::DoNotOptimize(output.data());
    }
}

// ============================================================
// Benchmark: torus64 FFT (scale + raw fft + f64→i64 conversion)
// ============================================================
template <int N>
static void BM_RawFFT64(benchmark::State& state)
{
    static thread_local FFT_Processor_Spqlios proc(N);
    alignas(64) std::array<double, N> input;
    alignas(64) std::array<uint64_t, N> output;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1e15, 1e15);
    for (auto& v : input) v = dist(rng);
    for (auto _ : state) {
        // Need mutable copy since execute_direct_torus64 modifies input in-place
        alignas(64) std::array<double, N> tmp;
        memcpy(tmp.data(), input.data(), sizeof(tmp));
        proc.execute_direct_torus64(output.data(), tmp.data());
        benchmark::DoNotOptimize(output.data());
    }
}

BENCHMARK(BM_IFFT_lvl1)
    ->Iterations(10000)
    ->Repetitions(5)
    ->DisplayAggregatesOnly(true);
BENCHMARK(BM_FFT_lvl1)
    ->Iterations(10000)
    ->Repetitions(5)
    ->DisplayAggregatesOnly(true);
BENCHMARK(BM_MulInFD_lvl1)
    ->Iterations(10000)
    ->Repetitions(5)
    ->DisplayAggregatesOnly(true);
BENCHMARK(BM_FMAInFD_lvl1)
    ->Iterations(10000)
    ->Repetitions(5)
    ->DisplayAggregatesOnly(true);
BENCHMARK(BM_PolyMul_lvl1)
    ->Iterations(10000)
    ->Repetitions(5)
    ->DisplayAggregatesOnly(true);
BENCHMARK(BM_IFFTMulFFT_lvl1)
    ->Iterations(10000)
    ->Repetitions(5)
    ->DisplayAggregatesOnly(true);

BENCHMARK(BM_RawIFFT<256>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK(BM_RawFFT<256>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK(BM_RawIFFTMulFFT<256>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK(BM_RawIFFT<512>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK(BM_RawFFT<512>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK(BM_RawIFFTMulFFT<512>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK(BM_RawIFFT<1024>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK(BM_RawFFT<1024>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK(BM_RawIFFTMulFFT<1024>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
// Isolated conversion benchmarks
BENCHMARK(BM_ConvI64toF64<512>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK(BM_ConvF64toI64<512>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK(BM_ConvI64toF64<1024>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK(BM_ConvF64toI64<1024>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
// Torus64 FFT benchmarks (to compare overhead vs torus32)
BENCHMARK(BM_RawIFFT64<512>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK(BM_RawFFT64<512>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK(BM_RawIFFT64<1024>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK(BM_RawFFT64<1024>)->Iterations(10000)->Repetitions(5)->DisplayAggregatesOnly(true);
BENCHMARK_MAIN();
