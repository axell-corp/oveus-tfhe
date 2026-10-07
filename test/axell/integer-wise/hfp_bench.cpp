#include <algorithm>
#include <axell/integer-wise/float/hfp.hpp>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using brP = TFHEpp::lvlh2param;
using iksP = TFHEpp::lvl2hparam;
using cbbrP = TFHEpp::cblvlh2param;
using ahP = TFHEpp::AHlvl2param;
using P = typename iksP::domainP;

struct Stats {
    double min_ms = 0.0;
    double mean_ms = 0.0;
    double max_ms = 0.0;
};

Stats stats(const std::vector<double> &samples)
{
    assert(!samples.empty());
    const auto [min_it, max_it] =
        std::minmax_element(samples.cbegin(), samples.cend());
    double sum = 0.0;
    for (const auto sample : samples) sum += sample;
    return {*min_it, sum / samples.size(), *max_it};
}

double elapsed_ms(const std::chrono::steady_clock::time_point start,
                  const std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_stats(const std::string_view format, const std::string_view op,
                 const Stats &result)
{
    std::cout << format << "," << op << "," << result.min_ms << ","
              << result.mean_ms << "," << result.max_ms << std::endl;
}

template <class Format, class Ref>
void assert_same_bits(const TFHEpp::HFP<Format, P> &cipher, const Ref &expected,
                      const TFHEpp::SecretKey &sk)
{
    const auto actual = TFHEpp::hfp_decrypt(cipher, sk);
    if (actual.get_bits() != expected.get_bits()) {
        throw std::runtime_error(
            "unexpected HFP benchmark result: actual_bits=" +
            std::to_string(actual.get_bits()) +
            " expected_bits=" + std::to_string(expected.get_bits()) +
            " actual=" + std::to_string(actual.to_double()) +
            " expected=" + std::to_string(expected.to_double()));
    }
}

struct DivProfileTime {
    double before_intdiv_ms = 0.0;
    double inv_ms = 0.0;
    double div_by_mul_ms = 0.0;
    double intdiv_ms = 0.0;
    double after_intdiv_ms = 0.0;
    double total_ms = 0.0;
};

template <class Format>
DivProfileTime profile_hfp_div(TFHEpp::HFP<Format, P> &out,
                               const TFHEpp::HFP<Format, P> &lhs,
                               const TFHEpp::HFP<Format, P> &rhs,
                               const TFHEpp::EvalKey &ek)
{
    using intP = typename Format::word_param;
    using CbP = typename cbbrP::targetP;
    static_assert(TFHEpp::hfp_detail::same_tlwe_layout_v<P, CbP>);

    DivProfileTime profile;
    const auto total_start = std::chrono::steady_clock::now();

    TFHEpp::HFP<Format, P> raw;
    for (int i = 0; i <= P::k * P::n; i++)
        raw.sign[i] = lhs.sign[i] + rhs.sign[i];
    TFHEpp::hfp_detail::word_mod2<brP, iksP, intP>(raw.sign, raw.sign, ek);

    constexpr size_t div_words = 2 * Format::mantissa_words;
    static_assert(div_words == 4);
    std::vector<TFHEpp::TLWE<P>> numerator(div_words), denominator(div_words);
    for (size_t i = 0; i < Format::mantissa_words; i++) {
        numerator[i + Format::mantissa_words - 1] = lhs.mantissa[i];
        denominator[i] = rhs.mantissa[i];
    }

    const auto before_intdiv_end = std::chrono::steady_clock::now();
    profile.before_intdiv_ms = elapsed_ms(total_start, before_intdiv_end);

    std::vector<TFHEpp::TLWE<P>> quotient(div_words), remainder(div_words);
    std::vector<TFHEpp::TLWE<P>> reciprocal;
    const auto inv_start = std::chrono::steady_clock::now();
    TFHEpp::hfp_detail::inv_lut_nr_direct_2word<brP, iksP, cbbrP, ahP, intP>(
        reciprocal, denominator, ek);
    const auto inv_end = std::chrono::steady_clock::now();
    profile.inv_ms = elapsed_ms(inv_start, inv_end);

    const auto div_by_mul_start = std::chrono::steady_clock::now();
    TFHEpp::hfp_detail::div_by_mul_shifted_2word<brP, iksP, ahP, intP>(
        quotient, remainder, numerator, denominator, reciprocal, ek);
    const auto div_by_mul_end = std::chrono::steady_clock::now();
    profile.div_by_mul_ms = elapsed_ms(div_by_mul_start, div_by_mul_end);

    const auto intdiv_end = std::chrono::steady_clock::now();
    profile.intdiv_ms = elapsed_ms(before_intdiv_end, intdiv_end);

    raw.mantissa.resize(Format::mantissa_words);
    for (size_t i = 0; i < Format::mantissa_words; i++)
        raw.mantissa[i] = quotient[i];

    TFHEpp::TLWE<P> exp_borrow;
    TFHEpp::LongSub<brP, iksP, intP>(exp_borrow, raw.exponent, lhs.exponent,
                                     rhs.exponent, ek);
    TFHEpp::hfp_detail::add_plain_words<brP, iksP, intP>(
        raw.exponent, raw.exponent,
        static_cast<uint64_t>(Format::bias - Format::mantissa_words + 1), ek);

    TFHEpp::TRGSWFFT<CbP> underflow_selector;
    TFHEpp::hfp_detail::bit_selector<iksP, cbbrP, ahP, intP>(underflow_selector,
                                                             exp_borrow, 0, ek);
    const auto zero_m =
        TFHEpp::hfp_detail::trivial_words<P, intP>(Format::mantissa_words, 0);
    const auto zero_e =
        TFHEpp::hfp_detail::trivial_words<P, intP>(Format::exponent_words, 0);
    TFHEpp::hfp_detail::extended_cmux<CbP>(raw.mantissa, raw.mantissa, zero_m,
                                           underflow_selector);
    TFHEpp::hfp_detail::extended_cmux<CbP>(raw.exponent, raw.exponent, zero_e,
                                           underflow_selector);

    TFHEpp::HFPCarryPropagateNoMantissaOverflow<brP, iksP, Format>(out, raw,
                                                                  ek);

    const auto total_end = std::chrono::steady_clock::now();
    profile.after_intdiv_ms = elapsed_ms(intdiv_end, total_end);
    profile.total_ms = elapsed_ms(total_start, total_end);
    return profile;
}


void bench_bfloat16_div_profile(const int iterations, TFHEpp::EvalKey &ek,
                                const TFHEpp::SecretKey &sk)
{
    using Format = TFHEpp::BFloat16Format;
    using Ref = TFHEpp::RefBFloat16;

    const auto one = Ref::from_double(1.0);
    const auto two = Ref::from_double(2.0);
    const auto c_one = TFHEpp::hfp_encrypt<Format, P>(one, sk);
    const auto c_two = TFHEpp::hfp_encrypt<Format, P>(two, sk);

    std::vector<double> before_intdiv_times;
    std::vector<double> inv_times;
    std::vector<double> div_by_mul_times;
    std::vector<double> intdiv_times;
    std::vector<double> after_intdiv_times;
    std::vector<double> total_times;
    before_intdiv_times.reserve(iterations);
    inv_times.reserve(iterations);
    div_by_mul_times.reserve(iterations);
    intdiv_times.reserve(iterations);
    after_intdiv_times.reserve(iterations);
    total_times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        TFHEpp::HFP<Format, P> out;
        const auto profile = profile_hfp_div<Format>(out, c_two, c_one, ek);
        assert_same_bits<Format>(out, two, sk);

        before_intdiv_times.push_back(profile.before_intdiv_ms);
        inv_times.push_back(profile.inv_ms);
        div_by_mul_times.push_back(profile.div_by_mul_ms);
        intdiv_times.push_back(profile.intdiv_ms);
        after_intdiv_times.push_back(profile.after_intdiv_ms);
        total_times.push_back(profile.total_ms);
    }

    print_stats("BFloat16", "div_before_intdiv", stats(before_intdiv_times));
    print_stats("BFloat16", "div_inv", stats(inv_times));
    print_stats("BFloat16", "div_divbymul", stats(div_by_mul_times));
    print_stats("BFloat16", "div_intdiv", stats(intdiv_times));
    print_stats("BFloat16", "div_after_intdiv", stats(after_intdiv_times));
    print_stats("BFloat16", "div_profiled_total", stats(total_times));
}

int parse_iterations(int argc, char **argv)
{
    if (argc < 2) return 1;
    const int iterations = std::atoi(argv[1]);
    return std::max(1, iterations);
}

}  // namespace

int main(int argc, char **argv)
{
    const int iterations = parse_iterations(argc, argv);

    TFHEpp::SecretKey sk;
    TFHEpp::EvalKey ek;
    ek.emplacebkfft<brP>(sk);
    ek.emplacebkfft<cbbrP>(sk);
    ek.emplaceiksk<iksP>(sk);
    ek.emplaceahk<ahP>(sk);
    ek.emplacecbsk<ahP>(sk);

    std::cout << "format,operation,min_ms,mean_ms,max_ms" << std::endl;
    bench_bfloat16_div_profile(iterations, ek, sk);
}
