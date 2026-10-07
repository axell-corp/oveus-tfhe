#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <tfhe/circuitbootstrapping.hpp>
#include <tfhe/cloudkey.hpp>
#include <tfhe/cmuxmem.hpp>
#include <cmath>
#include <cstdint>
#include <tfhe/detwfa.hpp>
#include <tfhe/gatebootstrapping.hpp>
#include <tfhe/keyswitch.hpp>
#include <limits>
#include <ranges>
#include <tfhe/tlwe.hpp>
#include <tfhe/trlwe.hpp>
#include <type_traits>
#include <utils.hpp>
#include <vector>

#include "../add.hpp"
#include "../intkey.hpp"
#include "../long.hpp"

namespace TFHEpp {

template <uint32_t MantissaWords, uint32_t ExponentWords,
          int32_t Bias =
              ((1 << (ExponentWords * intparam::bit - 1)) + MantissaWords - 1),
          class WordParam = intparam>
struct HFPFormat {
    static_assert(WordParam::bit == 4,
                  "The Axell HFP formats intentionally use 4-bit words.");
    static_assert(MantissaWords >= 1,
                  "An Axell HFP format needs at least one mantissa word.");
    static_assert(ExponentWords >= 1,
                  "An Axell HFP format needs at least one exponent word.");
    static constexpr uint32_t word_bits = WordParam::bit;
    static constexpr uint32_t word_base = WordParam::t;
    static constexpr uint32_t mantissa_words = MantissaWords;
    static constexpr uint32_t exponent_words = ExponentWords;
    static constexpr uint32_t mantissa_bits = MantissaWords * WordParam::bit;
    static constexpr uint32_t exponent_bits = ExponentWords * WordParam::bit;
    static constexpr uint64_t mantissa_lower =
        uint64_t{1} << ((MantissaWords - 1) * WordParam::bit);
    static constexpr uint64_t mantissa_upper =
        uint64_t{1} << (MantissaWords * WordParam::bit);
    static constexpr uint32_t max_exponent =
        (uint32_t{1} << (ExponentWords * WordParam::bit)) - 1;
    static constexpr int32_t bias = Bias;
    using word_param = WordParam;
};

// 4-bit-word HFP presets. These are intentionally not IEEE754-exact:
// special values/subnormals are omitted, and the exponent is radix-16.
using BFloat16Format = HFPFormat<2, 2>;

template <class Format>
struct RefHFP {
    static constexpr uint32_t word_base = Format::word_base;
    static constexpr uint32_t word_bits = Format::word_bits;
    static constexpr uint64_t mantissa_lower = Format::mantissa_lower;
    static constexpr uint64_t mantissa_upper = Format::mantissa_upper;
    static constexpr uint32_t max_exponent = Format::max_exponent;

    uint8_t sign = 0;
    uint32_t exponent = 0;
    uint64_t mantissa = 0;

    static RefHFP zero(const uint8_t sign = 0)
    {
        RefHFP res;
        res.sign = sign & 1;
        return res;
    }

    bool is_zero() const { return exponent == 0 || mantissa == 0; }

    static RefHFP from_long_double(long double value)
    {
        RefHFP res;
        res.sign = std::signbit(value) ? 1 : 0;
        if (value == 0.0L || !std::isfinite(value)) {
            if (!std::isfinite(value)) {
                res.exponent = Format::max_exponent;
                res.mantissa = Format::mantissa_upper - 1;
            }
            return res;
        }

        long double abs_value = std::fabs(value);
        const long double base = static_cast<long double>(word_base);
        const long double low =
            static_cast<long double>(Format::mantissa_lower);
        const long double high =
            static_cast<long double>(Format::mantissa_upper);

        int32_t exponent = static_cast<int32_t>(std::floor(
                               std::log(abs_value / low) / std::log(base))) +
                           Format::bias;
        exponent = std::max<int32_t>(1, exponent);

        long double scaled =
            abs_value / std::pow(base, exponent - Format::bias);
        while (scaled >= high &&
               exponent < static_cast<int32_t>(Format::max_exponent)) {
            ++exponent;
            scaled /= base;
        }
        while (scaled < low && exponent > 1) {
            --exponent;
            scaled *= base;
        }

        if (exponent <= 0 || scaled < low) return zero(res.sign);
        if (exponent > static_cast<int32_t>(Format::max_exponent)) {
            res.exponent = Format::max_exponent;
            res.mantissa = Format::mantissa_upper - 1;
            return res;
        }

        res.exponent = static_cast<uint32_t>(exponent);
        res.mantissa = static_cast<uint64_t>(scaled);
        if (res.mantissa >= Format::mantissa_upper)
            res.mantissa = Format::mantissa_upper - 1;
        if (res.mantissa < Format::mantissa_lower) return zero(res.sign);
        return res;
    }

    static RefHFP from_double(const double value)
    {
        return from_long_double(static_cast<long double>(value));
    }

    long double to_long_double() const
    {
        if (is_zero()) return sign ? -0.0L : 0.0L;
        const long double base = static_cast<long double>(word_base);
        long double value =
            static_cast<long double>(mantissa) *
            std::pow(base, static_cast<int32_t>(exponent) - Format::bias);
        return sign ? -value : value;
    }

    double to_double() const { return static_cast<double>(to_long_double()); }

    uint64_t get_bits() const
    {
        return (uint64_t{sign & 1U}
                << (Format::exponent_bits + Format::mantissa_bits)) |
               (uint64_t{exponent} << Format::mantissa_bits) | mantissa;
    }

};

using RefBFloat16 = RefHFP<BFloat16Format>;

template <class Format, class P>
struct HFP {
    TLWE<P> sign = {};
    std::vector<TLWE<P>> exponent;
    std::vector<TLWE<P>> mantissa;

    HFP() : exponent(Format::exponent_words), mantissa(Format::mantissa_words)
    {
    }
};

namespace hfp_detail {

template <class P, class Q>
inline constexpr bool same_tlwe_layout_v =
    std::is_same_v<typename P::T, typename Q::T> && P::k == Q::k &&
    P::n == Q::n;

template <class P, class intP>
void add_plain_word(TLWE<P> &ct, const int64_t value)
{
    ct[P::k * P::n] += static_cast<typename P::T>(value * intμ<P, intP>());
}

template <class P, class intP>
TLWE<P> trivial_word(const int64_t value)
{
    TLWE<P> res = {};
    add_plain_word<P, intP>(res, value);
    return res;
}

template <class P, class intP>
std::vector<TLWE<P>> trivial_words(const size_t words, uint64_t value)
{
    std::vector<TLWE<P>> res(words);
    for (size_t i = 0; i < words; i++) {
        res[i] = trivial_word<P, intP>(value & (intP::t - 1));
        value >>= intP::bit;
    }
    return res;
}

template <class bkP, class iksP, class intP>
void add_plain_words(std::vector<TLWE<typename iksP::domainP>> &out,
                     const std::vector<TLWE<typename iksP::domainP>> &in,
                     const uint64_t value, const EvalKey &ek)
{
    const auto plain =
        trivial_words<typename iksP::domainP, intP>(in.size(), value);
    LongAdd<bkP, iksP, intP>(out, in, plain, ek);
}


template <class intP>
uint64_t words_to_uint(const std::vector<uint32_t> &words)
{
    uint64_t value = 0;
    for (size_t i = 0; i < words.size(); i++)
        value |= uint64_t{words[i] & (intP::t - 1)} << (i * intP::bit);
    return value;
}

template <class P, class intP, class Fn>
Polynomial<P> word_lut_poly(Fn &&fn)
{
    Polynomial<P> poly = {};
    constexpr uint32_t segment = 1U << (P::nbit - intP::bit);
    for (uint32_t i = 0; i < static_cast<uint32_t>(intP::t); i++) {
        const int64_t value = fn(i);
        for (uint32_t j = 0; j < segment; j++)
            poly[i * segment + j] = value * intμ<P, intP>();
    }
    return poly;
}

template <class bkP, class iksP, class intP, class Fn>
void apply_word_lut(TLWE<typename iksP::domainP> &out,
                    const TLWE<typename iksP::domainP> &in, const EvalKey &ek,
                    Fn &&fn)
{
    TLWE<typename iksP::targetP> shifted;
    IdentityKeySwitch<iksP>(shifted, in, ek.getiksk<iksP>());
    shifted[iksP::targetP::k * iksP::targetP::n] +=
        intμ<typename iksP::targetP, intP>() / 2;
    GateBootstrappingTLWE2TLWEFFT<bkP>(
        out, shifted, ek.getbkfft<bkP>(),
        word_lut_poly<typename bkP::targetP, intP>(std::forward<Fn>(fn)));
}

template <class P>
void extended_cmux_word(TLWE<P> &out, const TLWE<P> &c0, const TLWE<P> &c1,
                        const TRGSWFFT<P> &selector)
{
    TRLWE<P> t0, t1, tres;
    InvSampleExtractIndex<P>(t0, c0, 0);
    InvSampleExtractIndex<P>(t1, c1, 0);
    CMUXFFT<P>(tres, selector, t1, t0);
    SampleExtractIndex<P>(out, tres, 0);
}

template <class P>
void extended_cmux(std::vector<TLWE<P>> &out, const std::vector<TLWE<P>> &c0,
                   const std::vector<TLWE<P>> &c1, const TRGSWFFT<P> &selector)
{
    assert(c0.size() == c1.size());
    out.resize(c0.size());
    for (size_t i = 0; i < c0.size(); i++)
        extended_cmux_word<P>(out[i], c0[i], c1[i], selector);
}

template <class iksP, class cbbrP, class ahP, class intP>
void bit_selector(TRGSWFFT<typename cbbrP::targetP> &selector,
                  const TLWE<typename iksP::domainP> &word, const uint32_t bit,
                  const EvalKey &ek)
{
    static_assert(
        same_tlwe_layout_v<typename iksP::domainP, typename cbbrP::targetP>,
        "The CMux selector target must share the TLWE layout.");
    TLWE<typename iksP::domainP> bit_tlwe = {};
    const uint32_t shift = intP::bit - bit;
    for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        bit_tlwe[i] = -(word[i] << shift);
    bit_tlwe[iksP::domainP::k * iksP::domainP::n] -=
        (intμ<typename iksP::domainP, intP>() / 2) << shift;
    AnnihilateCircuitBootstrappingFFT<iksP, cbbrP, ahP>(selector, bit_tlwe, ek);
}

template <class bkP, class iksP, class intP>
void word_mod2(TLWE<typename iksP::domainP> &out,
               const TLWE<typename iksP::domainP> &in, const EvalKey &ek)
{
    apply_word_lut<bkP, iksP, intP>(
        out, in, ek, [](const uint32_t x) -> int64_t { return x & 1U; });
}

template <class bkP, class iksP, class intP>
void refresh_word(TLWE<typename iksP::domainP> &out,
                  const TLWE<typename iksP::domainP> &in, const EvalKey &ek)
{
    apply_word_lut<bkP, iksP, intP>(
        out, in, ek, [](const uint32_t x) -> int64_t { return x; });
}


template <class P, class intP>
constexpr std::array<std::array<Polynomial<P>, (1 << intP::bit)>, 2>
direct_inv8_tablegen()
{
    static_assert(intP::bit == 4);
    std::array<std::array<Polynomial<P>, (1 << intP::bit)>, 2> table = {};
    constexpr uint32_t segment = P::n / intP::t;
    constexpr uint32_t numerator = (uint32_t{1} << (4 * intP::bit)) - 1;
    for (uint32_t high = 0; high < (1U << intP::bit); high++) {
        for (uint32_t low = 0; low < (1U << intP::bit); low++) {
            const uint32_t denominator = (high << intP::bit) | low;
            const uint32_t reciprocal =
                denominator == 0 ? 0 : numerator / denominator;
            for (uint32_t pack = 0; pack < 2; pack++) {
                for (uint32_t i = 0; i < segment / 2; i++) {
                    table[pack][high][low * segment + 2 * i] =
                        ((reciprocal >> (2 * pack * intP::bit)) &
                         intP::intmax) *
                        intμ<P, intP>();
                    table[pack][high][low * segment + 2 * i + 1] =
                        ((reciprocal >> ((2 * pack + 1) * intP::bit)) &
                         intP::intmax) *
                        intμ<P, intP>();
                }
            }
        }
    }
    return table;
}

template <class bkP, class iksP, class cbbrP, class ahP, class intP>
void direct_inv8_seed(
    std::vector<TLWE<typename iksP::domainP>> &reciprocal,
    const std::vector<TLWE<typename iksP::domainP>> &denominator,
    const EvalKey &ek)
{
    static_assert(intP::bit == 4);
    using CbP = typename cbbrP::targetP;
    assert(denominator.size() == 4);

    alignas(64) std::array<TRGSWFFT<CbP>, intP::bit> address;
    for (int bit = 0; bit < intP::bit; bit++) {
        TLWE<typename bkP::targetP> tlwebit;
        for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
            tlwebit[j] = denominator[1][j] << (intP::bit - bit);
        tlwebit[bkP::targetP::k * bkP::targetP::n] +=
            (intμ<typename bkP::targetP, intP>() / 2) << (intP::bit - bit);
        AnnihilateCircuitBootstrappingFFT<iksP, cbbrP, ahP>(address[bit],
                                                            tlwebit, ek);
    }

    TLWE<typename iksP::targetP> brinput;
    IdentityKeySwitch<iksP>(brinput, denominator[0], ek.getiksk<iksP>());
    brinput[iksP::targetP::k * iksP::targetP::n] +=
        intμ<typename iksP::targetP, intP>() / 2;

    static const auto table = direct_inv8_tablegen<CbP, intP>();
    reciprocal.resize(4);
    for (int pack = 0; pack < 2; pack++) {
        std::array<TRLWE<CbP>, (1 << intP::bit)> packed = {};
        for (int high = 0; high < (1 << intP::bit); high++)
            packed[high][CbP::k] = table[pack][high];

        TRLWE<CbP> selected;
        UROMUX<CbP, intP::bit, 0>(selected, address, packed);
        std::array<TLWE<typename iksP::domainP>, 2> part;
        GateBootstrappingManyLUT<bkP, 2>(part, brinput, ek.getbkfft<bkP>(),
                                         selected);
        reciprocal[2 * pack] = part[0];
        reciprocal[2 * pack + 1] = part[1];
    }
}

template <class bkP, class iksP, class cbbrP, class ahP, class intP>
void inv_lut_nr_direct_2word(
    std::vector<TLWE<typename iksP::domainP>> &reciprocal,
    const std::vector<TLWE<typename iksP::domainP>> &denominator,
    const EvalKey &ek)
{
    assert(denominator.size() == 4);

    direct_inv8_seed<bkP, iksP, cbbrP, ahP, intP>(reciprocal, denominator, ek);

    using P = typename iksP::domainP;
    const std::vector<TLWE<P>> denominator_mantissa = {denominator[0],
                                                       denominator[1]};
    std::vector<TLWE<P>> full_product(6), product(4), diff(4);
    LongMul<bkP, iksP, ahP, intP>(full_product, reciprocal,
                                  denominator_mantissa, ek);
    std::copy_n(full_product.begin(), product.size(), product.begin());
    TLWE<P> borrow;
    LongSub<bkP, iksP, intP>(borrow, diff, product, denominator, ek);
    TLWESub<P>(reciprocal[0], reciprocal[0], borrow);
    LongBorrowPropagation<bkP, iksP, intP>(reciprocal, reciprocal, ek);
}

template <class bkP, class iksP, class ahP, class intP>
void div_by_mul_shifted_2word(
    std::vector<TLWE<typename iksP::domainP>> &quotient,
    std::vector<TLWE<typename iksP::domainP>> &remainder,
    const std::vector<TLWE<typename iksP::domainP>> &numerator,
    const std::vector<TLWE<typename iksP::domainP>> &denominator,
    const std::vector<TLWE<typename iksP::domainP>> &reciprocal,
    const EvalKey &ek)
{
    using P = typename iksP::domainP;
    assert(numerator.size() == 4);
    assert(denominator.size() == 4);
    assert(reciprocal.size() == 4);

    const std::vector<TLWE<P>> numerator_mantissa = {numerator[1],
                                                     numerator[2]};
    const std::vector<TLWE<P>> denominator_mantissa = {denominator[0],
                                                       denominator[1]};

    std::vector<TLWE<P>> numerator_reciprocal_product(6);
    LongMul<bkP, iksP, ahP, intP>(numerator_reciprocal_product, reciprocal,
                                  numerator_mantissa, ek);

    std::vector<TLWE<P>> quotient_low = {
        numerator_reciprocal_product[3], numerator_reciprocal_product[4]};

    std::vector<TLWE<P>> denominator_times_quotient(4);
    LongMul<bkP, iksP, ahP, intP>(denominator_times_quotient,
                                  denominator_mantissa, quotient_low, ek);

    LongSub<bkP, iksP, intP>(remainder, numerator,
                             denominator_times_quotient, ek);

    TLWE<P> correction;
    std::vector<TLWE<P>> diff(4);
    LongSub<bkP, iksP, intP>(correction, diff, remainder, denominator, ek);
    correction[P::k * P::n] -= intμ<P, intP>() / 2;
    for (int k = 0; k <= P::k * P::n; k++) correction[k] *= -1;
    correction[P::k * P::n] += intμ<P, intP>() / 2;

    for (int k = 0; k <= P::k * P::n; k++)
        quotient_low[0][k] += correction[k];
    LongCaryyPropagation<bkP, iksP, intP>(quotient_low, quotient_low, ek);

    std::vector<TLWE<P>> correction_times_denominator_low;
    LongInvReLU<bkP, iksP, ahP, intP>(correction_times_denominator_low,
                                      correction, denominator_mantissa, ek);
    std::vector<TLWE<P>> correction_times_denominator(4);
    correction_times_denominator[0] = correction_times_denominator_low[0];
    correction_times_denominator[1] = correction_times_denominator_low[1];
    LongSub<bkP, iksP, intP>(remainder, remainder,
                             correction_times_denominator, ek);

    quotient.assign(4, TLWE<P>{});
    quotient[0] = quotient_low[0];
    quotient[1] = quotient_low[1];
}

}  // namespace hfp_detail

template <class Format, class P>
void hfp_encrypt(HFP<Format, P> &out, const RefHFP<Format> &plain,
                 const SecretKey &sk)
{
    using intP = typename Format::word_param;
    out = HFP<Format, P>();
    out.sign =
        tlweSymEncrypt<P>(plain.sign * intμ<P, intP>(), P::α, sk.key.get<P>());
    for (size_t i = 0; i < Format::exponent_words; i++) {
        const uint32_t word =
            (plain.exponent >> (i * intP::bit)) & (intP::t - 1);
        out.exponent[i] =
            tlweSymEncrypt<P>(word * intμ<P, intP>(), P::α, sk.key.get<P>());
    }
    for (size_t i = 0; i < Format::mantissa_words; i++) {
        const uint32_t word =
            (plain.mantissa >> (i * intP::bit)) & (intP::t - 1);
        out.mantissa[i] =
            tlweSymEncrypt<P>(word * intμ<P, intP>(), P::α, sk.key.get<P>());
    }
}

template <class Format, class P>
HFP<Format, P> hfp_encrypt(const RefHFP<Format> &plain, const SecretKey &sk)
{
    HFP<Format, P> out;
    hfp_encrypt(out, plain, sk);
    return out;
}

template <class Format, class P>
RefHFP<Format> hfp_decrypt(const HFP<Format, P> &cipher, const SecretKey &sk)
{
    using intP = typename Format::word_param;
    RefHFP<Format> plain;
    plain.sign = tlweIntDecrypt<P, intP>(cipher.sign, sk) & 1U;

    std::vector<uint32_t> exponent_words(Format::exponent_words);
    for (size_t i = 0; i < Format::exponent_words; i++)
        exponent_words[i] =
            tlweIntDecrypt<P, intP>(cipher.exponent[i], sk) & (intP::t - 1);
    plain.exponent =
        static_cast<uint32_t>(hfp_detail::words_to_uint<intP>(exponent_words));

    std::vector<uint32_t> mantissa_words(Format::mantissa_words);
    for (size_t i = 0; i < Format::mantissa_words; i++)
        mantissa_words[i] =
            tlweIntDecrypt<P, intP>(cipher.mantissa[i], sk) & (intP::t - 1);
    plain.mantissa = hfp_detail::words_to_uint<intP>(mantissa_words);
    return plain;
}

template <class bkP, class iksP, class Format>
void HFPCarryPropagateNoMantissaOverflow(
    HFP<Format, typename iksP::domainP> &out,
    const HFP<Format, typename iksP::domainP> &in, const EvalKey &ek)
{
    using intP = typename Format::word_param;

    out = HFP<Format, typename iksP::domainP>();
    hfp_detail::refresh_word<bkP, iksP, intP>(out.sign, in.sign, ek);

    TLWE<typename iksP::domainP> exponent_carry;
    LongCaryyPropagation<bkP, iksP, intP>(exponent_carry, out.exponent,
                                          in.exponent, ek);

    TLWE<typename iksP::domainP> mantissa_carry;
    LongCaryyPropagation<bkP, iksP, intP>(mantissa_carry, out.mantissa,
                                          in.mantissa, ek);
}


}  // namespace TFHEpp
