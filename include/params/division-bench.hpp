#pragma once

#include <cmath>
#include <cstdint>

struct lvl0param {
    static constexpr int32_t key_value_max = 1;
    static constexpr int32_t key_value_min = 0;
    static constexpr int32_t key_value_diff = key_value_max - key_value_min;
    static constexpr std::uint32_t n = 636;
    static constexpr std::uint32_t k = 1;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double α = 0.000'092'511'997'467'675'6;
    using T = uint16_t;
    static constexpr std::make_signed_t<T> μ =
        1LL << (std::numeric_limits<T>::digits - 3);
    static constexpr uint32_t plain_modulus = 8;
    static constexpr double Δ =
        static_cast<double>(1ULL << std::numeric_limits<T>::digits) /
        plain_modulus;
};

struct lvlhalfparam {
    static constexpr int32_t key_value_max = 1;
    static constexpr int32_t key_value_min = 0;
    static constexpr int32_t key_value_diff = key_value_max - key_value_min;
    static constexpr std::uint32_t n = 760;
    static constexpr std::uint32_t k = 1;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double α = std::pow(2.0, -17);
    using T = uint32_t;
    static constexpr T μ = 1U << (std::numeric_limits<T>::digits - 3);
    static constexpr uint32_t plain_modulus = 32;
    static constexpr double Δ =
        static_cast<double>(1ULL << std::numeric_limits<T>::digits) /
        plain_modulus;
};

struct lvl1param {
    static constexpr int32_t key_value_max = 1;
    static constexpr int32_t key_value_min = -1;
    static constexpr int32_t key_value_diff = key_value_max - key_value_min;
    static constexpr std::uint32_t nbit = 10;
    static constexpr std::uint32_t n = 1 << nbit;
    static constexpr std::uint32_t k = 1;
    static constexpr std::uint32_t l = 2;
    static constexpr std::uint32_t lₐ = l;
    static constexpr std::uint32_t Bgbit = 8;
    static constexpr std::uint32_t Bgₐbit = Bgbit;
    static constexpr std::uint32_t Bg = 1 << Bgbit;
    static constexpr std::uint32_t Bgₐ = 1 << Bgₐbit;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double α = std::pow(2.0, -25);
    using T = uint32_t;
    static constexpr std::make_signed_t<T> μ = 1 << 29;
    static constexpr uint32_t plain_modulus = 8;
    static constexpr double Δ =
        static_cast<double>(1ULL << std::numeric_limits<T>::digits) /
        plain_modulus;
    static constexpr std::uint32_t l̅ = 1;
    static constexpr std::uint32_t l̅ₐ = l̅;
    static constexpr std::uint32_t B̅gbit = std::numeric_limits<T>::digits;
    static constexpr std::uint32_t B̅gₐbit = B̅gbit;
};

struct AHlvl1param {
    using baseP = lvl1param;
    static constexpr int32_t key_value_max = baseP::key_value_max;
    static constexpr int32_t key_value_min = baseP::key_value_min;
    static constexpr std::uint32_t nbit = baseP::nbit;
    static constexpr std::uint32_t n = baseP::n;
    static constexpr std::uint32_t k = baseP::k;
    static constexpr std::uint32_t lₐ = 4;
    static constexpr std::uint32_t l = 4;
    static constexpr std::uint32_t Bgbit = 5;
    static constexpr std::uint32_t Bgₐbit = 5;
    static constexpr std::uint32_t Bg = 1 << Bgbit;
    static constexpr std::uint32_t Bgₐ = 1 << Bgₐbit;
    static constexpr ErrorDistribution errordist = baseP::errordist;
    static const inline double α = baseP::α;
    using T = typename baseP::T;
    static constexpr std::make_signed_t<T> μ = baseP::μ;
    static constexpr uint32_t plain_modulus = baseP::plain_modulus;
    static constexpr double Δ = baseP::Δ;
    static constexpr std::uint32_t l̅ = baseP::l̅;
    static constexpr std::uint32_t l̅ₐ = baseP::l̅ₐ;
    static constexpr std::uint32_t B̅gbit = baseP::B̅gbit;
    static constexpr std::uint32_t B̅gₐbit = baseP::B̅gₐbit;
};

struct lvl2param {
    static constexpr int32_t key_value_max = 1;
    static constexpr int32_t key_value_min = -1;
    static constexpr int32_t key_value_diff = key_value_max - key_value_min;
    static const std::uint32_t nbit = 11;
    static constexpr std::uint32_t n = 1 << nbit;
    static constexpr std::uint32_t k = 1;
    static constexpr std::uint32_t l = 2;
    static constexpr std::uint32_t lₐ = l;
    static constexpr std::uint32_t Bgbit = 17;
    static constexpr std::uint32_t Bgₐbit = Bgbit;
    static constexpr std::uint32_t Bg = 1 << Bgbit;
    static constexpr std::uint32_t Bgₐ = 1 << Bgₐbit;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double α = std::pow(2.0, -51);
    using T = uint64_t;
    static constexpr std::make_signed_t<T> μ = 1LL << 61;
    static constexpr uint32_t plain_modulus = 8;
    static constexpr double Δ =
        static_cast<double>(1ULL << (std::numeric_limits<T>::digits - 4));
    static constexpr std::uint32_t l̅ = 1;
    static constexpr std::uint32_t l̅ₐ = l̅;
    static constexpr std::uint32_t B̅gbit = std::numeric_limits<T>::digits;
    static constexpr std::uint32_t B̅gₐbit = B̅gbit;
};

#define USE_DIFFERENT_BR_PARAM
struct cblvl2param {
    static constexpr int32_t key_value_max = 1;
    static constexpr int32_t key_value_min = -1;
    static constexpr int32_t key_value_diff = key_value_max - key_value_min;
    static const std::uint32_t nbit = 11;
    static constexpr std::uint32_t n = 1 << nbit;
    static constexpr std::uint32_t k = 1;
    static constexpr std::uint32_t l = 4;
    static constexpr std::uint32_t lₐ = l;
    static constexpr std::uint32_t Bgbit = 10;
    static constexpr std::uint32_t Bgₐbit = Bgbit;
    static constexpr std::uint32_t Bg = 1 << Bgbit;
    static constexpr std::uint32_t Bgₐ = 1 << Bgₐbit;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double α = std::pow(2.0, -51);
    using T = uint64_t;
    static constexpr std::make_signed_t<T> μ = 1LL << 61;
    static constexpr uint32_t plain_modulus = 8;
    static constexpr double Δ =
        static_cast<double>(1ULL << (std::numeric_limits<T>::digits - 4));
    static constexpr std::uint32_t l̅ = 1;
    static constexpr std::uint32_t l̅ₐ = l̅;
    static constexpr std::uint32_t B̅gbit = std::numeric_limits<T>::digits;
    static constexpr std::uint32_t B̅gₐbit = B̅gbit;
};

struct AHlvl2param {
    using baseP = lvl2param;
    static constexpr int32_t key_value_max = baseP::key_value_max;
    static constexpr int32_t key_value_min = baseP::key_value_min;
    static constexpr std::uint32_t nbit = baseP::nbit;
    static constexpr std::uint32_t n = baseP::n;
    static constexpr std::uint32_t k = baseP::k;
    static constexpr std::uint32_t lₐ = 4;
    static constexpr std::uint32_t l = 4;
    static constexpr std::uint32_t Bgbit = 11;
    static constexpr std::uint32_t Bgₐbit = 11;
    static constexpr std::uint32_t Bg = 1 << Bgbit;
    static constexpr std::uint32_t Bgₐ = 1 << Bgₐbit;
    static constexpr ErrorDistribution errordist = baseP::errordist;
    static const inline double α = baseP::α;
    using T = typename baseP::T;
    static constexpr std::make_signed_t<T> μ = baseP::μ;
    static constexpr uint32_t plain_modulus = baseP::plain_modulus;
    static constexpr double Δ = baseP::Δ;
    static constexpr std::uint32_t l̅ = baseP::l̅;
    static constexpr std::uint32_t l̅ₐ = baseP::l̅ₐ;
    static constexpr std::uint32_t B̅gbit = baseP::B̅gbit;
    static constexpr std::uint32_t B̅gₐbit = baseP::B̅gₐbit;
};

struct lvl3param {
    static constexpr int32_t key_value_max = 1;
    static constexpr int32_t key_value_min = -1;
    static const std::uint32_t nbit = 13;
    static constexpr std::uint32_t n = 1 << nbit;
    static constexpr std::uint32_t k = 1;
    static constexpr std::uint32_t lₐ = 4;
    static constexpr std::uint32_t l = 4;
    static constexpr std::uint32_t Bgbit = 9;
    static constexpr std::uint32_t Bgₐbit = 9;
    static constexpr std::uint32_t Bg = 1 << Bgbit;
    static constexpr std::uint32_t Bgₐ = 1 << Bgₐbit;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double α = std::pow(2.0, -51);
    using T = uint64_t;
    static constexpr T μ = 1ULL << 61;
    static constexpr uint32_t plain_modulusbit = 31;
    static constexpr uint64_t plain_modulus = 1ULL << plain_modulusbit;
    static constexpr double Δ = 1ULL << (64 - plain_modulusbit - 1);
    static constexpr std::uint32_t l̅ = 1;
    static constexpr std::uint32_t l̅ₐ = l̅;
    static constexpr std::uint32_t B̅gbit = std::numeric_limits<T>::digits;
    static constexpr std::uint32_t B̅gₐbit = B̅gbit;
};

// mulfft.hpp names this optional parameter type in constexpr branches.  The
// division benchmark does not instantiate those branches, so a distinct type
// inheriting the lvl3 constants is sufficient to keep the generic header
// well-formed without importing the extended parameter set.
struct lvl3simdparam : lvl3param {};

struct lvl10param {
    static constexpr std::uint32_t t = 4;
    static constexpr std::uint32_t basebit = 3;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double α = lvl0param::α;
    using domainP = lvl1param;
    using targetP = lvl0param;
};

struct lvl1hparam {
    static constexpr std::uint32_t t = 10;
    static constexpr std::uint32_t basebit = 3;
    static const inline double α = lvlhalfparam::α;
    using domainP = lvl1param;
    using targetP = lvlhalfparam;
};

struct lvl11param {
    static constexpr std::uint32_t t = 6;
    static constexpr std::uint32_t basebit = 4;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double α = lvl1param::α;
    using domainP = lvl1param;
    using targetP = lvl1param;
};

struct lvl20param {
    static constexpr std::uint32_t t = 7;
    static constexpr std::uint32_t basebit = 2;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double α = lvl0param::α;
    using domainP = lvl2param;
    using targetP = lvl0param;
};

struct lvl2hparam {
    static constexpr std::uint32_t t = 5;
    static constexpr std::uint32_t basebit = 3;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double α = lvlhalfparam::α;
    using domainP = lvl2param;
    using targetP = lvlhalfparam;
};

struct lvl21param {
    static constexpr std::uint32_t t = 8;
    static constexpr std::uint32_t basebit = 3;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double α = lvl1param::α;
    using domainP = lvl2param;
    using targetP = lvl1param;
};

struct lvl22param {
    static constexpr std::uint32_t t = 38;
    static constexpr std::uint32_t basebit = 1;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double α = lvl2param::α;
    using domainP = lvl2param;
    using targetP = lvl2param;
};

struct lvl31param {
    static constexpr std::uint32_t t = 7;
    static constexpr std::uint32_t basebit = 2;
    static const inline double α = lvl1param::α;
    using domainP = lvl3param;
    using targetP = lvl1param;
};
