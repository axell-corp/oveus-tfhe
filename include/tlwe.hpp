#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "key.hpp"
#include "params.hpp"
#include "utils.hpp"

namespace TFHEpp {
template <class P>
void tlweSymEncrypt(TLWE<P> &res, const typename P::T p, const double α,
                    const Key<P> &key)
{
    res = {};
    res[P::k * P::n] = ModularGaussian<P>(p, α);
    for (int k = 0; k < P::k; k++)
        for (int i = 0; i < P::n; i++) {
            res[k * P::n + i] = UniformTorusRandom<P>();
            res[P::k * P::n] += res[k * P::n + i] * key[k * P::n + i];
        }
}

template <class P>
array<typename P::T, P::n + 1> tlweSymEncrypt(
    const typename P::T p, const double α,
    const array<typename P::T, P::n> &key);

template <class P>
void tlweSymEncrypt(TLWE<P> &res, const typename P::T p, const SecretKey &sk)
{
    tlweSymEncrypt<P>(res, p, sk.key.get<P>());
}

template <class P, uint plain_modulus = P::plain_modulus>
void tlweSymIntEncrypt(TLWE<P> &res, const typename P::T p, const double α,
                       const Key<P> &key)
{
    const double Δ = std::pow(2.0, std::numeric_limits<typename P::T>::digits) /
                     plain_modulus;
    tlweSymEncrypt<P>(res, static_cast<typename P::T>(p * Δ), α, key);
}

template <class P, uint plain_modulus = P::plain_modulus>
void tlweSymIntEncrypt(TLWE<P> &res, const typename P::T p, const uint η,
                       const Key<P> &key)
{
    constexpr double Δ =
        std::pow(2.0, std::numeric_limits<typename P::T>::digits) /
        plain_modulus;
    tlweSymEncrypt<P>(res, static_cast<typename P::T>(p * Δ), η, key);
}

template <class P, uint plain_modulus = P::plain_modulus>
void tlweSymIntEncrypt(TLWE<P> &res, const typename P::T p, const Key<P> &key)
{
    if constexpr (P::errordist == ErrorDistribution::ModularGaussian)
        tlweSymIntEncrypt<P, plain_modulus>(res, p, P::α, key);
    else
        tlweSymIntEncrypt<P, plain_modulus>(res, p, P::η, key);
}

template <class P, uint plain_modulus = P::plain_modulus>
void tlweSymIntEncrypt(TLWE<P> &res, const typename P::T p, const SecretKey &sk)
{
    tlweSymIntEncrypt<P, plain_modulus>(res, p, sk.key.get<P>());
}

template <class P = lvl1param>
void bootsSymEncrypt(std::vector<TLWE<P>> &c, const std::vector<uint8_t> &p,
                     const SecretKey &sk)
{
    bootsSymEncrypt<P>(c, p, sk.key.get<P>());
}

template <class P = lvl1param, std::make_signed_t<typename P::T> μ>
void bootsSymEncrypt(std::vector<TLWE<P>> &c, const std::vector<uint8_t> &p,
                     const SecretKey &sk)
{
    bootsSymEncrypt<P, μ>(c, p, sk.key.get<P>());
}

template <class P>
std::vector<uint8_t> bootsSymDecrypt(const std::vector<TLWE<P>> &c,
                                     const Key<P> &key)
{
    vector<uint8_t> p(c.size());
#pragma omp parallel for
    for (int i = 0; i < c.size(); i++) p[i] = tlweSymDecrypt<P>(c[i], key);
    return p;
}

template <class P = lvl1param>
vector<uint8_t> bootsSymDecrypt(const vector<TLWE<P>> &c, const SecretKey &sk);
vector<TLWE<lvl1param>> bootsSymEncryptHalf(const vector<uint8_t> &p,
                                            const SecretKey &sk);
}  // namespace TFHEpp