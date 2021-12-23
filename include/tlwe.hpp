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
TLWE<P> tlweSymEncrypt(const typename P::T p, const double α,
                       const array<typename P::T, P::n> &key);

template <class P>
TLWE<P> tlweSymIntEncrypt(const typename P::T p, const double α,
                          const array<typename P::T, P::n> &key);

template <class P>
bool tlweSymDecrypt(const TLWE<P> &c, const Key<P> &key);
template <class P>
typename P::T tlweSymIntDecrypt(const TLWE<P> &c, const Key<P> &key);

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