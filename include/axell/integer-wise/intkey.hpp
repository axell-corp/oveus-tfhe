#pragma once

#include <bits/stdint-uintn.h>

#include <array>
#include <tfhe/cloudkey.hpp>
#include <cmath>
#include <limits>
#include <params.hpp>
#include <utils.hpp>

namespace TFHEpp {

template <class P, class intP>
uint32_t tlweIntDecrypt(const TLWE<P> &ciphertext, const Key<P> &key)
{
    typename P::T phase = ciphertext[P::k * P::n];
    for (int i = 0; i < P::k * P::n; i++)
        phase -= ciphertext[i] * key[i];
    typename P::T result = std::round(
        phase / static_cast<double>(
                    1ULL << (std::numeric_limits<typename P::T>::digits - 1 -
                             intP::bit)));
    result %= 2 * intP::t;
    return static_cast<int>(result >= intP::t ? result - 2 * intP::t : result);
}

template <class P, class intP>
uint32_t tlweIntDecrypt(const TLWE<P> &ciphertext, const SecretKey &sk)
{
    return tlweIntDecrypt<P, intP>(ciphertext, sk.key.get<P>());
}

template <class P, class intP>
Polynomial<P> intmulpolygen()
{
    Polynomial<P> poly;
    constexpr uint32_t segment = 1U << (P::nbit - intP::bit);
    for (int i = 0; i < (1U << intP::bit); i++)
        for (int j = 0; j < segment; j++) poly[i * segment + j] = i;
    return poly;
}

template <class P, class intP>
constexpr Polynomial<P> mulpolygen()
{
    Polynomial<P> poly = intmulpolygen<P, intP>();
    for (typename P::T &i : poly)
        i *= 1ULL << (std::numeric_limits<typename P::T>::digits - 1 -
                      2 * intP::bit);
    return poly;
}

}  // namespace TFHEpp
