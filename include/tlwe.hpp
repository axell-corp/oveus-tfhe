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
std::vector<uint8_t> bootsSymDecrypt(const std::vector<TLWE<P>> &c,
                                     const SecretKey &sk)
{
    return bootsSymDecrypt<P>(c, sk.key.get<P>());
}

/**
 * @brief Adds an arbitrary number of TLWE ciphertexts element-wise.
 *
 * This function calculates res = c_1 + c_2 + ... + c_n.
 * It uses a C++17 fold expression within a loop to sum the elements
 * of all provided ciphertexts directly into the result.
 *
 * @tparam P The TLWE parameter type.
 * @tparam Args A parameter pack of TLWE<P> types.
 * @param res The output ciphertext where the sum is stored.
 * @param first The first ciphertext in the sum.
 * @param rest The remaining ciphertexts in the sum.
 */
template <class P, class... Args>
void TLWEAdd(TLWE<P> &res, const TLWE<P> &first, const Args &...rest)
{
    for (int i = 0; i <= P::k * P::n; i++) {
        // A binary fold expression sums all corresponding elements at once.
        res[i] = (first[i] + ... + rest[i]);
    }
}

/**
 * @brief Subtracts multiple TLWE ciphertexts element-wise.
 *
 * Calculates res = c1 - c2 - c3 - ...
 * NOTE: This implementation requires at least two input ciphertexts.
 */
template <class P, class... Args>
void TLWESub(TLWE<P> &res, const TLWE<P> &first, const Args &...rest)
{
    // A binary fold requires the parameter pack 'rest' to be non-empty.
    static_assert(
        sizeof...(Args) > 0,
        "This TLWESub implementation requires at least two arguments.");

    for (int i = 0; i <= P::k * P::n; i++) {
        // Binary fold over the '-' operator.
        // Expands to (((first[i] - c2[i]) - c3[i]) - ...)
        res[i] = (first[i] - ... - rest[i]);
    }
}

}  // namespace TFHEpp
