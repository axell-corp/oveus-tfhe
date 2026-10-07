#pragma once

#include <cstdint>
#include <limits>
struct intparam {
    static constexpr uint32_t bit = 4;
    static constexpr uint32_t halfbit = bit / 2;
    static constexpr int32_t t = 1 << bit;
    static constexpr int32_t halft = 1 << halfbit;
    static constexpr int32_t intmax = t - 1;
    static constexpr int32_t halfintmax = halft - 1;
    static constexpr int32_t intmulmax = intmax * intmax;
};

// struct lvl2Iparam {
//     static constexpr std::uint32_t t = 10;  // number of addition in
//                                             // keyswitching
//     static constexpr std::uint32_t basebit =
//         3;  // how many bit should be encrypted in keyswitching key
//     static constexpr ErrorDistribution errordist =
//         ErrorDistribution::ModularGaussian;
//     static const inline double α = lvlIparam::α;  // key noise
//     // static const inline double α = 0; //key noise
//     using domainP = lvl2param;
//     using targetP = lvlIparam;
// };

using lvlIparam = lvlhalfparam;
using lvl2Iparam = lvl2hparam;
using lvlI2param = lvlh2param;

template <class P, class intP>
constexpr std::make_signed_t<typename P::T> intμ()
{
    return 1LL << (std::numeric_limits<typename P::T>::digits - 1 - intP::bit);
}

template <class P, class intP>
constexpr std::make_signed_t<typename P::T> extendedintμ()
{
    return 1LL << (std::numeric_limits<typename P::T>::digits - 1 -
                   2 * intP::bit);
}
