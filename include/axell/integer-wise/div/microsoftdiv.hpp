#pragma once
#include <memory>

#include <tfhe/cmuxmem.hpp>

#include "../long.hpp"

namespace TFHEpp {

template <class P, class intP>
constexpr Polynomial<P> zeroeqandclzspoly()
{
    Polynomial<P> poly;
    constexpr uint segment = P::n / 2 / intP::t;
    for (int i = 0; i < intP::t; i++) {
        for (uint j = 0; j < segment; j++) {
            poly[2 * (i * segment + j)] = (i == 0) ? intμ<P, intP>() : 0;
            poly[2 * (i * segment + j) + 1] =
                (i == 0)
                    ? 0
                    : intμ<P, intP>()
                          << (std::countl_zero<uint>(i) -
                              (std::numeric_limits<uint>::digits - intP::bit));
        }
    }
    return poly;
}

template <class P, class intP>
constexpr Polynomial<P> clzpoly()
{
    Polynomial<P> poly;
    for (int i = 0; i < intP::t; i++)
        for (uint j = 0; j < P::n / intP::t; j++)
            poly[i * P::n / intP::t + j] =
                intμ<P, intP>() *
                (std::countl_zero<uint>(i) -
                 (std::numeric_limits<uint>::digits - intP::bit));
    return poly;
}

template <class P, class intP>
constexpr Polynomial<P> zeroeqandshiftpoly()
{
    Polynomial<P> poly = {};
    // for(int i = 0; i < P::n/4; i++)
    // poly[2*i] = intμ<P,intP>()* (intP::t/2);
    constexpr uint segment = P::n / 4 / (intP::t / 2);
    for (int i = 0; i <= intP::bit; i++) {
        for (uint j = 0; j < segment; j++) {
            poly[P::n / 2 + 2 * (i * segment + j)] =
                (i == intP::bit) ? intμ<P, intP>() : 0;
            poly[P::n / 2 + 2 * (i * segment + j) + 1] =
                (i == intP::bit) ? 0 : intμ<P, intP>() << i;
        }
    }
    return poly;
}

// Calculate cres = 1 << clz(a)
template <class brP, class iksP, class intP>
void HomIntCLZShift(std::vector<TLWE<typename iksP::domainP>> &cres,
                    const std::vector<TLWE<typename iksP::domainP>> &a,
                    const EvalKey &ek)
{
    static_assert(intP::t / 2 > intP::bit);
    const int ns = a.size();
    cres.resize(ns);
    // temporary
    TLWE<typename iksP::targetP> shifted;
    std::array<TLWE<typename iksP::domainP>, 2> temp;  //(uwzero, res)
    {
        IdentityKeySwitch<iksP>(shifted, a[ns - 1], ek.getiksk<iksP>());
        shifted[iksP::targetP::k * iksP::targetP::n] +=
            intμ<typename iksP::targetP, intP>() / 2;
        GateBootstrappingManyLUT<brP, 2>(
            temp, shifted, ek.getbkfft<brP>(),
            zeroeqandclzspoly<typename brP::targetP, intP>());
        cres[0] = temp[1];  // If MSW of input is 0, then LSW of res is 0
    }
    for (int i = ns - 2; i >= 0; i--) {
        TLWE<typename iksP::targetP> shifted;
        IdentityKeySwitch<iksP>(shifted, a[i], ek.getiksk<iksP>());
        shifted[iksP::targetP::k * iksP::targetP::n] +=
            intμ<typename iksP::targetP, intP>() / 2;
        TLWE<typename iksP::domainP> clz;
        GateBootstrappingTLWE2TLWEFFT<brP>(
            clz, shifted, ek.getbkfft<brP>(),
            clzpoly<typename brP::targetP, intP>());
        for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
            clz[i] += temp[0][i] * (intP::t / 2);
        IdentityKeySwitch<iksP>(shifted, clz, ek.getiksk<iksP>());
        shifted[iksP::targetP::k * iksP::targetP::n] +=
            intμ<typename iksP::targetP, intP>() / 2;
        GateBootstrappingManyLUT<brP, 2>(
            temp, shifted, ek.getbkfft<brP>(),
            zeroeqandshiftpoly<typename brP::targetP, intP>());
        cres[ns - i - 1] = temp[1];
    }
}

template <class P, class intP>
constexpr Polynomial<P> zeroeqandclzpolyinit()
{
    Polynomial<P> poly;
    constexpr uint segment = P::n / 2 / intP::t;
    for (int i = 0; i < intP::t; i++) {
        for (uint j = 0; j < segment; j++) {
            poly[2 * (i * segment + j)] = (i == 0) ? intμ<P, intP>() : 0;
            poly[2 * (i * segment + j) + 1] =
                (i == 0)
                    ? 0
                    : intμ<P, intP>() *
                          (std::countl_zero<uint>(i) -
                           (std::numeric_limits<uint>::digits - intP::bit));
        }
    }
    return poly;
}

template <class P, class intP>
constexpr Polynomial<P> zeroeqandclzpoly()
{
    Polynomial<P> poly = {};
    // for(int i = 0; i < P::n/4; i++)
    // poly[2*i] = intμ<P,intP>()* (intP::t/2);
    // constexpr uint segment = P::n / 4 / (intP::t / 2);
    constexpr uint segment = (P::n / intP::t) / 2;
    for (int i = 0; i <= intP::bit; i++) {
        for (uint j = 0; j < segment; j++) {
            poly[P::n / 2 + 2 * (i * segment + j)] =
                (i == intP::bit) ? intμ<P, intP>() : 0;
            poly[P::n / 2 + 2 * (i * segment + j) + 1] =
                (i == intP::bit) ? 0 : intμ<P, intP>() * i;
        }
    }
    return poly;
}

// clzu: word wise zero count
// clzl: inword zero count
template <class brP, class iksP, class intP>
void HomIntCLZ(TLWE<typename iksP::domainP> &clzu,
               TLWE<typename iksP::domainP> &clzl,
               const std::vector<TLWE<typename iksP::domainP>> &a,
               const EvalKey &ek)
{
    static_assert(intP::t / 2 > intP::bit);
    assert(a.size() <= intP::t);
    const int ns = a.size();
    // temporary
    TLWE<typename iksP::targetP> shifted;
    std::array<TLWE<typename iksP::domainP>, 2> temp;  //(uwzero, res)
    {
        IdentityKeySwitch<iksP>(shifted, a[ns - 1], ek.getiksk<iksP>());
        shifted[iksP::targetP::k * iksP::targetP::n] +=
            intμ<typename iksP::targetP, intP>() / 2;
        GateBootstrappingManyLUT<brP, 2>(
            temp, shifted, ek.getbkfft<brP>(),
            zeroeqandclzpolyinit<typename brP::targetP, intP>());
        clzu = temp[0];
        clzl = temp[1];
    }
    for (int i = ns - 2; i >= 0; i--) {
        TLWE<typename iksP::targetP> shifted;
        IdentityKeySwitch<iksP>(shifted, a[i], ek.getiksk<iksP>());
        shifted[iksP::targetP::k * iksP::targetP::n] +=
            intμ<typename iksP::targetP, intP>() / 2;
        TLWE<typename iksP::domainP> clz;
        GateBootstrappingTLWE2TLWEFFT<brP>(
            clz, shifted, ek.getbkfft<brP>(),
            clzpoly<typename brP::targetP, intP>());
        for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
            clz[i] += temp[0][i] * (intP::t / 2);
        IdentityKeySwitch<iksP>(shifted, clz, ek.getiksk<iksP>());
        shifted[iksP::targetP::k * iksP::targetP::n] +=
            intμ<typename iksP::targetP, intP>() / 2;
        GateBootstrappingManyLUT<brP, 2>(
            temp, shifted, ek.getbkfft<brP>(),
            zeroeqandclzpoly<typename brP::targetP, intP>());
        for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
            clzu[i] += temp[0][i];
        for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
            clzl[i] += temp[1][i];
    }
}

template <class brP, class iksP, class ahP, class intP, uint numiter>
void HomIntInvNR(std::vector<TLWE<typename iksP::domainP>> &cz,
                 const std::vector<TLWE<typename iksP::domainP>> &cy,
                 const EvalKey &ek)
{
    const uint ns = cy.size();
    cz.resize(ns);
    // 1.
    HomIntCLZShift<brP, iksP, intP>(cz, cy, ek);
    // 2.
    std::vector<TLWE<typename iksP::domainP>> cmy(ns);
    cmy = {};
    LongSub<brP, iksP, intP>(cmy, cmy, cy, ek);
    // 3.
    std::vector<TLWE<typename iksP::domainP>> mulhres, mullres;
    for (int i = 0; i < numiter; i++) {
        LongMulL<brP, iksP, ahP, intP>(mullres, cz, cmy, ek);
        LongMulH<brP, iksP, ahP, intP>(mulhres, cz, mullres, ek);
        LongAdd<brP, iksP, intP>(cz, cz, mulhres, ek);
    }
    // 4.
    mulhres = {};
    LongMulL<brP, iksP, ahP, intP>(mullres, cz, cy, ek);
    LongSub<brP, iksP, intP>(mulhres[0], mullres, mullres, cmy, ek);
    LongAdd<brP, iksP, intP>(cz, cz, mulhres, ek);
}

template <class P, class intP>
Polynomial<P> bitsegmentpolygen(const uint pos)
{
    Polynomial<P> poly = {};
    constexpr uint segment = P::n / 2 / intP::t;
    for (int i = 0; i < intP::t; i++)
        for (int j = 0; j < segment; j++) {
            poly[2 * (i * segment + j)] =
                ((i >> (2 * pos)) & 1)
                    ? 1ULL << (std::numeric_limits<typename P::T>::digits - 2)
                    : -1ULL << (std::numeric_limits<typename P::T>::digits - 2);
            poly[2 * (i * segment + j) + 1] =
                ((i >> (2 * pos + 1)) & 1)
                    ? 1ULL << (std::numeric_limits<typename P::T>::digits - 2)
                    : -1ULL << (std::numeric_limits<typename P::T>::digits - 2);
        }
    return poly;
}

template <class iksP, class brP, class intP>
void BitDecomp(std::array<TLWE<typename brP::targetP>, intP::bit> &cres,
               const TLWE<typename iksP::domainP> &a, const EvalKey &ek)
{
    static_assert((intP::bit & 1) == 0,
                  "BitDecomp currently only supports even bit size");
    TLWE<typename iksP::targetP> shifted;
    IdentityKeySwitch<iksP>(shifted, a, ek.getiksk<iksP>());
    shifted[iksP::targetP::k * iksP::targetP::n] +=
        intμ<typename iksP::targetP, intP>() / 2;
    for (int i = 0; i < intP::bit / 2; i++) {
        std::array<TLWE<typename brP::targetP>, 2> tlweseg;
        GateBootstrappingManyLUT<brP, 2>(
            tlweseg, shifted, ek.getbkfft<brP>(),
            bitsegmentpolygen<typename brP::targetP, intP>(i));
        cres[2 * i] = tlweseg[0];
        cres[2 * i + 1] = tlweseg[1];
    }
}

template <class brP, class iksP, class ahP, class intP>
void HomIntDivByMulNR(std::vector<TLWE<typename iksP::domainP>> &cq,
                      std::vector<TLWE<typename iksP::domainP>> &cr,
                      const std::vector<TLWE<typename iksP::domainP>> &cx,
                      const std::vector<TLWE<typename iksP::domainP>> &cy,
                      const std::vector<TLWE<typename iksP::domainP>> &cz,
                      const EvalKey &ek)
{
    assert(cx.size() == cy.size());
    const uint ns = cx.size();
    // 6.
    LongMulH<brP, iksP, ahP, intP>(cq, cx, cz, ek);
    // 7.
    std::vector<TLWE<typename iksP::domainP>> temp2(ns);
    LongMulL<brP, iksP, ahP, intP>(temp2, cy, cq, ek);
    LongSub<brP, iksP, intP>(cr, cx, temp2, ek);
    // 8.
    {
        TLWE<typename iksP::domainP> temp1;
        // less
        LongSub<brP, iksP, intP>(temp1, temp2, cr, cy, ek);  // cr < cy
        // Negate to get geq
        temp1[iksP::domainP::k * iksP::domainP::n] -=
            intμ<typename iksP::domainP, intP>() / 2;
        for (int k = 0; k <= iksP::domainP::k * iksP::domainP::n; k++)
            temp1[k] *= -1;
        temp1[iksP::domainP::k * iksP::domainP::n] +=
            intμ<typename iksP::domainP, intP>() / 2;
        // cq + cqeq
        for (int k = 0; k <= iksP::domainP::k * iksP::domainP::n; k++)
            cq[0][k] += temp1[k];
        LongCaryyPropagation<brP, iksP, intP>(cq, cq, ek);
        // cr - cqeq⋅cy
        LongInvReLU<brP, iksP, ahP, intP>(temp2, temp1, cy, ek);
        LongSub<brP, iksP, intP>(cr, cr, temp2, ek);
    }
}

template <class P, class intP>
inline Polynomial<P> z1poly()
{
    Polynomial<P> poly;
    constexpr uint segment = P::n / intP::t;
    for (int i = 0; i < intP::t; i++)
        for (uint j = 0; j < segment; j++)
            poly[i * segment + j] = (24 - (i + (i >> 3))) * intμ<P, intP>();
    return poly;
}

template <class P, class intP>
constexpr std::array<Polynomial<P>, (1 << 3)> reducednr32seedtablegen()
{
    static_assert(P::n >= (1 << 10));
    static_assert(intP::bit == 4);
    std::array<Polynomial<P>, (1 << 3)> table = {};
    constexpr uint32_t segment = P::n >> 4;
    for (uint32_t upper = 0; upper < (1U << 3); upper++) {
        for (uint32_t lower = 0; lower < (1U << 4); lower++) {
            const uint32_t hhy = (1U << 7) + (upper << intP::bit) + lower;
            uint32_t z = (1U << (4 * intP::bit - 1)) / hhy;
            // The exact reciprocal is 256 only when hhy == 128. The reduced-NR
            // path consumes two 4-bit words, so clamp the single edge case into
            // that representation.
            if (z > ((1U << (2 * intP::bit)) - 1))
                z = (1U << (2 * intP::bit)) - 1;
            for (uint32_t idx = 0; idx < segment / 2; idx++) {
                table[upper][lower * segment + 2 * idx] =
                    (z & intP::intmax) * intμ<P, intP>();
                table[upper][lower * segment + 2 * idx + 1] =
                    ((z >> intP::bit) & intP::intmax) * intμ<P, intP>();
            }
        }
    }
    return table;
}

template <class brP, class iksP, class cbbrP, class ahP, class intP>
void HomIntInvReducedNRSeed32(
    std::vector<TLWE<typename iksP::domainP>> &z1,
    const std::vector<TLWE<typename iksP::domainP>> &cyy, const EvalKey &ek)
{
    assert(cyy.size() == 8);
    alignas(64) std::array<TRGSWFFT<typename cbbrP::targetP>, 3> address;
    for (int i = 0; i < intP::bit - 1; i++) {
        TLWE<typename brP::targetP> tlwebit;
        for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
            tlwebit[j] = (cyy[7][j] << (intP::bit - i));
        tlwebit[brP::targetP::k * brP::targetP::n] +=
            (intμ<typename brP::targetP, intP>() / 2) << (intP::bit - i);
        AnnihilateCircuitBootstrappingFFT<iksP, cbbrP, ahP>(address[i],
                                                            tlwebit, ek);
    }

    TLWE<typename iksP::targetP> z1lowerlvlh;
    IdentityKeySwitch<iksP>(z1lowerlvlh, cyy[6], ek.getiksk<iksP>());
    z1lowerlvlh[iksP::targetP::k * iksP::targetP::n] +=
        intμ<typename iksP::targetP, intP>() / 2;

    std::array<TRLWE<typename cbbrP::targetP>, (1 << 3)> z1tabletrlwe = {};
    const auto z1table = reducednr32seedtablegen<typename cbbrP::targetP, intP>();
    for (int i = 0; i < (1 << 3); i++)
        z1tabletrlwe[i][cbbrP::targetP::k] = z1table[i];

    TRLWE<typename cbbrP::targetP> z1lowertable;
    UROMUX<typename cbbrP::targetP, 3, 0>(z1lowertable, address, z1tabletrlwe);

    std::array<TLWE<typename iksP::domainP>, 2> z1part;
    GateBootstrappingManyLUT<brP, 2>(z1part, z1lowerlvlh, ek.getbkfft<brP>(),
                                     z1lowertable);
    z1.assign(z1part.begin(), z1part.end());
}

// IntDivisionOracleNR4NoFlagOneFix
template <class brP, class iksP, class cbbrP, class ahP, class intP>
void HomIntInvReducedNR(std::vector<TLWE<typename iksP::domainP>> &cz,
                        const std::vector<TLWE<typename iksP::domainP>> &cy,
                        const EvalKey &ek)
{
    const size_t ns = cy.size();
    assert(ns == 4 || ns == 8);
    const size_t seedwords = ns / 4;
    const size_t halfwords = ns / 2;
    // 1.
    TLWE<typename iksP::domainP> clzu, clzl;
    HomIntCLZ<brP, iksP, intP>(clzu, clzl, cy, ek);
    std::vector<TLWE<typename iksP::domainP>> cyy;
    ULeftShift<brP, iksP, cbbrP, ahP, intP>(cyy, cy, clzu, clzl, ek);
    std::vector<TRLWE<typename brP::targetP>> cyytb;
    PreprocessMulTB<brP, iksP, ahP, intP>(
        cyytb, std::ranges::subrange(cyy.cbegin(), cyy.cend()), ek);
    auto hyytb =
        std::ranges::subrange(cyytb.cbegin() + seedwords, cyytb.cend());

    // 2. Initial seed.
    std::vector<TLWE<typename iksP::domainP>> z1;
    if (ns == 4) {
        z1.resize(1);
        TLWE<typename iksP::targetP> shifted;
        IdentityKeySwitch<iksP>(shifted, cyy[3], ek.getiksk<iksP>());
        shifted[iksP::targetP::k * iksP::targetP::n] +=
            intμ<typename iksP::targetP, intP>() / 2;
        GateBootstrappingTLWE2TLWEFFT<brP>(
            z1[0], shifted, ek.getbkfft<brP>(),
            z1poly<typename brP::targetP, intP>());
    }
    else {
        HomIntInvReducedNRSeed32<brP, iksP, cbbrP, ahP, intP>(z1, cyy, ek);
    }

    // 3. z = (z << W/4) - (z*(hy * z >> W/4)>>W/4)
    std::vector<TLWE<typename iksP::domainP>> z2(seedwords + halfwords);
    std::vector<TLWE<typename iksP::domainP>> temp(seedwords + halfwords);
    {
        LongMul<brP, iksP, intP>(
            std::ranges::subrange(temp.begin(), temp.end()),
            std::ranges::subrange(z1.cbegin(), z1.cend()), hyytb, ek);
        LongMul<brP, iksP, ahP, intP>(
            z2, z1,
            std::ranges::subrange(temp.cbegin() + seedwords, temp.cend()), ek);
        z2.erase(z2.begin(), z2.begin() + seedwords);
        temp.assign(halfwords, {});
        for (size_t i = 0; i < seedwords; i++) temp[seedwords + i] = z1[i];
        LongSub<brP, iksP, intP>(z2, temp, z2, ek);
    }

    // 4. z = ((z<<3) << W/2) - (z*(yy * z >> (W/2-4))>>W/2)
    {
        temp.resize(z2.size() + cyy.size());
        LongMul<brP, iksP, intP>(temp, z2, cyytb, ek);
        temp.erase(temp.begin(), temp.begin() + (halfwords - 1));
        // reuse z1
        z1.resize(z2.size() + temp.size());
        LongMul<brP, iksP, ahP, intP>(z1, z2, temp, ek);
        z1.erase(z1.begin(), z1.begin() + halfwords);
        temp.assign(ns + 1, {});
        for (size_t i = 0; i < halfwords; i++) {
            TLWE<typename iksP::domainP> upper, lower;
            IntMulPlain<brP, iksP, intP>(upper, lower, z2[i], 8, ek);
            TLWEAdd<typename iksP::domainP>(temp[halfwords + i],
                                            temp[halfwords + i], lower);
            TLWEAdd<typename iksP::domainP>(temp[halfwords + i + 1],
                                            temp[halfwords + i + 1], upper);
        }
        LongSub<brP, iksP, intP>(cz, temp, z1, ek);
    }
    // 5.
    cz.resize(ns + 1);
    {
        ULeftShiftH<brP, iksP, cbbrP, ahP, intP>(cz, cz, clzu, clzl, ek);
        for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
            cz[ns][i] *= intP::t;
        cz[ns][iksP::domainP::k * iksP::domainP::n] +=
            intP::t * (intμ<typename iksP::domainP, intP>() / 2);
        TLWE<typename iksP::targetP> czmsblvlh;
        IdentityKeySwitch<iksP>(czmsblvlh, cz[ns], ek.getiksk<iksP>());
        Polynomial<typename brP::targetP> czmsbpoly;
        for (int i = 0; i < brP::targetP::n; i++)
            czmsbpoly[i] =
                -(intP::t - 1) * intμ<typename brP::targetP, intP>() / 2;
        GateBootstrappingTLWE2TLWEFFT<brP>(cz[ns], czmsblvlh,
                                           ek.getbkfft<brP>(), czmsbpoly);
        cz[ns][brP::targetP::k * brP::targetP::n] +=
            (intP::t - 1) * intμ<typename brP::targetP, intP>() / 2;
        for (size_t i = 0; i < ns; i++)
            for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
                cz[i][j] += cz[ns][j];
        cz.resize(ns);
    }
    // 6.
    {
        // reuse cyy
        LongMulL<brP, iksP, ahP, intP>(cyy, cy, cz, ek);
        TLWE<typename iksP::domainP> ccarry;
        LongAdd<brP, iksP, intP>(ccarry, cyy, cyy, cy, ek);
        for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
            cz[0][i] -= ccarry[i];
        cz[0][iksP::domainP::k * iksP::domainP::n] +=
            intμ<typename iksP::domainP, intP>();
        LongCaryyPropagation<brP, iksP, intP>(cz, cz, ek);
    }
}

inline uint32_t inv8bit(uint8_t y8)
{
    // These constants are equivalent to the ones in the Python script.
    // Using 'constexpr' allows them to be evaluated at compile time.
    // constexpr int L = 16;
    // constexpr int w = 1 << L;  // 2^16
    // Assert that the input y8 meets the specified constraint.
    assert(y8 >= 128);

    // Calculate 'y' by left-shifting y8.
    // We cast y8 to a 32-bit integer before shifting to prevent overflow.
    // Equivalent to Python: y = y8 << 8
    const uint32_t y = static_cast<uint32_t>(y8) << 8;

    // Calculate 'l', the position of the most significant bit.
    // For the given constraints on y8, 'l' will always be 16.
    // Equivalent to Python: l = floor( log2( y ) ) + 1
    // int l = static_cast<int>(floor(log2(y))) + 1;

    // Normalize y to 'yd' by shifting it.
    // Since L=16 and l=16, this shift is by 0, so yd will be equal to y.
    // Equivalent to Python: yd = y * (2 ** (L - l))
    // uint32_t yd = y << (L - l);
    const uint32_t yd = y;

    // Initial approximation for z.
    // Equivalent to Python: z = 48 - ((yd>>11) + (yd>>14))
    int32_t z = 48 - ((y8 >> 3) + (y8 >> 6));
    assert(0 < z && z < (1 << 5));

    // First refinement step.
    // We use a signed integer 't' because it can become negative after
    // subtraction. Equivalent to Python: t = ((yd>>8) * z) >> 5
    int32_t t = ((yd >> 8) * z) >> 5;
    assert(0 < t && t < (1 << 8));

    // Equivalent to Python: t = t - (1<<7)
    t = t - (1 << 7);
    assert(std::abs(t) < (1 << 5));

    // Equivalent to Python: z = (z << 4) - (t * z >> 3)
    z = (z << 4) - ((t * z) >> 3);
    assert(0 < z && z < (1 << 9));

    // Second refinement step.
    // Equivalent to Python: t = ((yd >> 8) * z) >> 4
    t = ((yd >> 8) * z) >> 4;
    assert(0 < t && t < (1 << 13));

    // Equivalent to Python: t = t - (1<<12)
    t = t - (1 << 12);
    assert(std::abs(t) < (1 << 9));

    // Equivalent to Python: z = (z << 8) - ((t * z) >> 4)
    z = (z << 8) - ((t * z) >> 4);
    assert(0 < z && z < (1 << 17));

    // Fix in advance
    if (((z * y) & ((1 << 16) - 1)) < y) z -= 1;
    if ((z * y) < ((1 << 16) - y)) z += 1;
    assert((1 << 16) - y < z * y);
    assert(z * y < (1 << 16) + 2 * y);

    // Return the final result.
    return z;
}

template <class P, class intP>
constexpr Polynomial<P> inv8bitpolygen()
{
    static_assert(P::n >= (1 << 10));
    Polynomial<P> poly = {};
    constexpr uint32_t segment = P::n >> 7;
    static_assert(segment >= 4);
    for (int i = 0; i < 1 << 7; i++) {
        const uint32_t z = ((1ULL << 32) - 1) / ((128 + i) << 8);
        for (int j = 0; j < 4; j++)
            poly[i * segment + j] = ((z >> (4 * j)) & 0xF) * intμ<P, intP>();
    }
    return poly;
}

template <class P, class intP>
constexpr std::array<std::array<Polynomial<P>,(1<<3)>,2> inv8bittablegen()
{
    static_assert(P::n >= (1 << 10));
    std::array<std::array<Polynomial<P>,(1<<3)>,2> table = {};
    constexpr uint32_t segment = P::n >> 4;
    for (int i = 0; i < (1 << 3); i++) {
        for (int j = 0; j < (1<<4); j++){
            for(int k = 0; k < 2; k++){
                const uint32_t z = ((1ULL << 32) - 1) / ((128 + (i<<4)+j) << 8);
                for(int l = 0; l < segment/2; l++){
                    table[k][i][j * segment + 2*l] = ((z >> (4 * 2*k)) & 0xF) * intμ<P, intP>();
                    table[k][i][j * segment + 2*l+1] = ((z >> (4 * (2*k+1))) & 0xF) * intμ<P, intP>();
                }
            }
        }
    }
    return table;
}

template <class P, class intP>
const auto &inv16biturom()
{
    static_assert(P::nbit >= 11);
    static_assert(intP::bit == 4);

    constexpr uint32_t cmux_address_bit = 11;
    constexpr uint32_t num_tables = 1 << cmux_address_bit;
    constexpr uint32_t segment = P::n / intP::t;
    constexpr uint64_t numerator = std::numeric_limits<uint64_t>::max();
    using Table = std::array<std::array<TRLWE<P>, num_tables>, intP::bit>;

    static const std::unique_ptr<Table> table = [] {
        auto res = std::make_unique<Table>();

        // The 32-bit LUT seed depends on the top 16 bits of the normalized
        // divisor. The leading bit is fixed to 1, so 11 CMUX bits plus the
        // full 4-bit BR input cover the remaining 15 address bits.
        for (uint32_t upper = 0; upper < num_tables; upper++) {
            for (uint32_t br = 0; br < intP::t; br++) {
                const uint32_t prefix = (1U << 15) | (upper << 4) | br;
                const uint64_t z =
                    numerator / (static_cast<uint64_t>(prefix) << 16);

                for (uint32_t pack = 0; pack < intP::bit; pack++) {
                    auto &poly = (*res)[pack][upper][P::k];
                    const uint32_t lo =
                        static_cast<uint32_t>((z >> (8 * pack)) & 0xF);
                    const uint32_t hi =
                        static_cast<uint32_t>((z >> (8 * pack + 4)) & 0xF);

                    for (uint32_t l = 0; l < segment / 2; l++) {
                        poly[br * segment + 2 * l] = lo * intμ<P, intP>();
                        poly[br * segment + 2 * l + 1] = hi * intμ<P, intP>();
                    }
                }
            }
        }
        return res;
    }();
    return *table;
}

template <class brP, class iksP, class cbbrP, class ahP, class intP>
void HomIntInvLUTSeed16Normalized(
    std::vector<TLWE<typename iksP::domainP>> &cz,
    const std::vector<TLWE<typename iksP::domainP>> &cyy, const EvalKey &ek)
{
    assert(cyy.size() == 4);

    alignas(64) std::array<TRGSWFFT<typename cbbrP::targetP>, 3> address;
    for (int i = 0; i < intP::bit - 1; i++) {
        TLWE<typename brP::targetP> tlwebit;
        for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
            tlwebit[j] = (cyy[3][j] << (intP::bit - i));
        tlwebit[brP::targetP::k * brP::targetP::n] +=
            (intμ<typename brP::targetP, intP>() / 2) << (intP::bit - i);

        AnnihilateCircuitBootstrappingFFT<iksP, cbbrP, ahP>(address[i],
                                                            tlwebit, ek);
    }

    cz.resize(4);
    TLWE<typename iksP::targetP> czlowerlvlh;
    IdentityKeySwitch<iksP>(czlowerlvlh, cyy[2], ek.getiksk<iksP>());
    czlowerlvlh[iksP::targetP::k * iksP::targetP::n] +=
        intμ<typename iksP::targetP, intP>() / 2;

    const auto inv8bittable = inv8bittablegen<typename cbbrP::targetP, intP>();
    for (int i = 0; i < 2; i++) {
        std::array<TRLWE<typename brP::targetP>, (1 << 3)>
            inv8bittabletrlwe = {};
        for (int j = 0; j < (1 << 3); j++)
            inv8bittabletrlwe[j][cbbrP::targetP::k] = inv8bittable[i][j];
        TRLWE<typename cbbrP::targetP> inv8bitlowertable;
        UROMUX<typename cbbrP::targetP, 3, 0>(inv8bitlowertable, address,
                                              inv8bittabletrlwe);
        std::array<TLWE<typename iksP::domainP>, 2> czpart;
        GateBootstrappingManyLUT<brP, 2>(czpart, czlowerlvlh,
                                         ek.getbkfft<brP>(),
                                         inv8bitlowertable);
        for (int j = 0; j < 2; j++) cz[2 * i + j] = czpart[j];
    }
}

template <class brP, class iksP, class cbbrP, class ahP, class intP>
void HomIntInvLUTSeed32Normalized(
    std::vector<TLWE<typename iksP::domainP>> &cz,
    const std::vector<TLWE<typename iksP::domainP>> &cyy, const EvalKey &ek)
{
    static_assert(intP::bit == 4);
    assert(cyy.size() == 8);

    constexpr uint32_t cmux_address_bit = 11;
    alignas(64) std::array<TRGSWFFT<typename cbbrP::targetP>, cmux_address_bit>
        address;
    for (int word = 0; word < 3; word++) {
        for (int bit = 0; bit < intP::bit; bit++) {
            const int address_index = word * intP::bit + bit;
            if (address_index >= cmux_address_bit) break;

            TLWE<typename brP::targetP> tlwebit;
            for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
                tlwebit[j] = cyy[5 + word][j] << (intP::bit - bit);
            tlwebit[brP::targetP::k * brP::targetP::n] +=
                (intμ<typename brP::targetP, intP>() / 2)
                << (intP::bit - bit);
            AnnihilateCircuitBootstrappingFFT<iksP, cbbrP, ahP>(
                address[address_index], tlwebit, ek);
        }
    }

    TLWE<typename iksP::targetP> brinput;
    IdentityKeySwitch<iksP>(brinput, cyy[4], ek.getiksk<iksP>());
    brinput[iksP::targetP::k * iksP::targetP::n] +=
        intμ<typename iksP::targetP, intP>() / 2;

    const auto &inv16bittable = inv16biturom<typename cbbrP::targetP, intP>();
    cz.resize(2 * intP::bit);
    for (int pack = 0; pack < intP::bit; pack++) {
        TRLWE<typename cbbrP::targetP> selectedtable;
        UROMUX<typename cbbrP::targetP, cmux_address_bit, 0>(
            selectedtable, address, inv16bittable[pack]);

        std::array<TLWE<typename iksP::domainP>, 2> czpart;
        GateBootstrappingManyLUT<brP, 2>(czpart, brinput, ek.getbkfft<brP>(),
                                         selectedtable);
        cz[2 * pack] = czpart[0];
        cz[2 * pack + 1] = czpart[1];
    }
}

template <class brP, class iksP, class cbbrP, class ahP, class intP>
void HomIntInvLUTNR(std::vector<TLWE<typename iksP::domainP>> &cz,
                    const std::vector<TLWE<typename iksP::domainP>> &cy,
                    const EvalKey &ek)
{
    const size_t ns = cy.size();
    assert(ns == 4 || ns == 8);
    // 1.
    TLWE<typename iksP::domainP> clzu, clzl;
    HomIntCLZ<brP, iksP, intP>(clzu, clzl, cy, ek);
    std::vector<TLWE<typename iksP::domainP>> cyy;
    ULeftShift<brP, iksP, cbbrP, ahP, intP>(cyy, cy, clzu, clzl, ek);

    if (ns == 8) {
        HomIntInvLUTSeed32Normalized<brP, iksP, cbbrP, ahP, intP>(cz, cyy, ek);
        cz.resize(9);
        cz[8] = {};
        cz[8][brP::targetP::k * brP::targetP::n] =
            intμ<typename brP::targetP, intP>();

        // 3. Undo the normalization shift and apply the final downward fix.
        ULeftShiftH<brP, iksP, cbbrP, ahP, intP>(cz, cz, clzu, clzl, ek);
        cz.resize(ns);

        std::vector<TLWE<typename iksP::domainP>> cyz(ns), temp(ns);
        TLWE<typename iksP::domainP> cborrow;
        LongMulL<brP, iksP, ahP, intP>(cyz, cy, cz, ek);
        LongSub<brP, iksP, intP>(cborrow, temp, cyz, cy, ek);
        TLWESub<typename iksP::domainP>(cz[0], cz[0], cborrow);
        LongBorrowPropagation<brP, iksP, intP>(cz, cz, ek);
        return;
    }

    HomIntInvLUTSeed16Normalized<brP, iksP, cbbrP, ahP, intP>(cz, cyy, ek);
    cz.resize(5);
    cz[4] = {};
    cz[4][brP::targetP::k * brP::targetP::n] =
        intμ<typename brP::targetP, intP>();

    // 4.
    ULeftShiftH<brP, iksP, cbbrP, ahP, intP>(cz, cz, clzu, clzl, ek);
    cz.resize(4);

    // 6.
    {
        std::vector<TLWE<typename iksP::domainP>> cyz(4);
        // Downward Fix
        //  reuse cyy
        LongMulL<brP, iksP, ahP, intP>(cyz, cy, cz, ek);
        // reuse clzu, larger than or equals to 2^W
        LongSub<brP, iksP, intP>(clzu, cyy, cyz, cy, ek);
        TLWESub<typename iksP::domainP>(cz[0], cz[0], clzu);
        LongBorrowPropagation<brP, iksP, intP>(cz, cz, ek);
    }
}

}  // namespace TFHEpp
