#pragma once
#include "../long_par.hpp"
#include "../shift_par.hpp"
#include "microsoftdiv.hpp"

namespace TFHEpp {

// Parallel HomIntCLZ: pre-computes all first-stage IKS+GB in parallel,
// then sequentially combines with zero-flag cascade.
template <class brP, class iksP, class intP>
void HomIntCLZPar(TLWE<typename iksP::domainP> &clzu,
                  TLWE<typename iksP::domainP> &clzl,
                  const std::vector<TLWE<typename iksP::domainP>> &a,
                  const EvalKey &ek)
{
    static_assert(intP::t / 2 > intP::bit);
    assert(a.size() <= intP::t);
    const int ns = static_cast<int>(a.size());

    // Phase 1: Parallel first-stage IKS + GB for all words
    std::vector<TLWE<typename iksP::domainP>> first_gb(ns);
    std::array<TLWE<typename iksP::domainP>, 2> msw_temp;
#pragma omp parallel for
    for (int i = 0; i < ns; i++) {
        TLWE<typename iksP::targetP> shifted;
        IdentityKeySwitch<iksP>(shifted, a[i], ek.getiksk<iksP>());
        shifted[iksP::targetP::k * iksP::targetP::n] +=
            intμ<typename iksP::targetP, intP>() / 2;
        if (i == ns - 1) {
            GateBootstrappingManyLUT<brP, 2>(
                msw_temp, shifted, ek.getbkfft<brP>(),
                zeroeqandclzpolyinit<typename brP::targetP, intP>());
        }
        else {
            GateBootstrappingTLWE2TLWEFFT<brP>(
                first_gb[i], shifted, ek.getbkfft<brP>(),
                clzpoly<typename brP::targetP, intP>());
        }
    }

    // Phase 2: Initialize from MSW result
    std::array<TLWE<typename iksP::domainP>, 2> temp = msw_temp;
    clzu = temp[0];
    clzl = temp[1];

    // Phase 3: Sequential combine — only the second IKS+GB per word
    for (int i = ns - 2; i >= 0; i--) {
        TLWE<typename iksP::domainP> clz = first_gb[i];
        for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
            clz[j] += temp[0][j] * (intP::t / 2);
        TLWE<typename iksP::targetP> shifted;
        IdentityKeySwitch<iksP>(shifted, clz, ek.getiksk<iksP>());
        shifted[iksP::targetP::k * iksP::targetP::n] +=
            intμ<typename iksP::targetP, intP>() / 2;
        GateBootstrappingManyLUT<brP, 2>(
            temp, shifted, ek.getbkfft<brP>(),
            zeroeqandclzpoly<typename brP::targetP, intP>());
        for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
            clzu[j] += temp[0][j];
        for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
            clzl[j] += temp[1][j];
    }
}

// Parallel version of HomIntInvNR
template <class brP, class iksP, class ahP, class intP, uint numiter>
void HomIntInvNRPar(std::vector<TLWE<typename iksP::domainP>> &cz,
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
        LongMulLPar<brP, iksP, ahP, intP>(mullres, cz, cmy, ek);
        LongMulHPar<brP, iksP, ahP, intP>(mulhres, cz, mullres, ek);
        LongAdd<brP, iksP, intP>(cz, cz, mulhres, ek);
    }
    // 4.
    mulhres = {};
    LongMulLPar<brP, iksP, ahP, intP>(mullres, cz, cy, ek);
    LongSub<brP, iksP, intP>(mulhres[0], mullres, mullres, cmy, ek);
    LongAdd<brP, iksP, intP>(cz, cz, mulhres, ek);
}

// Parallel version of HomIntDivByMulNR
template <class brP, class iksP, class ahP, class intP>
void HomIntDivByMulNRPar(
    std::vector<TLWE<typename iksP::domainP>> &cq,
    std::vector<TLWE<typename iksP::domainP>> &cr,
    const std::vector<TLWE<typename iksP::domainP>> &cx,
    const std::vector<TLWE<typename iksP::domainP>> &cy,
    const std::vector<TLWE<typename iksP::domainP>> &cz, const EvalKey &ek)
{
    assert(cx.size() == cy.size());
    const uint ns = cx.size();
    // 6.
    LongMulHPar<brP, iksP, ahP, intP>(cq, cx, cz, ek);
    // 7.
    std::vector<TLWE<typename iksP::domainP>> temp2(ns);
    LongMulLPar<brP, iksP, ahP, intP>(temp2, cy, cq, ek);
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
        // LongCaryyPropagation(cq) and LongInvReLU(temp2, temp1, cy)
        // have no data conflict — run concurrently
#pragma omp parallel sections num_threads(2)
        {
#pragma omp section
            { LongCaryyPropagation<brP, iksP, intP>(cq, cq, ek); }
#pragma omp section
            {
                LongInvReLUPar<brP, iksP, ahP, intP>(temp2, temp1, cy,
                                                      ek);
            }
        }
        LongSub<brP, iksP, intP>(cr, cr, temp2, ek);
    }
}

// Parallel version of HomIntInvReducedNRSeed32
template <class brP, class iksP, class cbbrP, class ahP, class intP>
void HomIntInvReducedNRSeed32Par(
    std::vector<TLWE<typename iksP::domainP>> &z1,
    const std::vector<TLWE<typename iksP::domainP>> &cyy, const EvalKey &ek)
{
    assert(cyy.size() == 8);
    alignas(64) std::array<TRGSWFFT<typename cbbrP::targetP>, 3> address;
#pragma omp parallel for
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
    const auto z1table =
        reducednr32seedtablegen<typename cbbrP::targetP, intP>();
    for (int i = 0; i < (1 << 3); i++)
        z1tabletrlwe[i][cbbrP::targetP::k] = z1table[i];

    TRLWE<typename cbbrP::targetP> z1lowertable;
    UROMUX<typename cbbrP::targetP, 3, 0>(z1lowertable, address, z1tabletrlwe);

    std::array<TLWE<typename iksP::domainP>, 2> z1part;
    GateBootstrappingManyLUT<brP, 2>(z1part, z1lowerlvlh, ek.getbkfft<brP>(),
                                     z1lowertable);
    z1.assign(z1part.begin(), z1part.end());
}

// Parallel version of HomIntInvReducedNR
template <class brP, class iksP, class cbbrP, class ahP, class intP>
void HomIntInvReducedNRPar(
    std::vector<TLWE<typename iksP::domainP>> &cz,
    const std::vector<TLWE<typename iksP::domainP>> &cy, const EvalKey &ek)
{
    const size_t ns = cy.size();
    assert(ns == 4 || ns == 8);
    const size_t seedwords = ns / 4;
    const size_t halfwords = ns / 2;
    // 1.
    TLWE<typename iksP::domainP> clzu, clzl;
    HomIntCLZPar<brP, iksP, intP>(clzu, clzl, cy, ek);
    std::vector<TLWE<typename iksP::domainP>> cyy;
    ULeftShiftPar<brP, iksP, cbbrP, ahP, intP>(cyy, cy, clzu, clzl, ek);

    // 1b + 2: PreprocessMulTB and seed generation are independent — concurrent
    std::vector<TRLWE<typename brP::targetP>> cyytb;
    std::vector<TLWE<typename iksP::domainP>> z1;
#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        {
            PreprocessMulTBPar<brP, iksP, ahP, intP>(
                cyytb, std::ranges::subrange(cyy.cbegin(), cyy.cend()), ek);
        }
#pragma omp section
        {
            if (ns == 4) {
                z1.resize(1);
                TLWE<typename iksP::targetP> shifted;
                IdentityKeySwitch<iksP>(shifted, cyy[3],
                                        ek.getiksk<iksP>());
                shifted[iksP::targetP::k * iksP::targetP::n] +=
                    intμ<typename iksP::targetP, intP>() / 2;
                GateBootstrappingTLWE2TLWEFFT<brP>(
                    z1[0], shifted, ek.getbkfft<brP>(),
                    z1poly<typename brP::targetP, intP>());
            }
            else {
                HomIntInvReducedNRSeed32Par<brP, iksP, cbbrP, ahP, intP>(
                    z1, cyy, ek);
            }
        }
    }
    auto hyytb =
        std::ranges::subrange(cyytb.cbegin() + seedwords, cyytb.cend());

    // 3. z = (z << W/4) - (z*(hy * z >> W/4)>>W/4)
    std::vector<TLWE<typename iksP::domainP>> z2(seedwords + halfwords);
    std::vector<TLWE<typename iksP::domainP>> temp(seedwords + halfwords);
    {
        LongMulPar<brP, iksP, intP>(
            std::ranges::subrange(temp.begin(), temp.end()),
            std::ranges::subrange(z1.cbegin(), z1.cend()), hyytb, ek);
        LongMulPar<brP, iksP, ahP, intP>(
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
        LongMulPar<brP, iksP, intP>(temp, z2, cyytb, ek);
        temp.erase(temp.begin(), temp.begin() + (halfwords - 1));
        // reuse z1
        z1.resize(z2.size() + temp.size());
        LongMulPar<brP, iksP, ahP, intP>(z1, z2, temp, ek);
        z1.erase(z1.begin(), z1.begin() + halfwords);
        temp.assign(ns + 1, {});
        {
            // Parallel IntMulPlain — each z2[i] independent
            const int hw = static_cast<int>(halfwords);
            std::vector<TLWE<typename iksP::domainP>> mp_uppers(hw),
                mp_lowers(hw);
#pragma omp parallel for
            for (int i = 0; i < hw; i++) {
                IntMulPlain<brP, iksP, intP>(mp_uppers[i], mp_lowers[i],
                                             z2[i], 8, ek);
            }
            for (size_t i = 0; i < halfwords; i++) {
                TLWEAdd<typename iksP::domainP>(temp[halfwords + i],
                                                temp[halfwords + i],
                                                mp_lowers[i]);
                TLWEAdd<typename iksP::domainP>(temp[halfwords + i + 1],
                                                temp[halfwords + i + 1],
                                                mp_uppers[i]);
            }
        }
        LongSub<brP, iksP, intP>(cz, temp, z1, ek);
    }
    // 5.
    cz.resize(ns + 1);
    {
        ULeftShiftHPar<brP, iksP, cbbrP, ahP, intP>(cz, cz, clzu, clzl, ek);
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
        LongMulLPar<brP, iksP, ahP, intP>(cyy, cy, cz, ek);
        TLWE<typename iksP::domainP> ccarry;
        LongAdd<brP, iksP, intP>(ccarry, cyy, cyy, cy, ek);
        for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
            cz[0][i] -= ccarry[i];
        cz[0][iksP::domainP::k * iksP::domainP::n] +=
            intμ<typename iksP::domainP, intP>();
        LongCaryyPropagation<brP, iksP, intP>(cz, cz, ek);
    }
}

// Parallel version of HomIntInvLUTNR
template <class brP, class iksP, class cbbrP, class ahP, class intP>
void HomIntInvLUTNRPar(std::vector<TLWE<typename iksP::domainP>> &cz,
                       const std::vector<TLWE<typename iksP::domainP>> &cy,
                       const EvalKey &ek)
{
    assert(cy.size() == 4);
    // 1.
    TLWE<typename iksP::domainP> clzu, clzl;
    HomIntCLZPar<brP, iksP, intP>(clzu, clzl, cy, ek);
    std::vector<TLWE<typename iksP::domainP>> cyy;
    ULeftShiftPar<brP, iksP, cbbrP, ahP, intP>(cyy, cy, clzu, clzl, ek);

    // 2. Hybrid version - parallel CB loop
    alignas(64) std::array<TRGSWFFT<typename cbbrP::targetP>, 3> address;
#pragma omp parallel for
    for (int i = 0; i < intP::bit - 1; i++) {
        TLWE<typename brP::targetP> tlwebit;
        for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
            tlwebit[j] = (cyy[3][j] << (intP::bit - i));
        tlwebit[brP::targetP::k * brP::targetP::n] +=
            (intμ<typename brP::targetP, intP>() / 2) << (intP::bit - i);
        AnnihilateCircuitBootstrappingFFT<iksP, cbbrP, ahP>(address[i],
                                                            tlwebit, ek);
    }

    // 3.
    cz.resize(5);
    cz[4] = {};
    cz[4][brP::targetP::k * brP::targetP::n] =
        intμ<typename brP::targetP, intP>();

    // Hybrid version
    TLWE<typename iksP::targetP> czlowerlvlh;
    IdentityKeySwitch<iksP>(czlowerlvlh, cyy[2], ek.getiksk<iksP>());
    czlowerlvlh[iksP::targetP::k * iksP::targetP::n] +=
        intμ<typename iksP::targetP, intP>() / 2;

    std::array<std::array<Polynomial<typename cbbrP::targetP>, (1 << 3)>, 2>
        inv8bittable =
            inv8bittablegen<typename cbbrP::targetP, intP>();
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

    // 4.
    ULeftShiftHPar<brP, iksP, cbbrP, ahP, intP>(cz, cz, clzu, clzl, ek);
    cz.resize(4);

    // 6.
    {
        std::vector<TLWE<typename iksP::domainP>> cyz(4);
        // Downward Fix
        LongMulLPar<brP, iksP, ahP, intP>(cyz, cy, cz, ek);
        // reuse clzu, larger than or equals to 2^W
        LongSub<brP, iksP, intP>(clzu, cyy, cyz, cy, ek);
        TLWESub<typename iksP::domainP>(cz[0], cz[0], clzu);
        LongBorrowPropagation<brP, iksP, intP>(cz, cz, ek);
    }
}

}  // namespace TFHEpp
