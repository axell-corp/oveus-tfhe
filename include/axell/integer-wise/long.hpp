#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <tfhe/circuitbootstrapping.hpp>
#include <tfhe/cloudkey.hpp>
#include <tfhe/detwfa.hpp>
#include <tfhe/gatebootstrapping.hpp>
#include <tfhe/keyswitch.hpp>
#include <limits>
#include <params.hpp>
#include <ranges>
#include <type_traits>
#include <utility>
#include <utils.hpp>

#include "add.hpp"
#include "mul.hpp"

namespace TFHEpp {

template <class brP, class iksP, class ahP, class intP>
void LongInvReLU(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const TLWE<typename iksP::domainP> &csign,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const EvalKey &ek)
{
    assert(cres.size() == a.size());
    TRLWE<typename brP::targetP> maskedbysign;
    {
        std::array<TLWE<typename iksP::domainP>, intP::t> cisgntlwes;
        cisgntlwes[0] = {};
        for (int i = 1; i < intP::t; i++)
            for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
                cisgntlwes[i][j] = i * csign[j];
        TLWE2TablePacking<ahP, intP::t>(maskedbysign, cisgntlwes,
                                        ek.getahk<ahP>());
    }
    for (int seg = 0; seg < a.size(); seg++) {
        TLWE<typename iksP::targetP> shifteda;
        IdentityKeySwitch<iksP>(shifteda, a[seg], ek.getiksk<iksP>());
        shifteda[brP::domainP::k * brP::domainP::n] +=
            intμ<typename brP::domainP, intP>() / 2;
        GateBootstrappingTLWE2TLWEFFTfunc<brP>(
            cres[seg], shifteda, ek.getbkfft<brP>(), maskedbysign);
    }
}

template <class bkP, class iksP, class ahP, class intP>
void LongInvReLU(std::vector<TLWE<typename iksP::domainP>> &cres,
                 const TLWE<typename iksP::domainP> &csign,
                 const std::vector<TLWE<typename iksP::domainP>> &a,
                 const EvalKey &ek)
{
    cres.resize(a.size());
    LongInvReLU<bkP, iksP, ahP, intP>(
        std::ranges::subrange(cres.begin(), cres.end()), csign,
        std::ranges::subrange(a.cbegin(), a.cend()), ek);
}

template <class bkP, class iksP, class intP>
void LongCaryyPropagation(
    TLWE<typename iksP::domainP> &ccarry,
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const EvalKey &ek)
{
    ccarry = {};
    for (int i = 0; i < a.size(); i++) {
        IntAdd<bkP, iksP, intP>(ccarry, cres[i], ccarry, a[i], ek);
    }
}

template <class bkP, class iksP, class intP>
void LongCaryyPropagation(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const EvalKey &ek)
{
    TLWE<typename iksP::domainP> temp;
    LongCaryyPropagation<bkP, iksP, intP>(temp, cres, a, ek);
}

template <class bkP, class iksP, class intP>
void LongCaryyPropagation(TLWE<typename iksP::domainP> &ccarry,
                          std::vector<TLWE<typename iksP::domainP>> &cres,
                          const std::vector<TLWE<typename iksP::domainP>> &a,
                          const EvalKey &ek)
{
    const size_t numsegment = a.size();
    cres.resize(numsegment);
    LongCaryyPropagation<bkP, iksP, intP>(
        ccarry, std::ranges::subrange(cres.begin(), cres.end()),
        std::ranges::subrange(a.cbegin(), a.cend()), ek);
}

template <class bkP, class iksP, class intP>
void LongCaryyPropagation(std::vector<TLWE<typename iksP::domainP>> &cres,
                          const std::vector<TLWE<typename iksP::domainP>> &a,
                          const EvalKey &ek)
{
    TLWE<typename iksP::domainP> temp;
    LongCaryyPropagation<bkP, iksP, intP>(temp, cres, a, ek);
}

template <class bkP, class iksP, class intP>
void LongAdd(
    TLWE<typename iksP::domainP> &ccarry,
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        b,
    const EvalKey &ek)
{
    const size_t minnumsegment = std::min(a.size(), b.size());
    IntAdd<bkP, iksP, intP>(ccarry, cres[0], a[0], b[0], ek);
    for (int i = 1; i < minnumsegment; i++) {
        TLWEAdd<typename iksP::domainP>(ccarry, ccarry, a[i]);
        IntAdd<bkP, iksP, intP>(ccarry, cres[i], ccarry, b[i], ek);
    }
    const size_t maxnumsegment = std::max(a.size(), b.size());
    assert(maxnumsegment == cres.size());
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        longer = (a.size() > b.size()) ? a : b;
    for (int i = minnumsegment; i < maxnumsegment; i++) {
        IntAdd<bkP, iksP, intP>(ccarry, cres[i], ccarry, longer[i], ek);
    }
}

template <class bkP, class iksP, class intP>
void LongAdd(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        b,
    const EvalKey &ek)
{
    TLWE<typename iksP::domainP> temp;
    LongAdd<bkP, iksP, intP>(temp, cres, a, b, ek);
}

template <class bkP, class iksP, class intP>
void LongAdd(TLWE<typename iksP::domainP> &ccarry,
             std::vector<TLWE<typename iksP::domainP>> &cres,
             const std::vector<TLWE<typename iksP::domainP>> &a,
             const std::vector<TLWE<typename iksP::domainP>> &b,
             const EvalKey &ek)
{
    const size_t numsegment = std::max(a.size(), b.size());
    cres.resize(numsegment);
    LongAdd<bkP, iksP, intP>(ccarry,
                             std::ranges::subrange(cres.begin(), cres.end()),
                             std::ranges::subrange(a.cbegin(), a.cend()),
                             std::ranges::subrange(b.cbegin(), b.cend()), ek);
}

template <class bkP, class iksP, class intP>
void LongAdd(std::vector<TLWE<typename iksP::domainP>> &cres,
             const std::vector<TLWE<typename iksP::domainP>> &a,
             const std::vector<TLWE<typename iksP::domainP>> &b,
             const EvalKey &ek)
{
    TLWE<typename iksP::domainP> temp;
    LongAdd<bkP, iksP, intP>(temp, cres, a, b, ek);
}

template <class bkP, class iksP, class intP>
void LongBorrowPropagation(
    TLWE<typename iksP::domainP> &cborrow,
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const EvalKey &ek)
{
    cborrow = {};
    for (int i = 0; i < a.size(); i++) {
        IntSub<bkP, iksP, intP>(cborrow, cres[i], a[i], cborrow, ek);
    }
}

template <class bkP, class iksP, class intP>
void LongBorrowPropagation(std::vector<TLWE<typename iksP::domainP>> &cres,
                           const std::vector<TLWE<typename iksP::domainP>> &a,
                           const EvalKey &ek)
{
    TLWE<typename iksP::domainP> temp;
    LongBorrowPropagation<bkP, iksP, intP>(temp, cres, a, ek);
}

template <class bkP, class iksP, class intP>
void LongSub(
    TLWE<typename iksP::domainP> &cborrow,
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        b,
    const EvalKey &ek)
{
    const size_t numsegment = std::max(a.size(), b.size());
    IntSub<bkP, iksP, intP>(cborrow, cres[0], a[0], b[0], ek);
    for (int i = 1; i < numsegment; i++) {
        for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
            cborrow[j] = a[i][j] - cborrow[j];
        IntSub<bkP, iksP, intP>(cborrow, cres[i], cborrow, b[i], ek);
    }
}

template <class bkP, class iksP, class intP>
void LongSub(TLWE<typename iksP::domainP> &cborrow,
             std::vector<TLWE<typename iksP::domainP>> &cres,
             const std::vector<TLWE<typename iksP::domainP>> &a,
             const std::vector<TLWE<typename iksP::domainP>> &b,
             const EvalKey &ek)
{
    const size_t numsegment = std::max(a.size(), b.size());
    cres.resize(numsegment);
    LongSub<bkP, iksP, intP>(cborrow,
                             std::ranges::subrange(cres.begin(), cres.end()),
                             std::ranges::subrange(a.cbegin(), a.cend()),
                             std::ranges::subrange(b.cbegin(), b.cend()), ek);
}

template <class bkP, class iksP, class intP>
void LongSub(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        b,
    const EvalKey &ek)
{
    TLWE<typename iksP::domainP> temp;
    LongSub<bkP, iksP, intP>(temp, cres, a, b, ek);
}

template <class bkP, class iksP, class intP>
void LongSub(std::vector<TLWE<typename iksP::domainP>> &cres,
             const std::vector<TLWE<typename iksP::domainP>> &a,
             const std::vector<TLWE<typename iksP::domainP>> &b,
             const EvalKey &ek)
{
    TLWE<typename iksP::domainP> temp;
    LongSub<bkP, iksP, intP>(temp, cres, a, b, ek);
}

template <class bkP, class iksP, class intP>
void PartialLongMul(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::targetP>>::const_iterator>
        shifteda,
    const TRLWE<typename bkP::targetP> &btb, const EvalKey &ek)
{
    const uint numsegment = shifteda.size();
    assert(cres.size() == shifteda.size() + 1);
    for (size_t i = 0; i < numsegment; i++) {
        std::array<TLWE<typename iksP::domainP>, 2> cmul;
        TLWE<typename iksP::domainP> ccarry;
        IntMulSharedIKSTB<bkP, iksP, intP>(cmul[1], cmul[0], shifteda[i], btb,
                                           ek);
        IntAdd<bkP, iksP, intP>(ccarry, cres[i], cres[i], cmul[0], ek);
        for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
            cres[i + 1][j] += cmul[1][j] + ccarry[j];
        if (i != numsegment - 1) {
            IntCarryExtract<bkP, iksP, intP>(ccarry, cres[i + 1], cres[i + 1],
                                             ek);
            TLWEAdd<typename iksP::domainP>(cres[i + 2], cres[i + 2], ccarry);
        }
    }
}

template <class brP, class iksP, class ahP, class intP>
void IntCombinedMulSquareTBgen(TRLWE<typename brP::targetP> &cres,
                               const TLWE<typename iksP::domainP> &bupper,
                               const TLWE<typename iksP::domainP> &blower,
                               const EvalKey &ek)
{
    std::array<std::array<TLWE<typename brP::targetP>, intP::t>, 2> tabletlwes;
    tabletlwes[0][0] = {};
    tabletlwes[1][0] = {};
    {
        // IKS
        TLWE<typename iksP::targetP> shiftedb;
        IdentityKeySwitch<iksP>(shiftedb, blower, ek.getiksk<iksP>());

        // 1st stage
        shiftedb[brP::domainP::k * brP::domainP::n] +=
            intμ<typename brP::domainP, intP>() / 2;
        TLWE<typename brP::targetP> extendedb;
        GateBootstrappingTLWE2TLWEFFT<brP>(
            extendedb, shiftedb, ek.getbkfft<brP>(),
            mulpolygen<typename brP::targetP, intP>());
        tabletlwes[0][1] = extendedb;
    }
    {
        // IKS
        TLWE<typename iksP::targetP> shiftedb;
        IdentityKeySwitch<iksP>(shiftedb, bupper, ek.getiksk<iksP>());

        // 1st stage
        shiftedb[brP::domainP::k * brP::domainP::n] +=
            intμ<typename brP::domainP, intP>() / 2;
        TLWE<typename brP::targetP> extendedb;
        GateBootstrappingTLWE2TLWEFFT<brP>(
            extendedb, shiftedb, ek.getbkfft<brP>(),
            mulpolygen<typename brP::targetP, intP>());
        tabletlwes[1][1] = extendedb;
    }

    for (int i = 2; i < intP::t; i++)
        for (int j = 0; j <= brP::targetP::k * brP::targetP::n; j++) {
            tabletlwes[0][i][j] = i * tabletlwes[0][1][j];
            tabletlwes[1][i][j] = i * tabletlwes[1][1][j];
        }
    TLWE2TablePackingManyLUT<ahP, intP::t, 2>(cres, tabletlwes,
                                              ek.getahk<ahP>());
}

template <class bkP, class iksP, class intP>
void PartialLongMul2(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::targetP>>::const_iterator>
        shifteda,
    const TRLWE<typename bkP::targetP> &btb, const EvalKey &ek)
{
    const uint numsegment = shifteda.size();
    assert(cres.size() == shifteda.size() + 2);
    // Compute in 2t^2
    for (size_t i = 0; i < numsegment; i++) {
        std::array<TLWE<typename iksP::domainP>, 2> cmul;
        GateBootstrappingManyLUT<bkP, 2>(cmul, shifteda[i], ek.getbkfft<bkP>(),
                                         btb);
        if (i == 0)
            cres[i] = cmul[0];
        else {
            for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
                cres[i][j] += cmul[0][j];
        }
        cres[i + 1] = cmul[1];
    }
    // Convert to 2t
    TLWE<typename iksP::domainP> cupper;
    cupper = cres[0];
    IntSquare2Words<bkP, iksP, intP>(cupper, cres[0], cupper, ek);
    for (size_t i = 1; i < numsegment; i++) {
        TLWE<typename iksP::domainP> temp = cres[i];
        IntSquare2Words<bkP, iksP, intP>(temp, cres[i], temp, ek);
        IntAddFMA<bkP, iksP, intP>(cupper, cres[i], cupper, cres[i], ek);
        TLWEAdd<typename iksP::domainP>(cres[i + 1], cres[i + 1], cupper);
        IntCarryExtract<bkP, iksP, intP>(temp, cupper, temp, ek);
        TLWEAdd<typename iksP::domainP>(cres[i + 1], cres[i + 1], temp);
    }
    {
        TLWE<typename iksP::targetP> shiftedcupper;
        IdentityKeySwitch<iksP>(shiftedcupper, cupper, ek.getiksk<iksP>());
        shiftedcupper[bkP::domainP::k * bkP::domainP::n] +=
            intμ<typename bkP::domainP, intP>() / 2;
        GateBootstrappingTLWE2TLWEFFT<bkP>(
            cupper, shiftedcupper, ek.getbkfft<bkP>(),
            mulpolygen<typename bkP::targetP, intP>());
    }
    TLWEAdd<typename iksP::domainP>(cres[numsegment], cres[numsegment], cupper);
    // Avoid memory access confrict
    cres[numsegment + 1] = cres[numsegment];
    IntSquare2Words<bkP, iksP, intP>(cres[numsegment + 1], cres[numsegment],
                                     cres[numsegment + 1], ek);
}

// Assume Unsigned
template <class bkP, class iksP, class intP>
void LongMul(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const std::ranges::subrange<
        typename std::vector<TRLWE<typename bkP::targetP>>::const_iterator>
        btb,
    const EvalKey &ek, const bool isoddlenb = false)
{
    if (a.size() == 1 && btb.size() == 1 && isoddlenb) {
        IntMul<bkP, iksP, intP>(cres[1], cres[0], a[0], btb[0], ek);
    }
    else {
        // 1st
        // Shared IKS
        std::vector<TLWE<typename iksP::targetP>> shifteda(a.size());
        for (size_t i = 0; i < a.size(); i++) {
            IdentityKeySwitch<iksP>(shifteda[i], a[i], ek.getiksk<iksP>());
            shifteda[i][bkP::domainP::k * bkP::domainP::n] +=
                intμ<typename bkP::domainP, intP>() / 2;
        }
        std::vector<TLWE<typename iksP::domainP>> partmul(a.size() + 2);
        // 2nd
        for (size_t i = 0; i < btb.size() - (isoddlenb ? 1 : 0); i++) {
            PartialLongMul2<bkP, iksP, intP>(
                std::ranges::subrange(partmul.begin(), partmul.end()),
                std::ranges::subrange(shifteda.cbegin(), shifteda.cend()),
                btb[i], ek);
            if (i == 0)
                for (size_t j = 0; j < a.size() + 2; j++) cres[j] = partmul[j];
            else {
                TLWE<typename iksP::domainP> ccarry;
                LongAdd<bkP, iksP, intP>(
                    ccarry,
                    std::ranges::subrange(cres.begin() + 2 * i,
                                          cres.begin() + 2 * i + a.size()),
                    std::ranges::subrange(cres.begin() + 2 * i,
                                          cres.begin() + 2 * i + a.size()),
                    std::ranges::subrange(partmul.begin(),
                                          partmul.begin() + a.size()),
                    ek);
                IntAdd<bkP, iksP, intP>(ccarry, cres[2 * i + a.size()],
                                        partmul[a.size()], ccarry, ek);
                for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
                    cres[2 * i + a.size() + 1][j] =
                        partmul[a.size() + 1][j] + ccarry[j];
            }
        }
        if (isoddlenb) {
            if (btb.size() == 1) {
                for (size_t j = 0; j < cres.size(); j++) cres[j] = {};
            }
            else {
                for (size_t j = 2 * (btb.size() - 2) + a.size() + 2;
                     j < cres.size(); j++)
                    cres[j] = {};
            }
            PartialLongMul<bkP, iksP, intP>(
                std::ranges::subrange(
                    cres.begin() + (btb.size() - 1) * 2 + 1 - 1,
                    cres.begin() + (btb.size() - 1) * 2 + 1 - 1 + a.size() + 1),
                std::ranges::subrange(shifteda.cbegin(), shifteda.cend()),
                btb[btb.size() - 1], ek);
        }
    }
}

template <class bkP, class iksP, class ahP, class intP>
void PreprocessMulTB(
    std::vector<TRLWE<typename bkP::targetP>> &btb,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        b,
    const EvalKey &ek)
{
    btb.resize(b.size() / 2 + (b.size() & 1));
    for (size_t i = 0; i < btb.size() - (b.size() & 1); i++) {
        IntCombinedMulSquareTBgen<bkP, iksP, ahP, intP>(btb[i], b[2 * i + 1],
                                                        b[2 * i], ek);
    }
    if (b.size() & 1) {
        IntMulSquareTBgen<bkP, iksP, ahP, intP>(btb[btb.size() - 1],
                                                b[b.size() - 1], ek);
    }
}

// Assume Unsigned
template <class bkP, class iksP, class ahP, class intP>
void LongMul(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        b,
    const EvalKey &ek)
{
    if (a.size() < b.size()) {
        LongMul<bkP, iksP, ahP, intP>(cres, b, a, ek);
        return;
    }
    assert(cres.size() == a.size() + b.size());
    if (cres.size() == 2) {
        IntMul<bkP, iksP, ahP, intP>(cres[1], cres[0], a[0], b[0], ek);
    }
    else {
        std::vector<TRLWE<typename bkP::targetP>> btb(b.size() / 2 +
                                                      (b.size() & 1));
        // 2nd
        PreprocessMulTB<bkP, iksP, ahP, intP>(btb, b, ek);
        LongMul<bkP, iksP, intP>(
            cres, a, std::ranges::subrange(btb.cbegin(), btb.cend()), ek,
            b.size() & 1);
    }
}

template <class bkP, class iksP, class ahP, class intP>
void LongMul(std::vector<TLWE<typename iksP::domainP>> &cres,
             const std::vector<TLWE<typename iksP::domainP>> &a,
             const std::vector<TLWE<typename iksP::domainP>> &b,
             const EvalKey &ek)
{
    assert(cres.size() == a.size() + b.size());
    cres.resize(a.size() + b.size());
    LongMul<bkP, iksP, ahP, intP>(
        std::ranges::subrange(cres.begin(), cres.end()),
        std::ranges::subrange(a.cbegin(), a.cend()),
        std::ranges::subrange(b.cbegin(), b.cend()), ek);
}

template <class bkP, class iksP, class ahP, class intP>
void PartialLongMulL(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::targetP>>::const_iterator>
        shifteda,
    const TLWE<typename bkP::targetP> &b, const EvalKey &ek)
{
    const uint numsegment = shifteda.size();
    assert(cres.size() == shifteda.size());
    // TBshare
    TRLWE<typename bkP::targetP> btb;
    IntMulSquareTBgen<bkP, iksP, ahP, intP>(btb, b, ek);
    for (size_t i = 0; i < numsegment; i++) {
        std::array<TLWE<typename iksP::domainP>, 2> cmul;
        TLWE<typename iksP::domainP> ccarry;
        IntMulSharedIKSTB<bkP, iksP, intP>(cmul[1], cmul[0], shifteda[i], btb,
                                           ek);
        IntAdd<bkP, iksP, intP>(ccarry, cres[i], cres[i], cmul[0], ek);
        if (i != numsegment - 1) {
            for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
                cres[i + 1][j] += cmul[1][j] + ccarry[j];
            IntCarryExtract<bkP, iksP, intP>(ccarry, cres[i + 1], cres[i + 1],
                                             ek);
            if (i != numsegment - 2)
                TLWEAdd<typename iksP::domainP>(cres[i + 2], cres[i + 2],
                                                ccarry);
        }
    }
}

template <class bkP, class iksP, class ahP, class intP>
void PartialLongMulL2(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::targetP>>::const_iterator>
        shifteda,
    const TLWE<typename bkP::targetP> &bupper,
    const TLWE<typename bkP::targetP> &blower, const EvalKey &ek)
{
    const uint numsegment = shifteda.size();
    assert(cres.size() == shifteda.size());
    // TBshare
    TRLWE<typename bkP::targetP> btb;
    IntCombinedMulSquareTBgen<bkP, iksP, ahP, intP>(btb, bupper, blower, ek);
    // Compute in 2t^2
    for (size_t i = 0; i < numsegment; i++) {
        std::array<TLWE<typename iksP::domainP>, 2> cmul;
        GateBootstrappingManyLUT<bkP, 2>(cmul, shifteda[i], ek.getbkfft<bkP>(),
                                         btb);
        if (i == 0)
            cres[i] = cmul[0];
        else {
            for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
                cres[i][j] += cmul[0][j];
        }
        if (i != numsegment - 1) cres[i + 1] = cmul[1];
    }
    // Convert to 2t
    TLWE<typename iksP::domainP> cupper;
    cupper = cres[0];
    IntSquare2Words<bkP, iksP, intP>(cupper, cres[0], cupper, ek);
    for (size_t i = 1; i < numsegment; i++) {
        TLWE<typename iksP::domainP> temp = cres[i];
        IntSquare2Words<bkP, iksP, intP>(temp, cres[i], temp, ek);
        IntAddFMA<bkP, iksP, intP>(cupper, cres[i], cupper, cres[i], ek);
        if (i != numsegment - 1)
            TLWEAdd<typename iksP::domainP>(cres[i + 1], cres[i + 1], cupper);
        IntCarryExtract<bkP, iksP, intP>(temp, cupper, temp, ek);
        if (i != numsegment - 1)
            TLWEAdd<typename iksP::domainP>(cres[i + 1], cres[i + 1], temp);
    }
}

template <class bkP, class iksP, class ahP, class intP>
void LongMulL(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        b,
    const EvalKey &ek)
{
    const uint numsegment = a.size();
    assert(b.size() == numsegment);
    assert(cres.size() == numsegment);
    const uint numsegmentbit = std::countr_zero(numsegment);
    if (numsegmentbit == 0) {
        IntMulL<bkP, iksP, ahP, intP>(cres[0], a[0], b[0], ek);
    }
    else {
        // 1st
        // Shared IKS
        std::vector<TLWE<typename iksP::targetP>> shifteda(a.size());
        for (size_t i = 0; i < a.size(); i++) {
            IdentityKeySwitch<iksP>(shifteda[i], a[i], ek.getiksk<iksP>());
            shifteda[i][bkP::domainP::k * bkP::domainP::n] +=
                intμ<typename bkP::domainP, intP>() / 2;
        }
        std::vector<TLWE<typename iksP::domainP>> partmul(a.size());
        // 2nd
        for (size_t i = 0; i < b.size() / 2; i++) {
            PartialLongMulL2<bkP, iksP, ahP, intP>(
                std::ranges::subrange(partmul.begin(), partmul.end() - 2 * i),
                std::ranges::subrange(shifteda.cbegin(),
                                      shifteda.cend() - 2 * i),
                b[2 * i + 1], b[2 * i], ek);
            if (i == 0)
                for (size_t j = 0; j < a.size(); j++) cres[j] = partmul[j];
            else {
                TLWE<typename iksP::domainP> ccarry;
                LongAdd<bkP, iksP, intP>(
                    ccarry,
                    std::ranges::subrange(cres.begin() + 2 * i, cres.end() - 1),
                    std::ranges::subrange(cres.begin() + 2 * i, cres.end() - 1),
                    std::ranges::subrange(partmul.begin(),
                                          partmul.end() - 2 * i - 1),
                    ek);
                for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
                    cres[a.size() - 1][j] +=
                        partmul[a.size() - 2 * i - 1][j] + ccarry[j];
            }
        }
        if (b.size() & 1) {
            PartialLongMulL<bkP, iksP, ahP, intP>(
                std::ranges::subrange(
                    cres.begin() + b.size() - 1,
                    cres.begin() + b.size() - 1 + a.size() + 1),
                std::ranges::subrange(shifteda.cbegin(), shifteda.cend()),
                b[b.size() - 1], ek);
        }
        else {
            TLWE<typename iksP::domainP> ccarry;
            IntCarryExtract<bkP, iksP, intP>(ccarry, cres[b.size() - 1],
                                             cres[b.size() - 1], ek);
        }
    }
}

template <class bkP, class iksP, class ahP, class intP>
void LongMulL(std::vector<TLWE<typename iksP::domainP>> &cres,
              const std::vector<TLWE<typename iksP::domainP>> &a,
              const std::vector<TLWE<typename iksP::domainP>> &b,
              const EvalKey &ek)
{
    const uint numsegment = a.size();
    cres.resize(numsegment);
    LongMulL<bkP, iksP, ahP, intP>(
        std::ranges::subrange(cres.begin(), cres.end()),
        std::ranges::subrange(a.cbegin(), a.cend()),
        std::ranges::subrange(b.cbegin(), b.cend()), ek);
}

template <class bkP, class iksP, class ahP, class intP>
void LongMulH(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        b,
    const EvalKey &ek)
{
    const uint numsegment = a.size();
    assert(numsegment == cres.size());
    std::vector<TLWE<typename iksP::domainP>> mulh(2 * numsegment);
    LongMul<bkP, iksP, ahP, intP>(
        std::ranges::subrange(mulh.begin(), mulh.end()), a, b, ek);
    for (int i = 0; i < numsegment; i++) cres[i] = mulh[i + numsegment];
}

template <class bkP, class iksP, class ahP, class intP>
void LongMulH(std::vector<TLWE<typename iksP::domainP>> &cres,
              const std::vector<TLWE<typename iksP::domainP>> &a,
              const std::vector<TLWE<typename iksP::domainP>> &b,
              const EvalKey &ek)
{
    const uint numsegment = a.size();
    cres.resize(numsegment);
    std::vector<TLWE<typename iksP::domainP>> mulh(2 * numsegment);
    LongMul<bkP, iksP, ahP, intP>(
        std::ranges::subrange(mulh.begin(), mulh.end()),
        std::ranges::subrange(a.cbegin(), a.cend()),
        std::ranges::subrange(b.cbegin(), b.cend()), ek);
    for (int i = 0; i < numsegment; i++) cres[i] = mulh[i + numsegment];
}

}  // namespace TFHEpp
