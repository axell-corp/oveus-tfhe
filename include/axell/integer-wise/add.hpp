#pragma once

#include <array>
#include <tfhe/cloudkey.hpp>
#include <tfhe/gatebootstrapping.hpp>
#include <tfhe/keyswitch.hpp>
#include <limits>
#include <params.hpp>
#include <utils.hpp>

#include "intkey.hpp"

namespace TFHEpp {


template <class P, class intP>
constexpr Polynomial<P> carryandsignpoly()
{
    Polynomial<P> poly;
    for (int i = 0; i < P::n / 2; i++) {
        poly[2 * i] =
            -(1ULL << (std::numeric_limits<typename P::T>::digits - 2));
        poly[2 * i + 1] = -(intμ<P, intP>() / 2);
    }
    return poly;
}

template <class bkP, class iksP, class intP>
void IntCarryExtract(TLWE<typename iksP::domainP> &ccarry,
                     TLWE<typename iksP::domainP> &cres,
                     const TLWE<typename iksP::domainP> &ca, const EvalKey &ek)
{
    TLWE<typename iksP::targetP> calvlI;
    IdentityKeySwitch<iksP>(calvlI, ca, ek.getiksk<iksP>());
    calvlI[iksP::targetP::k * iksP::targetP::n] +=
        intμ<typename iksP::targetP, intP>() / 2;
    std::array<TLWE<typename iksP::domainP>, 2> csc;
    GateBootstrappingManyLUT<bkP, 2>(
        csc, calvlI, ek.getbkfft<bkP>(),
        carryandsignpoly<typename bkP::targetP, intP>());
    for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        cres[i] = ca[i] - csc[0][i];
    cres[iksP::domainP::k * iksP::domainP::n] -=
        1ULL << (std::numeric_limits<typename bkP::targetP::T>::digits - 2);
    ccarry = csc[1];
    ccarry[iksP::domainP::k * iksP::domainP::n] +=
        intμ<typename iksP::domainP, intP>() / 2;
}

// a+b
template <class bkP, class iksP, class intP>
void IntAdd(TLWE<typename iksP::domainP> &ccarry,
            TLWE<typename iksP::domainP> &cres,
            const TLWE<typename iksP::domainP> &a,
            const TLWE<typename iksP::domainP> &b, const EvalKey &ek)
{
    TLWE<typename iksP::domainP> cadd;
    for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        cadd[i] = a[i] + b[i];
    IntCarryExtract<bkP, iksP, intP>(ccarry, cres, cadd, ek);
}

template <class bkP, class iksP, class intP>
void IntAdd(TLWE<typename iksP::domainP> &cres,
            const TLWE<typename iksP::domainP> &a,
            const TLWE<typename iksP::domainP> &b, const EvalKey &ek)
{
    TLWE<typename iksP::domainP> temp;
    IntAdd<bkP, iksP, intP>(temp, cres, a, b, ek);
}

template <class P, class intP>
constexpr Polynomial<P> carryandsignfmapoly()
{
    Polynomial<P> poly;
    for (int i = 0; i < P::n / 2; i++) {
        poly[2 * i] =
            -(1ULL << (std::numeric_limits<typename P::T>::digits - 2));
        poly[2 * i + 1] = -(extendedintμ<P, intP>() / 2);
    }
    return poly;
}

template <class bkP, class iksP, class intP>
void IntCarryExtractFMA(TLWE<typename iksP::domainP> &ccarry,
                        TLWE<typename iksP::domainP> &cres,
                        const TLWE<typename iksP::domainP> &ca,
                        const EvalKey &ek)
{
    TLWE<typename iksP::targetP> calvlI;
    IdentityKeySwitch<iksP>(calvlI, ca, ek.getiksk<iksP>());
    calvlI[iksP::targetP::k * iksP::targetP::n] +=
        intμ<typename iksP::targetP, intP>() / 2;
    std::array<TLWE<typename iksP::domainP>, 2> csc;
    GateBootstrappingManyLUT<bkP, 2>(
        csc, calvlI, ek.getbkfft<bkP>(),
        carryandsignfmapoly<typename bkP::targetP, intP>());
    for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        cres[i] = ca[i] - csc[0][i];
    cres[iksP::domainP::k * iksP::domainP::n] -=
        1ULL << (std::numeric_limits<typename bkP::targetP::T>::digits - 2);
    ccarry = csc[1];
    ccarry[iksP::domainP::k * iksP::domainP::n] +=
        extendedintμ<typename iksP::domainP, intP>() / 2;
}

// a+b
template <class bkP, class iksP, class intP>
void IntAddFMA(TLWE<typename iksP::domainP> &ccarry,
               TLWE<typename iksP::domainP> &cres,
               const TLWE<typename iksP::domainP> &a,
               const TLWE<typename iksP::domainP> &b, const EvalKey &ek)
{
    TLWE<typename iksP::domainP> cadd;
    for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        cadd[i] = a[i] + b[i];
    IntCarryExtractFMA<bkP, iksP, intP>(ccarry, cres, cadd, ek);
}

// a-b
template <class bkP, class iksP, class intP>
void IntSub(TLWE<typename iksP::domainP> &cborrow,
            TLWE<typename iksP::domainP> &cres,
            const TLWE<typename iksP::domainP> &a,
            const TLWE<typename iksP::domainP> &b, const EvalKey &ek)
{
    TLWE<typename iksP::domainP> csub;
    for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        csub[i] = a[i] - b[i];
    TLWE<typename iksP::targetP> csublvlI;
    IdentityKeySwitch<iksP>(csublvlI, csub, ek.getiksk<iksP>());
    csublvlI[iksP::targetP::k * iksP::targetP::n] +=
        intμ<typename iksP::targetP, intP>() / 2;
    std::array<TLWE<typename iksP::domainP>, 2> csc;
    GateBootstrappingManyLUT<bkP, 2>(
        csc, csublvlI, ek.getbkfft<bkP>(),
        carryandsignpoly<typename bkP::targetP, intP>());
    for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        cres[i] = csub[i] - csc[0][i];
    cres[iksP::domainP::k * iksP::domainP::n] -=
        1ULL << (std::numeric_limits<typename bkP::targetP::T>::digits - 2);
    cborrow = csc[1];
    cborrow[iksP::domainP::k * iksP::domainP::n] +=
        intμ<typename iksP::domainP, intP>() / 2;
}

template <class bkP, class iksP, class intP>
void IntSub(TLWE<typename iksP::domainP> &cres,
            const TLWE<typename iksP::domainP> &a,
            const TLWE<typename iksP::domainP> &b, const EvalKey &ek)
{
    TLWE<typename iksP::domainP> temp;
    IntSub<bkP, iksP, intP>(temp, cres, a, b, ek);
}

}  // namespace TFHEpp
