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

template <class bkP, class iksP, class ahP, class intP>
void IntMulSquareTBgen(TRLWE<typename iksP::domainP> &cres,
                       const TLWE<typename iksP::domainP> &b, const EvalKey &ek)
{
    // IKS
    TLWE<typename iksP::targetP> shiftedb;
    IdentityKeySwitch<iksP>(shiftedb, b, ek.getiksk<iksP>());

    // 1st stage
    shiftedb[bkP::domainP::k * bkP::domainP::n] +=
        intμ<typename bkP::domainP, intP>() / 2;
    TLWE<typename bkP::targetP> extendedb;
    GateBootstrappingTLWE2TLWEFFT<bkP>(
        extendedb, shiftedb, ek.getbkfft<bkP>(),
        mulpolygen<typename bkP::targetP, intP>());

    // 2nd stage
    std::array<TLWE<typename bkP::targetP>, intP::t> extendedbmul;
    extendedbmul[0] = {};
    for (int i = 1; i < intP::t; i++)
        for (int j = 0; j <= bkP::targetP::k * bkP::targetP::n; j++)
            extendedbmul[i][j] = i * extendedb[j];
    TLWE2TablePacking<ahP, intP::t>(cres, extendedbmul, ek.getahk<ahP>());
}

template <class brP, class iksP, class intP>
void IntMulSquareSharedTB(TLWE<typename iksP::domainP> &cres,
                          const TLWE<typename iksP::domainP> &a,
                          const TRLWE<typename brP::targetP> &btb,
                          const EvalKey &ek)
{
    // 3rd stage
    TLWE<typename iksP::targetP> shifteda;
    IdentityKeySwitch<iksP>(shifteda, a, ek.getiksk<iksP>());
    shifteda[brP::domainP::k * brP::domainP::n] +=
        intμ<typename brP::domainP, intP>() / 2;
    GateBootstrappingTLWE2TLWEFFTfunc<brP>(cres, shifteda, ek.getbkfft<brP>(),
                                           btb);
}

template <class brP, class iksP, class ahP, class intP>
void IntMulSquare(TLWE<typename iksP::domainP> &cres,
                  const TLWE<typename iksP::domainP> &a,
                  const TLWE<typename iksP::domainP> &b, const EvalKey &ek)
{
    alignas(64) TRLWE<typename brP::targetP> extendedbmul;
    IntMulSquareTBgen<brP, iksP, ahP, intP>(extendedbmul, b, ek);
    IntMulSquareSharedTB<brP, iksP, intP>(cres, a, extendedbmul, ek);
}

// cupper and a can be the same
template <class bkP, class iksP, class intP>
void IntSquare2Words(TLWE<typename iksP::domainP> &cupper,
                     TLWE<typename iksP::domainP> &clower,
                     TLWE<typename iksP::domainP> &clowersquare,
                     const TLWE<typename iksP::domainP> &a, const EvalKey &ek)
{
    assert(&clower != &a);
    // 4th stage
    TLWE<typename iksP::domainP> scaledcreslvl2, ulsblvl2;
    TLWE<typename iksP::targetP> scaledcreslvl1;
    for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        scaledcreslvl2[i] = a[i] * intP::t;
    IdentityKeySwitch<iksP>(scaledcreslvl1, scaledcreslvl2, ek.getiksk<iksP>());
    scaledcreslvl1[iksP::targetP::k * iksP::targetP::n] +=
        intμ<typename iksP::targetP, intP>() / 2;
    GateBootstrappingTLWE2TLWEFFT<bkP>(
        ulsblvl2, scaledcreslvl1, ek.getbkfft<bkP>(),
        μpolygen<lvl2param, 1ULL << (std::numeric_limits<
                                         typename bkP::targetP::T>::digits -
                                     2)>());
    ulsblvl2[bkP::targetP::k * bkP::targetP::n] +=
        (1ULL << (std::numeric_limits<typename bkP::targetP::T>::digits - 1)) +
        (1ULL << (std::numeric_limits<typename bkP::targetP::T>::digits - 2));

    // 5th stage
    TLWE<typename iksP::targetP> lreslvlI;
    for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        clower[i] = scaledcreslvl2[i] - ulsblvl2[i];
    IdentityKeySwitch<iksP>(lreslvlI, clower, ek.getiksk<iksP>());
    lreslvlI[iksP::targetP::k * iksP::targetP::n] +=
        intμ<typename iksP::targetP, intP>() / 2;
    GateBootstrappingTLWE2TLWEFFT<bkP>(
        clowersquare, lreslvlI, ek.getbkfft<bkP>(),
        mulpolygen<typename iksP::domainP, intP>());

    // 6th stage
    for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        cupper[i] = a[i] - clowersquare[i];
}

template <class bkP, class iksP, class intP>
void IntSquare2Words(TLWE<typename iksP::domainP> &cupper,
                     TLWE<typename iksP::domainP> &clower,
                     const TLWE<typename iksP::domainP> &a, const EvalKey &ek)
{
    TLWE<typename iksP::domainP> temp;
    IntSquare2Words<bkP, iksP, intP>(cupper, clower, temp, a, ek);
}

// a*b
template <class brP, class iksP, class intP>
void IntMulSharedIKSTB(TLWE<typename iksP::domainP> &cupper,
                       TLWE<typename iksP::domainP> &clower,
                       const TLWE<typename iksP::targetP> &shifteda,
                       const TRLWE<typename brP::targetP> &btb,
                       const EvalKey &ek)
{
    TLWE<typename brP::targetP> creslvl2;
    GateBootstrappingTLWE2TLWEFFTfunc<brP>(creslvl2, shifteda,
                                           ek.getbkfft<brP>(), btb);
    IntSquare2Words<brP, iksP, intP>(cupper, clower, creslvl2, ek);
}

// a*b
template <class bkP, class iksP, class intP>
void IntMul(TLWE<typename iksP::domainP> &cupper,
            TLWE<typename iksP::domainP> &clower,
            const TLWE<typename iksP::domainP> &a,
            const TRLWE<typename bkP::targetP> &btb, const EvalKey &ek)
{
    TLWE<typename bkP::targetP> creslvl2;
    IntMulSquareSharedTB<bkP, iksP, intP>(creslvl2, a, btb, ek);
    IntSquare2Words<bkP, iksP, intP>(cupper, clower, creslvl2, ek);
}

// a*b
template <class bkP, class iksP, class ahP, class intP>
void IntMul(TLWE<typename iksP::domainP> &cupper,
            TLWE<typename iksP::domainP> &clower,
            const TLWE<typename iksP::domainP> &a,
            const TLWE<typename iksP::domainP> &b, const EvalKey &ek)
{
    TLWE<typename bkP::targetP> creslvl2;
    IntMulSquare<bkP, iksP, ahP, intP>(creslvl2, a, b, ek);
    IntSquare2Words<bkP, iksP, intP>(cupper, clower, creslvl2, ek);
}

template <class P, class intP>
Polynomial<P> mulplainlpolygen(uint b)
{
    Polynomial<P> poly;
    constexpr uint32_t segment = 1U << (P::nbit - intP::bit);
    for (int i = 0; i < intP::t; i++)
        for (int j = 0; j < segment; j++)
            poly[i * segment + j] = (i * b % intP::t) * intμ<P, intP>();
    return poly;
}

template <class P, class intP>
Polynomial<P> mulplainhpolygen(uint b)
{
    Polynomial<P> poly;
    constexpr uint32_t segment = 1U << (P::nbit - intP::bit);
    for (int i = 0; i < intP::t; i++)
        for (int j = 0; j < segment; j++)
            poly[i * segment + j] = (i * b / intP::t) * intμ<P, intP>();
    return poly;
}

template <class bkP, class iksP, class intP>
void IntMulPlain(TLWE<typename iksP::domainP> &cupper,
                 TLWE<typename iksP::domainP> &clower,
                 const TLWE<typename iksP::domainP> &a, const uint b,
                 const EvalKey &ek)
{
    assert(b < intP::t);
    TLWE<typename iksP::targetP> shifteda;
    IdentityKeySwitch<iksP>(shifteda, a, ek.getiksk<iksP>());
    shifteda[bkP::domainP::k * bkP::domainP::n] +=
        intμ<typename bkP::domainP, intP>() / 2;
    GateBootstrappingTLWE2TLWEFFT<bkP>(
        clower, shifteda, ek.getbkfft<bkP>(),
        mulplainlpolygen<typename bkP::targetP, intP>(b));
    GateBootstrappingTLWE2TLWEFFT<bkP>(
        cupper, shifteda, ek.getbkfft<bkP>(),
        mulplainhpolygen<typename bkP::targetP, intP>(b));
}

template <class bkP, class iksP, class ahP, class intP>
void IntMulL(TLWE<typename iksP::domainP> &cres,
             const TLWE<typename iksP::domainP> &a,
             const TLWE<typename iksP::domainP> &b, const EvalKey &ek)
{
    // IKS
    TLWE<typename iksP::targetP> shifteda;
    IdentityKeySwitch<iksP>(shifteda, a, ek.getiksk<iksP>());
    shifteda[bkP::domainP::k * bkP::domainP::n] +=
        intμ<typename bkP::domainP, intP>() / 2;

    // 1st stage
    TRLWE<typename bkP::targetP> bmul;
    {
        std::array<TLWE<typename iksP::domainP>, intP::t> extendedbmul;
        extendedbmul[0] = {};
        for (int i = 1; i < intP::t; i++)
            for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
                extendedbmul[i][j] = i * b[j];
        TLWE2TablePacking<ahP, intP::t>(bmul, extendedbmul, ek.getahk<ahP>());
    }

    // 2nd stage
    GateBootstrappingTLWE2TLWEFFTfunc<bkP>(cres, shifteda, ek.getbkfft<bkP>(),
                                           bmul);

    // 3rd stage
    TLWE<typename bkP::targetP> ulsb;
    IdentityKeySwitch<iksP>(shifteda, cres, ek.getiksk<iksP>());
    shifteda[bkP::domainP::k * bkP::domainP::n] +=
        intμ<typename bkP::domainP, intP>() / 2;
    GateBootstrappingTLWE2TLWEFFT<bkP>(
        ulsb, shifteda, ek.getbkfft<bkP>(),
        μpolygen<lvl2param, 1ULL << (std::numeric_limits<
                                         typename bkP::targetP::T>::digits -
                                     2)>());
    ulsb[bkP::targetP::k * bkP::targetP::n] +=
        (1ULL << (std::numeric_limits<typename bkP::targetP::T>::digits - 1)) +
        (1ULL << (std::numeric_limits<typename bkP::targetP::T>::digits - 2));
    for (size_t i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        cres[i] -= ulsb[i];
}

}  // namespace TFHEpp
