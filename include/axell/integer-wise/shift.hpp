#pragma once
#include <limits>

#include "intkey.hpp"

namespace TFHEpp {
template <class brP, class iksP, class ahP, class intP>
void LeftShiftSquare(TLWE<typename iksP::domainP> &cres,
                     const TLWE<typename iksP::domainP> &ca,
                     const TLWE<typename iksP::domainP> &cb, const EvalKey &ek)
{
    // scalet to switch Mt to Mbase
    TLWE<typename iksP::domainP> bitb;
    for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        bitb[i] = (intP::t / intP::bit) * cb[i];

    // IKS
    TLWE<typename iksP::targetP> shifteda, shiftedb;
    IdentityKeySwitch<iksP>(shifteda, ca, ek.getiksk<iksP>());
    IdentityKeySwitch<iksP>(shiftedb, bitb, ek.getiksk<iksP>());

    // 1st stage
    shifteda[brP::domainP::k * brP::domainP::n] +=
        intμ<typename brP::domainP, intP>() / 2;
    TLWE<typename brP::targetP> extendeda;
    GateBootstrappingTLWE2TLWEFFT<brP>(
        extendeda, shifteda, ek.getbkfft<brP>(),
        mulpolygen<typename brP::targetP, intP>());

    // 2nd stage
    TRLWE<typename brP::targetP> extendedamul;
    std::array<TLWE<typename brP::targetP>, intP::bit> extendedamularray;
    for (int i = 0; i < intP::bit; i++)
        for (int j = 0; j <= brP::targetP::k * brP::targetP::n; j++)
            extendedamularray[i][j] = (1ULL << i) * extendeda[j];
    TLWE2TablePacking<ahP, intP::bit>(extendedamul, extendedamularray,
                                      ek.getahk<ahP>());

    // 3rd stage
    shiftedb[brP::domainP::k * brP::domainP::n] +=
        intμ<typename brP::domainP, intP>() / 2;
    GateBootstrappingTLWE2TLWEFFTfunc<brP>(cres, shiftedb, ek.getbkfft<brP>(),
                                           extendedamul);
}

template <class bkP, class iksP, class ahP, class intP>
void IntLeftShift(TLWE<typename iksP::domainP> &cupper,
                  TLWE<typename iksP::domainP> &clower,
                  const TLWE<typename iksP::domainP> &ca,
                  const TLWE<typename iksP::domainP> &cb, const EvalKey &ek)
{
    TLWE<typename bkP::targetP> creslvl2;
    LeftShiftSquare<bkP, iksP, ahP, intP>(creslvl2, ca, cb, ek);
    IntSquare2Words<bkP, iksP, intP>(cupper, clower, creslvl2, ek);
}

template <class bkP, class iksP, class ahP, class intP>
void IntLeftShift(TLWE<typename iksP::domainP> &cres,
                  const TLWE<typename iksP::domainP> &ca,
                  const TLWE<typename iksP::domainP> &cb, const EvalKey &ek)
{
    TLWE<typename bkP::targetP> creslvl2;
    LeftShiftSquare<bkP, iksP, ahP, intP>(creslvl2, ca, cb, ek);
    // 4th stage
    TLWE<typename iksP::domainP> scaledcreslvl2, ulsblvl2;
    TLWE<typename iksP::targetP> scaledcreslvl1;
    for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        scaledcreslvl2[i] = creslvl2[i] * intP::t;
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
    for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        cres[i] = scaledcreslvl2[i] - ulsblvl2[i];
}

template <class P>
void MUXlshift(
    std::ranges::subrange<typename std::vector<TLWE<P>>::iterator> cres,
    const std::ranges::subrange<typename std::vector<TLWE<P>>::const_iterator>
        cin,
    const std::vector<TRGSWFFT<P>, AlignedAllocator<TRGSWFFT<P>, 64>>
        &shiftamount)
{
    const uint numword = cres.size();
    std::vector<TRLWE<P>> temp(numword);
    for (int i = 0; i < numword; i++)
        InvSampleExtractIndex<P>(temp[i], cin[i], 0);
    constexpr TRLWE<P> zero = {};

    for (int bit = 0; bit < shiftamount.size(); bit++) {
        const int stride = 1 << bit;
        for (int index = numword - 1; index >= 0; index--) {
            alignas(64) TRLWE<P> element;
            CMUXFFT<P>(element, shiftamount[bit],
                       index >= stride ? temp[index - stride] : zero,
                       temp[index]);
            temp[index] = element;
        }
    }

    for (int i = 0; i < numword; i++)
        SampleExtractIndex<P>(cres[i], temp[i], 0);
}

template <class P>
void MUXlshiftH(
    std::ranges::subrange<typename std::vector<TLWE<P>>::iterator> cres,
    const std::ranges::subrange<typename std::vector<TLWE<P>>::const_iterator>
        cin,
    const std::vector<TRGSWFFT<P>, AlignedAllocator<TRGSWFFT<P>, 64>>
        &shiftamount)
{
    const uint numword = cres.size();
    std::vector<TRLWE<P>> temp(numword);
    for (int i = 0; i < numword; i++)
        InvSampleExtractIndex<P>(temp[i], cin[i], 0);
    constexpr TRLWE<P> zero = {};

    for (int bit = 0; bit < shiftamount.size(); bit++) {
        const int stride = 1 << bit;
        for (int index = 0; index < numword; index++) {
            TRLWE<P> element;
            CMUXFFT<P>(element, shiftamount[bit], temp[index],
                       index + stride >= numword ? zero
                                                 : temp[index + stride]);
            temp[index] = element;
        }
    }

    for (int i = 0; i < numword; i++)
        SampleExtractIndex<P>(cres[i], temp[i], 0);
}

template <class brP, class iksP, class cbbrP, class ahP, class intP>
void ULeftShift(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        cin,
    const TLWE<typename iksP::domainP> &shiftu,
    const TLWE<typename iksP::domainP> &shiftl, const EvalKey &ek)
{
    const uint numword = cres.size();
    assert(numword == cin.size());
    assert(numword <= intP::t);
    // assert(std::has_single_bit(numword));
    const uint numdigit =
        // std::numeric_limits<uint>::digits - std::countl_zero(numword) - 1;
        std::bit_width(numword - 1);
    std::vector<TRGSWFFT<typename cbbrP::targetP>,
                AlignedAllocator<TRGSWFFT<typename cbbrP::targetP>, 64>>
        shiftamount(numdigit);

    for (int i = 0; i < numdigit; i++) {
        alignas(64) TLWE<typename brP::targetP> shiftamounti;
        for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
            shiftamounti[j] = -(shiftu[j] << (intP::bit - i));
        shiftamounti[brP::targetP::k * brP::targetP::n] -=
            (intμ<typename brP::targetP, intP>() / 2) << (intP::bit - i);
        AnnihilateCircuitBootstrappingFFT<iksP, cbbrP, ahP>(shiftamount[i],
                                                            shiftamounti, ek);
    }
    MUXlshift<typename cbbrP::targetP>(cres, cin, shiftamount);
    TLWE<typename iksP::domainP> temp;
    IntLeftShift<brP, iksP, ahP, intP>(temp, cres[numword - 1],
                                       cres[numword - 1], shiftl, ek);
    for (int i = numword - 2; i >= 0; i--) {
        IntLeftShift<brP, iksP, ahP, intP>(temp, cres[i], cres[i], shiftl, ek);
        TLWEAdd<typename iksP::domainP>(cres[i + 1], cres[i + 1], temp);
    }
}

template <class brP, class iksP, class cbbrP, class ahP, class intP>
void ULeftShift(std::vector<TLWE<typename iksP::domainP>> &cres,
                const std::vector<TLWE<typename iksP::domainP>> &cin,
                const TLWE<typename iksP::domainP> &shiftu,
                const TLWE<typename iksP::domainP> &shiftl, const EvalKey &ek)
{
    cres.resize(cin.size());
    ULeftShift<brP, iksP, cbbrP, ahP, intP>(
        std::ranges::subrange(cres.begin(), cres.end()),
        std::ranges::subrange(cin.cbegin(), cin.cend()), shiftu, shiftl, ek);
}

template <class brP, class iksP, class cbbrP, class ahP, class intP>
void ULeftShiftH(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        cin,
    const TLWE<typename iksP::domainP> &shiftu,
    const TLWE<typename iksP::domainP> &shiftl, const EvalKey &ek)
{
    const uint numword = cres.size();
    assert(numword == cin.size());
    assert(numword <= intP::t);
    const uint numdigit =
        std::numeric_limits<uint>::digits - std::countl_zero(numword) - 1;
    std::vector<TRGSWFFT<typename cbbrP::targetP>,
                AlignedAllocator<TRGSWFFT<typename cbbrP::targetP>, 64>>
        shiftamount(numdigit);

    for (int i = 0; i < numdigit; i++) {
        alignas(64) TLWE<typename brP::targetP> shiftamounti;
        for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
            shiftamounti[j] = -(shiftu[j] << (intP::bit - i));
        shiftamounti[brP::targetP::k * brP::targetP::n] -=
            (intμ<typename brP::targetP, intP>() / 2) << (intP::bit - i);
        AnnihilateCircuitBootstrappingFFT<iksP, cbbrP, ahP>(shiftamount[i],
                                                            shiftamounti, ek);
    }
    MUXlshiftH<typename cbbrP::targetP>(cres, cin, shiftamount);
    TLWE<typename iksP::domainP> temp;
    for (int i = 0; i < numword; i++) {
        IntLeftShift<brP, iksP, ahP, intP>(cres[i], temp, cres[i], shiftl, ek);
        if (i != 0)
            TLWEAdd<typename iksP::domainP>(cres[i - 1], cres[i - 1], temp);
    };
}


}  // namespace TFHEpp
