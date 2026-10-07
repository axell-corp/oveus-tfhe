#pragma once
#include "shift.hpp"

namespace TFHEpp {

// Parallel ULeftShift: parallel CB loop (stage 1) + parallel IntLeftShift (stage 2)
template <class brP, class iksP, class cbbrP, class ahP, class intP>
void ULeftShiftPar(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        cin,
    const TLWE<typename iksP::domainP> &shiftu,
    const TLWE<typename iksP::domainP> &shiftl, const EvalKey &ek)
{
    const int numword = static_cast<int>(cres.size());
    assert(numword == static_cast<int>(cin.size()));
    assert(numword <= static_cast<int>(intP::t));
    const int numdigit = static_cast<int>(std::bit_width(
        static_cast<uint>(numword) - 1));
    std::vector<TRGSWFFT<typename cbbrP::targetP>,
                AlignedAllocator<TRGSWFFT<typename cbbrP::targetP>, 64>>
        shiftamount(numdigit);

    // Stage 1: parallel CB
#pragma omp parallel for
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

    // Stage 2: parallel IntLeftShift — each word independent
    std::vector<TLWE<typename iksP::domainP>> overflows(numword);
#pragma omp parallel for
    for (int i = 0; i < numword; i++) {
        IntLeftShift<brP, iksP, ahP, intP>(overflows[i], cres[i], cres[i],
                                           shiftl, ek);
    }
    // Sequential overflow accumulation (cheap TLWEAdd, no bootstrapping)
    for (int i = 0; i < numword - 1; i++) {
        TLWEAdd<typename iksP::domainP>(cres[i + 1], cres[i + 1], overflows[i]);
    }
}

template <class brP, class iksP, class cbbrP, class ahP, class intP>
void ULeftShiftPar(std::vector<TLWE<typename iksP::domainP>> &cres,
                   const std::vector<TLWE<typename iksP::domainP>> &cin,
                   const TLWE<typename iksP::domainP> &shiftu,
                   const TLWE<typename iksP::domainP> &shiftl,
                   const EvalKey &ek)
{
    cres.resize(cin.size());
    ULeftShiftPar<brP, iksP, cbbrP, ahP, intP>(
        std::ranges::subrange(cres.begin(), cres.end()),
        std::ranges::subrange(cin.cbegin(), cin.cend()), shiftu, shiftl, ek);
}

// Parallel ULeftShiftH: parallel CB + parallel IntLeftShift
template <class brP, class iksP, class cbbrP, class ahP, class intP>
void ULeftShiftHPar(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        cin,
    const TLWE<typename iksP::domainP> &shiftu,
    const TLWE<typename iksP::domainP> &shiftl, const EvalKey &ek)
{
    const int numword = static_cast<int>(cres.size());
    assert(numword == static_cast<int>(cin.size()));
    assert(numword <= static_cast<int>(intP::t));
    const int numdigit = static_cast<int>(
        std::numeric_limits<uint>::digits - std::countl_zero(static_cast<uint>(numword)) - 1);
    std::vector<TRGSWFFT<typename cbbrP::targetP>,
                AlignedAllocator<TRGSWFFT<typename cbbrP::targetP>, 64>>
        shiftamount(numdigit);

#pragma omp parallel for
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

    // Stage 2: parallel IntLeftShift — each word independent
    // IntLeftShift(upper, lower, ca, cb): upper=cres[i], lower=temp
    // In the H variant, upper goes into cres[i], lower overflow goes to cres[i-1]
    std::vector<TLWE<typename iksP::domainP>> uppers(numword), lowers(numword);
#pragma omp parallel for
    for (int i = 0; i < numword; i++) {
        IntLeftShift<brP, iksP, ahP, intP>(uppers[i], lowers[i], cres[i],
                                           shiftl, ek);
    }
    for (int i = 0; i < numword; i++) cres[i] = uppers[i];
    for (int i = 1; i < numword; i++) {
        TLWEAdd<typename iksP::domainP>(cres[i - 1], cres[i - 1], lowers[i]);
    }
}

template <class brP, class iksP, class cbbrP, class ahP, class intP>
void ULeftShiftHPar(std::vector<TLWE<typename iksP::domainP>> &cres,
                    const std::vector<TLWE<typename iksP::domainP>> &cin,
                    const TLWE<typename iksP::domainP> &shiftu,
                    const TLWE<typename iksP::domainP> &shiftl,
                    const EvalKey &ek)
{
    cres.resize(cin.size());
    ULeftShiftHPar<brP, iksP, cbbrP, ahP, intP>(
        std::ranges::subrange(cres.begin(), cres.end()),
        std::ranges::subrange(cin.cbegin(), cin.cend()), shiftu, shiftl, ek);
}


}  // namespace TFHEpp
