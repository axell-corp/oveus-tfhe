#pragma once

#include <cstdint>

#include "cloudkey.hpp"
#include "gatebootstrapping.hpp"
#include "keyswitch.hpp"

namespace TFHEpp {

template <class iksP, class bkP, class privksP>
void CircuitBootstrappingPartial(TRLWE<typename privksP::targetP> &trgswupper,
                                 TRLWE<typename privksP::targetP> &trgswlower,
                                 const TLWE<typename iksP::domainP> &tlwe,
                                 const EvalKey &ek, const uint32_t digit);

template <class iksP, class bkP, class privksP>
void CircuitBootstrapping(TRGSW<typename privksP::targetP> &trgsw,
                          const TLWE<typename iksP::domainP> &tlwe,
                          const EvalKey &ek)
{
    TLWE<typename bkP::domainP> tlwelvl0;
    IdentityKeySwitch<iksP>(tlwelvl0, tlwe, ek.getiksk<iksP>());
    CircuitBootstrapping<bkP, privksP>(trgsw, tlwelvl0, ek);
}

template <class iksP, class brP, class ahP>
void AnnihilateCircuitBootstrapping(TRGSW<typename brP::targetP> &trgsw,
                                    const TLWE<typename iksP::domainP> &tlwe,
                                    const EvalKey &ek)
{
    TLWE<typename brP::domainP> tlwelvl0;
    IdentityKeySwitch<iksP>(tlwelvl0, tlwe, ek.getiksk<iksP>());
    AnnihilateCircuitBootstrapping<brP, ahP>(trgsw, tlwelvl0, ek);
}

template <class brP, class privksP>
void CircuitBootstrapping(TRGSWFFT<typename privksP::targetP> &trgswfft,
                          const TLWE<typename brP::domainP> &tlwe,
                          const EvalKey &ek)
{
    alignas(64) TRGSW<typename privksP::targetP> trgsw;
    CircuitBootstrapping<brP, privksP>(trgsw, tlwe, ek);
    ApplyFFT2trgsw<typename privksP::targetP>(trgswfft, trgsw);
}

template <class iksP, class bkP, class privksP>
void CircuitBootstrapping(TRGSWFFT<typename privksP::targetP> &trgswfft,
                          const TLWE<typename iksP::domainP> &tlwe,
                          const EvalKey &ek)
{
    alignas(64) TRGSW<typename privksP::targetP> trgsw;
    CircuitBootstrapping<iksP, bkP, privksP>(trgsw, tlwe, ek);
    ApplyFFT2trgsw<typename privksP::targetP>(trgswfft, trgsw);
}

template <class brP, class ahP>
void AnnihilateCircuitBootstrapping(
    TRGSWFFT<typename brP::targetP> &trgswfft,
    const TLWE<typename brP::domainP> &tlwe, const EvalKey &ek)
{
    alignas(64) TRGSW<typename brP::targetP> trgsw;
    AnnihilateCircuitBootstrapping<brP, ahP>(trgsw, tlwe, ek);
    ApplyFFT2trgsw<typename brP::targetP>(trgswfft, trgsw);
}

template <class iksP, class brP, class ahP>
void AnnihilateCircuitBootstrapping(
    TRGSWFFT<typename brP::targetP> &trgswfft,
    const TLWE<typename iksP::domainP> &tlwe, const EvalKey &ek)
{
    alignas(64) TRGSW<typename brP::targetP> trgsw;
    AnnihilateCircuitBootstrapping<iksP, brP, ahP>(trgsw, tlwe, ek);
    ApplyFFT2trgsw<typename brP::targetP>(trgswfft, trgsw);
}

template <class iksP, class bkP, class privksP>
void CircuitBootstrappingSub(TRGSW<typename privksP::targetP> &trgsw,
                             const TLWE<typename iksP::domainP> &tlwe,
                             const EvalKey &ek)
{
    alignas(64) TLWE<typename bkP::domainP> tlwelvl0;
    IdentityKeySwitch<iksP>(tlwelvl0, tlwe, ek.getiksk<iksP>());
    alignas(64) std::array<TLWE<typename bkP::targetP>, privksP::targetP::l>
        temp;
    GateBootstrappingManyLUT<bkP, privksP::targetP::l>(
        temp, tlwelvl0, ek.getbkfft<bkP>(), CBtestvector<privksP>());
    for (int i = 0; i < privksP::targetP::l; i++) {
        temp[i][privksP::domainP::k * privksP::domainP::n] +=
            1ULL << (numeric_limits<typename privksP::domainP::T>::digits -
                     (i + 1) * privksP::targetP::Bgbit - 1);
        for (int k = 0; k < privksP::targetP::k + 1; k++) {
            alignas(64) TLWE<typename privksP::targetP> subsettlwe;
            SubsetIdentityKeySwitch<privksP>(subsettlwe, temp[i],
                                             ek.getsubiksk<privksP>());
            SubsetPrivKeySwitch<privksP>(
                trgsw[i + k * privksP::targetP::l], subsettlwe,
                ek.getsubprivksk<privksP>("subprivksk4cb_" +
                                          std::to_string(k)));
        }
    }
}

template <class iksP, class bkP, class privksP>
void CircuitBootstrappingFFTInv(
    TRGSWFFT<typename privksP::targetP> &invtrgswfft,
    const TLWE<typename iksP::domainP> &tlwe, const EvalKey &ek);

template <class iksP, class bkP, class privksP>
void CircuitBootstrappingFFTwithInvPartial(
    TRLWEInFD<typename privksP::targetP> &trgswfftupper,
    TRLWEInFD<typename privksP::targetP> &trgswfftlower,
    TRLWEInFD<typename privksP::targetP> &invtrgswfftupper,
    TRLWEInFD<typename privksP::targetP> &invtrgswfftlower,
    const TLWE<typename iksP::domainP> &tlwe, const EvalKey &ck,
    const uint32_t digit);

template <class iksP, class bkP, class privksP>
void CircuitBootstrappingFFTwithInv(
    TRGSWFFT<typename privksP::targetP> &trgswfft,
    TRGSWFFT<typename privksP::targetP> &invtrgswfft,
    const TLWE<typename iksP::domainP> &tlwe, const EvalKey &ek);

}  // namespace TFHEpp