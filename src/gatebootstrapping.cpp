#include <gatebootstrapping.hpp>

namespace TFHEpp {

#define INST(P)                                     \
    template void GateBootstrappingTLWE2TLWEFFT<P>( \
        TLWE<typename P::targetP> & res,            \
        const TLWE<typename P::domainP> &tlwe,      \
        const BootstrappingKeyFFT<P> &bkfft,        \
        const Polynomial<typename P::targetP> &testvector)
TFHEPP_EXPLICIT_INSTANTIATION_BLIND_ROTATE(INST)
#undef INST

template <class P>
void GateBootstrappingTLWE2TLWEFFTfunc(
    TLWE<typename P::targetP> &res, const TLWE<typename P::domainP> &tlwe,
    const BootstrappingKeyFFT<P> &bkfft,
    const TRLWE<typename P::targetP> &testvector)
{
    TRLWE<typename P::targetP> acc;
    BlindRotate<P>(acc, tlwe, bkfft, testvector);
    SampleExtractIndex<typename P::targetP>(res, acc, 0);
}
#define INST(P)                                         \
    template void GateBootstrappingTLWE2TLWEFFTfunc<P>( \
        TLWE<typename P::targetP> & res,                \
        const TLWE<typename P::domainP> &tlwe,          \
        const BootstrappingKeyFFT<P> &bkfft,            \
        const TRLWE<typename P::targetP> &testvector)
TFHEPP_EXPLICIT_INSTANTIATION_BLIND_ROTATE(INST);
#undef INST

}  // namespace TFHEpp
