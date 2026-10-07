#pragma once

namespace TFHEpp{
#define INST(bkP, iksP, intP)                                     \
    extern template void IntAdd<bkP, iksP, intP>(TLWE<typename iksP::domainP> &ccarry, \
            TLWE<typename iksP::domainP> &cres,					  \
            const TLWE<typename iksP::domainP> &a,				  \
            const TLWE<typename iksP::domainP> &b,				  \
            const EvalKey &ek)
TFHEPP_EXPLICIT_INSTANTIATION_INTADD(INST)
#undef INST


#define INST(bkP, iksP, intP)                                      \
    extern template void IntSub<bkP, iksP, intP>(TLWE<typename iksP::domainP> &cborrow, \
            TLWE<typename iksP::domainP> &cres,					   \
            const TLWE<typename iksP::domainP> &a,				   \
            const TLWE<typename iksP::domainP> &b,				   \
            const EvalKey &ek)
TFHEPP_EXPLICIT_INSTANTIATION_INTADD(INST)
#undef INST

}
