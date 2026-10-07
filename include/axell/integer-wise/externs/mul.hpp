#pragma once

namespace TFHEpp{
    #define INST(bkP, iksP, ahP, intP)                                     \
        extern template void IntMulSquare<bkP, iksP, ahP, intP>(TLWE<typename iksP::domainP> &cres, \
                const TLWE<typename iksP::domainP> &a,									   \
                const TLWE<typename iksP::domainP> &b,									   \
                const EvalKey &ek)
    TFHEPP_EXPLICIT_INSTANTIATION_INTMUL_AH(INST)
    #undef INST


    #define INST(bkP, iksP, intP)                                     \
        extern template void IntSquare2Words<bkP, iksP, intP>(TLWE<typename iksP::domainP> &cupper, \
                TLWE<typename iksP::domainP> &clower,										 \
                TLWE<typename iksP::domainP> &clowersquare, 								 \
                const TLWE<typename iksP::domainP> &a,										 \
                const EvalKey &ek)
    TFHEPP_EXPLICIT_INSTANTIATION_INTMUL(INST)
    #undef INST

    #define INST(bkP, iksP, ahP, intP)                                     \
        extern template void IntMul<bkP, iksP, ahP, intP>(TLWE<typename iksP::domainP> &cupper, \
                TLWE<typename iksP::domainP> &clower,										 \
                const TLWE<typename iksP::domainP> &a,										 \
                const TLWE<typename iksP::domainP> &b,										 \
                const EvalKey &ek)
    TFHEPP_EXPLICIT_INSTANTIATION_INTMUL_AH(INST)
    #undef INST

}
