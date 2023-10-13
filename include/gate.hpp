#pragma once

#include "axell/gate.hpp"
#include "axell/mpparam.hpp"
#include "cloudkey.hpp"

namespace TFHEpp {
template <class P, int casign, int cbsign, uint64_t offset>
inline void HomGate(TLWE<P> &res, const TLWE<P> &ca, const TLWE<P> &cb,
                    const EvalKey &ek)
{
    for (int i = 0; i <= P::k * P::n; i++)
        res[i] = casign * ca[i] + cbsign * cb[i];
    res[P::k * P::n] += offset;
    if constexpr (std::is_same_v<P, lvlMparam>) {
        GateBootstrapping<lvlM0param,lvl0Mparam,lvlMparam::μ>(res, res, ek);
    }else{
        GateBootstrapping(res, res, ek);
    }
}
template <class P = lvl1param>
void HomCONSTANTONE(TLWE<P> &res);
template <class P = lvl1param>
void HomCONSTANTZERO(TLWE<P> &res);
template <class P = lvl1param>
void HomNOT(TLWE<P> &res, const TLWE<P> &ca);
template <class P = lvl1param>
void HomCOPY(TLWE<P> &res, const TLWE<P> &ca);
template <class P = lvl1param>
void HomNAND(TLWE<P> &res, const TLWE<P> &ca, const TLWE<P> &cb,
             const EvalKey &ek);
template <class P = lvl1param>
void HomNOR(TLWE<P> &res, const TLWE<P> &ca, const TLWE<P> &cb,
            const EvalKey &ek);
template <class P = lvl1param>
void HomXNOR(TLWE<P> &res, const TLWE<P> &ca, const TLWE<P> &cb,
             const EvalKey &ek);
template <class P = lvl1param>
void HomAND(TLWE<P> &res, const TLWE<P> &ca, const TLWE<P> &cb,
            const EvalKey &ek);
template <class P = lvl1param>
void HomOR(TLWE<P> &res, const TLWE<P> &ca, const TLWE<P> &cb,
           const EvalKey &ek);
template <class P = lvl1param>
void HomXOR(TLWE<P> &res, const TLWE<P> &ca, const TLWE<P> &cb,
            const EvalKey &ek);
template <class P = lvl1param>
void HomANDNY(TLWE<P> &res, const TLWE<P> &ca, const TLWE<P> &cb,
              const EvalKey &ek);
template <class P = lvl1param>
void HomANDYN(TLWE<P> &res, const TLWE<P> &ca, const TLWE<P> &cb,
              const EvalKey &ek);
template <class P = lvl1param>
void HomORNY(TLWE<P> &res, const TLWE<P> &ca, const TLWE<P> &cb,
             const EvalKey &ek);
template <class P = lvl1param>
void HomORYN(TLWE<P> &res, const TLWE<P> &ca, const TLWE<P> &cb,
             const EvalKey &ek);
template <class P = lvl1param>
void HomMUX(TLWE<P> &res, const TLWE<P> &cs, const TLWE<P> &c1,
            const TLWE<P> &c0, const EvalKey &ek)
{
    TLWE<P> temp;
    for (int i = 0; i <= P::k * P::n; i++) temp[i] = cs[i] + c1[i];
    for (int i = 0; i <= P::k * P::n; i++) res[i] = -cs[i] + c0[i];
    temp[P::k * P::n] -= P::μ;
    res[P::k * P::n] -= P::μ;
    if constexpr (std::is_same_v<P, lvl1param> || std::is_same_v<P, lvlMparam>) {
        TLWE<lvl0param> and1, and0;
        IdentityKeySwitch<lvl10param>(and1, temp, *ek.iksklvl10);
        IdentityKeySwitch<lvl10param>(and0, res, *ek.iksklvl10);
        GateBootstrappingTLWE2TLWEFFT<lvl01param>(
            temp, and1, *ek.bkfftlvl01, μpolygen<lvl1param, P::μ>());
        GateBootstrappingTLWE2TLWEFFT<lvl01param>(
            res, and0, *ek.bkfftlvl01, μpolygen<lvl1param, P::μ>());
        for (int i = 0; i <= P::k * P::n; i++) res[i] += temp[i];
        res[P::k * P::n] += P::μ;
    }
    else if constexpr (std::is_same_v<P, lvl0param>) {
        TLWE<lvl1param> and1, and0;
        GateBootstrappingTLWE2TLWEFFT<lvl01param>(
            and1, temp, *ek.bkfftlvl01, μpolygen<lvl1param, lvl1param::μ>());
        GateBootstrappingTLWE2TLWEFFT<lvl01param>(
            and0, res, *ek.bkfftlvl01, μpolygen<lvl1param, lvl1param::μ>());
        for (int i = 0; i <= lvl1param::k * lvl1param::n; i++)
            and0[i] += and1[i];
        IdentityKeySwitch<lvl10param>(res, and0, *ek.iksklvl10);
        res[P::k * P::n] += P::μ;
    }else{
        static_assert(false_v<typename P::T>, "Undefined HomMUX!");
    }
}
template <class P = lvl1param>
void HomNMUX(TLWE<P> &res, const TLWE<P> &cs, const TLWE<P> &c1,
             const TLWE<P> &c0, const EvalKey &ek);
template <class iksP, class bkP>
void HomMUXwoIKSandSE(TRLWE<typename bkP::targetP> &res,
                      const TLWE<typename bkP::domainP> &cs,
                      const TLWE<typename bkP::domainP> &c1,
                      const TLWE<typename bkP::domainP> &c0, const EvalKey &ek);
template <class iksP, class bkP>
void HomMUXwoSE(TRLWE<typename bkP::targetP> &res,
                const TLWE<typename iksP::domainP> &cs,
                const TLWE<typename iksP::domainP> &c1,
                const TLWE<typename iksP::domainP> &c0, const EvalKey &ek);
void ExtractSwitchAndHomMUX(TRLWE<lvl1param> &res, const TRLWE<lvl1param> &csr,
                            const TRLWE<lvl1param> &c1r,
                            const TRLWE<lvl1param> &c0r, const EvalKey &ek);
}  // namespace TFHEpp
