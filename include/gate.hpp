#pragma once

#include "cloudkey.hpp"
#include "gatebootstrapping.hpp"
#include "keyswitch.hpp"

#ifdef ENABLE_AXELL
#include "axell/gate.hpp"
#endif

namespace TFHEpp {
template <class brP, typename brP::targetP::T μ, class iksP, int casign,
          int cbsign, std::make_signed_t<typename brP::domainP::T> offset>
inline void HomGate(TLWE<typename iksP::targetP> &res,
                    const TLWE<typename brP::domainP> &ca,
                    const TLWE<typename brP::domainP> &cb, const EvalKey &ek)
{
    for (int i = 0; i <= brP::domainP::k * brP::domainP::n; i++)
        res[i] = casign * ca[i] + cbsign * cb[i];
    res[brP::domainP::k * brP::domainP::n] += offset;
    GateBootstrapping<brP, μ, iksP>(res, res, ek);
}

template <class iksP, class brP, typename brP::targetP::T μ, int casign,
          int cbsign, std::make_signed_t<typename iksP::domainP::T> offset>
inline void HomGate(TLWE<typename brP::targetP> &res,
                    const TLWE<typename iksP::domainP> &ca,
                    const TLWE<typename iksP::domainP> &cb, const EvalKey &ek)
{
    for (int i = 0; i <= iksP::domainP::k * iksP::domainP::n; i++)
        res[i] = casign * ca[i] + cbsign * cb[i];
    res[iksP::domainP::k * iksP::domainP::n] += offset;
    GateBootstrapping<iksP, brP, μ>(res, res, ek);
}

template <class P = lvl1param>
void HomCONSTANTONE(TLWE<P> &res)
{
    res = {};
    res[P::k * P::n] = P::μ;
}

template <class P = lvl1param>
void HomCONSTANTZERO(TLWE<P> &res)
{
    res = {};
    res[P::k * P::n] = -P::μ;
}

template <class P = lvl1param>
void HomNOT(TLWE<P> &res, const TLWE<P> &ca)
{
    for (int i = 0; i <= P::k * P::n; i++) res[i] = -ca[i];
}

template <class P = lvl1param>
void HomCOPY(TLWE<P> &res, const TLWE<P> &ca)
{
    for (int i = 0; i <= P::k * P::n; i++) res[i] = ca[i];
}

template <class brP = lvl01param, typename brP::targetP::T μ = lvl1param::μ,
          class iksP = lvl10param>
void HomNAND(TLWE<typename iksP::targetP> &res,
             const TLWE<typename brP::domainP> &ca,
             const TLWE<typename brP::domainP> &cb, const EvalKey &ek)
{
    HomGate<brP, μ, iksP, -1, -1, brP::domainP::μ>(res, ca, cb, ek);
}

template <class iksP = lvl10param, class brP = lvl01param,
          typename brP::targetP::T μ = lvl1param::μ>
void HomNAND(TLWE<typename brP::targetP> &res,
             const TLWE<typename iksP::domainP> &ca,
             const TLWE<typename iksP::domainP> &cb, const EvalKey &ek)
{
    HomGate<iksP, brP, μ, -1, -1, iksP::domainP::μ>(res, ca, cb, ek);
}

template <class brP = lvl01param, typename brP::targetP::T μ = lvl1param::μ,
          class iksP = lvl10param>
void HomNOR(TLWE<typename iksP::targetP> &res,
            const TLWE<typename brP::domainP> &ca,
            const TLWE<typename brP::domainP> &cb, const EvalKey &ek)
{
    HomGate<brP, μ, iksP, -1, -1, -brP::domainP::μ>(res, ca, cb, ek);
}

template <class iksP = lvl10param, class brP = lvl01param,
          typename brP::targetP::T μ = lvl1param::μ>
void HomNOR(TLWE<typename brP::targetP> &res,
            const TLWE<typename iksP::domainP> &ca,
            const TLWE<typename iksP::domainP> &cb, const EvalKey &ek)
{
    HomGate<iksP, brP, μ, -1, -1, -iksP::domainP::μ>(res, ca, cb, ek);
}

template <class brP = lvl01param, typename brP::targetP::T μ = lvl1param::μ,
          class iksP = lvl10param>
void HomXNOR(TLWE<typename iksP::targetP> &res,
             const TLWE<typename brP::domainP> &ca,
             const TLWE<typename brP::domainP> &cb, const EvalKey &ek)
{
    HomGate<brP, μ, iksP, -2, -2, -2 * brP::domainP::μ>(res, ca, cb, ek);
}

template <class iksP = lvl10param, class brP = lvl01param,
          typename brP::targetP::T μ = lvl1param::μ>
void HomXNOR(TLWE<typename brP::targetP> &res,
             const TLWE<typename iksP::domainP> &ca,
             const TLWE<typename iksP::domainP> &cb, const EvalKey &ek)
{
    HomGate<iksP, brP, μ, -2, -2, -2 * iksP::domainP::μ>(res, ca, cb, ek);
}

template <class brP = lvl01param, typename brP::targetP::T μ = lvl1param::μ,
          class iksP = lvl10param>
void HomAND(TLWE<typename iksP::targetP> &res,
            const TLWE<typename brP::domainP> &ca,
            const TLWE<typename brP::domainP> &cb, const EvalKey &ek)
{
    HomGate<brP, μ, iksP, 1, 1, -brP::domainP::μ>(res, ca, cb, ek);
}

template <class iksP = lvl10param, class brP = lvl01param,
          typename brP::targetP::T μ = lvl1param::μ>
void HomAND(TLWE<typename brP::targetP> &res,
            const TLWE<typename iksP::domainP> &ca,
            const TLWE<typename iksP::domainP> &cb, const EvalKey &ek)
{
    HomGate<iksP, brP, μ, 1, 1, -iksP::domainP::μ>(res, ca, cb, ek);
}

template <class brP = lvl01param, typename brP::targetP::T μ = lvl1param::μ,
          class iksP = lvl10param>
void HomOR(TLWE<typename iksP::targetP> &res,
           const TLWE<typename brP::domainP> &ca,
           const TLWE<typename brP::domainP> &cb, const EvalKey &ek)
{
    HomGate<brP, μ, iksP, 1, 1, brP::domainP::μ>(res, ca, cb, ek);
}

template <class iksP = lvl10param, class brP = lvl01param,
          typename brP::targetP::T μ = lvl1param::μ>
void HomOR(TLWE<typename brP::targetP> &res,
           const TLWE<typename iksP::domainP> &ca,
           const TLWE<typename iksP::domainP> &cb, const EvalKey &ek)
{
    HomGate<iksP, brP, μ, 1, 1, iksP::domainP::μ>(res, ca, cb, ek);
}

template <class brP = lvl01param, typename brP::targetP::T μ = lvl1param::μ,
          class iksP = lvl10param>
void HomXOR(TLWE<typename iksP::targetP> &res,
            const TLWE<typename brP::domainP> &ca,
            const TLWE<typename brP::domainP> &cb, const EvalKey &ek)
{
    HomGate<brP, μ, iksP, 2, 2, 2 * brP::domainP::μ>(res, ca, cb, ek);
}

template <class iksP = lvl10param, class brP = lvl01param,
          typename brP::targetP::T μ = lvl1param::μ>
void HomXOR(TLWE<typename brP::targetP> &res,
            const TLWE<typename iksP::domainP> &ca,
            const TLWE<typename iksP::domainP> &cb, const EvalKey &ek)
{
    HomGate<iksP, brP, μ, 2, 2, 2 * iksP::domainP::μ>(res, ca, cb, ek);
}

template <class brP = lvl01param, typename brP::targetP::T μ = lvl1param::μ,
          class iksP = lvl10param>
void HomANDNY(TLWE<typename iksP::targetP> &res,
              const TLWE<typename brP::domainP> &ca,
              const TLWE<typename brP::domainP> &cb, const EvalKey &ek)
{
    HomGate<brP, μ, iksP, -1, 1, -brP::domainP::μ>(res, ca, cb, ek);
}

template <class iksP = lvl10param, class brP = lvl01param,
          typename brP::targetP::T μ = lvl1param::μ>
void HomANDNY(TLWE<typename brP::targetP> &res,
              const TLWE<typename iksP::domainP> &ca,
              const TLWE<typename iksP::domainP> &cb, const EvalKey &ek)
{
    HomGate<iksP, brP, μ, -1, 1, -iksP::domainP::μ>(res, ca, cb, ek);
}

template <class brP = lvl01param, typename brP::targetP::T μ = lvl1param::μ,
          class iksP = lvl10param>
void HomANDYN(TLWE<typename iksP::targetP> &res,
              const TLWE<typename brP::domainP> &ca,
              const TLWE<typename brP::domainP> &cb, const EvalKey &ek)
{
    HomGate<brP, μ, iksP, 1, -1, -brP::domainP::μ>(res, ca, cb, ek);
}

template <class iksP = lvl10param, class brP = lvl01param,
          typename brP::targetP::T μ = lvl1param::μ>
void HomANDYN(TLWE<typename brP::targetP> &res,
              const TLWE<typename iksP::domainP> &ca,
              const TLWE<typename iksP::domainP> &cb, const EvalKey &ek)
{
    HomGate<iksP, brP, μ, 1, -1, -iksP::domainP::μ>(res, ca, cb, ek);
}

template <class brP = lvl01param, typename brP::targetP::T μ = lvl1param::μ,
          class iksP = lvl10param>
void HomORNY(TLWE<typename iksP::targetP> &res,
             const TLWE<typename brP::domainP> &ca,
             const TLWE<typename brP::domainP> &cb, const EvalKey &ek)
{
    HomGate<brP, μ, iksP, -1, 1, brP::domainP::μ>(res, ca, cb, ek);
}

template <class iksP = lvl10param, class brP = lvl01param,
          typename brP::targetP::T μ = lvl1param::μ>
void HomORNY(TLWE<typename brP::targetP> &res,
             const TLWE<typename iksP::domainP> &ca,
             const TLWE<typename iksP::domainP> &cb, const EvalKey &ek)
{
    HomGate<iksP, brP, μ, -1, 1, iksP::domainP::μ>(res, ca, cb, ek);
}

template <class brP = lvl01param, typename brP::targetP::T μ = lvl1param::μ,
          class iksP = lvl10param>
void HomORYN(TLWE<typename iksP::targetP> &res,
             const TLWE<typename brP::domainP> &ca,
             const TLWE<typename brP::domainP> &cb, const EvalKey &ek)
{
    HomGate<brP, μ, iksP, 1, -1, brP::domainP::μ>(res, ca, cb, ek);
}

template <class iksP = lvl10param, class brP = lvl01param,
          typename brP::targetP::T μ = lvl1param::μ>
void HomORYN(TLWE<typename brP::targetP> &res,
             const TLWE<typename iksP::domainP> &ca,
             const TLWE<typename iksP::domainP> &cb, const EvalKey &ek)
{
    HomGate<iksP, brP, μ, 1, -1, iksP::domainP::μ>(res, ca, cb, ek);
}

template <class P = lvl1param>
void HomMUX(TLWE<P> &res, const TLWE<P> &cs, const TLWE<P> &c1,
            const TLWE<P> &c0, const EvalKey &ek)
{
    TLWE<P> temp;
    for (int i = 0; i <= P::k * P::n; i++) temp[i] = cs[i] + c1[i];
    for (int i = 0; i <= P::k * P::n; i++) res[i] = -cs[i] + c0[i];
    temp[P::k * P::n] -= P::μ;
    res[P::k * P::n] -= P::μ;
    if constexpr (std::is_same_v<P, lvl1param>) {
        TLWE<lvl0param> and1, and0;
        IdentityKeySwitch<lvl10param>(and1, temp, ek.getiksk<lvl10param>());
        IdentityKeySwitch<lvl10param>(and0, res, ek.getiksk<lvl10param>());
        GateBootstrappingTLWE2TLWEFFT<lvl01param>(
            temp, and1, ek.getbkfft<lvl01param>(),
            μpolygen<lvl1param, lvl1param::μ>());
        GateBootstrappingTLWE2TLWEFFT<lvl01param>(
            res, and0, ek.getbkfft<lvl01param>(),
            μpolygen<lvl1param, lvl1param::μ>());
        for (int i = 0; i <= P::k * lvl1param::n; i++) res[i] += temp[i];
        res[P::k * P::n] += P::μ;
    }
    else if constexpr (std::is_same_v<P, lvl0param>) {
        TLWE<lvl1param> and1, and0;
        GateBootstrappingTLWE2TLWEFFT<lvl01param>(
            and1, temp, ek.getbkfft<lvl01param>(),
            μpolygen<lvl1param, lvl1param::μ>());
        GateBootstrappingTLWE2TLWEFFT<lvl01param>(
            and0, res, ek.getbkfft<lvl01param>(),
            μpolygen<lvl1param, lvl1param::μ>());
        for (int i = 0; i <= lvl1param::k * lvl1param::n; i++)
            and0[i] += and1[i];
        IdentityKeySwitch<lvl10param>(res, and0, ek.getiksk<lvl10param>());
        res[P::k * P::n] += P::μ;
    }
    else {
        static_assert(false_v<typename P::T>, "Undefined HomMUX!");
    }
}

template <class P = lvl1param>
void HomNMUX(TLWE<P> &res, const TLWE<P> &cs, const TLWE<P> &c1,
             const TLWE<P> &c0, const EvalKey &ek);

template <class bkP>
void HomMUXwoIKSandSE(TRLWE<typename bkP::targetP> &res,
                      const TLWE<typename bkP::domainP> &cs,
                      const TLWE<typename bkP::domainP> &c1,
                      const TLWE<typename bkP::domainP> &c0,
                      const EvalKey &ek);

template <class iksP, class bkP>
void HomMUXwoSE(TRLWE<typename bkP::targetP> &res,
                const TLWE<typename iksP::domainP> &cs,
                const TLWE<typename iksP::domainP> &c1,
                const TLWE<typename iksP::domainP> &c0,
                const EvalKey &ek);

void ExtractSwitchAndHomMUX(TRLWE<lvl1param> &res, const TRLWE<lvl1param> &csr,
                            const TRLWE<lvl1param> &c1r,
                            const TRLWE<lvl1param> &c0r, const EvalKey &ek);
}  // namespace TFHEpp
