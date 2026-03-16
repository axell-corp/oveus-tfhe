#pragma once

#include "trgsw.hpp"

namespace TFHEpp {
template <class P>
void CMUXFFT(TRLWE<P> &res, const TRGSWFFT<P> &cs, const TRLWE<P> &c1,
             const TRLWE<P> &c0)
{
    for (int k = 0; k < P::k + 1; k++)
        for (int i = 0; i < P::n; i++) res[k][i] = c1[k][i] - c0[k][i];
    ExternalProduct<P>(res, res, cs);
    for (int k = 0; k < P::k + 1; k++)
        for (int i = 0; i < P::n; i++) res[k][i] += c0[k][i];
}

template <class P>
TRGSWFFT<P> TRGSWFFTOneGen()
{
    TRGSW<P> trgsw = {};
    trgswhoneadd<P>(trgsw);
    return ApplyFFT2trgsw<P>(trgsw);
}

alignas(64) const TRGSWFFT<lvl1param> trgswonelvl1 =
    TRGSWFFTOneGen<lvl1param>();
alignas(64) const TRGSWFFT<lvl2param> trgswonelvl2 =
    TRGSWFFTOneGen<lvl2param>();

// Fused CMUX for l̅==1 path: rotation + ExternalProduct + accumulation
// in one pass, avoiding full TRLWE temp allocation and separate add-back loop.
template <class P>
void CMUXFFTwithPolynomialMulByXaiMinusOne(
    TRLWE<P> &acc, const TRGSWFFT<P> &trgswfft, const typename P::T a,
    const TRGSWFFT<P> *next_trgswfft = nullptr)
{
    alignas(64) TRLWEInFD<P> restrlwefft;
    alignas(64) Polynomial<P> rotated;
    alignas(64) PolynomialInFD<P> decpolyfft;

    // Prefetch the next bootstrapping key element into L2/L3 cache.
    // Each TRGSWFFT element is large (~96KB for default params). Prefetch
    // the first portion; the hardware sequential prefetcher handles the rest.
    if (next_trgswfft) {
        const char *ptr = reinterpret_cast<const char *>(next_trgswfft);
        constexpr size_t prefetch_bytes = 32768;  // 32KB into L2
        constexpr size_t limit =
            sizeof(TRGSWFFT<P>) < prefetch_bytes ? sizeof(TRGSWFFT<P>) : prefetch_bytes;
        for (size_t offset = 0; offset < limit; offset += 64)
            __builtin_prefetch(ptr + offset, 0, 2);
    }

    // --- Nonce part: k_idx=0, initialize with MulInFD ---
    PolynomialMulByXaiMinusOne<P>(rotated, acc[0], a);
    {
        alignas(64) DecomposedNoncePolynomial<P> decpoly;
        NonceDecomposition<P>(decpoly, rotated);
        TwistIFFT<P>(decpolyfft, decpoly[0]);
        for (int m = 0; m < P::k + 1; m++)
            MulInFD<P::n>(restrlwefft[m], decpolyfft, trgswfft[0][m]);
        for (int i = 1; i < P::lₐ; i++) {
            TwistIFFT<P>(decpolyfft, decpoly[i]);
            for (int m = 0; m < P::k + 1; m++)
                FMAInFD<P::n>(restrlwefft[m], decpolyfft, trgswfft[i][m]);
        }
    }

    // --- Nonce part: k_idx=1..k-1, FMAInFD ---
    for (int k_idx = 1; k_idx < P::k; k_idx++) {
        PolynomialMulByXaiMinusOne<P>(rotated, acc[k_idx], a);
        alignas(64) DecomposedNoncePolynomial<P> decpoly;
        NonceDecomposition<P>(decpoly, rotated);
        for (int i = 0; i < P::lₐ; i++) {
            TwistIFFT<P>(decpolyfft, decpoly[i]);
            for (int m = 0; m < P::k + 1; m++)
                FMAInFD<P::n>(restrlwefft[m], decpolyfft,
                              trgswfft[i + k_idx * P::lₐ][m]);
        }
    }

    // --- Main part: k_idx=k ---
    PolynomialMulByXaiMinusOne<P>(rotated, acc[P::k], a);
    {
        alignas(64) DecomposedPolynomial<P> decpoly;
        Decomposition<P>(decpoly, rotated);
        for (int i = 0; i < P::l; i++) {
            TwistIFFT<P>(decpolyfft, decpoly[i]);
            for (int m = 0; m < P::k + 1; m++)
                FMAInFD<P::n>(restrlwefft[m], decpolyfft,
                              trgswfft[i + P::k * P::lₐ][m]);
        }
    }

    // --- Fused IFFT + accumulation ---
    for (int k = 0; k < P::k + 1; k++)
        TwistFFTAdd<P>(acc[k], restrlwefft[k]);
}

template <class bkP>
void CMUXwithPolynomialMulByXaiMinusOne(
    TRLWE<typename bkP::targetP> &acc,
    const BootstrappingKeyElementFFT<bkP> &cs, const int a,
    const TRGSWFFT<typename bkP::targetP> *next_trgswfft = nullptr)
{
    if constexpr (bkP::domainP::key_value_diff == 1) {
        CMUXFFTwithPolynomialMulByXaiMinusOne<typename bkP::targetP>(
            acc, cs[0], a, next_trgswfft);
    }
    else {
#ifdef USE_TERNARY_CMUX
        alignas(32) TRGSWFFT<typename bkP::targetP> trgsw;
        if constexpr (std::is_same_v<typename bkP::targetP, lvl1param>) {
            trgsw = trgswonelvl1;
        }
        else if constexpr (std::is_same_v<typename bkP::targetP, lvl2param>) {
            trgsw = trgswonelvl2;
        }
        int count = 0;
        alignas(32) PolynomialInFD<typename bkP::targetP> poly;
        for (int i = bkP::domainP::key_value_min;
             i <= bkP::domainP::key_value_max; i++) {
            if (i != 0) {
                for (int j = 0; j < (bkP::targetP::k + 1) * bkP::targetP::l;
                     j++) {
                    for (int k = 0; k < bkP::targetP::k + 1; k++) {
                        PolynomialMulByXaiMinusOneInFD<typename bkP::targetP>(
                            poly, cs[count][j][k], a * i);
                        for (int l = 0; l < bkP::targetP::n; l++)
                            trgsw[j][k][l] += poly[l];
                    }
                }
                count++;
            }
        }
        ExternalProduct<typename bkP::targetP>(acc, acc, trgsw);
#else
        alignas(32) TRLWE<typename bkP::targetP> temp;
        int count = 0;
        for (int i = bkP::domainP::key_value_min;
             i <= bkP::domainP::key_value_max; i++) {
            if (i != 0) {
                const int mod = (a * i) % (2 * bkP::targetP::n);
                const int index = mod > 0 ? mod : mod + (2 * bkP::targetP::n);
                for (int k = 0; k < bkP::targetP::k + 1; k++)
                    PolynomialMulByXaiMinusOne<typename bkP::targetP>(
                        temp[k], acc[k], index);
                ExternalProduct<typename bkP::targetP>(temp, temp,
                                                               cs[count]);
                for (int k = 0; k < bkP::targetP::k + 1; k++)
                    for (int i = 0; i < bkP::targetP::n; i++)
                        acc[k][i] += temp[k][i];
                count++;
            }
        }
#endif
    }
}

template <class P>
void CMUXwithPolynomialMulByXaiMinusOne(TRLWE<P> &acc, const TRGSWNTT<P> &cs,
                                        const typename P::T a)
{
    TRLWE<P> temp;
    for (int k = 0; k < P::k + 1; k++)
        PolynomialMulByXaiMinusOne<P>(temp[k], acc[k], a);
    ExternalProduct<P>(temp, temp, cs);
    for (int k = 0; k < P::k + 1; k++)
        for (int i = 0; i < P::n; i++) acc[k][i] += temp[k][i];
}

template <class P>
void CMUXwithPolynomialMulByXaiMinusOne(TRLWE<P> &acc,
                                        const TRGSWRAINTT<P> &cs,
                                        const typename P::T a)
{
    TRLWE<P> temp;
    for (int k = 0; k < P::k + 1; k++)
        PolynomialMulByXaiMinusOne<P>(temp[k], acc[k], a);
    ExternalProduct<P>(temp, temp, cs);
    for (int k = 0; k < P::k + 1; k++)
        for (int i = 0; i < P::n; i++) acc[k][i] += temp[k][i];
}

}  // namespace TFHEpp