#pragma once
#include <cstdint>

#include "keyswitch.hpp"
#include "mulfft.hpp"
#include "trgsw.hpp"

namespace TFHEpp {

// Standard TRLWE multiplication without relinearization
// Uses PolyMulRescaleUnsigned for rescaling by Δ
template <class P>
void LWEMultWithoutRelinerization(TRLWEMult<P> &res, const TRLWE<P> &a,
                                  const TRLWE<P> &b)
{
    alignas(64) PolynomialInFD<P> ffta, fftb, fftc;
    TwistIFFTUInt<P>(ffta, a[0]);
    TwistIFFTUInt<P>(fftb, b[1]);
    MulInFD<P::n>(fftc, ffta, fftb);
    TwistIFFTUInt<P>(ffta, a[1]);
    TwistIFFTUInt<P>(fftb, b[0]);
    FMAInFD<P::n>(fftc, ffta, fftb);
    TwistFFTrescale<P>(res[0], fftc);

    PolyMulRescaleUnsigned<P>(res[1], a[1], b[1]);
    PolyMulRescaleUnsigned<P>(res[2], a[0], b[0]);
}

// Relinearization key switch - automatically handles DD when l̅ > 1
template <class P>
inline void relinKeySwitch(TRLWE<P> &res, const Polynomial<P> &poly,
                           const relinKeyFFT<P> &relinkeyfft)
{
    alignas(64) DecomposedPolynomial<P> decvec;
    Decomposition<P>(decvec, poly);
    alignas(64) PolynomialInFD<P> decvecfft;

    if constexpr (P::l̅ > 1) {
        // Double Decomposition path: l̅ separate accumulators
        alignas(64) std::array<TRLWEInFD<P>, P::l̅> resfft_dd;

        // Initialize all accumulators to zero
        for (int j = 0; j < P::l̅; j++)
            for (int m = 0; m <= P::k; m++)
                for (int n = 0; n < P::n; n++)
                    resfft_dd[j][m][n] = 0.0;

        // Process with standard decomposition (l levels), accumulate into l̅ results
        for (int i = 0; i < P::l; i++) {
            TwistIFFT<P>(decvecfft, decvec[i]);
            // Each decomposition level i multiplies with l̅ relinkey rows
            for (int j = 0; j < P::l̅; j++) {
                const int row_idx = i * P::l̅ + j;
                for (int m = 0; m <= P::k; m++) {
                    FMAInFD<P::n>(resfft_dd[j][m], decvecfft,
                                  relinkeyfft[row_idx][m]);
                }
            }
        }

        // FFT back to coefficient domain for each accumulator and recombine
        std::array<TRLWE<P>, P::l̅> results_dd;
        for (int j = 0; j < P::l̅; j++)
            for (int k = 0; k <= P::k; k++)
                TwistFFT<P>(results_dd[j][k], resfft_dd[j][k]);

        // Recombine the l̅ TRLWEs back to single TRLWE
        RecombineTRLWEFromDD<P, false>(res, results_dd);
    }
    else {
        // Standard path
        TRLWEInFD<P> resfft;
        TwistIFFT<P>(decvecfft, decvec[0]);
        MulInFD<P::n>(resfft[0], decvecfft, relinkeyfft[0][0]);
        MulInFD<P::n>(resfft[1], decvecfft, relinkeyfft[0][1]);
        for (int i = 1; i < P::l; i++) {
            TwistIFFT<P>(decvecfft, decvec[i]);
            FMAInFD<P::n>(resfft[0], decvecfft, relinkeyfft[i][0]);
            FMAInFD<P::n>(resfft[1], decvecfft, relinkeyfft[i][1]);
        }
        TwistFFT<P>(res[0], resfft[0]);
        TwistFFT<P>(res[1], resfft[1]);
    }
}

// Relinearization - automatically handles DD when l̅ > 1
template <class P>
inline void Relinearization(TRLWE<P> &res, const TRLWEMult<P> &mult,
                            const relinKeyFFT<P> &relinkeyfft)
{
    TRLWE<P> squareterm;
    relinKeySwitch<P>(squareterm, mult[2], relinkeyfft);
    for (int i = 0; i < P::n; i++) res[0][i] = mult[0][i] + squareterm[0][i];
    for (int i = 0; i < P::n; i++) res[1][i] = mult[1][i] + squareterm[1][i];
}

// TRLWE multiplication with relinearization - automatically handles DD when l̅ > 1
template <class P>
inline void LWEMult(TRLWE<P> &res, const TRLWE<P> &a, const TRLWE<P> &b,
                    const relinKeyFFT<P> &relinkeyfft)
{
    TRLWEMult<P> resmult;
    LWEMultWithoutRelinerization<P>(resmult, a, b);
    Relinearization<P>(res, resmult, relinkeyfft);
}
}  // namespace TFHEpp