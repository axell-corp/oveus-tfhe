#pragma once
#include "long.hpp"

namespace TFHEpp {

// Parallel version of LongInvReLU: parallelizes per-segment GateBootstrapping
template <class bkP, class iksP, class ahP, class intP>
void LongInvReLUPar(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const TLWE<typename iksP::domainP> &csign,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const EvalKey &ek)
{
    assert(cres.size() == a.size());
    TRLWE<typename bkP::targetP> maskedbysign;
    {
        std::array<TLWE<typename iksP::domainP>, intP::t> cisgntlwes;
        cisgntlwes[0] = {};
        for (int i = 1; i < intP::t; i++)
            for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
                cisgntlwes[i][j] = i * csign[j];
        TLWE2TablePacking<ahP, intP::t>(maskedbysign, cisgntlwes,
                                        ek.getahk<ahP>());
    }
#pragma omp parallel for
    for (int seg = 0; seg < static_cast<int>(a.size()); seg++) {
        TLWE<typename iksP::targetP> shifteda;
        IdentityKeySwitch<iksP>(shifteda, a[seg], ek.getiksk<iksP>());
        shifteda[bkP::domainP::k * bkP::domainP::n] +=
            intμ<typename bkP::domainP, intP>() / 2;
        GateBootstrappingTLWE2TLWEFFTfunc<bkP>(
            cres[seg], shifteda, ek.getbkfft<bkP>(), maskedbysign);
    }
}

template <class bkP, class iksP, class ahP, class intP>
void LongInvReLUPar(std::vector<TLWE<typename iksP::domainP>> &cres,
                    const TLWE<typename iksP::domainP> &csign,
                    const std::vector<TLWE<typename iksP::domainP>> &a,
                    const EvalKey &ek)
{
    cres.resize(a.size());
    LongInvReLUPar<bkP, iksP, ahP, intP>(
        std::ranges::subrange(cres.begin(), cres.end()), csign,
        std::ranges::subrange(a.cbegin(), a.cend()), ek);
}

// Parallel version of PreprocessMulTB: parallelizes table generation loop
template <class bkP, class iksP, class ahP, class intP>
void PreprocessMulTBPar(
    std::vector<TRLWE<typename bkP::targetP>> &btb,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        b,
    const EvalKey &ek)
{
    btb.resize(b.size() / 2 + (b.size() & 1));
    const int even_count =
        static_cast<int>(btb.size()) - static_cast<int>(b.size() & 1);
#pragma omp parallel for
    for (int i = 0; i < even_count; i++) {
        IntCombinedMulSquareTBgen<bkP, iksP, ahP, intP>(btb[i], b[2 * i + 1],
                                                        b[2 * i], ek);
    }
    if (b.size() & 1) {
        IntMulSquareTBgen<bkP, iksP, ahP, intP>(btb[btb.size() - 1],
                                                b[b.size() - 1], ek);
    }
}

// Parallel version of LongMul (btb variant): flattens ALL GateBootstrappings
// across all btb entries into a single parallel region, then parallelizes
// carry propagation across independent partial products.
template <class bkP, class iksP, class intP>
void LongMulPar(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const std::ranges::subrange<
        typename std::vector<TRLWE<typename bkP::targetP>>::const_iterator>
        btb,
    const EvalKey &ek, const bool isoddlenb = false)
{
    if (a.size() == 1 && btb.size() == 1 && isoddlenb) {
        IntMul<bkP, iksP, intP>(cres[1], cres[0], a[0], btb[0], ek);
    }
    else {
        const int numseg = static_cast<int>(a.size());
        const int even_btb =
            static_cast<int>(btb.size()) - (isoddlenb ? 1 : 0);

        // Phase 1: Shared IKS — PARALLEL
        std::vector<TLWE<typename iksP::targetP>> shifteda(numseg);
#pragma omp parallel for
        for (int i = 0; i < numseg; i++) {
            IdentityKeySwitch<iksP>(shifteda[i], a[i], ek.getiksk<iksP>());
            shifteda[i][bkP::domainP::k * bkP::domainP::n] +=
                intμ<typename bkP::domainP, intP>() / 2;
        }

        // Phase 2: ALL GateBootstrappings flattened — PARALLEL
        //   even_btb × numseg independent calls
        const int total_gb = even_btb * numseg;
        std::vector<std::array<TLWE<typename iksP::domainP>, 2>> flat_cmul(
            total_gb);
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < total_gb; idx++) {
            const int k = idx / numseg;
            const int i = idx % numseg;
            GateBootstrappingManyLUT<bkP, 2>(
                flat_cmul[idx], shifteda[i], ek.getbkfft<bkP>(), btb[k]);
        }

        // Phase 3: Accumulate cmul + carry propagation per btb entry — PARALLEL
        //   Each partial product is independent.
        std::vector<TLWE<typename iksP::domainP>> all_partmul(
            even_btb * (numseg + 2));
#pragma omp parallel for
        for (int k = 0; k < even_btb; k++) {
            auto *partmul = all_partmul.data() + k * (numseg + 2);
            const auto *cmul_k = &flat_cmul[k * numseg];
            // Accumulate in 2t^2
            for (int i = 0; i < numseg; i++) {
                if (i == 0)
                    partmul[0] = cmul_k[i][0];
                else
                    for (int j = 0;
                         j <= iksP::domainP::k * iksP::domainP::n; j++)
                        partmul[i][j] += cmul_k[i][0][j];
                partmul[i + 1] = cmul_k[i][1];
            }
            // Convert to 2t (carry propagation)
            TLWE<typename iksP::domainP> cupper = partmul[0];
            IntSquare2Words<bkP, iksP, intP>(cupper, partmul[0], cupper, ek);
            for (int i = 1; i < numseg; i++) {
                TLWE<typename iksP::domainP> temp = partmul[i];
                IntSquare2Words<bkP, iksP, intP>(temp, partmul[i], temp, ek);
                IntAddFMA<bkP, iksP, intP>(cupper, partmul[i], cupper,
                                           partmul[i], ek);
                TLWEAdd<typename iksP::domainP>(partmul[i + 1], partmul[i + 1],
                                                cupper);
                IntCarryExtract<bkP, iksP, intP>(temp, cupper, temp, ek);
                TLWEAdd<typename iksP::domainP>(partmul[i + 1], partmul[i + 1],
                                                temp);
            }
            {
                TLWE<typename iksP::targetP> shiftedcupper;
                IdentityKeySwitch<iksP>(shiftedcupper, cupper,
                                        ek.getiksk<iksP>());
                shiftedcupper[bkP::domainP::k * bkP::domainP::n] +=
                    intμ<typename bkP::domainP, intP>() / 2;
                GateBootstrappingTLWE2TLWEFFT<bkP>(
                    cupper, shiftedcupper, ek.getbkfft<bkP>(),
                    mulpolygen<typename bkP::targetP, intP>());
            }
            TLWEAdd<typename iksP::domainP>(partmul[numseg], partmul[numseg],
                                            cupper);
            partmul[numseg + 1] = partmul[numseg];
            IntSquare2Words<bkP, iksP, intP>(partmul[numseg + 1],
                                             partmul[numseg],
                                             partmul[numseg + 1], ek);
        }

        // Phase 4: Merge partial products into cres — sequential
        for (int k = 0; k < even_btb; k++) {
            auto *partmul = all_partmul.data() + k * (numseg + 2);
            const auto partmul_begin = all_partmul.cbegin() + k * (numseg + 2);
            if (k == 0)
                for (int j = 0; j < numseg + 2; j++)
                    cres[j] = partmul[j];
            else {
                TLWE<typename iksP::domainP> ccarry;
                LongAdd<bkP, iksP, intP>(
                    ccarry,
                    std::ranges::subrange(cres.begin() + 2 * k,
                                          cres.begin() + 2 * k + numseg),
                    std::ranges::subrange(cres.begin() + 2 * k,
                                          cres.begin() + 2 * k + numseg),
                    std::ranges::subrange(partmul_begin,
                                          partmul_begin + numseg),
                    ek);
                IntAdd<bkP, iksP, intP>(ccarry, cres[2 * k + numseg],
                                        partmul[numseg], ccarry, ek);
                for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
                    cres[2 * k + numseg + 1][j] =
                        partmul[numseg + 1][j] + ccarry[j];
            }
        }

        // Handle odd-length b
        if (isoddlenb) {
            if (btb.size() == 1) {
                for (size_t j = 0; j < cres.size(); j++) cres[j] = {};
            }
            else {
                for (size_t j = 2 * (btb.size() - 2) + a.size() + 2;
                     j < cres.size(); j++)
                    cres[j] = {};
            }
            PartialLongMul<bkP, iksP, intP>(
                std::ranges::subrange(
                    cres.begin() + (btb.size() - 1) * 2 + 1 - 1,
                    cres.begin() + (btb.size() - 1) * 2 + 1 - 1 + a.size() +
                        1),
                std::ranges::subrange(shifteda.cbegin(), shifteda.cend()),
                btb[btb.size() - 1], ek);
        }
    }
}

// Parallel version of LongMul (raw b variant): calls PreprocessMulTBPar + LongMulPar
template <class bkP, class iksP, class ahP, class intP>
void LongMulPar(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        b,
    const EvalKey &ek)
{
    if (a.size() < b.size()) {
        LongMulPar<bkP, iksP, ahP, intP>(cres, b, a, ek);
        return;
    }
    assert(cres.size() == a.size() + b.size());
    if (cres.size() == 2) {
        IntMul<bkP, iksP, ahP, intP>(cres[1], cres[0], a[0], b[0], ek);
    }
    else {
        std::vector<TRLWE<typename bkP::targetP>> btb(b.size() / 2 +
                                                      (b.size() & 1));
        PreprocessMulTBPar<bkP, iksP, ahP, intP>(btb, b, ek);
        LongMulPar<bkP, iksP, intP>(
            cres, a, std::ranges::subrange(btb.cbegin(), btb.cend()), ek,
            b.size() & 1);
    }
}

template <class bkP, class iksP, class ahP, class intP>
void LongMulPar(std::vector<TLWE<typename iksP::domainP>> &cres,
                const std::vector<TLWE<typename iksP::domainP>> &a,
                const std::vector<TLWE<typename iksP::domainP>> &b,
                const EvalKey &ek)
{
    cres.resize(a.size() + b.size());
    LongMulPar<bkP, iksP, ahP, intP>(
        std::ranges::subrange(cres.begin(), cres.end()),
        std::ranges::subrange(a.cbegin(), a.cend()),
        std::ranges::subrange(b.cbegin(), b.cend()), ek);
}

// Parallel version of LongMulH: calls LongMulPar
template <class bkP, class iksP, class ahP, class intP>
void LongMulHPar(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        b,
    const EvalKey &ek)
{
    const uint numsegment = a.size();
    assert(numsegment == cres.size());
    std::vector<TLWE<typename iksP::domainP>> mulh(2 * numsegment);
    LongMulPar<bkP, iksP, ahP, intP>(
        std::ranges::subrange(mulh.begin(), mulh.end()), a, b, ek);
    for (int i = 0; i < static_cast<int>(numsegment); i++)
        cres[i] = mulh[i + numsegment];
}

template <class bkP, class iksP, class ahP, class intP>
void LongMulHPar(std::vector<TLWE<typename iksP::domainP>> &cres,
                 const std::vector<TLWE<typename iksP::domainP>> &a,
                 const std::vector<TLWE<typename iksP::domainP>> &b,
                 const EvalKey &ek)
{
    const uint numsegment = a.size();
    cres.resize(numsegment);
    std::vector<TLWE<typename iksP::domainP>> mulh(2 * numsegment);
    LongMulPar<bkP, iksP, ahP, intP>(
        std::ranges::subrange(mulh.begin(), mulh.end()),
        std::ranges::subrange(a.cbegin(), a.cend()),
        std::ranges::subrange(b.cbegin(), b.cend()), ek);
    for (int i = 0; i < static_cast<int>(numsegment); i++)
        cres[i] = mulh[i + numsegment];
}

// Parallel version of LongMulL: flattens ALL GateBootstrappings for lower
// product across all word pairs, parallelizes btb generation and carry
// propagation across independent partial products.
template <class bkP, class iksP, class ahP, class intP>
void LongMulLPar(
    std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::iterator>
        cres,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        a,
    const std::ranges::subrange<
        typename std::vector<TLWE<typename iksP::domainP>>::const_iterator>
        b,
    const EvalKey &ek)
{
    const int numseg = static_cast<int>(a.size());
    assert(b.size() == a.size());
    assert(cres.size() == a.size());
    const uint numsegmentbit = std::countr_zero(static_cast<uint>(numseg));
    if (numsegmentbit == 0) {
        IntMulL<bkP, iksP, ahP, intP>(cres[0], a[0], b[0], ek);
    }
    else {
        const int num_pairs = numseg / 2;

        // Phase 1: Shared IKS — PARALLEL
        std::vector<TLWE<typename iksP::targetP>> shifteda(numseg);
#pragma omp parallel for
        for (int i = 0; i < numseg; i++) {
            IdentityKeySwitch<iksP>(shifteda[i], a[i], ek.getiksk<iksP>());
            shifteda[i][bkP::domainP::k * bkP::domainP::n] +=
                intμ<typename bkP::domainP, intP>() / 2;
        }

        // Phase 2: Generate ALL btb tables — PARALLEL
        std::vector<TRLWE<typename bkP::targetP>> all_btb(num_pairs);
#pragma omp parallel for
        for (int k = 0; k < num_pairs; k++) {
            IntCombinedMulSquareTBgen<bkP, iksP, ahP, intP>(
                all_btb[k], b[2 * k + 1], b[2 * k], ek);
        }

        // Phase 3: ALL GateBootstrappings flattened — PARALLEL
        //   Pair k uses (numseg - 2*k) shifteda entries
        //   Build offset table for flat indexing
        std::vector<int> gb_off(num_pairs + 1);
        gb_off[0] = 0;
        for (int k = 0; k < num_pairs; k++)
            gb_off[k + 1] = gb_off[k] + (numseg - 2 * k);
        const int total_gb = gb_off[num_pairs];

        std::vector<std::array<TLWE<typename iksP::domainP>, 2>> flat_cmul(
            total_gb);
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < total_gb; idx++) {
            // find pair k for this flat index
            int k = 0;
            while (k + 1 < num_pairs && idx >= gb_off[k + 1]) k++;
            const int i = idx - gb_off[k];
            GateBootstrappingManyLUT<bkP, 2>(
                flat_cmul[idx], shifteda[i], ek.getbkfft<bkP>(), all_btb[k]);
        }

        // Phase 4: Accumulate cmul + carry propagation per pair — PARALLEL
        std::vector<int> partmul_off(num_pairs + 1);
        partmul_off[0] = 0;
        for (int k = 0; k < num_pairs; k++)
            partmul_off[k + 1] = partmul_off[k] + (numseg - 2 * k);
        std::vector<TLWE<typename iksP::domainP>> all_partmul(
            partmul_off[num_pairs]);
#pragma omp parallel for
        for (int k = 0; k < num_pairs; k++) {
            const int seg = numseg - 2 * k;
            auto *partmul = all_partmul.data() + partmul_off[k];
            const auto *cmul_k = &flat_cmul[gb_off[k]];
            // Accumulate in 2t^2
            for (int i = 0; i < seg; i++) {
                if (i == 0)
                    partmul[0] = cmul_k[i][0];
                else
                    for (int j = 0;
                         j <= iksP::domainP::k * iksP::domainP::n; j++)
                        partmul[i][j] += cmul_k[i][0][j];
                if (i != seg - 1) partmul[i + 1] = cmul_k[i][1];
            }
            // Convert to 2t (carry propagation)
            TLWE<typename iksP::domainP> cupper = partmul[0];
            IntSquare2Words<bkP, iksP, intP>(cupper, partmul[0], cupper, ek);
            for (int i = 1; i < seg; i++) {
                TLWE<typename iksP::domainP> temp = partmul[i];
                IntSquare2Words<bkP, iksP, intP>(temp, partmul[i], temp, ek);
                IntAddFMA<bkP, iksP, intP>(cupper, partmul[i], cupper,
                                           partmul[i], ek);
                if (i != seg - 1)
                    TLWEAdd<typename iksP::domainP>(partmul[i + 1],
                                                    partmul[i + 1], cupper);
                IntCarryExtract<bkP, iksP, intP>(temp, cupper, temp, ek);
                if (i != seg - 1)
                    TLWEAdd<typename iksP::domainP>(partmul[i + 1],
                                                    partmul[i + 1], temp);
            }
        }

        // Phase 5: Merge partial products into cres — sequential
        for (int k = 0; k < num_pairs; k++) {
            const int seg = numseg - 2 * k;
            auto *partmul = all_partmul.data() + partmul_off[k];
            const auto partmul_begin = all_partmul.cbegin() + partmul_off[k];
            if (k == 0)
                for (int j = 0; j < numseg; j++)
                    cres[j] = partmul[j];
            else {
                TLWE<typename iksP::domainP> ccarry;
                LongAdd<bkP, iksP, intP>(
                    ccarry,
                    std::ranges::subrange(cres.begin() + 2 * k,
                                          cres.end() - 1),
                    std::ranges::subrange(cres.begin() + 2 * k,
                                          cres.end() - 1),
                    std::ranges::subrange(partmul_begin,
                                          partmul_begin + seg - 1),
                    ek);
                for (int j = 0; j <= iksP::domainP::k * iksP::domainP::n; j++)
                    cres[numseg - 1][j] +=
                        partmul[seg - 1][j] + ccarry[j];
            }
        }
        if (b.size() & 1) {
            PartialLongMulL<bkP, iksP, ahP, intP>(
                std::ranges::subrange(
                    cres.begin() + b.size() - 1,
                    cres.begin() + b.size() - 1 + a.size() + 1),
                std::ranges::subrange(shifteda.cbegin(), shifteda.cend()),
                b[b.size() - 1], ek);
        }
        else {
            TLWE<typename iksP::domainP> ccarry;
            IntCarryExtract<bkP, iksP, intP>(ccarry, cres[b.size() - 1],
                                             cres[b.size() - 1], ek);
        }
    }
}

template <class bkP, class iksP, class ahP, class intP>
void LongMulLPar(std::vector<TLWE<typename iksP::domainP>> &cres,
                 const std::vector<TLWE<typename iksP::domainP>> &a,
                 const std::vector<TLWE<typename iksP::domainP>> &b,
                 const EvalKey &ek)
{
    const uint numsegment = a.size();
    cres.resize(numsegment);
    LongMulLPar<bkP, iksP, ahP, intP>(
        std::ranges::subrange(cres.begin(), cres.end()),
        std::ranges::subrange(a.cbegin(), a.cend()),
        std::ranges::subrange(b.cbegin(), b.cend()), ek);
}

}  // namespace TFHEpp
