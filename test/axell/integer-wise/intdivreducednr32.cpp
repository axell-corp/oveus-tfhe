#include <bits/stdint-uintn.h>

#include <cassert>
#include <chrono>
#include <limits>
#include <tfhe++.hpp>
#include <axell/integerops.hpp>

#ifdef USE_PERF
#include <gperftools/profiler.h>
#endif

int main()
{
    using bkP = TFHEpp::lvlh2param;
    using iksP = TFHEpp::lvl2hparam;
    using cbbrP = TFHEpp::cblvlh2param;
    using ahP = TFHEpp::AHlvl2param;
    constexpr uint numsegmentbit = 3;
    constexpr size_t numsegment = 1 << numsegmentbit;
    constexpr uint32_t num_test = 10;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::uniform_int_distribution<uint32_t> intdist(0,
                                                    TFHEpp::intparam::intmax);

    const TFHEpp::SecretKey *sk = new TFHEpp::SecretKey();

    std::vector<std::array<uint32_t, numsegment>> pa(num_test);
    std::vector<std::array<uint32_t, numsegment>> pb(num_test);
    for (std::array<uint32_t, numsegment> &pai : pa)
        for (uint32_t &i : pai) i = intdist(engine);
    for (std::array<uint32_t, numsegment> &pbi : pb) {
        bool allzero = true;
        for (uint32_t &i : pbi) {
            i = intdist(engine);
            allzero &= (i == 0);
        }
        if (allzero) pbi[0] = 1;
    }

    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>> ca(num_test);
    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>> cb(num_test);
    for (int i = 0; i < num_test; i++) ca[i].resize(numsegment);
    for (int i = 0; i < num_test; i++) cb[i].resize(numsegment);
    for (int i = 0; i < num_test; i++)
        for (int j = 0; j < numsegment; j++)
            ca[i][j] = TFHEpp::tlweSymEncrypt<typename iksP::domainP>(
                pa[i][j] *
                    TFHEpp::intμ<typename iksP::domainP, TFHEpp::intparam>(),
                iksP::domainP::α, sk->key.get<typename iksP::domainP>());
    for (int i = 0; i < num_test; i++)
        for (int j = 0; j < numsegment; j++)
            cb[i][j] = TFHEpp::tlweSymEncrypt<typename iksP::domainP>(
                pb[i][j] *
                    TFHEpp::intμ<typename iksP::domainP, TFHEpp::intparam>(),
                iksP::domainP::α, sk->key.get<typename iksP::domainP>());

    TFHEpp::EvalKey ek;
    ek.emplacebkfft<bkP>(*sk);
    ek.emplacebkfft<cbbrP>(*sk);
    ek.emplaceiksk<iksP>(*sk);
    ek.emplaceahk<ahP>(*sk);
    ek.emplacecbsk<ahP>(*sk);

    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>> cq(num_test);
    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>> cr(num_test);
    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>> cz(num_test);
    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>> cseed(num_test);
    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>> ctop(num_test);
    std::vector<std::array<TFHEpp::TLWE<typename iksP::domainP>, TFHEpp::intparam::bit>> ctopbits(num_test);
    for (int i = 0; i < num_test; i++) cq[i].resize(numsegment);
    for (int i = 0; i < num_test; i++) cr[i].resize(numsegment);
    std::vector<double> inv_times(num_test), div_times(num_test);

#ifdef USE_PERF
    ProfilerStart("reducednr32.prof");
#endif
    std::cout << "Inv" << std::endl;
    for (int test = 0; test < num_test; test++) {
        std::cout << test << std::endl;
        auto t0 = std::chrono::system_clock::now();
        TFHEpp::TLWE<typename iksP::domainP> clzu, clzl;
        std::vector<TFHEpp::TLWE<typename iksP::domainP>> cyy;
        TFHEpp::HomIntCLZ<bkP, iksP, TFHEpp::intparam>(clzu, clzl, cb[test], ek);
        TFHEpp::ULeftShift<bkP, iksP, cbbrP, ahP, TFHEpp::intparam>(
            cyy, cb[test], clzu, clzl, ek);
        ctop[test] = {cyy[6], cyy[7]};
        TFHEpp::BitDecomp<iksP, bkP, TFHEpp::intparam>(ctopbits[test], cyy[7],
                                                       ek);
        TFHEpp::HomIntInvReducedNRSeed32<bkP, iksP, cbbrP, ahP,
                                         TFHEpp::intparam>(cseed[test], cyy, ek);
        TFHEpp::HomIntInvReducedNR<bkP, iksP, cbbrP, ahP, TFHEpp::intparam>(
            cz[test], cb[test], ek);
        auto t1 = std::chrono::system_clock::now();
        inv_times[test] = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    }
    std::cout << "MulDiv" << std::endl;
    for (int test = 0; test < num_test; test++) {
        std::cout << test << std::endl;
        auto t0 = std::chrono::system_clock::now();
        TFHEpp::HomIntDivByMulNR<bkP, iksP, ahP, TFHEpp::intparam>(
            cq[test], cr[test], ca[test], cb[test], cz[test], ek);
        auto t1 = std::chrono::system_clock::now();
        div_times[test] = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    }
#ifdef USE_PERF
    ProfilerStop();
#endif

    std::vector<std::array<uint32_t, numsegment>> pq(num_test), pr(num_test);
    std::vector<std::array<uint32_t, numsegment>> pz(num_test);
    std::vector<std::array<uint32_t, 2>> pseed(num_test);
    std::vector<std::array<uint32_t, 2>> ptop(num_test);
    std::vector<std::array<uint32_t, TFHEpp::intparam::bit>> ptopbits(num_test);
    for (int i = 0; i < num_test; i++)
        for (int j = 0; j < numsegment; j++) {
            pq[i][j] = TFHEpp::tlweIntDecrypt<typename iksP::domainP,
                                              TFHEpp::intparam>(
                cq[i][j], sk->key.get<typename iksP::domainP>());
            pr[i][j] = TFHEpp::tlweIntDecrypt<typename iksP::domainP,
                                              TFHEpp::intparam>(
                cr[i][j], sk->key.get<typename iksP::domainP>());
            pz[i][j] = TFHEpp::tlweIntDecrypt<typename iksP::domainP,
                                              TFHEpp::intparam>(
                cz[i][j], sk->key.get<typename iksP::domainP>());
            if (j < 2)
                pseed[i][j] = TFHEpp::tlweIntDecrypt<typename iksP::domainP,
                                                     TFHEpp::intparam>(
                    cseed[i][j], sk->key.get<typename iksP::domainP>());
            if (j < 2)
                ptop[i][j] = TFHEpp::tlweIntDecrypt<typename iksP::domainP,
                                                    TFHEpp::intparam>(
                    ctop[i][j], sk->key.get<typename iksP::domainP>());
        }
    for (int i = 0; i < num_test; i++)
        for (int j = 0; j < TFHEpp::intparam::bit; j++)
            ptopbits[i][j] = TFHEpp::tlweSymDecrypt<typename iksP::domainP>(
                ctopbits[i][j], sk->key.get<typename iksP::domainP>());

    int count = 0;
    for (int i = 0; i < num_test; i++) {
        uint64_t q = pq[i][0];
        for (int j = 1; j < numsegment; j++)
            q |= static_cast<uint64_t>(pq[i][j]) << (TFHEpp::intparam::bit * j);
        uint64_t r = pr[i][0];
        for (int j = 1; j < numsegment; j++)
            r |= static_cast<uint64_t>(pr[i][j]) << (TFHEpp::intparam::bit * j);
        uint64_t a = pa[i][0];
        for (int j = 1; j < numsegment; j++)
            a |= static_cast<uint64_t>(pa[i][j]) << (TFHEpp::intparam::bit * j);
        uint64_t b = pb[i][0];
        for (int j = 1; j < numsegment; j++)
            b |= static_cast<uint64_t>(pb[i][j]) << (TFHEpp::intparam::bit * j);
        uint64_t z = pz[i][0];
        for (int j = 1; j < numsegment; j++)
            z |= static_cast<uint64_t>(pz[i][j]) << (TFHEpp::intparam::bit * j);
        uint64_t seed = pseed[i][0] | (static_cast<uint64_t>(pseed[i][1]) << 4);

        uint64_t tq = a / b;
        uint64_t tr = a % b;
        count += (q == tq) ? 0 : 1;
        count += (r == tr) ? 0 : 1;
        std::cout << "(" << q << "," << r << ")"
                  << ":"
                  << "(" << tq << "," << tr << ")"
                  << ":seed=" << seed
                  << ":top=" << ptop[i][1] << "," << ptop[i][0]
                  << ":topbits=" << ptopbits[i][0] << ptopbits[i][1]
                  << ptopbits[i][2] << ptopbits[i][3]
                  << ":z=" << z << ":tz=" << (((1ULL << 32) - 1) / b);
        for (int j = 0; j < numsegment; j++) std::cout << ":" << pa[i][j];
        for (int j = 0; j < numsegment; j++) std::cout << ":" << pb[i][j];
        std::cout << ":" << a << ":" << b << std::endl;
        std::cout << std::endl;
    }
    auto stats = [](const std::vector<double>& v) -> std::pair<double,double> {
        double sum = 0, sum2 = 0;
        for (double x : v) { sum += x; sum2 += x*x; }
        double mean = sum / v.size();
        double var = sum2 / v.size() - mean * mean;
        return {mean, std::sqrt(var < 0 ? 0 : var)};
    };
    auto [inv_mean, inv_std] = stats(inv_times);
    auto [div_mean, div_std] = stats(div_times);
    std::vector<double> sum_times(num_test);
    for (int i = 0; i < num_test; i++) sum_times[i] = inv_times[i] + div_times[i];
    auto [sum_mean, sum_std] = stats(sum_times);
    std::cout << "Inv:" << inv_mean << "ms (std:" << inv_std << "ms)" << std::endl;
    std::cout << "DivByMul:" << div_mean << "ms (std:" << div_std << "ms)" << std::endl;
    std::cout << "Sum:" << sum_mean << "ms (std:" << sum_std << "ms)" << std::endl;
    std::cout << "count:" << count << std::endl;
    assert(count == 0);

    std::cout << "Passed" << std::endl;
}
