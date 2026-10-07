#include <bits/stdint-uintn.h>

#include <cassert>
#include <chrono>
#include <limits>
#include <tfhe++.hpp>
#include <axell/integerops.hpp>

int main()
{
    using bkP = TFHEpp::lvlh2param;
    using iksP = TFHEpp::lvl2hparam;
    using ahP = TFHEpp::AHlvl2param;
    constexpr uint numsegmentbit = 3;
    constexpr uint numiter = 4;
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
    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>> cz(num_test);
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
    ek.emplaceiksk<iksP>(*sk);
    ek.emplaceahk<ahP>(*sk);

    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>> cq(num_test);
    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>> cr(num_test);
    for (int i = 0; i < num_test; i++) cq[i].resize(numsegment);
    for (int i = 0; i < num_test; i++) cr[i].resize(numsegment);
    std::vector<double> inv_times(num_test), div_times(num_test);

    std::cout << "Inv" << std::endl;
    for (int test = 0; test < num_test; test++) {
        std::cout << test << std::endl;
        auto t0 = std::chrono::system_clock::now();
        TFHEpp::HomIntInvNR<bkP, iksP, ahP, TFHEpp::intparam, numiter>(
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

    std::vector<std::array<uint32_t, numsegment>> pq(num_test), pr(num_test);
    for (int i = 0; i < num_test; i++)
        for (int j = 0; j < numsegment; j++) {
            pq[i][j] = TFHEpp::tlweIntDecrypt<typename iksP::domainP,
                                              TFHEpp::intparam>(
                cq[i][j], sk->key.get<typename iksP::domainP>());
            pr[i][j] = TFHEpp::tlweIntDecrypt<typename iksP::domainP,
                                              TFHEpp::intparam>(
                cr[i][j], sk->key.get<typename iksP::domainP>());
        }

    int count = 0;
    for (int i = 0; i < num_test; i++) {
        uint64_t q = pq[i][0];
        for (int j = 1; j < numsegment; j++)
            q |= static_cast<uint64_t>(pq[i][j])
                 << (TFHEpp::intparam::bit * j);
        uint64_t r = pr[i][0];
        for (int j = 1; j < numsegment; j++)
            r |= static_cast<uint64_t>(pr[i][j])
                 << (TFHEpp::intparam::bit * j);
        uint64_t a = pa[i][0];
        for (int j = 1; j < numsegment; j++)
            a |= static_cast<uint64_t>(pa[i][j])
                 << (TFHEpp::intparam::bit * j);
        uint64_t b = pb[i][0];
        for (int j = 1; j < numsegment; j++)
            b |= static_cast<uint64_t>(pb[i][j])
                 << (TFHEpp::intparam::bit * j);

        const uint64_t tq = a / b;
        const uint64_t tr = a % b;
        count += (q == tq) ? 0 : 1;
        count += (r == tr) ? 0 : 1;
        std::cout << "(" << q << "," << r << ")"
                  << ":"
                  << "(" << tq << "," << tr << ")";
        for (int j = 0; j < numsegment; j++) std::cout << ":" << pa[i][j];
        for (int j = 0; j < numsegment; j++) std::cout << ":" << pb[i][j];
        std::cout << ":" << a << ":" << b << std::endl;
        std::cout << std::endl;
    }
    std::cout << "count:" << count << std::endl;
    assert(count == 0);

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
