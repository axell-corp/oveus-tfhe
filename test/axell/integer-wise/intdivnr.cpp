#include <bits/stdint-uintn.h>

#include <cassert>
#include <chrono>
#include <limits>
#include <tfhe++.hpp>
#include <axell/integerops.hpp>

int main()
{
    // using bkP = TFHEpp::lvl12param;
    // using privksP = TFHEpp::lvl22param;
    // using iksP = TFHEpp::lvl21param;
    // using bkP = TFHEpp::lvl02param;
    // using privksP = TFHEpp::lvl22param;
    // using iksP = TFHEpp::lvl20param;
    using bkP = TFHEpp::lvlh2param;
    using iksP = TFHEpp::lvl2hparam;
    using ahP = TFHEpp::AHlvl2param;
    // constexpr uint numsegmentbit = 1;
    // constexpr uint numiter = 3;
    constexpr uint numsegmentbit = 2;
    constexpr uint numiter = 4;
    constexpr size_t numsegment = 1<<numsegmentbit;
    constexpr uint32_t num_test = 10;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::uniform_int_distribution<uint32_t> intdist(0,
                                                    TFHEpp::intparam::intmax);
    std::uniform_int_distribution<uint32_t> intdistnz(1,
                                                    TFHEpp::intparam::intmax);

    const TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();

    std::vector<std::array<uint32_t, numsegment>> pa(num_test);
    std::vector<std::array<uint32_t, numsegment>> pb(num_test);
    for (std::array<uint32_t, numsegment>& pai : pa)
        for (uint32_t& i : pai) i = intdist(engine);
    for (std::array<uint32_t, numsegment>& pbi : pb){
        pbi[0] = intdist(engine);
        for (int i = 1; i < numsegment; i++) pbi[i] = intdist(engine);
    }

    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>>
        ca(num_test);
    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>>
        cb(num_test);
    std::vector<
        std::vector<TFHEpp::TLWE<typename iksP::domainP>>>
        cz(num_test);
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

    std::vector<
        std::vector<TFHEpp::TLWE<typename iksP::domainP>>>
        cq(num_test);
    std::vector<
        std::vector<TFHEpp::TLWE<typename iksP::domainP>>>
        cr(num_test);
    for (int i = 0; i < num_test; i++) cq[i].resize(numsegment);
    for (int i = 0; i < num_test; i++) cr[i].resize(numsegment);
    std::vector<double> inv_times(num_test), div_times(num_test);

    std::cout<<"Inv"<<std::endl;
    for (int test = 0; test < num_test; test++) {
        std::cout<<test<<std::endl;
        auto t0 = std::chrono::system_clock::now();
        TFHEpp::HomIntInvNR<bkP, iksP, ahP, TFHEpp::intparam,numiter>(cz[test], cb[test], ek);
        auto t1 = std::chrono::system_clock::now();
        inv_times[test] = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    }
    std::cout<<"MulDiv"<<std::endl;
    for (int test = 0; test < num_test; test++) {
        std::cout<<test<<std::endl;
        auto t0 = std::chrono::system_clock::now();
        TFHEpp::HomIntDivByMulNR<bkP, iksP, ahP, TFHEpp::intparam>(cq[test], cr[test], ca[test], cb[test], cz[test], ek);
        auto t1 = std::chrono::system_clock::now();
        div_times[test] = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    }

    std::vector<std::array<uint32_t, numsegment>> pq(num_test), pr(num_test);
    for (int i = 0; i < num_test; i++)
        for (int j = 0; j < numsegment; j++){
            pq[i][j] = TFHEpp::tlweIntDecrypt<typename iksP::domainP,
                                                TFHEpp::intparam>(
                cq[i][j], sk->key.get<typename iksP::domainP>());
            pr[i][j] = TFHEpp::tlweIntDecrypt<typename iksP::domainP,
                                                TFHEpp::intparam>(
                cr[i][j], sk->key.get<typename iksP::domainP>());
        }
    // for(int i = 0; i<num_test;i++)
    // std::cout<<plres[i]<<":"<<pures[i]<<":"<<(pures[i]<<TFHEpp::intparam::bit)+plres[i]<<":"<<pa[i]<<":"<<pb[i]<<":"<<pa[i]*pb[i]<<":"<<((pa[i]*pb[i])&((1<<TFHEpp::intparam::bit)-1))<<":"<<((pa[i]*pb[i])>>TFHEpp::intparam::bit)<<std::endl;
    int count = 0;
    for (int i = 0; i < num_test; i++) {
        uint64_t q = pq[i][0];
        for (int j = 1; j < numsegment; j++)
            q |= pq[i][j] << (TFHEpp::intparam::bit * j);
        uint64_t r = pr[i][0];
        for (int j = 1; j < numsegment; j++)
            r |= pr[i][j] << (TFHEpp::intparam::bit * j);
        uint64_t a = pa[i][0];
        for (int j = 1; j < numsegment; j++)
            a |= pa[i][j] << (TFHEpp::intparam::bit * j);
        uint64_t b = pb[i][0];
        for (int j = 1; j < numsegment; j++)
            b |= pb[i][j] << (TFHEpp::intparam::bit * j);

        uint64_t tq = a/b;
        uint64_t tr = a%b;
        count += (q == tq) ? 0 : 1;
        count += (r == tr) ? 0 : 1;
        std::cout << "("<<q<<","<<r<<")" << ":" << "("<<tq<<","<<tr<<")";
        // for (int j = 0; j < numsegment; j++) std::cout << ":" << pres[i][j];
        // for (int j = 0; j < numsegment; j++) std::cout << ":" << ((tres>>TFHEpp::intparam::bit*j)&((1ull<<TFHEpp::intparam::bit)-1));
        for (int j = 0; j < numsegment; j++) std::cout << ":" << pa[i][j];
        for (int j = 0; j < numsegment; j++) std::cout << ":" << pb[i][j];
        std::cout<<":"<<a<<":"<<b<<std::endl;
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
    std::cout <<"Inv:"<<inv_mean << "ms (std:" << inv_std << "ms)" << std::endl;
    std::cout <<"DivByMul:"<<div_mean << "ms (std:" << div_std << "ms)" << std::endl;
    std::cout <<"Sum:"<<sum_mean << "ms (std:" << sum_std << "ms)" << std::endl;
    std::cout << "count:" << count << std::endl;
    assert(count == 0);
    std::cout << "Passed" << std::endl;
}
