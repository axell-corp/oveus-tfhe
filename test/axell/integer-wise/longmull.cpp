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
    using bkP = TFHEpp::lvlI2param;
    using iksP = TFHEpp::lvl2Iparam;
    using ahP = TFHEpp::AHlvl2param;
    constexpr int numsegmentbit = 2;
    constexpr int numsegment = 1<<numsegmentbit;
    constexpr uint32_t num_test = 10;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::uniform_int_distribution<uint32_t> intdist(0,
                                                    TFHEpp::intparam::intmax);

    const TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();

    std::vector<std::array<uint32_t, numsegment>> pa(num_test);
    std::vector<std::array<uint32_t, numsegment>> pb(num_test);
    for (std::array<uint32_t, numsegment>& pai : pa)
        for (uint32_t& i : pai) i = intdist(engine);
    for (std::array<uint32_t, numsegment>& pbi : pb)
        for (uint32_t& i : pbi) i = intdist(engine);

    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>>
        ca(num_test);
    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>>
        cb(num_test);
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
        res(num_test);
    for (int i = 0; i < num_test; i++) res[i].resize(numsegment);
    std::chrono::system_clock::time_point start, end;

    start = std::chrono::system_clock::now();
    // ProfilerStart("mul.prof");
    for (int test = 0; test < num_test; test++) {
        TFHEpp::LongMulL<bkP, iksP, ahP, TFHEpp::intparam>(res[test], ca[test],
                                                       cb[test], ek);
    }
    // ProfilerStop();
    end = std::chrono::system_clock::now();

    double elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    std::vector<std::array<uint32_t, numsegment>> pres(num_test);
    for (int i = 0; i < num_test; i++)
        for (int j = 0; j < numsegment; j++)
            pres[i][j] = TFHEpp::tlweIntDecrypt<typename iksP::domainP,
                                                TFHEpp::intparam>(
                res[i][j], sk->key.get<typename iksP::domainP>());

    // for(int i = 0; i<num_test;i++)
    // std::cout<<plres[i]<<":"<<pures[i]<<":"<<(pures[i]<<TFHEpp::intparam::bit)+plres[i]<<":"<<pa[i]<<":"<<pb[i]<<":"<<pa[i]*pb[i]<<":"<<((pa[i]*pb[i])&((1<<TFHEpp::intparam::bit)-1))<<":"<<((pa[i]*pb[i])>>TFHEpp::intparam::bit)<<std::endl;
    int count = 0;
    for (int i = 0; i < num_test; i++) {
        uint64_t res = pres[i][0];
        for (int j = 1; j < numsegment; j++)
            res += static_cast<uint64_t>(pres[i][j]) << (TFHEpp::intparam::bit * j);

        uint64_t tres = 0;
        for (int j = 0; j < numsegment; j++)
            for (int k = 0; k < numsegment; k++)
                tres += (j + k != 0) ? static_cast<uint64_t>(pa[i][j] * pb[i][k])
                                           << (TFHEpp::intparam::bit * (j + k))
                                     : pa[i][0] * pb[i][0];
        tres %= 1ull<<(TFHEpp::intparam::bit*numsegment);
        count += (res == tres) ? 0 : 1;
        std::cout << res << ":" << tres <<", pres";
        for (int j = 0; j < numsegment; j++) std::cout << ":" << pres[i][j];
        std::cout<<", tres";
        for (int j = 0; j < numsegment; j++) std::cout << ":" << ((tres>>TFHEpp::intparam::bit*j)&((1ull<<TFHEpp::intparam::bit)-1));
        std::cout<<", pa";
        for (int j = 0; j < numsegment; j++) std::cout << ":" << pa[i][j];
        std::cout<<", pb";
        for (int j = 0; j < numsegment; j++) std::cout << ":" << pb[i][j];
        std::cout << std::endl;
    }
    std::cout << elapsed / num_test << "ms" << std::endl;
    std::cout << "count:" << count << std::endl;
    assert(count == 0);

    std::cout << "Passed" << std::endl;
}
