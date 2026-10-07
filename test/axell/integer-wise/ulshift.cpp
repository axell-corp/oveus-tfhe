#ifdef USE_PERF
#include <gperftools/profiler.h>
#endif

#include <cassert>
#include <chrono>
#include <limits>
#include <tfhe++.hpp>
#include <axell/integerops.hpp>

int main()
{
    using brP = TFHEpp::lvlh2param;
    using iksP = TFHEpp::lvl2hparam;
    using cbbrP = TFHEpp::cblvlh2param;
    using ahP = TFHEpp::AHlvl2param;
    using P = typename iksP::domainP;
    constexpr uint numwordbit = 2;
    constexpr uint numword = 1U<<numwordbit;
    constexpr uint num_test = 100;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::uniform_int_distribution<uint32_t> intdist(0,
                                                    TFHEpp::intparam::t-1);
    std::uniform_int_distribution<uint32_t> shiftudist(0,
                                                numword-1);
    std::uniform_int_distribution<uint32_t> shiftldist(0,
                                                TFHEpp::intparam::bit-1);

    const TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();
    TFHEpp::EvalKey ek;
    ek.emplacebkfft<brP>(*sk);
    ek.emplacebkfft<cbbrP>(*sk);
    ek.emplaceiksk<iksP>(*sk);
    ek.emplaceahk<ahP>(*sk);
    ek.emplacecbsk<ahP>(*sk);

    std::vector<std::array<uint32_t,numword>> pa(num_test);
    std::vector<uint32_t> psu(num_test);
    std::vector<uint32_t> psl(num_test);
    for (std::array<uint32_t,numword>& i : pa) for(uint32_t& j : i) j = intdist(engine);
    for (uint32_t& i : psu) i = shiftudist(engine);
    for (uint32_t& i : psl) i = shiftldist(engine);

    std::vector<std::vector<TFHEpp::TLWE<P>>> ca(num_test);
    std::vector<TFHEpp::TLWE<P>> csu(num_test);
    std::vector<TFHEpp::TLWE<P>> csl(num_test);
    for (int i = 0; i < num_test; i++){
        ca[i].resize(numword);
        for(int j = 0; j < numword; j++)
        ca[i][j] = TFHEpp::tlweSymEncrypt<P>(
            pa[i][j] * TFHEpp::intμ<P, TFHEpp::intparam>(),
            P::α, sk->key.get<P>());
    }
    for (int i = 0; i < num_test; i++)
        csu[i] = TFHEpp::tlweSymEncrypt<P>(
            psu[i] * TFHEpp::intμ<P, TFHEpp::intparam>(),
            P::α, sk->key.get<P>());
    for (int i = 0; i < num_test; i++)
        csl[i] = TFHEpp::tlweSymEncrypt<P>(
            psl[i] * TFHEpp::intμ<P, TFHEpp::intparam>(),
            P::α, sk->key.get<P>());


    std::vector<std::vector<TFHEpp::TLWE<P>>> res(num_test);
    for (int test = 0; test < num_test; test++) res[test].resize(numword);
    std::chrono::system_clock::time_point start, end;

    start = std::chrono::system_clock::now();
    // ProfilerStart("mul.prof");
    for (int test = 0; test < num_test; test++) {
        TFHEpp::ULeftShift<brP, iksP, cbbrP, ahP, TFHEpp::intparam>(res[test], ca[test], csu[test], csl[test], ek);
    }
    // ProfilerStop();
    end = std::chrono::system_clock::now();

    double elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    std::vector<std::array<int,numword>> pres(num_test);
    for (int i = 0; i < num_test; i++)
        for(int j = 0; j < numword; j++)
        pres[i][j] =
            TFHEpp::tlweIntDecrypt<P, TFHEpp::intparam>(
                res[i][j], sk->key.get<P>());

    int count = 0;
    for (int i = 0; i < num_test; i++){
        uint sres = 0;
        for(int j = 0; j < numword; j++) sres |= pres[i][j]<<(j*TFHEpp::intparam::bit);
        uint spa = 0;
        for(int j = 0; j < numword; j++) spa |= pa[i][j]<<(j*TFHEpp::intparam::bit);

        const uint tres = (spa<<(psu[i]*TFHEpp::intparam::bit+psl[i]))&((1ULL<<(numword*TFHEpp::intparam::bit))-1);
        // const uint tres = (spa<<(psu[i]*TFHEpp::intparam::bit))&((1ULL<<(numword*TFHEpp::intparam::bit))-1);
        std::cout<<spa<<", psu:"<<psu[i]<<", psl:"<<psl[i]<<", res:"<<sres<<", tres:"<<tres<<", divide:"<<static_cast<double>(sres)/spa<<std::endl;
        count +=
            (sres == tres)
                ? 0
                : 1;
    }
    std::cout << "count:" << count << std::endl;
    assert(count == 0);

    std::cout << "Passed" << std::endl;
    std::cout << elapsed / num_test << "ms" << std::endl;
}
