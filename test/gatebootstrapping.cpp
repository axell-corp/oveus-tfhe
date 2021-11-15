#ifdef USE_PERF
#include <gperftools/profiler.h>
#endif

#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <tfhe++.hpp>

int main()
{
    constexpr uint32_t num_test = 1000;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::uniform_int_distribution<uint32_t> binary(0, 1);

    using bkP = TFHEpp::lvl02param;
    using iksP = TFHEpp::lvl20param;

    TFHEpp::SecretKey sk;
    TFHEpp::EvalKey ek;
    ek.emplacebkfft<bkP>(sk);
    ek.emplaceiksk<iksP>(sk);
    std::vector<TFHEpp::TLWE<typename iksP::domainP>> tlwe(num_test),
        bootedtlwe(num_test);
    std::array<bool, num_test> p;
    for (int i = 0; i < num_test; i++) p[i] = binary(engine) > 0;
    for (int i = 0; i < num_test; i++)
        TFHEpp::tlweSymEncrypt<typename iksP::domainP>(
            tlwe[i], p[i] ? iksP::domainP::μ : -iksP::domainP::μ,
            sk.key.get<typename iksP::domainP>());

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
#ifdef USE_PERF
    ProfilerStart("gb.prof");
#endif
    for (int test = 0; test < num_test; test++) {
        bool p = binary(engine) > 0;
        TFHEpp::TLWE<TFHEpp::lvl1param> tlwe =
            TFHEpp::tlweSymEncrypt<TFHEpp::lvl1param>(
                p ? TFHEpp::lvl1param::μ : -TFHEpp::lvl1param::μ,
                TFHEpp::lvl1param::α, sk.key.lvl1);
        TFHEpp::TLWE<TFHEpp::lvl1param> bootedtlwe;
        TFHEpp::GateBootstrapping(bootedtlwe, tlwe, ek);
        bool p2 =
            TFHEpp::tlweSymDecrypt<TFHEpp::lvl1param>(bootedtlwe, sk.key.lvl1);
        assert(p == p2);
    }
    std::cout << "Passed" << std::endl;
}