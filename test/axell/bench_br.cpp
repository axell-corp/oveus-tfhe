#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <tfhe++.hpp>

template <class brP, class iksP>
void bench_pbs(const char* name, const TFHEpp::SecretKey& sk,
               const TFHEpp::EvalKey& ek, int num_test) {
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::uniform_int_distribution<uint32_t> binary(0, 1);

    // Encrypt test data
    std::vector<TFHEpp::TLWE<typename brP::domainP>> ca(num_test);
    for (int i = 0; i < num_test; i++)
        ca[i] = TFHEpp::tlweSymEncrypt<typename brP::domainP>(
            binary(engine) ? typename brP::domainP::T(brP::targetP::μ)
                           : typename brP::domainP::T(-brP::targetP::μ),
            brP::domainP::α, sk.key.get<typename brP::domainP>());

    TFHEpp::TRLWE<typename brP::targetP> trlwe_res;
    TFHEpp::TLWE<typename brP::targetP> tlwe_big;
    TFHEpp::TLWE<typename iksP::targetP> tlwe_small;

    auto stats = [](const std::vector<double>& v) -> std::pair<double, double> {
        double sum = 0, sum2 = 0;
        for (double x : v) { sum += x; sum2 += x * x; }
        double mean = sum / v.size();
        double var = sum2 / v.size() - mean * mean;
        return {mean, std::sqrt(var < 0 ? 0 : var)};
    };

    // Warmup
    TFHEpp::BlindRotate<brP>(
        trlwe_res, ca[0], ek.getbkfft<brP>(),
        TFHEpp::μpolygen<typename brP::targetP, brP::targetP::μ>());
    TFHEpp::SampleExtractIndex<typename brP::targetP>(tlwe_big, trlwe_res, 0);
    TFHEpp::IdentityKeySwitch<iksP>(tlwe_small, tlwe_big, ek.getiksk<iksP>());

    // BR benchmark
    std::vector<double> br_times(num_test);
    for (int i = 0; i < num_test; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        TFHEpp::BlindRotate<brP>(
            trlwe_res, ca[i], ek.getbkfft<brP>(),
            TFHEpp::μpolygen<typename brP::targetP, brP::targetP::μ>());
        auto t1 = std::chrono::high_resolution_clock::now();
        br_times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // IKS benchmark (extract + keyswitch)
    std::vector<double> iks_times(num_test);
    for (int i = 0; i < num_test; i++) {
        TFHEpp::BlindRotate<brP>(
            trlwe_res, ca[i], ek.getbkfft<brP>(),
            TFHEpp::μpolygen<typename brP::targetP, brP::targetP::μ>());
        TFHEpp::SampleExtractIndex<typename brP::targetP>(tlwe_big, trlwe_res, 0);
        auto t0 = std::chrono::high_resolution_clock::now();
        TFHEpp::IdentityKeySwitch<iksP>(tlwe_small, tlwe_big, ek.getiksk<iksP>());
        auto t1 = std::chrono::high_resolution_clock::now();
        iks_times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Full PBS benchmark (BR + SEI + IKS)
    std::vector<double> pbs_times(num_test);
    for (int i = 0; i < num_test; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        TFHEpp::BlindRotate<brP>(
            trlwe_res, ca[i], ek.getbkfft<brP>(),
            TFHEpp::μpolygen<typename brP::targetP, brP::targetP::μ>());
        TFHEpp::SampleExtractIndex<typename brP::targetP>(tlwe_big, trlwe_res, 0);
        TFHEpp::IdentityKeySwitch<iksP>(tlwe_small, tlwe_big, ek.getiksk<iksP>());
        auto t1 = std::chrono::high_resolution_clock::now();
        pbs_times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    auto [br_mean, br_std] = stats(br_times);
    auto [iks_mean, iks_std] = stats(iks_times);
    auto [pbs_mean, pbs_std] = stats(pbs_times);

    std::cout << "--- " << name << " ---" << std::endl;
    std::cout << "BR:" << br_mean << "ms (std:" << br_std << "ms)" << std::endl;
    std::cout << "IKS:" << iks_mean << "ms (std:" << iks_std << "ms)" << std::endl;
    std::cout << "PBS:" << pbs_mean << "ms (std:" << pbs_std << "ms)" << std::endl;
    std::cout << std::endl;
}

int main() {
    constexpr int num_test = 100;

    std::cout << "Generating keys..." << std::endl;
    const auto sk = std::make_unique<TFHEpp::SecretKey>();
    TFHEpp::EvalKey ek;

    // lvlh2param (used in division: lvlhalf -> lvl2, N=2048)
    std::cout << "Setting up lvlh2/lvl2h keys..." << std::endl;
    ek.emplacebkfft<TFHEpp::lvlh2param>(*sk);
    ek.emplaceiksk<TFHEpp::lvl2hparam>(*sk);
    bench_pbs<TFHEpp::lvlh2param, TFHEpp::lvl2hparam>(
        "lvlh2param (N=2048, division-bench)", *sk, ek, num_test);

    // lvl01param (N=1024, standard NAND gate)
    std::cout << "Setting up lvl01/lvl10 keys..." << std::endl;
    ek.emplacebkfft<TFHEpp::lvl01param>(*sk);
    ek.emplaceiksk<TFHEpp::lvl10param>(*sk);
    bench_pbs<TFHEpp::lvl01param, TFHEpp::lvl10param>(
        "lvl01param (N=1024, NAND gate)", *sk, ek, num_test);

    return 0;
}
