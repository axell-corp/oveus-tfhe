#include <cassert>
#include <chrono>
#include <limits>
#include <tfhe++.hpp>
#include <axell/integerops.hpp>


int main()
{
    using brP = TFHEpp::lvlh2param;
    using iksP = TFHEpp::lvl2hparam;
    constexpr int numsegment = 4;
    constexpr uint32_t num_test = 100;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::uniform_int_distribution<uint32_t> intdist(0,
                                                    TFHEpp::intparam::intmax);

    const TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();

    std::vector<std::array<uint32_t, numsegment>> pa(num_test);
    // for (std::array<uint32_t, numsegment>& pai : pa)
        // for (uint32_t& i : pai) i = intdist(engine);
    for (std::array<uint32_t, numsegment>& pai : pa){
        for (int i = 0; i < numsegment-1; i++) pai[i] = intdist(engine);
        pai[numsegment-1] = 0;
    }

    std::vector<std::vector<TFHEpp::TLWE<typename iksP::domainP>>>
        ca(num_test);
    for (int i = 0; i < num_test; i++){
        ca[i].resize(numsegment);
        for (int j = 0; j < numsegment; j++)
            ca[i][j] = TFHEpp::tlweSymEncrypt<typename iksP::domainP>(
                pa[i][j] *
                    TFHEpp::intμ<typename iksP::domainP, TFHEpp::intparam>(),
                iksP::domainP::α, sk->key.get<typename iksP::domainP>());
    }

    TFHEpp::EvalKey ek;
    ek.emplacebkfft<brP>(*sk);
    ek.emplaceiksk<iksP>(*sk);

    std::vector<TFHEpp::TLWE<typename iksP::domainP>>
        resu(num_test);
    std::vector<TFHEpp::TLWE<typename iksP::domainP>>
        resl(num_test);
    std::chrono::system_clock::time_point start, end;

    start = std::chrono::system_clock::now();

    for (int test = 0; test < num_test; test++) {
        TFHEpp::HomIntCLZ<brP, iksP, TFHEpp::intparam>(resu[test],resl[test],ca[test], ek);
    }

    end = std::chrono::system_clock::now();

    double elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    std::vector<uint32_t> presu(num_test);
    std::vector<uint32_t> presl(num_test);
    for (int i = 0; i < num_test; i++)
        presu[i] = TFHEpp::tlweIntDecrypt<typename iksP::domainP,
                                            TFHEpp::intparam>(
            resu[i], sk->key.get<typename iksP::domainP>());
    for (int i = 0; i < num_test; i++)
        presl[i] = TFHEpp::tlweIntDecrypt<typename iksP::domainP,
                                            TFHEpp::intparam>(
            resl[i], sk->key.get<typename iksP::domainP>());

    int count = 0;
    for (int i = 0; i < num_test; i++) {
        uint32_t res = (presu[i]<<std::countr_zero(TFHEpp::intparam::bit))+presl[i];
        uint32_t val = 0;
        for(int j = 0; j < numsegment; j++) val |= pa[i][j] << (j*TFHEpp::intparam::bit);
        uint32_t tres = std::countl_zero(val) - (std::numeric_limits<uint32_t>::digits - TFHEpp::intparam::bit*numsegment);
        count += (res == tres) ? 0 : 1;
        std::cout << res << ":" << tres<<", "<<presu[i]<<":"<<presl[i];
        std::cout << std::endl;
    }
    std::cout << "count:" << count << std::endl;
    assert(count == 0);

    std::cout << "Passed" << std::endl;
    std::cout << elapsed / num_test << "ms" << std::endl;
}
