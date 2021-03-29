#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <tfhe++.hpp>

using namespace std;
using namespace TFHEpp;

int main()
{
    constexpr uint32_t num_test = 100;
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> binary(0, 1);

    SecretKey *sk = new SecretKey;
    vector<int32_t> ps(num_test);
    vector<array<uint8_t, lvl1param::n>> p1(num_test);
    vector<array<uint8_t, lvl1param::n>> p0(num_test);

    vector<array<uint32_t, lvl1param::n>> pmu1(num_test);
    vector<array<uint32_t, lvl1param::n>> pmu0(num_test);
    array<bool, lvl1param::n> pres;

    for (int32_t &p : ps) p = binary(engine);
    for (array<uint8_t, lvl1param::n> &i : p1)
        for (uint8_t &p : i) p = binary(engine);
    for (array<uint8_t, lvl1param::n> &i : p0)
        for (uint8_t &p : i) p = binary(engine);

    for (int i = 0; i < num_test; i++)
        for (int j = 0; j < lvl1param::n; j++)
            pmu1[i][j] = (p1[i][j] > 0) ? lvl1param::μ : -lvl1param::μ;
    for (int i = 0; i < num_test; i++)
        for (int j = 0; j < lvl1param::n; j++)
            pmu0[i][j] = (p0[i][j] > 0) ? lvl1param::μ : -lvl1param::μ;
    vector<TRGSWFFT<lvl1param>> cs(num_test);
    vector<TRLWE<lvl1param>> c1(num_test);
    vector<TRLWE<lvl1param>> c0(num_test);
    vector<TRLWE<lvl1param>> cres(num_test);

    for (int i = 0; i < num_test; i++)
        cs[i] = trgswfftSymEncrypt<lvl1param>(ps[i], lvl1param::α, sk->key.lvl1);
    for (int i = 0; i < num_test; i++)
        c1[i] = trlweSymEncryptlvl1(pmu1[i], lvl1param::α, sk->key.lvl1);
    for (int i = 0; i < num_test; i++)
        c0[i] = trlweSymEncryptlvl1(pmu0[i], lvl1param::α, sk->key.lvl1);

    chrono::system_clock::time_point start, end;
    start = chrono::system_clock::now();
    for (int test = 0; test < num_test; test++) {
        CMUXFFTlvl1(cres[test], cs[test], c1[test], c0[test]);
    }
    end = chrono::system_clock::now();

    for (int test = 0; test < num_test; test++) {
        pres = trlweSymDecryptlvl1(cres[test], sk->key.lvl1);
        for (int i = 0; i < lvl1param::n; i++)
            assert(pres[i] == ((ps[test] > 0) ? p1[test][i] : p0[test][i]) > 0);
    }
    cout << "Passed" << endl;
    double elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    cout << elapsed / num_test << "μs" << endl;
}