#pragma once

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

#include "evalkeygens.hpp"

namespace TFHEpp {

template <class P>
void bkgen(BootstrappingKey<P> &bk, const SecretKey &sk);

template <class P>
void bkfftgen(BootstrappingKeyFFT<P> &bkfft, const SecretKey &sk);

template <class P>
void bknttgen(BootstrappingKeyNTT<P> &bkntt, const SecretKey &sk);

template <class P>
void tlwe2trlweikskkgen(TLWE2TRLWEIKSKey<P> &iksk, const SecretKey &sk)
{
    for (int i = 0; i < P::domainP::n; i++)
        for (int j = 0; j < P::t; j++)
            for (uint32_t k = 0; k < (1 << P::basebit) - 1; k++) {
                Polynomial<typename P::targetP> p = {};
                p[0] =
                    sk.key.get<typename P::domainP>()[i] * (k + 1) *
                    (1ULL << (numeric_limits<typename P::targetP::T>::digits -
                              (j + 1) * P::basebit));
                iksk[i][j][k] = trlweSymEncrypt<typename P::targetP>(
                    p, P::α, sk.key.get<typename P::targetP>());
            }
}

template <class P>
inline void annihilatekeyegen(AnnihilateKey<P> &ahk, const SecretKey &sk)
{
    for (int i = 0; i < P::nbit; i++) {
        Polynomial<P> autokey;
        Automorphism<P>(autokey, sk.key.get<P>(), (1 << (P::nbit - i)) + 1);
        ahk[i] = trgswfftSymEncrypt<P>(autokey, P::α, sk.key.get<P>());
    }
}

template <class P>
void ikskgen(KeySwitchingKey<P> &ksk, const SecretKey &sk);

template <class P>
void privkskgen(PrivateKeySwitchingKey<P> &privksk,
                const Polynomial<typename P::targetP> &func,
                const SecretKey &sk);

template <class P>
inline relinKey<P> relinKeygen(const Key<P> &key)
{
    constexpr std::array<typename P::T, P::l> h = hgen<P>();

    Polynomial<P> keysquare;
    PolyMulNaieve<P>(keysquare, key, key);
    relinKey<P> relinkey;
    for (TRLWE<P> &ctxt : relinkey) ctxt = trlweSymEncryptZero<P>(P::α, key);
    for (int i = 0; i < P::l; i++)
        for (int j = 0; j < P::n; j++)
            relinkey[i][1][j] +=
                static_cast<typename P::T>(keysquare[j]) * h[i];
    return relinkey;
}

template <class P>
inline relinKeyFFT<P> relinKeyFFTgen(const Key<P> &key)
{
    relinKey<P> relinkey = relinKeygen<P>(key);
    relinKeyFFT<P> relinkeyfft;
    for (int i = 0; i < P::l; i++)
        for (int j = 0; j < 2; j++)
            TwistIFFT<P>(relinkeyfft[i][j], relinkey[i][j]);
    return relinkeyfft;
}

struct EvalKey {
    lweParams params;
    std::unique_ptr<BootstrappingKey<lvl01param>> bklvl01;
    std::unique_ptr<BootstrappingKey<lvl02param>> bklvl02;
    std::unique_ptr<BootstrappingKeyFFT<lvl01param>> bkfftlvl01;
    std::unique_ptr<BootstrappingKeyFFT<lvl02param>> bkfftlvl02;
    std::unique_ptr<BootstrappingKeyNTT<lvl01param>> bknttlvl01;
    std::unique_ptr<BootstrappingKeyNTT<lvl02param>> bknttlvl02;
    std::unique_ptr<KeySwitchingKey<lvl10param>> iksklvl10;
    std::unique_ptr<KeySwitchingKey<lvl11param>> iksklvl11;
    std::unique_ptr<KeySwitchingKey<lvl20param>> iksklvl20;
    std::unique_ptr<KeySwitchingKey<lvl21param>> iksklvl21;
    std::unique_ptr<KeySwitchingKey<lvl22param>> iksklvl22;
    std::unordered_map<std::string,
                       std::unique_ptr<PrivateKeySwitchingKey<lvl11param>>>
        privksklvl11;
    std::unordered_map<std::string,
                       std::unique_ptr<PrivateKeySwitchingKey<lvl21param>>>
        privksklvl21;
    std::unordered_map<std::string,
                       std::unique_ptr<PrivateKeySwitchingKey<lvl22param>>>
        privksklvl22;

    EvalKey(SecretKey sk) { params = sk.params; }
    EvalKey() {}

    template <class P>
    void emplacebk(const SecretKey &sk);
    template <class P>
    void emplacebkfft(const SecretKey &sk);
    template <class P>
    void emplacebkntt(const SecretKey &sk);
    template <class P>
    void emplacebk2bkfft();
    template <class P>
    void emplacebk2bkntt();
    template <class P>
    void emplaceiksk(const SecretKey &sk);
    template <class P>
    void emplaceprivksk(const std::string &key,
                        const Polynomial<typename P::targetP> &func,
                        const SecretKey &sk);
    template <class P, uint index>
    void emplaceprivksk(const SecretKey &sk)
    {
        if constexpr (index == 0) {
            emplaceprivksk<P>("identity", {1}, sk);
        }
        else if constexpr (index == 1) {
            TFHEpp::Polynomial<typename P::targetP> poly;
            for (int i = 0; i < P::targetP::n; i++)
                poly[i] = -sk.key.get<typename P::targetP>()[i];
            emplaceprivksk<P>("secret key", poly, sk);
        }
        else {
            static_assert(
                false_v<P>,
                "Not a predefined function for Private Key Switching!");
        }
    }

    template <class P>
    BootstrappingKey<P> &getbk() const;
    template <class P>
    BootstrappingKeyFFT<P> &getbkfft() const;
    template <class P>
    BootstrappingKeyNTT<P> &getbkntt() const;
    template <class P>
    KeySwitchingKey<P> &getiksk() const;
    template <class P>
    PrivateKeySwitchingKey<P> &getprivksk(const std::string &key) const;

    template <class Archive>
    void serialize(Archive &archive)
    {
        archive(params, bklvl01, bklvlh1, bklvl02, bklvlh2, bkfftlvl01,
                bkfftlvlh1, bkfftlvl02, bkfftlvlh2, bknttlvl01, bknttlvlh1,
                bknttlvl02, bknttlvlh2, iksklvl10, iksklvl1h, iksklvl20,
                iksklvl21, iksklvl22, iksklvl31, privksklvl11, privksklvl21,
                privksklvl22);
    }

    // emplace keys
    template <class P>
    void emplacebk(const SecretKey& sk)
    {
        if constexpr (std::is_same_v<P, lvl01param> ||
                      std::is_same_v<P, lvl0Mparam>) {
            bklvl01 =
                std::make_unique_for_overwrite<BootstrappingKey<lvl01param>>();
            bkgen<lvl01param>(*bklvl01, sk);
        }
        else if constexpr (std::is_same_v<P, lvlh1param>) {
            bklvlh1 =
                std::make_unique_for_overwrite<BootstrappingKey<lvlh1param>>();
            bkgen<lvlh1param>(*bklvlh1, sk);
        }
        else if constexpr (std::is_same_v<P, lvl02param>) {
            bklvl02 =
                std::make_unique_for_overwrite<BootstrappingKey<lvl02param>>();
            bkgen<lvl02param>(*bklvl02, sk);
        }
        else if constexpr (std::is_same_v<P, lvlh2param>) {
            bklvlh2 =
                std::make_unique_for_overwrite<BootstrappingKey<lvlh2param>>();
            bkgen<lvlh2param>(*bklvlh2, sk);
        }
        else
            static_assert(false_v<typename P::targetP::T>,
                          "Not predefined parameter!");
    }
    template <class P>
    void emplacebkfft(const SecretKey& sk)
    {
        if constexpr (std::is_same_v<P, lvl01param>) {
            bkfftlvl01 = std::make_unique_for_overwrite<BootstrappingKeyFFT<lvl01param>>();
            bkfftgen<lvl01param>(*bkfftlvl01, sk);
        }
        else if constexpr (std::is_same_v<P, lvlh1param>||
                      std::is_same_v<P, lvl0Mparam>) {
            bkfftlvlh1 = std::make_unique_for_overwrite<BootstrappingKeyFFT<lvlh1param>>();
            bkfftgen<lvlh1param>(*bkfftlvlh1, sk);
        }
        else if constexpr (std::is_same_v<P, lvl02param>) {
            bkfftlvl02 = std::make_unique_for_overwrite<BootstrappingKeyFFT<lvl02param>>();
            bkfftgen<lvl02param>(*bkfftlvl02, sk);
        }
        else if constexpr (std::is_same_v<P, lvlh2param>) {
            bkfftlvlh2 = std::make_unique_for_overwrite<BootstrappingKeyFFT<lvlh2param>>();
            bkfftgen<lvlh2param>(*bkfftlvlh2, sk);
        }
        else
            static_assert(false_v<typename P::targetP::T>,
                          "Not predefined parameter!");
    }
    template <class P>
    void emplacebkntt(const SecretKey& sk)
    {
        if constexpr (std::is_same_v<P, lvl01param> ||
                      std::is_same_v<P, lvl0Mparam>) {
            bknttlvl01 = std::make_unique_for_overwrite<
                BootstrappingKeyNTT<lvl01param>>();
            bknttgen<lvl01param>(*bknttlvl01, sk);
        }
        else if constexpr (std::is_same_v<P, lvlh1param>) {
            bknttlvlh1 = std::make_unique_for_overwrite<
                BootstrappingKeyNTT<lvlh1param>>();
            bknttgen<lvlh1param>(*bknttlvlh1, sk);
        }
        else if constexpr (std::is_same_v<P, lvl02param>) {
            bknttlvl02 = std::make_unique_for_overwrite<
                BootstrappingKeyNTT<lvl02param>>();
            bknttgen<lvl02param>(*bknttlvl02, sk);
        }
        else if constexpr (std::is_same_v<P, lvlh2param>) {
            bknttlvlh2 = std::make_unique_for_overwrite<
                BootstrappingKeyNTT<lvlh2param>>();
            bknttgen<lvlh2param>(*bknttlvlh2, sk);
        }
        else
            static_assert(false_v<typename P::targetP::T>,
                          "Not predefined parameter!");
    }
    template <class P>
    void emplacebk2bkfft()
    {
        if constexpr (std::is_same_v<P, lvl01param> ||
                      std::is_same_v<P, lvl0Mparam>) {
            bkfftlvl01 = std::make_unique_for_overwrite<
                BootstrappingKeyFFT<lvl01param>>();
            for (int i = 0; i < lvl01param::domainP::n; i++)
                (*bkfftlvl01)[i][0] =
                    ApplyFFT2trgsw<lvl1param>((*bklvl01)[i][0]);
        }
        else if constexpr (std::is_same_v<P, lvlh1param>) {
            bkfftlvlh1 = std::make_unique_for_overwrite<
                BootstrappingKeyFFT<lvlh1param>>();
            for (int i = 0; i < lvlh1param::domainP::n; i++)
                (*bkfftlvlh1)[i][0] =
                    ApplyFFT2trgsw<lvl1param>((*bklvlh1)[i][0]);
        }
        else if constexpr (std::is_same_v<P, lvl02param>) {
            bkfftlvl02 = std::make_unique_for_overwrite<
                BootstrappingKeyFFT<lvl02param>>();
            for (int i = 0; i < lvl02param::domainP::n; i++)
                (*bkfftlvl02)[i][0] =
                    ApplyFFT2trgsw<lvl2param>((*bklvl02)[i][0]);
        }
        else if constexpr (std::is_same_v<P, lvlh2param>) {
            bkfftlvlh2 = std::make_unique_for_overwrite<
                BootstrappingKeyFFT<lvlh2param>>();
            for (int i = 0; i < lvlh2param::domainP::n; i++)
                (*bkfftlvlh2)[i][0] =
                    ApplyFFT2trgsw<lvl2param>((*bklvlh2)[i][0]);
        }
        else
            static_assert(false_v<typename P::targetP::T>,
                          "Not predefined parameter!");
    }
    template <class P>
    void emplacebk2bkntt()
    {
        if constexpr (std::is_same_v<P, lvl01param> ||
                      std::is_same_v<P, lvl0Mparam>) {
            bknttlvl01 = std::make_unique_for_overwrite<
                BootstrappingKeyNTT<lvl01param>>();
            for (int i = 0; i < lvl01param::domainP::n; i++)
                (*bknttlvl01)[i] = ApplyNTT2trgsw<lvl1param>((*bklvl01)[i][0]);
        }
        else if constexpr (std::is_same_v<P, lvlh1param>) {
            bknttlvlh1 = std::make_unique_for_overwrite<
                BootstrappingKeyNTT<lvlh1param>>();
            for (int i = 0; i < lvlh1param::domainP::n; i++)
                (*bknttlvlh1)[i] = ApplyNTT2trgsw<lvl1param>((*bklvlh1)[i][0]);
        }
        else if constexpr (std::is_same_v<P, lvl02param>) {
            bknttlvl02 = std::make_unique_for_overwrite<
                BootstrappingKeyNTT<lvl02param>>();
            for (int i = 0; i < lvl02param::domainP::n; i++)
                (*bknttlvl02)[i] = ApplyNTT2trgsw<lvl2param>((*bklvl02)[i][0]);
        }
        else if constexpr (std::is_same_v<P, lvlh2param>) {
            bknttlvlh2 = std::make_unique_for_overwrite<
                BootstrappingKeyNTT<lvlh2param>>();
            for (int i = 0; i < lvlh2param::domainP::n; i++)
                (*bknttlvlh2)[i] = ApplyNTT2trgsw<lvl2param>((*bklvlh2)[i][0]);
        }
        else
            static_assert(false_v<typename P::T>, "Not predefined parameter!");
    }
    template <class P>
    void emplaceiksk(const SecretKey& sk)
    {
        if constexpr (std::is_same_v<P, lvl10param> ||
                      std::is_same_v<P, lvlM0param>) {
            iksklvl10 = std::make_unique_for_overwrite<KeySwitchingKey<lvl10param>>();
            ikskgen<lvl10param>(*iksklvl10, sk);
        }
        else if constexpr (std::is_same_v<P, lvl1hparam>) {
            iksklvl1h = std::make_unique_for_overwrite<KeySwitchingKey<lvl1hparam>>();
            ikskgen<lvl1hparam>(*iksklvl1h, sk);
        }
        else if constexpr (std::is_same_v<P, lvl20param>) {
            iksklvl20 = std::make_unique_for_overwrite<KeySwitchingKey<lvl20param>>();
            ikskgen<lvl20param>(*iksklvl20, sk);
        }
        else if constexpr (std::is_same_v<P, lvl2hparam>) {
            iksklvl2h =
                std::make_unique_for_overwrite<KeySwitchingKey<lvl2hparam>>();
            ikskgen<lvl2hparam>(*iksklvl2h, sk);
        }
        else if constexpr (std::is_same_v<P, lvl21param>) {
            iksklvl21 = std::make_unique_for_overwrite<KeySwitchingKey<lvl21param>>();
            ikskgen<lvl21param>(*iksklvl21, sk);
        }
        else if constexpr (std::is_same_v<P, lvl22param>) {
            iksklvl22 = std::make_unique_for_overwrite<KeySwitchingKey<lvl22param>>();
            ikskgen<lvl22param>(*iksklvl22, sk);
        }
        else if constexpr (std::is_same_v<P, lvl31param>) {
            iksklvl31 = std::make_unique_for_overwrite<KeySwitchingKey<lvl31param>>();
            ikskgen<lvl31param>(*iksklvl31, sk);
        }
        else
            static_assert(false_v<typename P::T>, "Not predefined parameter!");
    }
    template <class P>
    void emplacesubiksk(const SecretKey& sk)
    {
        if constexpr (std::is_same_v<P, lvl21param>) {
            subiksklvl21 = std::make_unique_for_overwrite<
                SubsetKeySwitchingKey<lvl21param>>();
            subikskgen<lvl21param>(*subiksklvl21, sk);
        }
        else
            static_assert(false_v<typename P::T>, "Not predefined parameter!");
    }
    template <class P>
    void emplaceprivksk(const std::string& key,
                        const Polynomial<typename P::targetP>& func,
                        const SecretKey& sk)
    {
        if constexpr (std::is_same_v<P, lvl11param>) {
            privksklvl11[key] =
                std::unique_ptr<PrivateKeySwitchingKey<lvl11param>>(new (
                    std::align_val_t(64)) PrivateKeySwitchingKey<lvl11param>());
            privkskgen<lvl11param>(*privksklvl11[key], func, sk);
        }
        else if constexpr (std::is_same_v<P, lvl21param>) {
            privksklvl21[key] =
                std::unique_ptr<PrivateKeySwitchingKey<lvl21param>>(new (
                    std::align_val_t(64)) PrivateKeySwitchingKey<lvl21param>());
            privkskgen<lvl21param>(*privksklvl21[key], func, sk);
        }
        else if constexpr (std::is_same_v<P, lvl22param>) {
            privksklvl22[key] =
                std::unique_ptr<PrivateKeySwitchingKey<lvl22param>>(new (
                    std::align_val_t(64)) PrivateKeySwitchingKey<lvl22param>());
            privkskgen<lvl22param>(*privksklvl22[key], func, sk);
        }
        else
            static_assert(false_v<typename P::targetP::T>,
                          "Not predefined parameter!");
    }
    template <class P>
    void emplacesubprivksk(const std::string& key,
                           const Polynomial<typename P::targetP>& func,
                           const SecretKey& sk)
    {
        if constexpr (std::is_same_v<P, lvl21param>) {
            subprivksklvl21[key] =
                std::make_unique_for_overwrite<SubsetPrivateKeySwitchingKey<lvl21param>>();
            subprivkskgen<lvl21param>(*subprivksklvl21[key], func, sk);
        }
        else
            static_assert(false_v<typename P::T>, "Not predefined parameter!");
    }
    template <class P>
    void emplaceprivksk4cb(const SecretKey& sk)
    {
        for (int k = 0; k < P::targetP::k; k++) {
            Polynomial<typename P::targetP> partkey;
            for (int i = 0; i < P::targetP::n; i++)
                partkey[i] =
                    -sk.key.get<typename P::targetP>()[k * P::targetP::n + i];
            emplaceprivksk<P>("privksk4cb_" + std::to_string(k), partkey, sk);
        }
        emplaceprivksk<P>("privksk4cb_" + std::to_string(P::targetP::k), {1},
                          sk);
    }
    template <class P>
    void emplacesubprivksk4cb(const SecretKey& sk)
    {
        for (int k = 0; k < P::targetP::k; k++) {
            Polynomial<typename P::targetP> partkey;
            for (int i = 0; i < P::targetP::n; i++)
                partkey[i] =
                    -sk.key.get<typename P::targetP>()[k * P::targetP::n + i];
            emplacesubprivksk<P>("subprivksk4cb_" + std::to_string(k), partkey,
                                 sk);
        }
        emplacesubprivksk<P>("subprivksk4cb_" + std::to_string(P::targetP::k),
                             {1}, sk);
    }

    template <class P>
    void emplaceahk(const SecretKey& sk)
    {
        if constexpr (std::is_same_v<P, lvl1param>) {
            ahklvl1 =
                std::make_unique_for_overwrite<AnnihilateKey<lvl1param>>();
            annihilatekeygen<lvl1param>(*ahklvl1, sk);
        }
        else if constexpr (std::is_same_v<P, lvl2param>) {
            ahklvl2 =
                std::make_unique_for_overwrite<AnnihilateKey<lvl2param>>();
            annihilatekeygen<lvl2param>(*ahklvl2, sk);
        }
        else
            static_assert(false_v<typename P::T>, "Not predefined parameter!");
    }

    template <class P>
    void emplacecbsk(const SecretKey& sk)
    {
        if constexpr (std::is_same_v<P, lvl1param>) {
            cbsklvl1 =
                std::make_unique_for_overwrite<CBswitchingKey<lvl1param>>();
            for (int i = 0; i < lvl1param::k; i++) {
                Polynomial<P> partkey;
                for (int j = 0; j < P::n; j++)
                    partkey[j] = -sk.key.get<P>()[i * P::n + j];
                (*cbsklvl1)[i] =
                    trgswfftSymEncrypt<P>(partkey, sk.key.get<P>());
            }
        }
        else if constexpr (std::is_same_v<P, lvl2param>) {
            cbsklvl2 =
                std::make_unique_for_overwrite<CBswitchingKey<lvl2param>>();
            for (int i = 0; i < lvl2param::k; i++) {
                Polynomial<P> partkey;
                for (int j = 0; j < P::n; j++)
                    partkey[j] = -sk.key.get<P>()[i * P::n + j];
                (*cbsklvl2)[i] =
                    trgswfftSymEncrypt<P>(partkey, sk.key.get<P>());
            }
        }
        else
            static_assert(false_v<typename P::T>, "Not predefined parameter!");
    }

    // get keys
    template <class P>
    BootstrappingKey<P>& getbk() const
    {
        if constexpr (std::is_same_v<P, lvl01param> ||
                      std::is_same_v<P, lvl0Mparam>) {
            return *bklvl01;
        }
        else if constexpr (std::is_same_v<P, lvlh1param>) {
            return *bklvlh1;
        }
        else if constexpr (std::is_same_v<P, lvl02param>) {
            return *bklvl02;
        }
        else if constexpr (std::is_same_v<P, lvlh2param>) {
            return *bklvlh2;
        }
        else
            static_assert(false_v<typename P::T>, "Not predefined parameter!");
    }
    template <class P>
    BootstrappingKeyFFT<P>& getbkfft() const
    {
        if constexpr (std::is_same_v<P, lvl01param> ||
                      std::is_same_v<P, lvl0Mparam>) {
            return *bkfftlvl01;
        }
        else if constexpr (std::is_same_v<P, lvlh1param>) {
            return *bkfftlvlh1;
        }
        else if constexpr (std::is_same_v<P, lvl02param>) {
            return *bkfftlvl02;
        }
        else if constexpr (std::is_same_v<P, lvlh2param>) {
            return *bkfftlvlh2;
        }
        else
            static_assert(false_v<typename P::T>, "Not predefined parameter!");
    }
    template <class P>
    BootstrappingKeyNTT<P>& getbkntt() const
    {
        if constexpr (std::is_same_v<P, lvl01param> ||
                      std::is_same_v<P, lvl0Mparam>) {
            return *bknttlvl01;
        }
        else if constexpr (std::is_same_v<P, lvlh1param>) {
            return *bknttlvlh1;
        }
        else if constexpr (std::is_same_v<P, lvl02param>) {
            return *bknttlvl02;
        }
        else if constexpr (std::is_same_v<P, lvlh2param>) {
            return *bknttlvlh2;
        }
        else
            static_assert(false_v<typename P::T>, "Not predefined parameter!");
    }
    template <class P>
    KeySwitchingKey<P>& getiksk() const
    {
        if constexpr (std::is_same_v<P, lvl10param> ||
                      std::is_same_v<P, lvlM0param>) {
            return *iksklvl10;
        }
        else if constexpr (std::is_same_v<P, lvl1hparam>) {
            return *iksklvl1h;
        }
        else if constexpr (std::is_same_v<P, lvl20param>) {
            return *iksklvl20;
        }
        else if constexpr (std::is_same_v<P, lvl2hparam>) {
            return *iksklvl2h;
        }
        else if constexpr (std::is_same_v<P, lvl21param>) {
            return *iksklvl21;
        }
        else if constexpr (std::is_same_v<P, lvl22param>) {
            return *iksklvl22;
        }
        else if constexpr (std::is_same_v<P, lvl31param>) {
            return *iksklvl31;
        }
        else
            static_assert(false_v<typename P::T>, "Not predefined parameter!");
    }
    template <class P>
    SubsetKeySwitchingKey<P>& getsubiksk() const
    {
        if constexpr (std::is_same_v<P, lvl21param>) {
            return *subiksklvl21;
        }
        else
            static_assert(false_v<typename P::T>, "Not predefined parameter!");
    }
    template <class P>
    PrivateKeySwitchingKey<P>& getprivksk(const std::string& key) const
    {
        if constexpr (std::is_same_v<P, lvl11param>) {
            return *(privksklvl11.at(key));
        }
        else if constexpr (std::is_same_v<P, lvl21param>) {
            return *(privksklvl21.at(key));
        }
        else if constexpr (std::is_same_v<P, lvl22param>) {
            return *(privksklvl22.at(key));
        }
        else
            static_assert(false_v<typename P::T>, "Not predefined parameter!");
    }
    template <class P>
    SubsetPrivateKeySwitchingKey<P>& getsubprivksk(const std::string& key) const
    {
        if constexpr (std::is_same_v<P, lvl21param>) {
            return *(subprivksklvl21.at(key));
        }
        else
            static_assert(false_v<typename P::targetP::T>,
                          "Not predefined parameter!");
    }
    template <class P>
    AnnihilateKey<P>& getahk() const
    {
        if constexpr (std::is_same_v<P, lvl1param>) {
            return *ahklvl1;
        }
        else if constexpr (std::is_same_v<P, lvl2param>) {
            return *ahklvl2;
        }
        else
            static_assert(false_v<typename P::T>, "Not predefined parameter!");
    }
    template <class P>
    CBswitchingKey<P>& getcbsk() const
    {
        if constexpr (std::is_same_v<P, lvl1param>) {
            return *cbsklvl1;
        }
        else if constexpr (std::is_same_v<P, lvl2param>) {
            return *cbsklvl2;
        }
        else
            static_assert(false_v<typename P::T>, "Not predefined parameter!");
    }
};

}  // namespace TFHEpp
