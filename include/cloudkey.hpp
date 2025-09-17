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

    // Tuple containing all keys
    std::tuple<
        // BootstrappingKey
        std::shared_ptr<BootstrappingKey<lvl01param>>,  // 0
        std::shared_ptr<BootstrappingKey<lvlh1param>>,  // 1
        std::shared_ptr<BootstrappingKey<lvl02param>>,  // 2
        std::shared_ptr<BootstrappingKey<lvlh2param>>,  // 3
#ifdef USE_DIFFERENT_BR_PARAM
        std::shared_ptr<BootstrappingKey<cblvl02param>>,
        std::shared_ptr<BootstrappingKey<cblvlh2param>>,
#endif
        // BootstrappingKeyFFT
        std::shared_ptr<BootstrappingKeyFFT<lvl01param>>,  // 4
        std::shared_ptr<BootstrappingKeyFFT<lvlh1param>>,  // 5
        std::shared_ptr<BootstrappingKeyFFT<lvl02param>>,  // 6
        std::shared_ptr<BootstrappingKeyFFT<lvlh2param>>,  // 7
#ifdef USE_DIFFERENT_BR_PARAM
        std::shared_ptr<BootstrappingKeyFFT<cblvl02param>>,  // 6
        std::shared_ptr<BootstrappingKeyFFT<cblvlh2param>>,  // 7
#endif
        // BootstrappingKeyNTT
        std::shared_ptr<BootstrappingKeyNTT<lvl01param>>,  // 8
        std::shared_ptr<BootstrappingKeyNTT<lvlh1param>>,  // 9
        std::shared_ptr<BootstrappingKeyNTT<lvl02param>>,  // 10
        std::shared_ptr<BootstrappingKeyNTT<lvlh2param>>,  // 11
#ifdef USE_DIFFERENT_BR_PARAM
        std::shared_ptr<BootstrappingKeyNTT<cblvl02param>>,  // 10
        std::shared_ptr<BootstrappingKeyNTT<cblvlh2param>>,  // 11
#endif
        // KeySwitchingKey
        std::shared_ptr<KeySwitchingKey<lvl10param>>,  // 12
        std::shared_ptr<KeySwitchingKey<lvl1hparam>>,  // 13
        std::shared_ptr<KeySwitchingKey<lvl20param>>,  // 14
        std::shared_ptr<KeySwitchingKey<lvl2hparam>>,  // 15
        std::shared_ptr<KeySwitchingKey<lvl21param>>,  // 16
        std::shared_ptr<KeySwitchingKey<lvl22param>>,  // 17
        std::shared_ptr<KeySwitchingKey<lvl31param>>,  // 18
        // SubsetKeySwitchingKey
        std::shared_ptr<SubsetKeySwitchingKey<lvl21param>>,  // 19
        // PrivateKeySwitchingKey
        std::unordered_map<std::string, std::shared_ptr<PrivateKeySwitchingKey<
                                            lvl11param>>>,  // 20
        std::unordered_map<std::string, std::shared_ptr<PrivateKeySwitchingKey<
                                            lvl21param>>>,  // 21
        std::unordered_map<std::string, std::shared_ptr<PrivateKeySwitchingKey<
                                            lvl22param>>>,  // 22
        // SubsetPrivateKeySwitchingKey
        std::unordered_map<
            std::string,
            std::shared_ptr<SubsetPrivateKeySwitchingKey<lvl21param>>>,  // 23
        // AnnihilateKey
        std::shared_ptr<AnnihilateKey<AHlvl1param>>,  // 24
        std::shared_ptr<AnnihilateKey<AHlvl2param>>,  // 25
#ifdef USE_DIFFERENT_AH_PARAM
        std::shared_ptr<AnnihilateKey<cbAHlvl2param>>,  // 25
#endif
        // CBswitchingKey
        std::shared_ptr<CBswitchingKey<AHlvl1param>>,  // 26
        std::shared_ptr<CBswitchingKey<AHlvl2param>>   // 27
#ifdef USE_DIFFERENT_AH_PARAM
        ,
        std::shared_ptr<CBswitchingKey<cbAHlvl2param>>  // 27
#endif
        >
        keys;

    EvalKey(SecretKey sk) { params = sk.params; }
    EvalKey() {}

    // Helper function for cleaner tuple access
    template <typename T>
    auto& get()
    {
        return std::get<std::shared_ptr<T>>(keys);
    }

    // Special overload for unordered_map types
    template <typename T>
    auto& get_map()
    {
        return std::get<std::unordered_map<std::string, std::shared_ptr<T>>>(
            keys);
    }

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
        get<BootstrappingKey<P>>() =
            std::make_unique_for_overwrite<BootstrappingKey<P>>();
        bkgen<P>(*get<BootstrappingKey<P>>(), sk);
    }
    template <class P>
    void emplacebkfft(const SecretKey& sk)
    {
        get<BootstrappingKeyFFT<P>>() =
            std::make_unique_for_overwrite<BootstrappingKeyFFT<P>>();
        bkfftgen<P>(*get<BootstrappingKeyFFT<P>>(), sk);
    }
    template <class P>
    void emplacebkntt(const SecretKey& sk)
    {
        get<BootstrappingKeyNTT<P>>() =
            std::make_unique_for_overwrite<BootstrappingKeyNTT<P>>();
        bknttgen<P>(*get<BootstrappingKeyNTT<P>>(), sk);
    }
    template <class P>
    void emplacebk2bkfft()
    {
        get<BootstrappingKeyFFT<P>>() =
            std::make_unique_for_overwrite<BootstrappingKeyFFT<P>>();
        for (int i = 0; i < P::domainP::n; i++)
            (*get<BootstrappingKeyFFT<P>>())[i][0] =
                ApplyFFT2trgsw<typename P::targetP>(
                    (*get<BootstrappingKey<P>>())[i][0]);
    }
    template <class P>
    void emplacebk2bkntt()
    {
        get<BootstrappingKeyNTT<P>>() =
            std::make_unique_for_overwrite<BootstrappingKeyNTT<P>>();
        for (int i = 0; i < P::domainP::n; i++)
            (*get<BootstrappingKeyNTT<P>>())[i] =
                ApplyNTT2trgsw<typename P::targetP>(
                    (*get<BootstrappingKey<P>>())[i][0]);
    }
    template <class P>
    void emplaceiksk(const SecretKey& sk)
    {
        get<KeySwitchingKey<P>>() =
            std::make_unique_for_overwrite<KeySwitchingKey<P>>();
        ikskgen<P>(*get<KeySwitchingKey<P>>(), sk);
    }
    template <class P>
    void emplacesubiksk(const SecretKey& sk)
    {
        get<SubsetKeySwitchingKey<P>>() =
            std::make_unique_for_overwrite<SubsetKeySwitchingKey<P>>();
        subikskgen<P>(*get<SubsetKeySwitchingKey<P>>(), sk);
    }
    template <class P>
    void emplaceprivksk(const std::string& key,
                        const Polynomial<typename P::targetP>& func,
                        const SecretKey& sk)
    {
        get_map<PrivateKeySwitchingKey<P>>()[key] =
            std::unique_ptr<PrivateKeySwitchingKey<P>>(
                new (std::align_val_t(64)) PrivateKeySwitchingKey<P>());
        privkskgen<P>(*get_map<PrivateKeySwitchingKey<P>>()[key], func, sk);
    }
    template <class P>
    void emplacesubprivksk(const std::string& key,
                           const Polynomial<typename P::targetP>& func,
                           const SecretKey& sk)
    {
        get_map<SubsetPrivateKeySwitchingKey<P>>()[key] =
            std::make_unique_for_overwrite<SubsetPrivateKeySwitchingKey<P>>();
        subprivkskgen<P>(*get_map<SubsetPrivateKeySwitchingKey<P>>()[key], func,
                         sk);
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
        get<AnnihilateKey<P>>() =
            std::make_unique_for_overwrite<AnnihilateKey<P>>();
        annihilatekeygen<P>(*get<AnnihilateKey<P>>(), sk);
    }

    template <class P>
    void emplacecbsk(const SecretKey& sk)
    {
        get<CBswitchingKey<P>>() =
            std::make_unique_for_overwrite<CBswitchingKey<P>>();
        for (int i = 0; i < P::k; i++) {
            Polynomial<P> partkey;
            for (int j = 0; j < P::n; j++)
                partkey[j] = -sk.key.get<P>()[i * P::n + j];
            (*get<CBswitchingKey<P>>())[i] =
                trgswfftSymEncrypt<P>(partkey, sk.key.get<P>());
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
        return *(
            const_cast<EvalKey*>(this)->get_map<PrivateKeySwitchingKey<P>>().at(
                key));
    }
    template <class P>
    SubsetPrivateKeySwitchingKey<P>& getsubprivksk(const std::string& key) const
    {
        return *(const_cast<EvalKey*>(this)
                     ->get_map<SubsetPrivateKeySwitchingKey<P>>()
                     .at(key));
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
