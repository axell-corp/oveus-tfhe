#pragma once

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <iostream>
#include <type_traits>
#include <tuple>

#include "evalkeygens.hpp"

namespace TFHEpp {

struct EvalKey {
    lweParams params;

    static constexpr std::size_t BkBaseCount = 4;
#ifdef ENABLE_AXELL
    static constexpr std::size_t BkAxellCount = 1;
#else
    static constexpr std::size_t BkAxellCount = 0;
#endif
#ifdef USE_DIFFERENT_BR_PARAM
    static constexpr std::size_t BrParamExtraCount = 2;
#else
    static constexpr std::size_t BrParamExtraCount = 0;
#endif
    static constexpr std::size_t BkAxellIndex = BkBaseCount;
    static constexpr std::size_t BkFFTBaseIndex =
        BkBaseCount + BkAxellCount + BrParamExtraCount;
    static constexpr std::size_t BkFFTAxellIndex = BkFFTBaseIndex + BkBaseCount;
    static constexpr std::size_t BkNTTBaseIndex =
        BkFFTBaseIndex + BkBaseCount + BkAxellCount + BrParamExtraCount;
    static constexpr std::size_t BkNTTAxellIndex = BkNTTBaseIndex + BkBaseCount;

    template <class P>
    auto& getbkptr()
    {
        if constexpr (std::is_same_v<P, lvl01param>)
            return std::get<0>(keys);
        else if constexpr (std::is_same_v<P, lvlh1param>)
            return std::get<1>(keys);
        else if constexpr (std::is_same_v<P, lvl02param>)
            return std::get<2>(keys);
        else if constexpr (std::is_same_v<P, lvlh2param>)
            return std::get<3>(keys);
#ifdef ENABLE_AXELL
        else if constexpr (std::is_same_v<P, lvl0Mparam>)
        {
            if constexpr (std::is_same_v<BootstrappingKey<lvl0Mparam>,
                                         BootstrappingKey<lvl01param>>)
                return std::get<0>(keys);
            else
                return std::get<BkAxellIndex>(keys);
        }
#endif
        else if constexpr (std::is_same_v<P, cblvl02param>)
        {
            if constexpr (std::is_same_v<BootstrappingKey<cblvl02param>,
                                         BootstrappingKey<lvl02param>>)
                return std::get<2>(keys);
            else
                return std::get<BkAxellIndex + BkAxellCount>(keys);
        }
        else if constexpr (std::is_same_v<P, cblvlh2param>)
        {
            if constexpr (std::is_same_v<BootstrappingKey<cblvlh2param>,
                                         BootstrappingKey<lvlh2param>>)
                return std::get<3>(keys);
            else
                return std::get<BkAxellIndex + BkAxellCount + 1>(keys);
        }
        else
            static_assert(!sizeof(P), "Unsupported bootstrapping key param");
    }

    template <class P>
    auto& getbkfftptr()
    {
        if constexpr (std::is_same_v<P, lvl01param>)
            return std::get<BkFFTBaseIndex + 0>(keys);
        else if constexpr (std::is_same_v<P, lvlh1param>)
            return std::get<BkFFTBaseIndex + 1>(keys);
        else if constexpr (std::is_same_v<P, lvl02param>)
            return std::get<BkFFTBaseIndex + 2>(keys);
        else if constexpr (std::is_same_v<P, lvlh2param>)
            return std::get<BkFFTBaseIndex + 3>(keys);
#ifdef ENABLE_AXELL
        else if constexpr (std::is_same_v<P, lvl0Mparam>)
        {
            if constexpr (std::is_same_v<BootstrappingKeyFFT<lvl0Mparam>,
                                         BootstrappingKeyFFT<lvl01param>>)
                return std::get<BkFFTBaseIndex + 0>(keys);
            else
                return std::get<BkFFTAxellIndex>(keys);
        }
#endif
        else if constexpr (std::is_same_v<P, cblvl02param>)
        {
            if constexpr (std::is_same_v<BootstrappingKeyFFT<cblvl02param>,
                                         BootstrappingKeyFFT<lvl02param>>)
                return std::get<BkFFTBaseIndex + 2>(keys);
            else
                return std::get<BkFFTAxellIndex + BkAxellCount>(keys);
        }
        else if constexpr (std::is_same_v<P, cblvlh2param>)
        {
            if constexpr (std::is_same_v<BootstrappingKeyFFT<cblvlh2param>,
                                         BootstrappingKeyFFT<lvlh2param>>)
                return std::get<BkFFTBaseIndex + 3>(keys);
            else
                return std::get<BkFFTAxellIndex + BkAxellCount + 1>(keys);
        }
        else
            static_assert(!sizeof(P), "Unsupported bootstrapping key FFT param");
    }

    template <class P>
    auto& getbknttptr()
    {
        if constexpr (std::is_same_v<P, lvl01param>)
            return std::get<BkNTTBaseIndex + 0>(keys);
        else if constexpr (std::is_same_v<P, lvlh1param>)
            return std::get<BkNTTBaseIndex + 1>(keys);
        else if constexpr (std::is_same_v<P, lvl02param>)
            return std::get<BkNTTBaseIndex + 2>(keys);
        else if constexpr (std::is_same_v<P, lvlh2param>)
            return std::get<BkNTTBaseIndex + 3>(keys);
#ifdef ENABLE_AXELL
        else if constexpr (std::is_same_v<P, lvl0Mparam>)
        {
            if constexpr (std::is_same_v<BootstrappingKeyNTT<lvl0Mparam>,
                                         BootstrappingKeyNTT<lvl01param>>)
                return std::get<BkNTTBaseIndex + 0>(keys);
            else
                return std::get<BkNTTAxellIndex>(keys);
        }
#endif
        else if constexpr (std::is_same_v<P, cblvl02param>)
        {
            if constexpr (std::is_same_v<BootstrappingKeyNTT<cblvl02param>,
                                         BootstrappingKeyNTT<lvl02param>>)
                return std::get<BkNTTBaseIndex + 2>(keys);
            else
                return std::get<BkNTTAxellIndex + BkAxellCount>(keys);
        }
        else if constexpr (std::is_same_v<P, cblvlh2param>)
        {
            if constexpr (std::is_same_v<BootstrappingKeyNTT<cblvlh2param>,
                                         BootstrappingKeyNTT<lvlh2param>>)
                return std::get<BkNTTBaseIndex + 3>(keys);
            else
                return std::get<BkNTTAxellIndex + BkAxellCount + 1>(keys);
        }
        else
            static_assert(!sizeof(P), "Unsupported bootstrapping key NTT param");
    }

    std::tuple<
        std::shared_ptr<BootstrappingKey<lvl01param>>,
        std::shared_ptr<BootstrappingKey<lvlh1param>>,
        std::shared_ptr<BootstrappingKey<lvl02param>>,
        std::shared_ptr<BootstrappingKey<lvlh2param>>,
#ifdef ENABLE_AXELL
        std::shared_ptr<BootstrappingKey<lvl0Mparam>>,
#endif
#ifdef USE_DIFFERENT_BR_PARAM
        std::shared_ptr<BootstrappingKey<cblvl02param>>,
        std::shared_ptr<BootstrappingKey<cblvlh2param>>,
#endif
        std::shared_ptr<BootstrappingKeyFFT<lvl01param>>,
        std::shared_ptr<BootstrappingKeyFFT<lvlh1param>>,
        std::shared_ptr<BootstrappingKeyFFT<lvl02param>>,
        std::shared_ptr<BootstrappingKeyFFT<lvlh2param>>,
#ifdef ENABLE_AXELL
        std::shared_ptr<BootstrappingKeyFFT<lvl0Mparam>>,
#endif
#ifdef USE_DIFFERENT_BR_PARAM
        std::shared_ptr<BootstrappingKeyFFT<cblvl02param>>,
        std::shared_ptr<BootstrappingKeyFFT<cblvlh2param>>,
#endif
        std::shared_ptr<BootstrappingKeyNTT<lvl01param>>,
        std::shared_ptr<BootstrappingKeyNTT<lvlh1param>>,
        std::shared_ptr<BootstrappingKeyNTT<lvl02param>>,
        std::shared_ptr<BootstrappingKeyNTT<lvlh2param>>,
#ifdef ENABLE_AXELL
        std::shared_ptr<BootstrappingKeyNTT<lvl0Mparam>>,
#endif
#ifdef USE_DIFFERENT_BR_PARAM
        std::shared_ptr<BootstrappingKeyNTT<cblvl02param>>,
        std::shared_ptr<BootstrappingKeyNTT<cblvlh2param>>,
#endif
        std::shared_ptr<KeySwitchingKey<lvl10param>>,
        std::shared_ptr<KeySwitchingKey<lvl1hparam>>,
        std::shared_ptr<KeySwitchingKey<lvl20param>>,
        std::shared_ptr<KeySwitchingKey<lvl2hparam>>,
        std::shared_ptr<KeySwitchingKey<lvl21param>>,
        std::shared_ptr<KeySwitchingKey<lvl22param>>,
        std::shared_ptr<KeySwitchingKey<lvl31param>>,
        std::shared_ptr<SubsetKeySwitchingKey<lvl21param>>,
        std::unordered_map<std::string, std::shared_ptr<PrivateKeySwitchingKey<
                                            lvl11param>>>,
        std::unordered_map<std::string, std::shared_ptr<PrivateKeySwitchingKey<
                                            lvl21param>>>,
        std::unordered_map<std::string, std::shared_ptr<PrivateKeySwitchingKey<
                                            lvl22param>>>,
        std::unordered_map<
            std::string,
            std::shared_ptr<SubsetPrivateKeySwitchingKey<lvl21param>>>,
        std::shared_ptr<AnnihilateKey<AHlvl1param>>,
        std::shared_ptr<AnnihilateKey<AHlvl2param>>,
#ifdef USE_DIFFERENT_AH_PARAM
        std::shared_ptr<AnnihilateKey<cbAHlvl2param>>,
#endif
        std::shared_ptr<CBswitchingKey<AHlvl1param>>,
        std::shared_ptr<CBswitchingKey<AHlvl2param>>
#ifdef USE_DIFFERENT_AH_PARAM
        ,
        std::shared_ptr<CBswitchingKey<cbAHlvl2param>>
#endif
        >
        keys;

    EvalKey(SecretKey sk) { params = sk.params; }
    EvalKey() {}

    template <typename T>
    auto& get()
    {
        return std::get<std::shared_ptr<T>>(keys);
    }

    template <typename T>
    auto& get_map()
    {
        return std::get<std::unordered_map<std::string, std::shared_ptr<T>>>(
            keys);
    }

    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(params, keys);
    }

    template <class P>
    void emplacebk(const SecretKey& sk)
    {
        auto& bk = getbkptr<P>();
        if (bk != nullptr) return;
        bk = std::make_unique_for_overwrite<BootstrappingKey<P>>();
        bkgen<P>(*bk, sk);
    }

    template <class P>
    void emplacebkfft(const SecretKey& sk)
    {
        auto& bkfft = getbkfftptr<P>();
        if (bkfft != nullptr) return;
        bkfft = std::make_unique_for_overwrite<BootstrappingKeyFFT<P>>();
        bkfftgen<P>(*bkfft, sk);
    }

    template <class P>
    void emplacebkntt(const SecretKey& sk)
    {
        auto& bkntt = getbknttptr<P>();
        if (bkntt != nullptr) return;
        bkntt = std::make_unique_for_overwrite<BootstrappingKeyNTT<P>>();
        bknttgen<P>(*bkntt, sk);
    }

    template <class P>
    void emplacebk2bkfft()
    {
        auto& bk = getbkptr<P>();
        auto& bkfft = getbkfftptr<P>();
        if (bkfft != nullptr) return;
        bkfft = std::make_unique_for_overwrite<BootstrappingKeyFFT<P>>();
        for (int i = 0; i < P::domainP::n; i++)
            (*bkfft)[i][0] =
                ApplyFFT2trgsw<typename P::targetP>((*bk)[i][0]);
    }

    template <class P>
    void emplacebk2bkntt()
    {
        auto& bk = getbkptr<P>();
        auto& bkntt = getbknttptr<P>();
        if (bkntt != nullptr) return;
        bkntt = std::make_unique_for_overwrite<BootstrappingKeyNTT<P>>();
        for (int i = 0; i < P::domainP::n; i++)
            (*bkntt)[i] =
                ApplyNTT2trgsw<typename P::targetP>((*bk)[i][0]);
    }

    template <class P>
    void emplaceiksk(const SecretKey& sk)
    {
        if (get<KeySwitchingKey<P>>() != nullptr) return;
        get<KeySwitchingKey<P>>() =
            std::make_unique_for_overwrite<KeySwitchingKey<P>>();
        ikskgen<P>(*get<KeySwitchingKey<P>>(), sk);
    }

    template <class P>
    void emplacesubiksk(const SecretKey& sk)
    {
        if (get<SubsetKeySwitchingKey<P>>() != nullptr) return;
        get<SubsetKeySwitchingKey<P>>() =
            std::make_unique_for_overwrite<SubsetKeySwitchingKey<P>>();
        subikskgen<P>(*get<SubsetKeySwitchingKey<P>>(), sk);
    }

    template <class P>
    void emplaceprivksk(const std::string& key,
                        const Polynomial<typename P::targetP>& func,
                        const SecretKey& sk)
    {
        if (get_map<PrivateKeySwitchingKey<P>>().find(key) !=
            get_map<PrivateKeySwitchingKey<P>>().end())
            return;
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
        if (get_map<SubsetPrivateKeySwitchingKey<P>>().find(key) !=
            get_map<SubsetPrivateKeySwitchingKey<P>>().end())
            return;
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
        if (get<AnnihilateKey<P>>() != nullptr) return;
        get<AnnihilateKey<P>>() =
            std::make_unique_for_overwrite<AnnihilateKey<P>>();
        annihilatekeygen<P>(*get<AnnihilateKey<P>>(), sk);
    }

    template <class P>
    void emplacecbsk(const SecretKey& sk)
    {
        if (get<CBswitchingKey<P>>() != nullptr) return;
        get<CBswitchingKey<P>>() =
            std::make_unique_for_overwrite<CBswitchingKey<P>>();
        for (int i = 0; i < P::k; i++) {
            Polynomial<P> partkey;
            for (int j = 0; j < P::n; j++)
                partkey[j] = -sk.key.get<P>()[i * P::n + j];
            trgswSymEncrypt<P>((*get<CBswitchingKey<P>>())[i], partkey,
                               sk.key.get<P>());
        }
    }

    template <class P>
    BootstrappingKey<P>& getbk() const
    {
        return *const_cast<EvalKey*>(this)->getbkptr<P>();
    }

    template <class P>
    BootstrappingKeyFFT<P>& getbkfft() const
    {
        return *const_cast<EvalKey*>(this)->getbkfftptr<P>();
    }

    template <class P>
    BootstrappingKeyNTT<P>& getbkntt() const
    {
        return *const_cast<EvalKey*>(this)->getbknttptr<P>();
    }

    template <class P>
    KeySwitchingKey<P>& getiksk() const
    {
        return *const_cast<EvalKey*>(this)->get<KeySwitchingKey<P>>();
    }

    template <class P>
    SubsetKeySwitchingKey<P>& getsubiksk() const
    {
        return *const_cast<EvalKey*>(this)->get<SubsetKeySwitchingKey<P>>();
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
        return *const_cast<EvalKey*>(this)->get<AnnihilateKey<P>>();
    }

    template <class P>
    CBswitchingKey<P>& getcbsk() const
    {
        return *const_cast<EvalKey*>(this)->get<CBswitchingKey<P>>();
    }
};

}  // namespace TFHEpp
