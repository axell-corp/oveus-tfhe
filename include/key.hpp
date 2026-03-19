#pragma once

#ifdef USE_RANDEN
#include <randen.h>
#endif

#include <algorithm>
#include <array>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/array.hpp>

#include "lweParams.hpp"
#include "params.hpp"
#include "random"
#include "utils.hpp"

namespace TFHEpp {
using namespace std;
struct lweKey {
    std::tuple<Key<lvl0param>, Key<lvlhalfparam>, Key<lvl1param>,
               Key<lvl2param>, Key<lvl3param>>
        keys;
    Key<lvl0param> &lvl0;
    Key<lvlhalfparam> &lvlhalf;
    Key<lvl1param> &lvl1;
    Key<lvl2param> &lvl2;
    Key<lvl3param> &lvl3;
    lweKey()
        : lvl0(std::get<Key<lvl0param>>(keys)),
          lvlhalf(std::get<Key<lvlhalfparam>>(keys)),
          lvl1(std::get<Key<lvl1param>>(keys)),
          lvl2(std::get<Key<lvl2param>>(keys)),
          lvl3(std::get<Key<lvl3param>>(keys))
    {
        std::uniform_int_distribution<int32_t> lvl0gen(
            lvl0param::key_value_min, lvl0param::key_value_max);
        std::uniform_int_distribution<int32_t> lvlhalfgen(
            lvlhalfparam::key_value_min, lvlhalfparam::key_value_max);
        std::uniform_int_distribution<int32_t> lvl1gen(
            lvl1param::key_value_min, lvl1param::key_value_max);
        std::uniform_int_distribution<int32_t> lvl2gen(
            lvl2param::key_value_min, lvl2param::key_value_max);
        for (typename lvl0param::T &i : std::get<Key<lvl0param>>(keys))
            i = lvl0gen(generator);
        for (typename lvlhalfparam::T &i : std::get<Key<lvlhalfparam>>(keys))
            i = lvlhalfgen(generator);
        for (typename lvl1param::T &i : std::get<Key<lvl1param>>(keys))
            i = lvl1gen(generator);
        for (typename lvl2param::T &i : std::get<Key<lvl2param>>(keys))
            i = lvl2gen(generator);
#ifdef USE_SUBSET_KEY
        for (int i = 0; i < lvl1param::k * lvl1param::n; i++)
            std::get<Key<lvl2param>>(keys)[i] =
                std::get<Key<lvl1param>>(keys)[i];
#endif
    }
    lweKey(const lweKey &other)
        : keys(other.keys),
          lvl0(std::get<Key<lvl0param>>(keys)),
          lvlhalf(std::get<Key<lvlhalfparam>>(keys)),
          lvl1(std::get<Key<lvl1param>>(keys)),
          lvl2(std::get<Key<lvl2param>>(keys)),
          lvl3(std::get<Key<lvl3param>>(keys))
    {
    }
    lweKey(lweKey &&other) noexcept
        : keys(std::move(other.keys)),
          lvl0(std::get<Key<lvl0param>>(keys)),
          lvlhalf(std::get<Key<lvlhalfparam>>(keys)),
          lvl1(std::get<Key<lvl1param>>(keys)),
          lvl2(std::get<Key<lvl2param>>(keys)),
          lvl3(std::get<Key<lvl3param>>(keys))
    {
    }
    lweKey &operator=(const lweKey &other)
    {
        keys = other.keys;
        return *this;
    }
    lweKey &operator=(lweKey &&other) noexcept
    {
        keys = std::move(other.keys);
        return *this;
    }
    template <class P>
    Key<P> get() const
    {
#ifdef ENABLE_AXELL
        if constexpr (std::is_same_v<P, lvlMparam>)
            return Key<P>(std::get<Key<lvl1param>>(keys));
        else
#endif
            return std::get<Key<P>>(keys);
    }
};

struct SecretKey {
    lweKey key;
    lweParams params;

    template <class Archive>
    void serialize(Archive &archive)
    {
        archive(key.keys, params);
    }
};
}  // namespace TFHEpp
