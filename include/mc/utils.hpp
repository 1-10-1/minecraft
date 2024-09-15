#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <vector>

#include "asserts.hpp"

namespace utils
{
    inline auto largeNumToHumanReadable(float num) -> std::string
    {
        std::array suffixes {
            "", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion"
        };

        size_t i = 0;

        while (num >= 1000.0f && i < suffixes.size() - 1)
        {
            num /= 1000.0f;
            ++i;
        }

        return std::format("{:.2f} {}", num, suffixes[i]);
    }

    template<typename SizeType = char>
    inline auto readBytes(std::filesystem::path const& filepath) -> std::vector<SizeType>
    {
        std::ifstream file(filepath, std::ios::ate | std::ios::binary);

        MC_ASSERT_MSG(file.is_open(), "Failed to read file '{}'", filepath.string());

        auto fileSize = file.tellg();
        std::vector<SizeType> buffer(static_cast<std::size_t>(fileSize));

        file.seekg(0);
        file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

        file.close();

        return buffer;
    }

    inline auto readFileIntoString(std::filesystem::path const& path) -> std::string
    {
        std::ifstream file(path, std::ios::ate | std::ios::binary);

        MC_ASSERT_MSG(file.is_open(), "Failed to read file '{}'", path.string());

        auto fileSize = file.tellg();
        file.seekg(0);

        std::string buffer;
        buffer.resize(static_cast<std::size_t>(fileSize));

        file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

        file.close();

        return buffer;
    }

    template<typename Class, typename Ret, typename... Args>
    auto captureThis(Ret (Class::*func)(Args...), Class* instance) -> std::function<Ret(Args...)>
    {
        return [func, instance](Args... args)
        {
            return (instance->*func)(args...);
        };
    }

    template<typename T>
    concept HasSize = requires(T const& t) {
        { t.size() } -> std::convertible_to<size_t>;
    };

    template<typename SizeType = uint32_t, typename T>
        requires HasSize<T> || std::is_array_v<T>
    constexpr auto size(T const& arg)
    {
        if constexpr (HasSize<T>)
        {
            return static_cast<SizeType>(arg.size());
        }
        else
        {
            return static_cast<SizeType>(std::size(arg));
        }
    }

    template<typename ReturnType = uint32_t, typename MemberType, typename ClassType>
    constexpr auto memberOffset(MemberType ClassType::*member) -> ReturnType
    {
#pragma GCC diagnostic ignored "-Wreturn-type"
        return static_cast<ReturnType>(reinterpret_cast<uint64_t>(
            reinterpret_cast<char const volatile*>(&((reinterpret_cast<ClassType*>(0))->*member))));
    }
}  // namespace utils
