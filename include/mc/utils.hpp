#pragma once

#include <cassert>
#include <cstdint>
#include <functional>

namespace Utils
{
    template<typename Class, typename Ret, typename... Args>
    auto captureThis(Ret (Class::*func)(Args...), Class* instance)
        -> std::function<Ret(Args...)>
    {
        return [func, instance](Args... args)
        {
            return (instance->*func)(args...);
        };
    }

    template<typename T>
    concept HasSize = requires(T const& t) {
        {
            t.size()
        } -> std::convertible_to<size_t>;
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

    template<typename ReturnType, typename MemberType, typename ClassType>
    auto member_offset(MemberType ClassType::*member) -> ReturnType
    {
#pragma GCC diagnostic ignored "-Wreturn-type"
        return static_cast<ReturnType>(
            reinterpret_cast<uint64_t>(reinterpret_cast<char const volatile*>(
                &((reinterpret_cast<ClassType*>(0))->*member))));
    }
}  // namespace Utils