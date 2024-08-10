#pragma once

#include "vk_result_messages.hpp"
#include <mc/asserts.hpp>
#include <mc/defines.hpp>

#include <source_location>

#include <vulkan/vulkan_raii.hpp>

namespace renderer::backend
{
    class ResultChecker
    {
    public:
        constexpr ResultChecker(std::source_location loc = std::source_location::current()) : loc { loc } {}

        std::source_location loc;
    };

    inline void operator>>(vk::Result res, ResultChecker token)
    {
        if constexpr (!kDebug)
        {
            return;
        }

        MC_ASSERT_MSG_LOC(token.loc, res == vk::Result::eSuccess, "{}", vkResultToStr(res));
    }

    inline void operator>>(VkResult res, ResultChecker token)
    {
        static_cast<vk::Result>(res) >> token;
    }

    template<typename T>
    T operator>>(vk::ResultValue<T>&& res, ResultChecker token)
    {
        if constexpr (!kDebug)
        {
            return res.value;
        }

        MC_ASSERT_MSG_LOC(token.loc, res.result == vk::Result::eSuccess, "{}", vkResultToStr(res.result));

        return res.value;
    }

    template<typename T>
    T operator>>(std::expected<T, vk::Result>&& res, ResultChecker token)
    {
        if constexpr (!kDebug)
        {
            return std::move(res.value());
        }

        MC_ASSERT_MSG_LOC(token.loc, res.has_value(), "{}", vkResultToStr(res.error()));

        return std::move(res.value());
    }
}  // namespace renderer::backend
