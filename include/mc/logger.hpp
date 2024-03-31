#pragma once

#ifndef NDEBUG
#    define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#else
#    define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_ERROR
#endif

#include <source_location>

#include <spdlog/logger.h>

#include "defines.hpp"

namespace logger
{
    using level = spdlog::level::level_enum;

    class LocationCapturingString
    {
    public:
        constexpr LocationCapturingString(  // NOLINT(google-explicit-constructor)
            char const* msg,
            std::source_location loc = std::source_location::current())
            : message(msg), location(loc)
        {
        }

        constexpr LocationCapturingString(  // NOLINT(google-explicit-constructor)
            std::string_view msg,
            std::source_location loc = std::source_location::current())
            : message(msg), location(loc)
        {
        }

        std::string_view message;
        std::source_location location;
    };

    class Logger
    {
    public:
        static auto get() -> std::shared_ptr<spdlog::logger> const& { return m_logger; }

        static void init();

    private:
        inline static std::shared_ptr<spdlog::logger> m_logger {};
    };

    template<level lvl, typename... Args>
    ALWAYS_INLINE constexpr void log(LocationCapturingString const& fmtstr, Args&&... args)
    {
        if constexpr (lvl < SPDLOG_ACTIVE_LEVEL)
        {
            return;
        }

        if constexpr (kDebug)
        {
            Logger::get()->log(
                spdlog::source_loc { std::string_view(fmtstr.location.file_name()).substr(std::size(ROOT_SOURCE_PATH)).data(),
                                     static_cast<int>(fmtstr.location.line()),
                                     nullptr },
                lvl,
                fmt::vformat(fmtstr.message, fmt::make_format_args(std::forward<Args>(args)...)));
        }
        else
        {
            Logger::get()->log(lvl, fmt::vformat(fmtstr.message, fmt::make_format_args(std::forward<Args>(args)...)));
        }
    }

    template<level lvl, typename... Args>
    ALWAYS_INLINE constexpr void logAt(std::source_location location, std::string_view const& fmtstr, Args&&... args)
    {
        if constexpr (lvl < SPDLOG_ACTIVE_LEVEL)
        {
            return;
        }

        if constexpr (kDebug)
        {
            Logger::get()->log(
                spdlog::source_loc { std::string_view(location.file_name()).substr(std::size(ROOT_SOURCE_PATH)).data(),
                                     static_cast<int>(location.line()),
                                     nullptr },
                lvl,
                fmt::vformat(fmtstr, fmt::make_format_args(std::forward<Args>(args)...)));
        }
        else
        {
            Logger::get()->log(lvl, fmt::vformat(fmtstr, fmt::make_format_args(std::forward<Args>(args)...)));
        }
    }

    template<typename... Args>
    ALWAYS_INLINE void trace(LocationCapturingString const& formatString, Args&&... args)
    {
        log<level::trace>(formatString, std::forward<Args>(args)...);
    }

    template<typename... Args>
    ALWAYS_INLINE void debug(LocationCapturingString const& formatString, Args&&... args)
    {
        log<level::debug>(formatString, std::forward<Args>(args)...);
    }

    template<typename... Args>
    ALWAYS_INLINE void info(LocationCapturingString const& formatString, Args&&... args)
    {
        log<level::info>(formatString, std::forward<Args>(args)...);
    }

    template<typename... Args>
    ALWAYS_INLINE void warn(LocationCapturingString const& formatString, Args&&... args)
    {
        log<level::warn>(formatString, std::forward<Args>(args)...);
    }

    template<typename... Args>
    ALWAYS_INLINE void error(LocationCapturingString const& formatString, Args&&... args)
    {
        log<level::err>(formatString, std::forward<Args>(args)...);
    }

    template<typename... Args>
    ALWAYS_INLINE void critical(LocationCapturingString const& formatString, Args&&... args)
    {
        log<level::critical>(formatString, std::forward<Args>(args)...);
    }
}  // namespace logger
