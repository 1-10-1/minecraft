#pragma once

#ifndef NDEBUG
#    define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#else
#    define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
#endif

#include <format>
#include <source_location>

#include <spdlog/logger.h>

#include "defines.hpp"

namespace logger
{
    using level = spdlog::level::level_enum;

    template<typename... Args>
    class LocationCapturingStringImpl
    {
    public:
        consteval LocationCapturingStringImpl(  // NOLINT(google-explicit-constructor)
            char const* msg,
            std::source_location loc = std::source_location::current())
            : message { msg }, location { loc }
        {
        }

        std::format_string<Args...> const message;
        std::source_location const location;
    };

    template<typename... Args>
    using LocationCapturingString = std::type_identity_t<LocationCapturingStringImpl<Args...>>;

    constexpr auto simplifyFunctionSignature(std::string const& sig) -> std::string;

    class Logger
    {
    public:
        static auto get() -> std::shared_ptr<spdlog::logger> const& { return m_logger; }

        static void init();

    private:
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
        inline static std::shared_ptr<spdlog::logger> m_logger {};
    };

    template<level lvl, typename... Args>
    ALWAYS_INLINE constexpr void log(LocationCapturingString<Args...> const& fmtstr, Args&&... args)
    {
        if constexpr (lvl < SPDLOG_ACTIVE_LEVEL)
        {
            return;
        }

        if constexpr (kDebug)
        {
            Logger::get()->log(
                spdlog::source_loc(
                    std::string_view(fmtstr.location.file_name()).substr(std::size(ROOT_SOURCE_PATH)).data(),
                    static_cast<int>(fmtstr.location.line()),
                    fmtstr.location.function_name()),
                lvl,
                std::format(fmtstr.message, std::forward<Args>(args)...));
        }
        else
        {
            Logger::get()->log(lvl, std::format(fmtstr.message, std::forward<Args>(args)...));
        }
    }

    template<level lvl, typename... Args>
    ALWAYS_INLINE constexpr void
    logAt(spdlog::source_loc location, std::format_string<Args...> const& fmtstr, Args&&... args)
    {
        if constexpr (lvl < SPDLOG_ACTIVE_LEVEL)
        {
            return;
        }

        if constexpr (kDebug)
        {
            if (std::string_view filename = location.filename; filename != "")
            {
                location.funcname = location.funcname;
                location.filename = filename.substr(std::size(ROOT_SOURCE_PATH)).data();
            }
            Logger::get()->log(location, lvl, std::format(fmtstr, std::forward<Args>(args)...));
        }
        else
        {
            Logger::get()->log(lvl, std::format(fmtstr, std::forward<Args>(args)...));
        }
    }

    template<level lvl, typename... Args>
    ALWAYS_INLINE constexpr void
    logAt(std::source_location location, std::format_string<Args...> const& fmtstr, Args&&... args)
    {
        if constexpr (lvl < SPDLOG_ACTIVE_LEVEL)
        {
            return;
        }

        logAt<lvl>(spdlog::source_loc(
                       location.file_name(), static_cast<int>(location.line()), location.function_name()),
                   fmtstr,
                   std::forward<Args>(args)...);
    }

    template<typename... Args>
    ALWAYS_INLINE void trace(LocationCapturingString<Args...> const& formatString, Args&&... args)
    {
        log<level::trace>(formatString, std::forward<Args>(args)...);
    }

    template<typename... Args>
    ALWAYS_INLINE void debug(LocationCapturingString<Args...> const& formatString, Args&&... args)
    {
        log<level::debug>(formatString, std::forward<Args>(args)...);
    }

    template<typename... Args>
    ALWAYS_INLINE void info(LocationCapturingString<Args...> const& formatString, Args&&... args)
    {
        log<level::info>(formatString, std::forward<Args>(args)...);
    }

    template<typename... Args>
    ALWAYS_INLINE void warn(LocationCapturingString<Args...> const& formatString, Args&&... args)
    {
        log<level::warn>(formatString, std::forward<Args>(args)...);
    }

    template<typename... Args>
    ALWAYS_INLINE void error(LocationCapturingString<Args...> const& formatString, Args&&... args)
    {
        log<level::err>(formatString, std::forward<Args>(args)...);
    }

    template<typename... Args>
    ALWAYS_INLINE void critical(LocationCapturingString<Args...> const& formatString, Args&&... args)
    {
        log<level::critical>(formatString, std::forward<Args>(args)...);
    }
}  // namespace logger
