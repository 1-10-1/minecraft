#pragma once

#include <source_location>
#include <stdexcept>
#include <string>

#include <fmt/core.h>
#include <magic_enum.hpp>

#include "logger.hpp"

// NOLINTBEGIN

#if DEBUG
#    define MC_TRY   try
#    define MC_CATCH catch
#    define MC_THROW throw
#else

class LogErrorAndExit
{
public:
    LogErrorAndExit(std::exception const& err)
    {
        logger::error(err.what());
        exit(-1);
    }
};

/*
Prolly not the place for this but make a macro called DEBUG_BLOCK which is used by MC_CATCH
this macro is more general purpose for debug-only code
and the same for release but do a bit of thinking cause i haven't
*/
#    define MC_TRY        if constexpr (true)
#    define MC_CATCH(...) if constexpr (false)
#    define MC_THROW      [[maybe_unused]] LogErrorAndExit err =
#endif

// NOLINTEND

enum ErrorType
{
    GenericError,
    GraphicsError,
    ApplicationError,
    WindowError,
    RendererError,
    EventError,
    AssetError
};

class Error : public std::runtime_error
{
public:
    Error(ErrorType type,
          std::string const& msg,
          std::source_location loc = std::source_location::current())
        : std::runtime_error { fmt::format("[{}] {}", magic_enum::enum_name(type), msg) },
          m_location(loc)
    {
    }

    ~Error() override = default;

    Error(Error const&) = default;
    Error(Error&&)      = default;

    auto operator=(Error const&) -> Error const& = delete;
    auto operator=(Error&&) -> Error const&      = delete;

    [[nodiscard]] auto getLocation() const -> auto const& { return m_location; }

private:
    std::source_location m_location {};
};