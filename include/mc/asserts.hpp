#pragma once

#include "logger.hpp"

#if _MSC_VER
#    include <intrin.h>
#    define debugBreak() __debugbreak()
#else
#    define debugBreak() __builtin_trap()
#endif

#if DEBUG
#    define MC_ASSERT(expr)                                        \
        {                                                          \
            if (expr)                                              \
            {                                                      \
            }                                                      \
            else                                                   \
            {                                                      \
                logger::critical("Assertion '{}' failed.", #expr); \
                debugBreak();                                      \
            }                                                      \
        }

#    define MC_ASSERT_MSG(expr, ...)                                                            \
        {                                                                                       \
            if (expr)                                                                           \
            {                                                                                   \
            }                                                                                   \
            else                                                                                \
            {                                                                                   \
                logger::critical("Assertion '{}' failed: {}", #expr, std::format(__VA_ARGS__)); \
                debugBreak();                                                                   \
            }                                                                                   \
        }

#    define MC_ASSERT_LOC(expr, location)                                                          \
        {                                                                                          \
            if (expr)                                                                              \
            {                                                                                      \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                logger::logAt<logger::level::critical>(location, "Assertion '{}' failed.", #expr); \
                debugBreak();                                                                      \
            }                                                                                      \
        }

#    define MC_ASSERT_MSG_LOC(location, expr, ...)                                           \
        {                                                                                    \
            if (expr)                                                                        \
            {                                                                                \
            }                                                                                \
            else                                                                             \
            {                                                                                \
                logger::logAt<logger::level::critical>(                                      \
                    location, "Assertion '{}' failed: {}", #expr, std::format(__VA_ARGS__)); \
                debugBreak();                                                                \
            }                                                                                \
        }

#else
#    define MC_ASSERT(expr)               (void)(expr);
#    define MC_ASSERT_LOC(expr, location) (void)(expr);

#    define MC_ASSERT_MSG(expr, ...)                                        \
        {                                                                   \
            (void)(expr);                                                   \
            /* So that the expressions in __VA_ARGS__ are also evaluated */ \
            std::tuple { __VA_ARGS__ };                                     \
        }

#    define MC_ASSERT_MSG_LOC(location, expr, ...)                          \
        {                                                                   \
            (void)(expr);                                                   \
            /* So that the expressions in __VA_ARGS__ are also evaluated */ \
            std::tuple { __VA_ARGS__ };                                     \
        }
#endif
