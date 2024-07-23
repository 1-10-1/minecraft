#pragma GCC system_header  // this is to silence warnings
#include <mc/asserts.hpp>

#define VMA_LEAK_LOG_FORMAT(format, ...)     \
    {                                        \
        char* str;                           \
        asprintf(&str, format, __VA_ARGS__); \
        logger::warn("[VMA LEAK] {}", str);  \
        free(str);                           \
    }

#include <vk_mem_alloc.h>
