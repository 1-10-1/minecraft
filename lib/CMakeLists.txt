# SPDLOG
set(SPDLOG_BUILD_WARNINGS OFF)
set(SPDLOG_NO_EXCEPTIONS ON)
set(SPDLOG_SYSTEM_INCLUDES ON)
set(SPDLOG_FMT_EXTERNAL OFF)
add_subdirectory(spdlog)

# GLM
add_subdirectory(glm)

# GLFW
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
add_subdirectory(glfw)

# Magic Enum
add_subdirectory(magic_enum)

# Tracy
if (NOT PROFILED_BUILD)
	set(TRACY_ENABLE OFF)
endif()

add_subdirectory(tracy)

# STB
add_subdirectory(stb)

# IMGUI
add_subdirectory(imgui)

# Vulkan Memory Allocator
add_subdirectory(vma)

# Tiny glTF
set(TINYGLTF_INSTALL OFF)
set(TINYGLTF_ENABLE_DRACO ON)
add_subdirectory(tinygltf)

# Basis Universal Codec
add_subdirectory(basisu)

# SPIRV Tools
# set(SPIRV_WERROR OFF)
# add_subdirectory(spirvtools)

# SPIRV Cross
# set(SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS ON)
# set(SPIRV_CROSS_CLI OFF)
# set(SPIRV_CROSS_ENABLE_TESTS OFF)
# add_subdirectory(spirvcross)

# set(TF_BUILD_PROFILER OFF)
# set(TF_BUILD_TESTS OFF)
# set(TF_BUILD_EXAMPLES OFF)
# add_subdirectory(taskflow)

option(ENKITS_BUILD_C_INTERFACE OFF)
option(ENKITS_BUILD_EXAMPLES OFF )
add_subdirectory(enkits)

