# SPDLOG
set(SPDLOG_BUILD_WARNINGS OFF)
set(SPDLOG_NO_EXCEPTIONS ON)
set(SPDLOG_SYSTEM_INCLUDES ON)
set(SPDLOG_FMT_EXTERNAL ON)
add_subdirectory(spdlog)

# GLM
add_subdirectory(glm)

# GLFW
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
add_subdirectory(glfw)

# Tiny OBJ Loader
add_subdirectory(tinyobjloader)

# Magic Enum
add_subdirectory(magic_enum)

# Tracy
if (PROFILED_BUILD)
add_subdirectory(tracy)
else()
set(TRACY_ENABLE OFF)
add_subdirectory(tracy)
endif()

# Assimp
if (MSVC)
	set(ASSIMP_BUILD_TESTS OFF)
	set(ASSIMP_WARNINGS_AS_ERRORS OFF)
	add_subdirectory(assimp)
endif()

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

