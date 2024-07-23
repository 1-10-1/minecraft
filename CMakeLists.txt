cmake_minimum_required(VERSION 3.26.0 FATAL_ERROR)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
project(minecraft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)

option(PROFILED_BUILD "" OFF)

if(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

add_subdirectory(lib)

list(APPEND SOURCE_FILES
    src/main.cpp
    src/timer.cpp
    src/key.cpp
    src/logger.cpp
    src/utils.cpp
    src/window.cpp
    src/camera.cpp

    src/game/game.cpp

    src/renderer/backend/renderer_backend.cpp
    src/renderer/backend/stb.cpp
    src/renderer/backend/gltfloader.cpp
    src/renderer/backend/render.cpp
    src/renderer/backend/instance.cpp
    src/renderer/backend/surface.cpp
    src/renderer/backend/image.cpp
    src/renderer/backend/buffer.cpp
    src/renderer/backend/command.cpp
    src/renderer/backend/pipeline.cpp
    src/renderer/backend/vma.cpp
    src/renderer/backend/allocator.cpp
    src/renderer/backend/descriptor.cpp
    src/renderer/backend/swapchain.cpp
    src/renderer/backend/device.cpp
    src/renderer/backend/utils.cpp
    src/renderer/renderer.cpp
)

# TODO: Figure out how to make this work
# This is the best I could do:
# clang-tidy -format-style=file -header-filter=../include/vkp/*.h -header-filter=../include/vkp/graphics/*.h -p ..\build\x64-Debug ./*.cpp ./graphics/*.cpp
# But its slow as fuck
# Or you can cd into the build folder and run python (Get-Command run-clang-tidy).Path

# set(CMAKE_CXX_CLANG_TIDY
#   clang-tidy;
#   -format-style='file';
#   --extra-arg-before=-std=c++2b;
#   -header-filter=${CMAKE_CURRENT_SOURCE_DIR}/include/*;
# )

find_package(ASSIMP REQUIRED)
find_package(Threads REQUIRED)

list(APPEND LIBS
    spdlog
    Vulkan::Vulkan
    glm::glm
    glfw
    magic_enum::magic_enum
    Tracy::TracyClient
    stb
    imgui
    GPUOpen::VulkanMemoryAllocator
    tinygltf
    assimp
    basisu
    Threads::Threads
)

if (MSVC)
    list(APPEND LIBS assimp)
else()
    list(APPEND LIBS stdc++exp)
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} ${LIBS})
target_compile_definitions(
    ${PROJECT_NAME} PUBLIC FMT_EXCEPTIONS=0
    GLM_FORCE_SIMD_AVX2 GLM_FORCE_AVX2 GLFW_INCLUDE_VULKAN GLM_ENABLE_EXPERIMENTAL
    NOMINMAX GLM_FORCE_RADIANS GLM_FORCE_DEPTH_ZERO_TO_ONE IMGUI_ENABLE_FREETYPE
    _CRT_SECURE_NO_WARNINGS MAGIC_ENUM_RANGE_MIN=-1 MAGIC_ENUM_RANGE_MAX=1000
    __cpp_lib_expected VULKAN_HPP_NO_EXCEPTIONS VULKAN_HPP_RAII_NO_EXCEPTIONS
    "VULKAN_HPP_ASSERT_ON_RESULT=(void)" VULKAN_HPP_NO_CONSTRUCTORS ROOT_SOURCE_PATH="${CMAKE_SOURCE_DIR}"
)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DEBUG=true ASSERTIONS_ENABLED)
else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC DEBUG=false)
endif()

if (PROFILED_BUILD)
    target_compile_definitions(${PROJECT_NAME} PRIVATE PROFILED=true)
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE PROFILED=false)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE "./include")

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /std:c++latest /arch:AVX2
        $<$<CONFIG:Release>:/O3 /EHs-c- /D_HAS_EXCEPTIONS=0>)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        -std=c++23 -Wall -Wunused -mavx2
        -Wno-format -Wno-switch
        -Wno-deprecated-declarations -march=native -flto=auto
        -Wno-sign-compare -pthread
        $<$<CONFIG:Release>:-fno-exceptions -ffast-math -finline-functions>
        $<$<CONFIG:Debug>:-mtune=native -g>) # -fsanitize=address -g -fno-omit-frame-pointer>)
endif()

if ((CMAKE_BUILD_TYPE MATCHES Release) OR PROFILED_BUILD)
    if (MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE /O2)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE -O3)
    endif()
endif()

target_link_options(${PROJECT_NAME} PRIVATE -pthread -flto=auto) #-fsanitize=address -g -fno-omit-frame-pointer)

file(GLOB SHADER_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/shaders/*")

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

foreach(FILE ${SHADER_FILES})
    if (NOT EXISTS ${FILE})
        message(FATAL_ERROR "Shader file '${FILE}' does not exist.")
    endif()
endforeach(FILE)

foreach(SHADER_FILE ${SHADER_FILES})
    get_filename_component(FILE_NAME_WITH_EXT ${SHADER_FILE} NAME)
    get_filename_component(FILE_WE ${SHADER_FILE} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER_FILE} EXT)

        # Determine the shader type based on the extension
    if(${SHADER_EXT} STREQUAL ".vert")
        set(SHADER_TYPE "vertex")
    elseif(${SHADER_EXT} STREQUAL ".tesc")
        set(SHADER_TYPE "tesscontrol")
    elseif(${SHADER_EXT} STREQUAL ".tese")
        set(SHADER_TYPE "tesseval")
    elseif(${SHADER_EXT} STREQUAL ".geom")
        set(SHADER_TYPE "geometry")
    elseif(${SHADER_EXT} STREQUAL ".frag")
        set(SHADER_TYPE "fragment")
    elseif(${SHADER_EXT} STREQUAL ".comp")
        set(SHADER_TYPE "compute")
    elseif(${SHADER_EXT} STREQUAL ".glsl")
        continue()
    else()
        message(FATAL_ERROR "Unknown shader extension: ${SHADER_EXT}")
    endif()

    add_custom_command(
        OUTPUT "${SHADER_OUTPUT_DIR}/${FILE_NAME_WITH_EXT}.spv"
        COMMAND glslc ${SHADER_FILE} -O
                -o "${SHADER_OUTPUT_DIR}/${FILE_NAME_WITH_EXT}.spv"
                -std=460
                --target-spv=spv1.6
                --target-env=vulkan1.3
                -fshader-stage=${SHADER_TYPE}
                -g
        DEPENDS ${SHADER_FILE}
        COMMENT "Compiling ${SHADER_TYPE} shader: ${SHADER_FILE}"
    )

    # Also output the assembly
    add_custom_command(
        OUTPUT "${SHADER_OUTPUT_DIR}/${FILE_NAME_WITH_EXT}.asm"
        COMMAND glslc ${SHADER_FILE} -O -S
                -o "${SHADER_OUTPUT_DIR}/${FILE_NAME_WITH_EXT}.asm"
                -std=460
                --target-spv=spv1.6
                --target-env=vulkan1.3
                -fshader-stage=${SHADER_TYPE}
            -g
        DEPENDS ${SHADER_FILE}
        COMMENT "Compiling ${SHADER_TYPE} shader assembly: ${SHADER_FILE}"
    )
    list(APPEND COMPILED_SHADERS "${SHADER_OUTPUT_DIR}/${FILE_NAME_WITH_EXT}.asm")

    list(APPEND COMPILED_SHADERS "${SHADER_OUTPUT_DIR}/${FILE_NAME_WITH_EXT}.spv")
endforeach()

add_custom_target(CompileShaders DEPENDS ${COMPILED_SHADERS})

add_dependencies(${PROJECT_NAME} CompileShaders)

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/res $<TARGET_FILE_DIR:${PROJECT_NAME}>/res
)
