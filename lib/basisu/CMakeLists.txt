cmake_minimum_required(VERSION 3.26.0)
project(basisu LANGUAGES C CXX)

add_library(basisu "./transcoder/basisu_transcoder.cpp" "./zstd/zstd.c")
target_include_directories(basisu PUBLIC "./transcoder" "./zstd")

if (MSVC)
	target_compile_options(basisu PRIVATE /WNoError /O3)
else()
	target_compile_options(basisu PRIVATE -Wno-error $<$<CONFIG:Release>:-O3>)
endif()

