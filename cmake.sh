#!/usr/bin/bash

set -e

generate() {
    cmake -B build/debug -S . -G Ninja -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Debug
    cmake -B build/release -S . -G Ninja -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release
    cmake -B build/release-profiled -S . -G Ninja -DPROFILED_BUILD=ON -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release
    cmake -B build/debug-profiled -S . -G Ninja -DPROFILED_BUILD=ON -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Debug
}

if [[ "$1" == "generate" ]]; then
    if [ ! -d "./build" ]; then
        generate;
    else
        rm -rf ./build
        generate;
    fi
elif [[ "$1" == "build-all" ]]; then
    cmake --build build/debug
    cmake --build build/release
    cmake --build build/release-profiled
    cmake --build build/debug-profiled
else
    echo "invalid command"
fi

