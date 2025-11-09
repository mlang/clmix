cmake_minimum_required(VERSION 3.25)
project(CLMIX VERSION 1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig)
pkg_check_modules(SndFile REQUIRED IMPORTED_TARGET sndfile)
pkg_check_modules(Aubio REQUIRED IMPORTED_TARGET aubio)
pkg_check_modules(Readline REQUIRED IMPORTED_TARGET readline)

add_executable(clmix main.cpp)
target_compile_definitions(clmix PRIVATE MDSPAN_USE_BRACKET_OPERATOR=1)
target_link_libraries(clmix PRIVATE PkgConfig::SndFile PkgConfig::Aubio PkgConfig::Readline)
