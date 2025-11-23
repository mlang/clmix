cmake_minimum_required(VERSION 3.25)
project(CLMIX VERSION 1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json REQUIRED)
find_package(PkgConfig)
pkg_check_modules(Aubio REQUIRED IMPORTED_TARGET aubio)
pkg_check_modules(Readline REQUIRED IMPORTED_TARGET readline)
pkg_check_modules(SampleRate REQUIRED IMPORTED_TARGET samplerate)
pkg_check_modules(SndFile REQUIRED IMPORTED_TARGET sndfile)

include(CheckIncludeFileCXX)
check_include_file_cxx("miniaudio.h" HAVE_MINIAUDIO_H)
if (NOT HAVE_MINIAUDIO_H)
  message(FATAL_ERROR "miniaudio.h not found. Install miniaudio or add its include directory to the build (e.g. set CMAKE_INCLUDE_PATH or use target_include_directories).")
endif()

add_executable(clmix main.cpp)
target_compile_definitions(clmix PRIVATE MDSPAN_USE_BRACKET_OPERATOR=1)
target_link_libraries(clmix
  PRIVATE nlohmann_json::nlohmann_json
          PkgConfig::Aubio
          PkgConfig::Readline
          PkgConfig::SampleRate
          PkgConfig::SndFile
)
