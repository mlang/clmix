cmake_minimum_required(VERSION 3.25)
project(CLMIX VERSION 1 LANGUAGES CXX)

option(CLMIX_ENABLE_MARCH_NATIVE "Enable -march=native for this build" ON)

find_package(nlohmann_json REQUIRED)
find_package(PkgConfig)
pkg_check_modules(Aubio REQUIRED IMPORTED_TARGET aubio)
pkg_check_modules(EBUR128 REQUIRED IMPORTED_TARGET libebur128)
pkg_check_modules(Readline REQUIRED IMPORTED_TARGET readline)
pkg_check_modules(SampleRate REQUIRED IMPORTED_TARGET samplerate)
pkg_check_modules(SndFile REQUIRED IMPORTED_TARGET sndfile)

include(CheckIncludeFileCXX)
check_include_file_cxx("miniaudio.h" HAVE_MINIAUDIO_H)
if (NOT HAVE_MINIAUDIO_H)
  message(FATAL_ERROR "miniaudio.h not found. Install miniaudio or add its include directory to the build (e.g. set CMAKE_INCLUDE_PATH or use target_include_directories).")
endif()

find_package(Boost REQUIRED CONFIG)
find_package(TBB REQUIRED CONFIG)

add_executable(clmix main.cpp)
target_compile_features(clmix PRIVATE cxx_std_23)
set_property(TARGET clmix PROPERTY CXX_STANDARD_REQUIRED ON)
target_compile_definitions(clmix
  PRIVATE MDSPAN_USE_BRACKET_OPERATOR=1
          _GLIBCXX_USE_TBB_PAR_BACKEND
)
if (CLMIX_ENABLE_MARCH_NATIVE)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(clmix PRIVATE -march=native)
  endif()
endif()
target_link_libraries(clmix
  PRIVATE nlohmann_json::nlohmann_json
          PkgConfig::Aubio
          PkgConfig::EBUR128
          PkgConfig::Readline
          PkgConfig::SampleRate
          PkgConfig::SndFile
          TBB::tbb
          Boost::boost
)
install(TARGETS clmix RUNTIME DESTINATION bin)
