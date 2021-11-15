cmake_minimum_required(VERSION 3.16)
project(tfhe++ CXX C)
set(CMAKE_CXX_STANDARD 20)
option(DEBUG "Debug mode" OFF)
# TFHEPP_MARCH: target architecture for -march flag (default: native).
# Set to e.g. "x86-64-v3" for a portable AVX2 build.
if(NOT DEFINED TFHEPP_MARCH)
  set(TFHEPP_MARCH "native")
endif()
if(DEBUG)
  set(CMAKE_CXX_FLAGS
      "-march=${TFHEPP_MARCH} -g -Wall -Wextra -pedantic -Wno-sign-compare")
else()
  set(CMAKE_CXX_FLAGS
      "-march=${TFHEPP_MARCH} -O3 -g -funroll-loops -Wall -Wextra -pedantic -Wno-sign-compare"
  )
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

OPTION(USE_80BIT_SECURITY "Use 80bit security parameter(faster)" OFF)
OPTION(USE_CGGI19 "Use the parameter set proposed in CGGI19" OFF)
OPTION(ENABLE_TEST "Build tests" OFF)
OPTION(ENABLE_BENCHMARK "Build benchmarks" OFF)
OPTION(ENABLE_TUTORIAL "Build tutorial" OFF)
OPTION(ENABLE_AXELL "Use codes developed in Axell" ON) 

if(USE_80BIT_SECURITY)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_80BIT_SECURITY"
      PARENT_SCOPE)
  add_compile_definitions(USE_80BIT_SECURITY)
elseif(USE_COMPRESS)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_COMPRESS"
      PARENT_SCOPE)
  add_compile_definitions(USE_COMPRESS)
elseif(USE_CGGI19)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_CGGI19"
      PARENT_SCOPE)
  add_compile_definitions(USE_CGGI19)
elseif(USE_CONCRETE)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_CONCRETE"
      PARENT_SCOPE)
  add_compile_definitions(USE_CONCRETE)
elseif(USE_TFHE_RS)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_TFHE_RS"
      PARENT_SCOPE)
  add_compile_definitions(USE_TFHE_RS)
elseif(USE_TERNARY_CMUX)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_TERNARY_CMUX;USE_TERNARY"
      PARENT_SCOPE)
  add_compile_definitions(USE_TERNARY)
  add_compile_definitions(USE_TERNARY_CMUX)
elseif(USE_TERNARY)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_TERNARY"
      PARENT_SCOPE)
  add_compile_definitions(USE_TERNARY)
endif()

if(NOT USE_TERNARY)
  if(USE_KEY_BUNDLE)
    set(TFHEpp_DEFINITIONS
        "${TFHEpp_DEFINITIONS};USE_KEY_BUNDLE"
        PARENT_SCOPE)
    add_compile_definitions(USE_KEY_BUNDLE)
  endif()
endif()

# if(USE_AVX512) string(APPEND CMAKE_CXX_FLAGS " -mprefer-vector-width=512")
# endif()

if(USE_FFTW3)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_FFTW3"
      PARENT_SCOPE)
  add_compile_definitions(USE_FFTW3)
  add_subdirectory(thirdparties/fftw)
elseif(USE_MKL)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_MKL;USE_INTERLEAVED_FORMAT"
      PARENT_SCOPE)
  add_compile_definitions(USE_MKL)
  add_compile_definitions(USE_INTERLEAVED_FORMAT)
  find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
  include_directories(${MKLROOT}/include)
  add_subdirectory(thirdparties/mkl)
elseif(USE_SPQLIOX_AARCH64)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_SPQLIOX_AARCH64"
      PARENT_SCOPE)
  add_compile_definitions(USE_SPQLIOX_AARCH64)
  add_subdirectory(thirdparties/spqliox_aarch64)
  # Check if the platform is macOS and the architecture is ARM64
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    include_directories(/opt/homebrew/include)
  endif()
elseif(USE_CONCRETE_FFT)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_CONCRETE_FFT;USE_INTERLEAVED_FORMAT"
      PARENT_SCOPE)
  add_compile_definitions(USE_CONCRETE_FFT)
  add_compile_definitions(USE_INTERLEAVED_FORMAT)
  add_subdirectory(thirdparties/concrete-fft)
elseif(USE_SPQLIOS_INTL)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_SPQLIOS_INTL;USE_INTERLEAVED_FORMAT"
      PARENT_SCOPE)
  add_compile_definitions(USE_SPQLIOS_INTL)
  add_compile_definitions(USE_INTERLEAVED_FORMAT)
  add_subdirectory(thirdparties/spqlios-intl)
elseif(USE_SPQLIOS_STOCKHAM)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_SPQLIOS_STOCKHAM"
      PARENT_SCOPE)
  add_compile_definitions(USE_SPQLIOS_STOCKHAM)
  add_subdirectory(thirdparties/spqlios-stockham)
elseif(USE_SPQLIOS_ARITHMETIC)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_SPQLIOS_ARITHMETIC"
      PARENT_SCOPE)
  add_compile_definitions(USE_SPQLIOS_ARITHMETIC)
  # Disable spqlios-arithmetic tests to avoid gtest dependency
  set(ENABLE_TESTING
      OFF
      CACHE BOOL "Disable spqlios-arithmetic tests" FORCE)
  add_subdirectory(thirdparties/spqlios-arithmetic)
  # Restore ENABLE_TEST for the parent project
  set(ENABLE_TESTING ${ENABLE_TEST})
else()
  if(USE_AVX512)
    set(TFHEpp_DEFINITIONS
        "${TFHEpp_DEFINITIONS};USE_AVX512"
        PARENT_SCOPE)
    add_compile_definitions(USE_AVX512)
  endif()
  add_subdirectory(thirdparties/spqlios)
endif()

if(USE_HEXL)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_HEXL"
      PARENT_SCOPE)
  add_compile_definitions(USE_HEXL)
  add_subdirectory(thirdparties/hexl)
  # set(CMAKE_CXX_FLAGS "-march=native -O3 -g -funroll-loops -Wall -Wextra
  # -pedantic -Wno-sign-compare -mprefer-vector-width=512" )
endif()

if(ENABLE_AXELL)
	add_compile_definitions(ENABLE_AXELL)
endif()

# For OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(USE_RANDEN)
  add_subdirectory(thirdparties/randen)
endif()
add_subdirectory(src)

if(ENABLE_TEST)
  add_subdirectory(test)
  add_subdirectory(thirdparties/AES)
endif()
if(ENABLE_BENCHMARK)
  add_subdirectory(benchmark)
endif()
if(ENABLE_TUTORIAL)
  add_subdirectory(tutorial)
endif()

install(TARGETS tfhe++ LIBRARY DESTINATION lib)
