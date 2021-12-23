cmake_minimum_required(VERSION 3.16)
project(tfhe++ CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "-march=native -O3 -g -funroll-loops -Wall -Wextra -pedantic -Wno-sign-compare"
)

option(USE_80BIT_SECURITY "Use 80bit security parameter(faster)" OFF)
option(USE_CGGI19 "Use the parameter set proposed in CGGI19" OFF)
option(USE_CONCRETE "Use the parameter set proposed in CONCRETE" OFF)
option(USE_TFHE_RS "Use the parameter set proposed in TFHE-rs" OFF)
option(USE_TERNARY "Use ternary secret keys" OFF)
option(USE_COMPRESS "Use compressed ciphertexts" OFF)
option(USE_SUBSET_KEY "Use subset key" OFF)

option(USE_TERNARY_CMUX "Use ternary cmux" OFF)
option(USE_KEY_BUNDLE "Use key bundle algorithm" OFF)

option(USE_BLAKE3 "Use blake3 as CSPRNG" ON)
option(USE_RANDEN "Use randen as CSPRNG" OFF)

# Detect AVX-512 support at configure time.
# Strategy (in priority order):
#   1. /proc/cpuinfo — works in containers (Apptainer/Docker) without executing
#      a compiled binary, because the host kernel's /proc is always visible.
#   2. check_cxx_source_compiles with -mavx512f — compile-only check for
#      non-Linux or cross-compile environments (macOS, Windows, etc.).
# check_cxx_source_runs is intentionally avoided: it fails in container build
# environments (Apptainer %post, Docker RUN) where sandbox restrictions prevent
# executing compiled binaries during cmake configuration.
if(NOT DEFINED AVX512_SUPPORTED)
  set(AVX512_SUPPORTED FALSE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    if(EXISTS "/proc/cpuinfo")
      file(READ "/proc/cpuinfo" TFHEPP_CPUINFO)
      if(TFHEPP_CPUINFO MATCHES "avx512f")
        set(AVX512_SUPPORTED TRUE)
      endif()
    else()
      # No /proc/cpuinfo — non-Linux environment (macOS, Windows, etc.).
      # Use a compile-only check as a proxy for AVX-512 availability.
      include(CheckCXXSourceCompiles)
      set(CMAKE_REQUIRED_FLAGS "-mavx512f")
      check_cxx_source_compiles("
        #include <immintrin.h>
        int main() {
          __m512d a = _mm512_setzero_pd();
          (void)a;
          return 0;
        }
      " AVX512_COMPILE_SUPPORTED)
      unset(CMAKE_REQUIRED_FLAGS)
      if(AVX512_COMPILE_SUPPORTED)
        set(AVX512_SUPPORTED TRUE)
      endif()
    endif()
  endif()
endif()

if(AVX512_SUPPORTED)
  message(STATUS "AVX-512 support detected, USE_AVX512 defaults to ON")
  set(USE_AVX512_DEFAULT ON)
else()
  message(STATUS "AVX-512 support not detected, USE_AVX512 defaults to OFF")
  set(USE_AVX512_DEFAULT OFF)
endif()

option(USE_AVX512 "Use AVX512 ver. SPQLIOS" ${USE_AVX512_DEFAULT})
option(USE_CONCRETE_FFT "Use concrete-fft" OFF)
option(USE_MKL "Use Intel MKL" OFF)
option(USE_FFTW3 "Use FFTW3" OFF)
option(USE_SPQLIOX_AARCH64 "Use spqliox_aarch64" OFF)
option(USE_SPQLIOS_ARITHMETIC "Use spqlios-arithmetic backend" OFF)
option(USE_SPQLIOS_INTL "Use interleaved-format SPQLIOS" ON)
option(USE_SPQLIOS_STOCKHAM "Use Stockham radix-4 SPQLIOS (split layout)" OFF)
option(USE_HEXL "Use Intel HEXL" OFF)

option(ENABLE_TEST "Build tests" OFF)
option(ENABLE_BENCHMARK "Build benchmarks" OFF)
option(ENABLE_TUTORIAL "Build tutorial" OFF)
option(ENABLE_AXELL "Use codes developed in Axell" ON)
option(USE_PERF "Use Google Profiler" OFF)

option(ENABLE_SHARED "Build as shared libraries" OFF)

set(TFHEpp_DEFINITIONS
    ""
    PARENT_SCOPE)

if(USE_BLAKE3)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_BLAKE3"
      PARENT_SCOPE)
  add_compile_definitions(USE_BLAKE3)
  add_subdirectory(thirdparties/BLAKE3/c)
elseif(USE_RANDEN)
  set(TFHEpp_DEFINITIONS
      "${TFHEpp_DEFINITIONS};USE_RANDEN"
      PARENT_SCOPE)
  add_compile_definitions(USE_RANDEN)
endif()

if(USE_RANDEN)
  add_compile_definitions(USE_RANDEN)
endif()

if(USE_80BIT_SECURITY)
  add_compile_definitions(USE_80BIT_SECURITY)
elseif(USE_CGGI19)
  add_compile_definitions(USE_CGGI19)
endif()

if(USE_FFTW3)
  add_compile_definitions(USE_FFTW3)
  add_subdirectory(thirdparties/fftw)
elseif(USE_SPQLIOX_AARCH64)
  add_compile_definitions(USE_SPQLIOX_AARCH64)
  add_subdirectory(thirdparties/spqliox_aarch64)
else()
  add_subdirectory(thirdparties/spqlios)
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
endif()
if(ENABLE_BENCHMARK)
  add_subdirectory(benchmark)
endif()
if(ENABLE_TUTORIAL)
  add_subdirectory(tutorial)
endif()

install(TARGETS tfhe++ LIBRARY DESTINATION lib)
