file(GLOB_RECURSE tfhe++_sources "*.cpp")

# cloudkey is template/header-driven in the current tree; the legacy .cpp is
# stale and conflicts with the current EvalKey layout.
list(REMOVE_ITEM tfhe++_sources
     ${CMAKE_CURRENT_SOURCE_DIR}/cloudkey.cpp)

# Exclude spqlios-arithmetic processor unless that backend is selected.
if(NOT USE_SPQLIOS_ARITHMETIC)
  list(REMOVE_ITEM tfhe++_sources
       ${CMAKE_CURRENT_SOURCE_DIR}/fft_processor_spqlios_arithmetic.cpp)
endif()

if(NOT ENABLE_AXELL)
  list(FILTER tfhe++_sources EXCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/axell/")
endif()

if(ENABLE_SHARED)
  add_library(tfhe++ SHARED ${tfhe++_sources})
else()
  add_library(tfhe++ STATIC ${tfhe++_sources})
endif()

target_include_directories(
  tfhe++
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparties/spqlios
    ${PROJECT_SOURCE_DIR}/thirdparties/fftw
    ${PROJECT_SOURCE_DIR}/thirdparties/concrete-fft/cpp
    ${PROJECT_BINARY_DIR}/thirdparties/concrete-fft
    ${PROJECT_SOURCE_DIR}/thirdparties/spqliox_aarch64
    ${PROJECT_SOURCE_DIR}/thirdparties/spqliox_aarch64/xbyak_aarch64/xbyak_aarch64
    ${PROJECT_SOURCE_DIR}/thirdparties/hexl/hexl/hexl/include
    ${PROJECT_SOURCE_DIR}/thirdparties/randen
    ${PROJECT_SOURCE_DIR}/thirdparties/AES/src
    ${PROJECT_SOURCE_DIR}/thirdparties/cereal/include)

if(USE_RANDEN)
  if(USE_FFTW3)
    target_link_libraries(tfhe++ INTERFACE fftwproc randen fftw3)
  elseif(USE_MKL)
    target_link_libraries(tfhe++ INTERFACE mklproc randen MKL::MKL)
  elseif(USE_CONCRETE_FFT)
    target_link_libraries(tfhe++ INTERFACE concrete-fft randen)
  elseif(USE_SPQLIOS_ARITHMETIC)
    target_link_libraries(tfhe++ INTERFACE libspqlios randen)
    target_include_directories(
      tfhe++ PUBLIC ${PROJECT_SOURCE_DIR}/thirdparties/spqlios-arithmetic)
  elseif(USE_SPQLIOS_INTL)
    target_link_libraries(tfhe++ INTERFACE spqlios-intl randen)
    target_include_directories(
      tfhe++ PUBLIC ${PROJECT_SOURCE_DIR}/thirdparties/spqlios-intl)
  elseif(USE_SPQLIOS_STOCKHAM)
    target_link_libraries(tfhe++ INTERFACE spqlios-stockham randen)
    target_include_directories(
      tfhe++ PUBLIC ${PROJECT_SOURCE_DIR}/thirdparties/spqlios-stockham)
  elseif(USE_HEXL)
    target_link_libraries(tfhe++ INTERFACE spqlios randen HEXL::hexl)
  elseif(USE_SPQLIOX_AARCH64)
    target_link_libraries(tfhe++ INTERFACE spqliox_aarch64 randen xbyak_aarch64)
  else()
    target_link_libraries(tfhe++ INTERFACE spqlios randen)
  endif()
else()
  if(USE_FFTW3)
    target_link_libraries(tfhe++ INTERFACE fftwproc fftw3)
  elseif(USE_MKL)
    target_link_libraries(tfhe++ INTERFACE mklproc MKL::MKL)
  elseif(USE_CONCRETE_FFT)
    target_link_libraries(tfhe++ INTERFACE concrete-fft)
  elseif(USE_SPQLIOS_ARITHMETIC)
    target_link_libraries(tfhe++ INTERFACE libspqlios)
    target_include_directories(
      tfhe++ PUBLIC ${PROJECT_SOURCE_DIR}/thirdparties/spqlios-arithmetic)
  elseif(USE_SPQLIOS_INTL)
    target_link_libraries(tfhe++ INTERFACE spqlios-intl)
    target_include_directories(
      tfhe++ PUBLIC ${PROJECT_SOURCE_DIR}/thirdparties/spqlios-intl)
  elseif(USE_SPQLIOS_STOCKHAM)
    target_link_libraries(tfhe++ INTERFACE spqlios-stockham)
    target_include_directories(
      tfhe++ PUBLIC ${PROJECT_SOURCE_DIR}/thirdparties/spqlios-stockham)
  elseif(USE_HEXL)
    target_link_libraries(tfhe++ INTERFACE spqlios HEXL::hexl)
  elseif(USE_SPQLIOX_AARCH64)
    target_link_libraries(tfhe++ INTERFACE spqliox_aarch64 xbyak_aarch64)
  else()
    target_link_libraries(tfhe++ INTERFACE spqlios)
  endif()
endif()
