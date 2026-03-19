# This is the nayuki fft processor for the tfhe library
enable_language(ASM)

if(USE_AVX512)
  set(SRCS_FMA spqlios-fft-avx512.s spqlios-ifft-avx512.s
               fft_processor_spqlios.cpp spqlios-fft-impl-avx512.cpp)
elseif(USE_TFHE_RS)
  # Stockham radix-4 FFT: faster for power-of-4 sizes (ns4=256 for tfhe-rs params)
  add_compile_definitions(USE_STOCKHAM_FFT)
  set(SRCS_FMA stockham_fft_avx2.cpp fft_processor_spqlios.cpp)
else()
  # Default: hand-tuned assembly radix-2 FFT (supports all power-of-2 sizes)
  set(SRCS_FMA spqlios-fft-fma.s spqlios-ifft-fma.s fft_processor_spqlios.cpp
               spqlios-fft-impl.cpp)
endif()

set(SPQLIOS_HEADERS fft_processor_spqlios.h x86.h)

add_library(spqlios STATIC ${SRCS_FMA} ${SPQLIOS_HEADERS})

target_include_directories(spqlios PUBLIC ${PROJECT_SOURCE_DIR}/include)
