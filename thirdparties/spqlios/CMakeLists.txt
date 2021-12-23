# This is the nayuki fft processor for the tfhe library
enable_language(ASM)

set(SRCS_FMA spqlios-fft-fma.s spqlios-ifft-fma.s fft_processor_spqlios.cpp
             spqlios-fft-impl.cpp)

set(SPQLIOS_HEADERS fft_processor_spqlios.h)

set(SPQLIOS_HEADERS fft_processor_spqlios.h x86.h)

target_include_directories(spqlios PUBLIC ${PROJECT_SOURCE_DIR}/include)
