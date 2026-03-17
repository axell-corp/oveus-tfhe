add_library(spqlios-intl STATIC
    fft_processor_spqlios_intl.cpp
)
target_include_directories(spqlios-intl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(spqlios-intl PRIVATE ${CMAKE_SOURCE_DIR}/include)
if(USE_AVX512)
    target_compile_options(spqlios-intl PRIVATE -mavx512f -mavx512dq -mfma)
    target_compile_definitions(spqlios-intl PRIVATE USE_AVX512)
else()
    target_compile_options(spqlios-intl PRIVATE -mavx2 -mfma)
endif()
