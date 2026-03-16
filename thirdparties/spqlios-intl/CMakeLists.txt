add_library(spqlios-intl STATIC
    fft_processor_spqlios_intl.cpp
)
target_include_directories(spqlios-intl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(spqlios-intl PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(spqlios-intl PRIVATE -mavx2 -mfma)
