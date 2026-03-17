add_library(spqlios-stockham STATIC
    fft_processor_spqlios.cpp
)
target_include_directories(spqlios-stockham PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(spqlios-stockham PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(spqlios-stockham PRIVATE -mavx2 -mfma)
