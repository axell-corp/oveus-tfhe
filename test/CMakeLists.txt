file(GLOB test_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "*.cpp")

file(COPY test.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FILE_PERMISSIONS OWNER_READ OWNER_EXECUTE)
file(COPY show_info.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FILE_PERMISSIONS OWNER_READ OWNER_EXECUTE)
file(COPY benchmark_docker.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FILE_PERMISSIONS OWNER_READ OWNER_EXECUTE)

set(core_test_sources
    INTorus.cpp
    annihilatecircuitbootstrapping.cpp
    circuitbootstrapping.cpp
    circuitbootstrappingsub.cpp
    gatebootstrappingntt.cpp
    gatebootstrappingtlwe2tlwe.cpp
    iks.cpp
    invcircuitbootstrapping.cpp
    polymul.cpp
    trgswenc.cpp)

set(enabled_test_sources ${core_test_sources})

if(USE_TFHE_RS)
    list(APPEND enabled_test_sources
        pbs_tfhers.cpp)
endif()

foreach(test_source ${test_sources})
    if(NOT test_source IN_LIST enabled_test_sources)
        continue()
    endif()
    string( REPLACE ".cpp" "" test_name ${test_source} )
    add_executable(${test_name} ${test_source})
    if(USE_PERF)
        add_compile_definitions(USE_PERF)
        target_link_libraries(${test_name} tfhe++ AES profiler)
    else()
        target_link_libraries(${test_name} tfhe++ AES)
    endif()
endforeach(test_source ${test_sources})

# For GMP used in INTorus test
find_library(gmp NAMES gmp gmpxx)
if(gmp)
    add_compile_definitions(ENABLE_LSHTEST)
    target_link_libraries(INTorus gmp)
endif()

if(ENABLE_AXELL)
    add_subdirectory(axell)
endif()
