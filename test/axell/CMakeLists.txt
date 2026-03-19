if(USE_80BIT_SECURITY)
    add_compile_definitions(USE_80BIT_SECURITY)
elseif(USE_CGGI19)
    add_compile_definitions(USE_CGGI19)
endif()

file(GLOB test_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "*.cpp")

set(axell_enabled_test_sources
    1brFullAdder.cpp
    2bit1brAdder.cpp
    2bit2brAdder.cpp
    2brFullAdder.cpp
    3inputfulladder.cpp
    4input.cpp
    Tree1brAdder.cpp
    ao3.cpp
    aoi3.cpp
    fulladder.cpp
    halfadder.cpp
    oai3.cpp
    swap.cpp
    xornandnor.cpp)

foreach(test_source ${test_sources})
    if(NOT test_source IN_LIST axell_enabled_test_sources)
        continue()
    endif()
    string( REPLACE ".cpp" "" test_name ${test_source} )
    add_executable(${test_name} ${test_source})
    if(USE_PERF)
        add_compile_definitions(USE_PERF)
        target_link_libraries(${test_name} tfhe++ profiler)
    else()
        target_link_libraries(${test_name} tfhe++)
    endif()
endforeach(test_source ${test_sources})
