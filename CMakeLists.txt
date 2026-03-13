cmake_minimum_required(VERSION 3.16)

# Detect standalone vs embedded (add_subdirectory) build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(gizmo LANGUAGES CXX)
    set(GIZMO_RTZ_STANDALONE ON)
    # Include Kokkos from the submodule when building standalone
    add_subdirectory(extern/kokkos)
else()
    set(GIZMO_RTZ_STANDALONE OFF)
endif()

###################################################################
# gizmo_rtz library
###################################################################
if(GIZMO_RTZ_STANDALONE)
    add_library(gizmo_rtz STATIC dyablo_api.cpp)
    set_target_properties(gizmo_rtz PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
    if(Kokkos_ENABLE_CUDA)
        target_compile_options(gizmo_rtz PRIVATE --expt-extended-lambda --expt-relaxed-constexpr -rdc=true)
        target_compile_definitions(gizmo_rtz PRIVATE DESUL_ATOMICS_ENABLE_CUDA_SEPARABLE_COMPILATION)
        target_link_options(gizmo_rtz PRIVATE -rdc=true)
    endif()
    target_compile_options(gizmo_rtz PRIVATE -Wall -Wextra)
else()
    # When embedded, provide only headers. The source is compiled as part of
    # PRISM (included from CoolingUpdate_PRISM.cpp) so that CUDA device
    # functions are visible within the same translation unit, avoiding the
    # need for relocatable device code (-rdc=true) and the duplicate
    # __constant__ memory symbols it causes with Kokkos.
    add_library(gizmo_rtz INTERFACE)
endif()
if(GIZMO_RTZ_STANDALONE)
    target_include_directories(gizmo_rtz PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(gizmo_rtz PUBLIC Kokkos::kokkos)
else()
    target_include_directories(gizmo_rtz INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(gizmo_rtz INTERFACE Kokkos::kokkos)
endif()

###################################################################
# Standalone executable and tests (only when building standalone)
###################################################################
if(GIZMO_RTZ_STANDALONE)
    add_executable(gizmo rtz_main.cpp)
    target_link_libraries(gizmo PRIVATE gizmo_rtz)
    set_target_properties(gizmo PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    if(Kokkos_ENABLE_CUDA)
        target_compile_options(gizmo PRIVATE --expt-extended-lambda --expt-relaxed-constexpr -rdc=true)
        target_compile_definitions(gizmo PRIVATE DESUL_ATOMICS_ENABLE_CUDA_SEPARABLE_COMPILATION)
        target_link_options(gizmo PRIVATE -rdc=true)
    endif()

    target_compile_options(gizmo PRIVATE -Wall -Wextra)

    # --- Testing ---
    enable_testing()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    set(DATA_DIR "${CMAKE_SOURCE_DIR}/data")
    set(GOLDEN_DIR "${CMAKE_SOURCE_DIR}/tests/golden")
    set(COMPARE_SCRIPT "${CMAKE_SOURCE_DIR}/tests/compare_output.py")

    # ISM equilibrium test (300 density steps)
    add_test(
        NAME ISM_test
        COMMAND ${CMAKE_COMMAND} -E env
            "OMP_NUM_THREADS=1" "OMP_PROC_BIND=false"
            ${CMAKE_COMMAND}
            -DEXE=$<TARGET_FILE:gizmo>
            -DTEST_NAME=ISM
            -DGOLDEN=${GOLDEN_DIR}/ism_reference.txt
            -DCOMPARE=${COMPARE_SCRIPT}
            -DPYTHON=${Python3_EXECUTABLE}
            -P ${CMAKE_SOURCE_DIR}/tests/run_test.cmake
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(ISM_test PROPERTIES TIMEOUT 1200)

    # CIE test (296 temperature steps, constant temperature)
    add_test(
        NAME CIE_test
        COMMAND ${CMAKE_COMMAND} -E env
            "OMP_NUM_THREADS=1" "OMP_PROC_BIND=false"
            ${CMAKE_COMMAND}
            -DEXE=$<TARGET_FILE:gizmo>
            -DTEST_NAME=CIE
            -DGOLDEN=${GOLDEN_DIR}/cie_reference.txt
            -DCOMPARE=${COMPARE_SCRIPT}
            -DPYTHON=${Python3_EXECUTABLE}
            -P ${CMAKE_SOURCE_DIR}/tests/run_test.cmake
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(CIE_test PROPERTIES TIMEOUT 1200)
endif()
