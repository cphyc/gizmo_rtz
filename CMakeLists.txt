cmake_minimum_required(VERSION 3.16)
option(GIZMO_RTZ_BUILD_PYTHON "Build the nanobind Python extension" OFF)
if(GIZMO_RTZ_BUILD_PYTHON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
   AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "gizmo_rtz: CMAKE_BUILD_TYPE not set; defaulting to Release (-O3 -DNDEBUG)")
endif()

# Detect standalone vs embedded (add_subdirectory) build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(gizmo LANGUAGES CXX)
    set(GIZMO_RTZ_STANDALONE ON)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/kokkos/CMakeLists.txt")
        add_subdirectory(extern/kokkos)
    else()
        include(FetchContent)
        FetchContent_Declare(
            Kokkos
            GIT_REPOSITORY https://github.com/kokkos/kokkos.git
            GIT_TAG 5.0.2
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(Kokkos)
    endif()
else()
    set(GIZMO_RTZ_STANDALONE OFF)
endif()
option(GIZMO_RTZ_FAST_MATH "Enable -ffast-math -march=native for the standalone solver" ON)
option(GIZMO_RTZ_REDUCED_PRECISION "Store chemistry state as float densities + log-encoded uint16 ion fractions (smaller GPU memory footprint, ~0.1% quantization)" OFF)
set(GIZMO_RTZ_PERF_FLAGS "")
if(GIZMO_RTZ_STANDALONE AND GIZMO_RTZ_FAST_MATH AND NOT Kokkos_ENABLE_CUDA)
    set(GIZMO_RTZ_PERF_FLAGS -ffast-math -march=native)
endif()

###################################################################
# gizmo_rtz library
###################################################################
if(GIZMO_RTZ_STANDALONE)
    add_library(gizmo_rtz STATIC dyablo_api.cpp)
    set_target_properties(gizmo_rtz PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    if(Kokkos_ENABLE_CUDA)
        target_compile_options(gizmo_rtz PRIVATE --expt-extended-lambda --expt-relaxed-constexpr -rdc=true)
        target_compile_definitions(gizmo_rtz PRIVATE DESUL_ATOMICS_ENABLE_CUDA_SEPARABLE_COMPILATION)
        target_link_options(gizmo_rtz PRIVATE -rdc=true)
    endif()
    target_compile_options(gizmo_rtz PRIVATE -Wall -Wextra ${GIZMO_RTZ_PERF_FLAGS})
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
    if(GIZMO_RTZ_REDUCED_PRECISION)
        target_compile_definitions(gizmo_rtz PUBLIC RTZ_REDUCED_PRECISION)
    endif()
else()
    target_include_directories(gizmo_rtz INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(gizmo_rtz INTERFACE Kokkos::kokkos)
    if(GIZMO_RTZ_REDUCED_PRECISION)
        target_compile_definitions(gizmo_rtz INTERFACE RTZ_REDUCED_PRECISION)
    endif()
endif()

###################################################################
# Standalone executable and tests (only when building standalone)
###################################################################
if(GIZMO_RTZ_STANDALONE)
    # rtz_main.cpp #includes dyablo_api_GPU.cpp directly (the same single
    # translation unit trick dyablo uses to embed PRISM), so it must NOT also
    # link gizmo_rtz -- that library is the reference dyablo_api.cpp build and
    # nothing in the executable references it any more.
    add_executable(gizmo rtz_main.cpp)
    target_include_directories(gizmo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(gizmo PRIVATE Kokkos::kokkos)
    if(GIZMO_RTZ_REDUCED_PRECISION)
        target_compile_definitions(gizmo PRIVATE RTZ_REDUCED_PRECISION)
    endif()
    set_target_properties(gizmo PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    if(Kokkos_ENABLE_CUDA)
        target_compile_options(gizmo PRIVATE --expt-extended-lambda --expt-relaxed-constexpr)
    endif()

    target_compile_options(gizmo PRIVATE -Wall -Wextra ${GIZMO_RTZ_PERF_FLAGS})

    if(GIZMO_RTZ_BUILD_PYTHON)
        find_package(Python COMPONENTS Interpreter Development REQUIRED)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            OUTPUT_VARIABLE nanobind_ROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE nanobind_result
        )
        if(NOT nanobind_result EQUAL 0)
            message(FATAL_ERROR
                "nanobind is required for GIZMO_RTZ_BUILD_PYTHON. "
                "Install it with: ${Python_EXECUTABLE} -m pip install nanobind"
            )
        endif()
        find_package(nanobind CONFIG REQUIRED)
        if(Kokkos_ENABLE_CUDA)
            nanobind_add_module(_rtz NOMINSIZE rtz_bindings.cpp)
        else()
            nanobind_add_module(_rtz rtz_bindings.cpp)
        endif()
        target_link_libraries(_rtz PRIVATE Kokkos::kokkos)
        target_compile_definitions(_rtz PRIVATE
            GIZMO_RTZ_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
        )
        target_compile_options(_rtz PRIVATE -Wall -Wextra ${GIZMO_RTZ_PERF_FLAGS})
        set_target_properties(_rtz PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
        )
        if(Kokkos_ENABLE_CUDA)
            target_compile_options(_rtz PRIVATE
                --expt-extended-lambda --expt-relaxed-constexpr
            )
        endif()
        install(TARGETS _rtz LIBRARY DESTINATION rtz RUNTIME DESTINATION rtz)
        install(FILES python/rtz/__init__.py DESTINATION rtz)
        install(DIRECTORY data/ DESTINATION rtz/data)
    endif()

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
