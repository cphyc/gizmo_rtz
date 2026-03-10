cmake_minimum_required(VERSION 3.16)
project(gizmo LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Include Kokkos from the submodule
add_subdirectory(extern/kokkos)

# Set source files
set(SOURCES
    dyablo_api.cpp
    rtz_main.cpp
)

# Define the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link Kokkos
target_link_libraries(${PROJECT_NAME} Kokkos::kokkos)

# Add compile flags if Kokkos_ENABLE_CUDA=ON
if(Kokkos_ENABLE_CUDA)
    target_compile_options(${PROJECT_NAME} PRIVATE --expt-extended-lambda --expt-relaxed-constexpr -rdc=true)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DESUL_ATOMICS_ENABLE_CUDA_SEPARABLE_COMPILATION)
    target_link_options(${PROJECT_NAME} PRIVATE -rdc=true)
endif()

# Warn all
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)

# --- Testing ---
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Data files needed at runtime (relative to source dir)
set(DATA_DIR "${CMAKE_SOURCE_DIR}/data")
set(GOLDEN_DIR "${CMAKE_SOURCE_DIR}/tests/golden")
set(COMPARE_SCRIPT "${CMAKE_SOURCE_DIR}/tests/compare_output.py")

# ISM equilibrium test (300 density steps)
add_test(
    NAME ISM_test
    COMMAND ${CMAKE_COMMAND} -E env
        "OMP_NUM_THREADS=1" "OMP_PROC_BIND=false"
        ${CMAKE_COMMAND}
        -DEXE=$<TARGET_FILE:${PROJECT_NAME}>
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
        -DEXE=$<TARGET_FILE:${PROJECT_NAME}>
        -DTEST_NAME=CIE
        -DGOLDEN=${GOLDEN_DIR}/cie_reference.txt
        -DCOMPARE=${COMPARE_SCRIPT}
        -DPYTHON=${Python3_EXECUTABLE}
        -P ${CMAKE_SOURCE_DIR}/tests/run_test.cmake
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(CIE_test PROPERTIES TIMEOUT 1200)
