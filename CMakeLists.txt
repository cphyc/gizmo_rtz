cmake_minimum_required(VERSION 3.16)
project(gizmo LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Include Kokkos from the submodule
add_subdirectory(extern/kokkos)

# Set source files
set(SOURCES
    rtz_main.cpp
)

# Define the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link Kokkos
target_link_libraries(${PROJECT_NAME} Kokkos::kokkos)

# Add compile flags if Kokkos_ENABLE_CUDA=ON
if(Kokkos_ENABLE_CUDA)
    target_compile_options(${PROJECT_NAME} PRIVATE --expt-extended-lambda --expt-relaxed-constexpr)
endif()

# Warn all
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
