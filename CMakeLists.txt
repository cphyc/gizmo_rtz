cmake_minimum_required(VERSION 3.16)
project(gizmo LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Include Kokkos from the submodule
add_subdirectory(extern/kokkos)

# Set source files
set(SOURCES
    charge_transfer.cpp
    collisional_ionization.cpp
    cooling.cpp
    cosmic_ray_ionization.cpp
    dust_recombination.cpp
    molecules.cpp
    photoionization_UVB.cpp
    recombination.cpp
    rtz_main.cpp
)

# Define the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link Kokkos
target_link_libraries(${PROJECT_NAME} Kokkos::kokkos)