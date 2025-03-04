// collisional_ionization.h
#pragma once

#include "types.hpp"

void init_collisional_ionization(TabulatedData& tabData);

// Declaration of collisional ionization functions
KOKKOS_FUNCTION
double collisional_ionization(double T, int ion, int element_idx, const TabulatedData& tabData);
