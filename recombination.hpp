// recombination.h
#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"

// Declaration of recombination functions
KOKKOS_FUNCTION
double recombination(double T, int ion, int element_idx, const TabulatedData& tabData);

void init_recombination_rates(TabulatedData& tabData);
