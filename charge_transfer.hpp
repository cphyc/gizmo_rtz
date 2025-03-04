// charge_transfer.h
#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"

// Getters
double (*getCTRecomb())[4][31];
double (*getCTIon())[3][31];

void load_ct_rates();
KOKKOS_FUNCTION
double charge_transfer_recombination(int ion, int nelem, double T, const TabulatedData& tabData);

KOKKOS_FUNCTION
double charge_transfer_ionization(int ion, int nelem, double T, const TabulatedData& tabData);
