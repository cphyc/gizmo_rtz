// molecules.h
#pragma once
#include <Kokkos_Core.hpp>
/////////////////////////
//         H2          //
/////////////////////////
// H2 formation on dust and via the primordial H- channel
KOKKOS_FUNCTION
double alpha_H2(double T, double dust_to_gas_mass_ratio_over_mw, double xe, double H2_cosmic_ray_ionization_rate, double G0, double xHI, double xHII, double nH);

KOKKOS_FUNCTION
double alpha_H2_dust(double T, double dust_to_gas_mass_ratio_over_mw);

KOKKOS_FUNCTION
double alpha_H2_prim(double T, double xe, double H2_cosmic_ray_ionization_rate, double G0, double xHI, double xHII);
// Collosional H2 destruction
KOKKOS_FUNCTION
double beta_H2(double T, double nH, double xHI, double xH2, double xHe, double ne, double nHI, double nH2, double nHeI);
KOKKOS_FUNCTION
double beta_H2_umist(double T, double nH, double ne, double nH2);

/////////////////////////
//         CO          //
/////////////////////////
// CO formation rate
KOKKOS_FUNCTION
double alpha_CO(double G0, double xi_cr_H2, double nCII, double nH2, double xO, double n);
// CO destruction rate
KOKKOS_FUNCTION
double beta_CO(double G0, double xi_cr_H2);
