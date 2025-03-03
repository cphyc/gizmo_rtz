// cooling.h
#ifndef COOLING_H
#define COOLING_H
#include <array>

#include <Kokkos_Core.hpp>


// Create shorthand for double kokkos views
#define MAX_ELEMENTS 27
using Array2D = std::array<std::array<double, MAX_ELEMENTS>, MAX_ELEMENTS>;

// Metal cooling table initialization
void initialize_high_temperature_metal_cooling();
void init_fine_structure_tables();
double all_cooling(double T, double ne, double aexp, double *element_number_densities, 
                   int *element_number_ions, Array2D& element_ion_fractions,
                   double G0, double f_dg, double xe, double xi_h_cr, double xi_h2_cr,
                   double ss_factor);

#endif // COOLING_H