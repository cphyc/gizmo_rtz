#pragma once

#include "types.hpp"
#include <Kokkos_Core.hpp>

#define MAX_ELEMENTS 27

KOKKOS_FUNCTION
void initialize_elements(Element *elements);

KOKKOS_FUNCTION
void initialize_ion_fracs(Element *elements, ParticleIonData *n_and_ion_fracs, double hdens, double metallicity, int itype);

template <
    bool constant_temperature,
    bool ramses_rt_T_scheme = true,
    bool rosenbrock_T_scheme = false,
    bool include_H2 = true,
    bool include_CO = false
>
class RTZ {
public:
    // Constructor: initializes elements, loads all tabulated data
    RTZ();

    // Update the UV background for a given redshift
    void need_to_update_UVB(double redshift);

    // Solve the chemistry network and cooling for a single cell
    KOKKOS_FUNCTION
    void solve_chemistry_and_cooling(double &TK,
                                     const double rho,
                                     const double metallicity,
                                     const double aexp,
                                     double nelements_loc[MAX_ELEMENTS],
                                     double xions_loc[MAX_ELEMENTS * MAX_ELEMENTS]) const;

private:
    // Tabulated rate data (lives on device)
    TabulatedData tabData;

    // UV background strength (sub-ionizing radiation field)
    double UV_background_G0;

    // Background cosmic-ray ionization rate (primary + secondary)
    double generic_cosmic_ray_ionization_rate;
};