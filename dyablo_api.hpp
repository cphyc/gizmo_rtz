#pragma once

#include "types.hpp"
#include <Kokkos_Core.hpp>

// MAX_ELEMENTS is already defined in types.hpp

KOKKOS_FUNCTION
void initialize_elements(std::array<Element, MAX_ELEMENTS> &elements);

KOKKOS_FUNCTION
void initialize_ion_fracs(std::array<Element, MAX_ELEMENTS> &elements, std::array<ParticleIonData, MAX_ELEMENTS> &n_and_ion_fracs, double hdens, double metallicity, int itype);

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
                                     const double metallicity,
                                     const double aexp,
                                     const std::array<double, MAX_ELEMENTS> &nelements_loc,
                                     std::array<double, MAX_ELEMENTS * MAX_ELEMENTS> &xions_loc) const;

private:
    // Tabulated rate data (lives on device)
    TabulatedData tabData;

    // UV background strength (sub-ionizing radiation field)
    double UV_background_G0;

    // Background cosmic-ray ionization rate (primary + secondary)
    double generic_cosmic_ray_ionization_rate;
};