#pragma once

#include "types.hpp"
#include <Kokkos_Core.hpp>

// MAX_ELEMENTS is already defined in types.hpp

KOKKOS_FUNCTION
void initialize_elements(
    std::array<Element, MAX_ELEMENTS> &elements,
    bool include_h2
);

KOKKOS_FUNCTION
void initialize_ion_fracs(
    std::array<Element, MAX_ELEMENTS> &elements,
    std::array<ParticleIonData, MAX_ELEMENTS> &n_and_ion_fracs,
    double hdens,
    double metallicity,
    int itype
);


template <bool constant_temperature,
          bool include_H2,
          bool include_CO,
          bool rt_advect,
          bool include_self_shielding>
class RTZ {
public:
    // Constructor: initializes elements, loads all tabulated data
    RTZ(std::string path_to_data);

    // Set the reduced speed of light factor (1.0 = no reduction)
    void set_reduced_speed_of_light_factor(double factor);

    // Set the static (sub-ionizing) UV background G0
    void set_UV_background_G0(double G0);

    // Update the UV background for a given redshift
    void need_to_update_UVB(double redshift);

    // Update cross sections
    void need_to_update_cross_sections(double T_blackbody);

    // Set the photon group energy bins
    void set_photon_groups(
        std::array<double, N_GROUPS> E_min,
        std::array<double, N_GROUPS> E_max
    );

    // Solve the chemistry network and cooling for a single cell
    KOKKOS_FUNCTION
    void solve_chemistry_and_cooling(
        double TK,
        double dx,
        const double aexp,
        const double total_dt,
        CompactIonData& n_and_ion_fracs_loc,
        double& nCO,
        std::array<double, N_GROUPS> &N_PHOT_loc,
        Array2D_RT &F_PHOT_loc,
        double& out_Tmu,
        double& out_mu,
        int conv_its,
        int max_its,
        int& total_iter_reached,
        PhysicsFlags flags
    ) const;

    // Constant element data (lives on device, shared across threads)
    Kokkos::View<Element*> elements_d;

private:

    // Tabulated rate data (lives on device)
    TabulatedData tabData;

    // UV background strength (sub-ionizing radiation field)
    double UV_background_G0;

    // Background cosmic-ray ionization rate (primary + secondary)
    double generic_cosmic_ray_ionization_rate;
};
