#pragma once

#include "types.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

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
    Real hdens,
    Real metallicity,
    int itype
);

KOKKOS_FUNCTION
Real get_ne(
    const Element* elements,
    const CompactIonData& n_and_ion_fracs,
    bool use_new
);

KOKKOS_FUNCTION
Real get_mu(
    Real ne,
    const Element* elements,
    const CompactIonData& n_and_ion_fracs,
    bool use_new,
    bool include_H2,
    Real nCO
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
    void set_reduced_speed_of_light_factor(Real factor);

    // Set the static (sub-ionizing) UV background G0
    void set_UV_background_G0(Real G0);

    // Update the UV background for a given redshift
    void need_to_update_UVB(Real redshift);

    // Update cross sections
    void need_to_update_cross_sections(Real T_blackbody,
                                      const std::vector<double>& sed_frequency = {},
                                      const std::vector<double>& sed_fnu = {});

    // Set the photon group energy bins
    void set_photon_groups(
        std::array<Real, N_GROUPS> E_min,
        std::array<Real, N_GROUPS> E_max
    );

    // Solve the chemistry network and cooling for a single cell
    KOKKOS_FUNCTION
    void solve_chemistry_and_cooling(
        Real TK,
        Real dx,
        const Real aexp,
        const Real total_dt,
        CompactIonData& n_and_ion_fracs_loc,
        Real& nCO,
        std::array<Real, N_GROUPS> &N_PHOT_loc,
        Array2D_RT &F_PHOT_loc,
        Real& out_Tmu,
        Real& out_mu,
        int conv_its,
        int max_its,
        int& total_iter_reached,
        PhysicsFlags flags,
        const bool rt_smooth,
        const std::array<Real, N_GROUPS>& dNpdt,
        const Array2D_RT& dFpdt
    ) const;

    // Constant element data (lives on device, shared across threads)
    Kokkos::View<Element*> elements_d;

    // TEMPORARY while GPU solver is being profiled/developed:
    // Accessors for drivers that launch the solver step functions as
    // separate kernels (see CoolingUpdate_PRISM_GPU.cpp)
    const TabulatedData& get_tabData() const { return tabData; }
    Real get_UV_background_G0() const { return UV_background_G0; }
    Real get_generic_cosmic_ray_ionization_rate() const { return generic_cosmic_ray_ionization_rate; }

private:

    // Tabulated rate data (lives on device)
    TabulatedData tabData;

    // UV background strength (sub-ionizing radiation field)
    Real UV_background_G0;

    // Background cosmic-ray ionization rate (primary + secondary)
    Real generic_cosmic_ray_ionization_rate;
};
