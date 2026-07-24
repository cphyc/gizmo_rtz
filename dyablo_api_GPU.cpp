
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <cstdint>
#include <array>
#include <fstream>
#include <cmath>

#include <Kokkos_Core.hpp>

// The gizmo_rtz headers are written for inclusion in a single translation
// unit: they define non-inline functions and global tables. Wrapping them,
// and this whole file, in a namespace renames every symbol so this GPU
// solver can link alongside the reference dyablo_api.cpp translation unit.
// External headers are pre-included at global scope above so their include
// guards keep them out of the namespace.
namespace prism_gpu {

#include "types.hpp"
#include "recombination.hpp"
#include "collisional_ionization.hpp"
#include "cosmic_ray_ionization.hpp"
#include "photoionization_UVB.hpp"
#include "charge_transfer.hpp"
#include "dust_recombination.hpp"
#include "cooling.hpp"
#include "dyablo_api.hpp"
#include "cross_sections.hpp"

// MAX_ELEMENTS is already defined in types.hpp (included via dyablo_api.hpp)
const double MIN_XION = 1E-20;         // Minimum ion fraction
const double X_PCT_RULE = 1E-1;        // Value required for convergence
const double X_FM = 1E-6;              // Damping value for X_PCT_RULE
const double T_MIN = 2.725;
const double T_MAX = 1E9;

#define uint unsigned int


template<typename T, typename V>
void copy_data_3D(T& src, V &dst) {
    auto dst_h = Kokkos::create_mirror_view(dst);
    for (uint i = 0; i < dst.extent(0); i++) {
        for (uint j = 0; j < dst.extent(1); j++) {
            for (uint k = 0; k < dst.extent(2); k++) {
                dst_h(i, j, k) = src[i][j][k];
            }
        }
    }
    Kokkos::deep_copy(dst, dst_h);
};

template<typename T, typename V>
void copy_data_2D(T& src, V &dst) {
    auto dst_h = Kokkos::create_mirror_view(dst);
    for (uint i = 0; i < dst.extent(0); i++) {
        for (uint j = 0; j < dst.extent(1); j++) {
            dst_h(i, j) = src[i][j];
        }
    }
    Kokkos::deep_copy(dst, dst_h);
};

template<typename T, typename T2>
void copy_data_1D(T& src, Kokkos::View<T2> &dst) {
    auto dst_h = Kokkos::create_mirror_view(dst);
    for (uint i = 0; i < dst.extent(0); i++) {
        dst_h(i) = src[i];
    }
    Kokkos::deep_copy(dst, dst_h);
};


// Depletion array for the elements, initialized to 1.0 (no depletion) but then modified based on the RR14 dust-to-gas mass ratio


// Structs for element properties
KOKKOS_FUNCTION
void initialize_elements(std::array<Element, MAX_ELEMENTS> &elements, bool include_h2)
{
    // Initialize
    for (int i = 0; i < MAX_ELEMENTS; i++) {
        elements[i].atomic_number = -1;
        elements[i].n_ions = -1;
        elements[i].atomic_mass = -1.0;
        elements[i].z_solar = -1.0;
        elements[i].G0_photo_rate = 0.0;
        elements[i].n_mol = 0;
        elements[i].depletion = 1.0;
    }

    // Element 1: Hydrogen
    elements[1].atomic_number = 1;
    elements[1].atomic_mass = 1.008;
    elements[1].z_solar = 1.0;
    elements[1].G0_photo_rate = 0.0; // No subionizing PI
    elements[1].n_ions = elements[1].atomic_number + 1;
    elements[1].n_mol = include_h2 ? 1 : 0;
    elements[1].depletion = 1.0;

    // Element 2: Helium
    elements[2].atomic_number = 2;
    elements[2].atomic_mass = 4.0026;
    elements[2].z_solar = 8.51E-02;
    elements[2].G0_photo_rate = 0.0; // No subionizing PI
    elements[2].n_ions = elements[2].atomic_number + 1;
    elements[2].depletion = 1.0;

    // Element 6: Carbon
    elements[6].atomic_number = 6;
    elements[6].atomic_mass = 12.0107;
    elements[6].z_solar = 2.69E-04;
    elements[6].G0_photo_rate = 3.39E-10;
    elements[6].n_ions = elements[6].atomic_number + 1;
    elements[6].depletion = 0.5;

    // Element 7: Nitrogen
    elements[7].atomic_number = 7;
    elements[7].atomic_mass = 14.0067;
    elements[7].z_solar = 6.76E-05;
    elements[7].G0_photo_rate = 0.0; // No subionizing PI
    elements[7].n_ions = elements[7].atomic_number + 1;
    elements[7].depletion = 0.6;

    // Element 8: Oxygen
    elements[8].atomic_number = 8;
    elements[8].atomic_mass = 15.9994;
    elements[8].z_solar = 4.90E-04;
    elements[8].G0_photo_rate = 0.0; // No subionizing PI
    elements[8].n_ions = elements[8].atomic_number + 1;
    elements[8].depletion = 0.73;

    // Element 10: Neon
    elements[10].atomic_number = 10;
    elements[10].atomic_mass = 20.1797;
    elements[10].z_solar = 8.51E-05;
    elements[10].G0_photo_rate = 0.0; // No subionizing PI
    elements[10].n_ions = elements[10].atomic_number + 1;
    elements[10].depletion = 1.0;

    // Element 12: Magnesium
    elements[12].atomic_number = 12;
    elements[12].atomic_mass = 24.305;
    elements[12].z_solar = 3.98E-05;
    elements[12].G0_photo_rate = 6.59E-11;
    elements[12].n_ions = elements[12].atomic_number + 1;
    elements[12].depletion = 0.16;

    // Element 14: Silicon
    elements[14].atomic_number = 14;
    elements[14].atomic_mass = 28.0855;
    elements[14].z_solar = 3.24E-05;
    elements[14].G0_photo_rate = 4.47E-09;
    elements[14].n_ions = elements[14].atomic_number + 1;
    elements[14].depletion = 0.1;

    // Element 16: Sulfur
    elements[16].atomic_number = 16;
    elements[16].atomic_mass = 32.065;
    elements[16].z_solar = 1.32E-05;
    elements[16].G0_photo_rate = 1.13E-09;
    elements[16].n_ions = elements[16].atomic_number + 1;
    elements[16].depletion = 1.0;

    // Element 26: Iron
    elements[26].atomic_number = 26;
    elements[26].atomic_mass = 55.854;
    elements[26].z_solar = 3.16E-05;
    elements[26].G0_photo_rate = 4.71E-10;
    elements[26].n_ions = elements[26].atomic_number + 1;
    elements[26].depletion = 0.01;
}


KOKKOS_FUNCTION
double get_dust_mass_and_depletion(
    const Element* elements,
    const double nO, const double nH, const double nCO)
{
    // Dust-to-gas mass ratio (relative to the MW value). The per-element
    // gas-phase depletion factor follows from the returned ratio alone as
    // 1 - (1 - elements[i].depletion) * ratio.
    if (elements[8].atomic_number < 1) {
        return 0.0; // no depletion without a dust budget
    }

    // Parameters for the RR14 dust-to-gas mass ratio
    double a = 2.21;
    double aH = 1.00;
    double b = 0.96;
    double aL = 3.10;
    double xt = 8.10;
    double xs = 8.69;
    double y;

    double loc_z = 12.0 + log10((nO + nCO + 1E-20) / (nH + 1E-20));
    double x = fmax(loc_z, 5.0);

    // Determine y (gas-to-dust mass ratio)
    if (loc_z > xt)
        y = a + aH * (xs - x);
    else
        y = b + aL * (xs - x);

    double G2D = pow(10.0, y);
    double G2D_sol = pow(10.0, a);

    double y_ratio = fmax(fmin(G2D_sol / G2D, 1.0), 0.0);

    return y_ratio;
}

KOKKOS_INLINE_FUNCTION
double depletion_factor(const Element& element, const double y_ratio)
{
    return 1.0 - ((1.0 - element.depletion) * y_ratio);
}

KOKKOS_FUNCTION
void apply_dust_depletion(const Element* elements, const double y_ratio,
                          CompactIonData& n_and_ion_fracs, const double nCO)
{
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (n_and_ion_fracs.n_element[i] == 0.0)
            continue;
        double depletion = depletion_factor(elements[i], y_ratio);
        if (i == 6 || i == 8)
            n_and_ion_fracs.n_element[i] = (n_and_ion_fracs.n_element[i] + nCO) * depletion - nCO;
        else
            n_and_ion_fracs.n_element[i] *= depletion;
    }
}

KOKKOS_FUNCTION
void remove_dust_depletion(const Element* elements, const double y_ratio,
                           CompactIonData& n_and_ion_fracs, const double nCO)
{
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (n_and_ion_fracs.n_element[i] == 0.0)
            continue;
        double depletion = depletion_factor(elements[i], y_ratio);
        if (i == 6 || i == 8)
            n_and_ion_fracs.n_element[i] = (n_and_ion_fracs.n_element[i] + nCO) / depletion - nCO;
        else
            n_and_ion_fracs.n_element[i] /= depletion;
    }
}

// Structs for element properties
KOKKOS_FUNCTION
void initialize_ion_fracs(std::array<Element, MAX_ELEMENTS> &elements, std::array<ParticleIonData, MAX_ELEMENTS> &n_and_ion_fracs, double hdens, double metallicity, int itype)
{
    // Loop over all elements
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;

        // Only scale metals by metallicity
        if (i < 3) {
            n_and_ion_fracs[i].n_element = hdens * elements[i].z_solar * elements[i].depletion;
        } else {
            n_and_ion_fracs[i].n_element = hdens * elements[i].z_solar * metallicity * elements[i].depletion;
        }

        switch (itype) {
        case 1:
            // fully ionized
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++) {
                n_and_ion_fracs[i].ion_fracs[j] = MIN_XION;
                n_and_ion_fracs[i].ion_fracs_new[j] = MIN_XION;
            }
            n_and_ion_fracs[i].ion_fracs[elements[i].n_ions - 1] = 1.0;
            n_and_ion_fracs[i].ion_fracs_new[elements[i].n_ions - 1] = 1.0;
            break;
        case 2:
            // fully neutral
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++) {
                n_and_ion_fracs[i].ion_fracs[j] = MIN_XION;
                n_and_ion_fracs[i].ion_fracs_new[j] = MIN_XION;
            }
            n_and_ion_fracs[i].ion_fracs[0] = 1.0;
            n_and_ion_fracs[i].ion_fracs_new[0] = 1.0;
            break;
        case 3:
            // fully molecular or neutral
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++) {
                n_and_ion_fracs[i].ion_fracs[j] = MIN_XION;
                n_and_ion_fracs[i].ion_fracs_new[j] = MIN_XION;
            }
            if (elements[i].n_mol > 0) {
                n_and_ion_fracs[i].ion_fracs[elements[i].n_ions + elements[i].n_mol - 1] = 1.0;
                n_and_ion_fracs[i].ion_fracs_new[elements[i].n_ions + elements[i].n_mol - 1] = 1.0;
            } else {
                n_and_ion_fracs[i].ion_fracs[0] = 1.0;
                n_and_ion_fracs[i].ion_fracs_new[0] = 1.0;
            }
            break;
        default:
            printf("YOU NEED TO SELECT AN APPROPRIATE CASE\n");
            break;
        }
    }
}

KOKKOS_FUNCTION
double get_ne(const Element* elements, const CompactIonData &n_and_ion_fracs, const bool use_new)
{
    /*
    Returns the electron number density by looping over all elements
    and summing their contributions
    */
    double ne = 0E0;

    // Loop over all elements
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (n_and_ion_fracs.n_element[i] == 0.0)
            continue;

        // Get the number of ions
        int n_ions = elements[i].n_ions;

        // Loop over all ions
        // Start loop at 1, no electrons in the ground state
        for (int j = 1; j < n_ions; j++) {
            if (use_new) {
                ne += n_and_ion_fracs[i].n_element_new * n_and_ion_fracs[i].ion_fracs_new[j] * j;
            } else {
                ne += n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs[j] * j;
            }
        } // End loop ion fracs
    } // End loop elements

    return ne;
}

KOKKOS_FUNCTION
double get_rho(const Element* elements,
               const CompactIonData& n_and_ion_fracs) {
    double rho = 0.0;
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (n_and_ion_fracs.n_element[i] == 0.0) continue;
        rho += n_and_ion_fracs[i].n_element * elements[i].atomic_mass; // amu/cm^3
    }
    return rho;
}

KOKKOS_FUNCTION
double get_mu(double ne,
              const Element* elements,
              const CompactIonData& n_and_ion_fracs,
              bool use_new, bool include_H2, double nCO) {
    /*
    Calculated the mean molecular weight
    */
    double m_bar = 0.0;
    double n_hat = 0.0;

    // Loop over all elements
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (n_and_ion_fracs.n_element[i] == 0.0) continue;

        // Get the number of ions
        int n_ions = elements[i].n_ions;

        // Loop over all ions
        for (int j = 0; j < n_ions; j++) {
            if (use_new){
                m_bar += n_and_ion_fracs[i].n_element_new * n_and_ion_fracs[i].ion_fracs_new[j] * elements[i].atomic_mass;
                n_hat += n_and_ion_fracs[i].n_element_new * n_and_ion_fracs[i].ion_fracs_new[j];
            } else {
                m_bar += n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs[j] * elements[i].atomic_mass;
                n_hat += n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs[j];
            }
        } // End loop over ions
    } // End loop over elements

    // Include electrons
    n_hat += ne;

    // Include contribution from H2
    if (include_H2) {
        if (use_new){
            m_bar += (n_and_ion_fracs[1].n_element_new * n_and_ion_fracs[1].ion_fracs_new[2] * elements[1].atomic_mass);
            n_hat += (0.5 * n_and_ion_fracs[1].n_element_new * n_and_ion_fracs[1].ion_fracs_new[2]);
        } else {
            m_bar += (n_and_ion_fracs[1].n_element * n_and_ion_fracs[1].ion_fracs[2] * elements[1].atomic_mass);
            n_hat += (0.5 * n_and_ion_fracs[1].n_element * n_and_ion_fracs[1].ion_fracs[2]);
        }
    }

    // Include contribution from CO
    m_bar += nCO * (elements[6].atomic_mass + elements[8].atomic_mass);
    n_hat += nCO;

    double mu = m_bar / n_hat;
    return mu;
}

KOKKOS_INLINE_FUNCTION
void reduce_xion(std::array<double, MAX_ELEMENTS>& ion_fracs, const int size)
{
    /*
    Makes sure that the ion fractions sum to one
    */
    double total_ion_frac = 0.0;

    // Loop over all ions and get the total
    for (int i = 0; i < size; i++) {
        total_ion_frac += ion_fracs[i];
    }
    // Loop again to normalize
    for (int i = 0; i < size; i++) {
        ion_fracs[i] *= (1.0 / total_ion_frac);
    }
}

KOKKOS_INLINE_FUNCTION
void reduce_xion_max(std::array<double, MAX_ELEMENTS>& ion_fracs, int size)
{
    /*
    Makes sure that the ion fractions sum to one and that
    the ion fractions don't go below some minimum value
    */
    double total_ion_frac = 0.0;

    int max_idx = 0;
    double max_frac = 0.0;

    // Loop over all ions and get the total
    for (int i = 0; i < size; i++) {
        ion_fracs[i] = fmax(ion_fracs[i], MIN_XION);
        total_ion_frac += ion_fracs[i];
        if (ion_fracs[i] > max_frac) {
            max_frac = ion_fracs[i];
            max_idx = i;
        }
    }

    // Now reset to 1
    ion_fracs[max_idx] -= (total_ion_frac - 1.0);
}


// Per-cell solver state shared between the step functions of subcycle_chemistry.
// The driver resets the per-iteration fields at the top of every outer
// iteration; each step reads and updates only the fields it owns, and gates
// itself on its compute_* flag, so the steps can be called unconditionally
// (and later launched independently, e.g. from CoolingUpdate_PRISM_GPU.cpp).
struct SubcycleState {
    // Thermal state
    double TK;                  // Temperature [K]
    double mu;                  // Mean molecular weight
    double Tmu_old;             // T/mu
    double Tmu_new;             // T/mu

    // Electrons
    double ne;                  // Electron number density
    double ne_initial;          // Electron number density at iteration start (for convergence)
    double xe;                  // Electron fraction (n_electron/nH)

    // Cosmic rays
    double total_cosmic_ray_ionization_rate;
    double H2_cosmic_ray_ionization_rate;
    double cosmic_ray_scale_factor; // only needed for secondary UVs

    // Radiation --> set by radiation_step
    double G0_local;
    double f_shd_H2;            // H2 self-shielding factor
    double f_shd_CO;            // CO self-shielding factor

    // Molecules --> set by molecular_step, consumed by ions_step
    double alpha_H2_loc;        // H2 formation rate [cm3 s-1]
    double de_H2;               // H2 destruction rate [s-1]
    double nCO_new;

    // Convergence
    double max_residual_all;
    int success_counter;
    bool x_percent_rule;
    bool compute_temperature;
    bool compute_molecules;
    bool compute_atoms;
};


/////////////////////////
//      Radiation      //
/////////////////////////
template <
    bool constant_temperature,
    bool include_H2,
    bool include_CO,
    bool rt_advect,
    bool include_self_shielding
>
KOKKOS_INLINE_FUNCTION
void radiation_step(
    const double ddt,
    const Element* elements,
    const CompactIonData& n_and_ion_fracs,
    const TabulatedData& tabData,
    const double nCO,
    const double dx_SS,
    const double dust_to_gas_mass_ratio_over_mw,
    const std::array<double, N_GROUPS>& N_PHOT,
    std::array<double, N_GROUPS>& N_phot_new,
    std::array<double, N_GROUPS>& phot_att,
    SubcycleState& state,
    // Optional: a group depleted below rad_residual_floor times its density at
    // the start of the hydro step (N_phot_start) is physically spent — in
    // optically thick cells its exponential decay would otherwise pin the
    // subcycle timestep at ddt*kappa ~ 0.07 for thousands of iterations.
    // Such groups still deplete normally but stop gating the controller.
    // rad_residual_floor = 0 (default) reproduces the historical behavior.
    const double rad_residual_floor = 0.0,
    const std::array<double, N_GROUPS>* N_phot_start = nullptr)
{
    // Self-shielding factors
    state.f_shd_H2 = 1.0;
    state.f_shd_CO = 1.0;

    double nHI = n_and_ion_fracs[1].n_element_new * n_and_ion_fracs[1].ion_fracs_new[0];
    double nH2 = 0.5 * n_and_ion_fracs[1].n_element_new * n_and_ion_fracs[1].ion_fracs_new[2];

    if constexpr (include_H2)
        state.f_shd_H2 = comp_SH2(nH2, dx_SS) * comp_Sd(nHI, nH2, dx_SS, dust_to_gas_mass_ratio_over_mw);
    if constexpr (include_CO)
        state.f_shd_CO = comp_SCO(tabData, nCO, nH2, dx_SS);

    double G_conv = 1.6E-3; // [erg/s/cm2] (Habing 1968)

    state.G0_local = 0.0;

    if constexpr (rt_advect) {
        double phAbs = 0.0;
        double group_egy;

        for (int k = 0; k < N_GROUPS; k++) { // Loop over photon groups

            // G0 from the radiation field
            group_egy = tabData.group_energy(k);
            if (group_egy > 5.6 && group_egy < 13.6) {
                state.G0_local += N_PHOT[k] * (C_CGS * tabData.c_frac())
                      * group_egy * EV_2_ERG / G_conv;
            }

            // Get the opacity
            phAbs = 0.0; // zero out absorption
            for (int i = 1; i < MAX_ELEMENTS; i++) { // Loop over all elements
                if (n_and_ion_fracs.n_element[i] == 0.0) continue;

                // Get the number of ions
                int n_ions = elements[i].n_ions;
                const int ei = elem_row(i);

                for (int j = 0; j < n_ions - 1; j++) { // Loop over the number of ions --> no absorption by final ion
                    phAbs += n_and_ion_fracs[i].n_element_new * n_and_ion_fracs[i].ion_fracs_new[j] * tabData.cs_ph(ei,j,k,0);
                } // End loop over ions
            } // End loop over elements

            // Deal with molecular hydrogen separately if needed
            if constexpr (include_H2) {
                if (tabData.group_energy(k) > 11.2 && tabData.group_energy(k) < 13.6)
                    phAbs += nH2 * tabData.cs_ph(elem_row(1), 2, k, 0) * state.f_shd_H2;
                else
                    phAbs += nH2 * tabData.cs_ph(elem_row(1), 2, k, 0);
            }

            // Dust photoabsorption (Zubko+2004 BARE-GR-S)
            double dustAbs = 0.0;
            if (state.TK < 1.0e6) {
                dustAbs = tabData.cs_dust(k) * n_and_ion_fracs[1].n_element_new * dust_to_gas_mass_ratio_over_mw;
            }

            // Next deplete the photons from the group
            phot_att[k] = 1.0 / (1.0 + ddt * (phAbs + dustAbs));
            N_phot_new[k] = N_PHOT[k] * phot_att[k];

            // Measure the redisual
            double residual = fabs((N_phot_new[k] - N_PHOT[k]) / (N_PHOT[k] + 1E-13));
            if (rad_residual_floor > 0.0 && N_phot_start != nullptr
                && N_PHOT[k] < rad_residual_floor * (*N_phot_start)[k])
                residual = 0.0; // group is spent: it no longer gates the timestep
            state.max_residual_all = fmax(state.max_residual_all, residual);

            if (residual > X_PCT_RULE){
                // Reset the success counter
                state.success_counter = 0;
                // Declare the x-percent rule was violated
                state.x_percent_rule = false;
                // Don't compute temperature
                state.compute_temperature = false;
                // Dont compute molecules
                state.compute_molecules = false;
                // Don't compute atoms
                state.compute_atoms = false;
            }
        } // End loop over photon groups
    }
}


/////////////////////////
//     Temperature     //
/////////////////////////
template <
    bool constant_temperature,
    bool include_H2,
    bool include_CO,
    bool rt_advect,
    bool include_self_shielding
>
KOKKOS_INLINE_FUNCTION
void cooling_step(
    const double ddt,
    const double aexp,
    const Element* elements,
    const CompactIonData& n_and_ion_fracs,
    const TabulatedData& tabData,
    const PhysicsFlags& flags,
    const double loc_rho,
    const double nCO,
    const double dust_to_gas_mass_ratio_over_mw,
    const double primary_cosmic_ray_ionization_rate,
    const double ss_factor,
    const double UV_background_G0,
    std::array<double, N_GROUPS>& N_phot_new,
    SubcycleState& state)
{
    if (constant_temperature || !state.compute_temperature) return;

    double cooling_rate = all_cooling(state.TK, state.ne, aexp,
                               n_and_ion_fracs, elements, N_phot_new,
                               nCO, state.G0_local, UV_background_G0, state.f_shd_H2,
                               dust_to_gas_mass_ratio_over_mw,
                               state.xe, primary_cosmic_ray_ionization_rate,
                               state.H2_cosmic_ray_ionization_rate,
                               ss_factor, rt_advect,
                               flags, tabData
                        );

    double cooling_rate_prime = all_cooling(1.001 * state.TK, state.ne, aexp,
                                     n_and_ion_fracs, elements, N_phot_new,
                                     nCO, state.G0_local, UV_background_G0, state.f_shd_H2,
                                     dust_to_gas_mass_ratio_over_mw,
                                     state.xe, primary_cosmic_ray_ionization_rate,
                                     state.H2_cosmic_ray_ionization_rate,
                                     ss_factor, rt_advect,
                                     flags, tabData
                                );
    cooling_rate_prime = (cooling_rate_prime - cooling_rate) / ((1.001*state.TK) - state.TK);


    double X_nHkb = 1.0 / (1.5 * loc_rho * 1.380649e-16);

    cooling_rate *= X_nHkb;
    cooling_rate_prime *= X_nHkb * state.mu;

    double dUU = fabs(fmax(T_MIN, state.Tmu_old + cooling_rate * ddt) - state.Tmu_old);
    state.Tmu_new = fmax(T_MIN, state.Tmu_old + cooling_rate * ddt/(1.0 - cooling_rate_prime * ddt));

    // ! New T2 value
    state.TK = state.Tmu_new * state.mu;

    double residual = fmax(dUU, fabs(state.Tmu_new - state.Tmu_old)) / (state.Tmu_old + T_MIN);
    residual = fmax(residual, fabs(cooling_rate_prime * ddt));

    // Convergence
    // X% rule

    // Max residual over all elements
    state.max_residual_all = fmax(state.max_residual_all, residual);

    if (residual > X_PCT_RULE){
        // Reset the success counter
        state.success_counter = 0;
        // Declare the x-percent rule was violated
        state.x_percent_rule = false;
        // Dont compute molecules
        state.compute_molecules = false;
        // Don't compute atoms
        state.compute_atoms = false;
    }
}


// The two all_cooling evaluations feeding the cooling Newton step (at T and
// 1.001*T for the numerical derivative) are independent, so they are computed
// by two threads writing rate_pair[0] and rate_pair[1]. lane selects which.
// Read-only in state, so two threads sharing a cell can call it concurrently.
template <
    bool constant_temperature,
    bool include_H2,
    bool include_CO,
    bool rt_advect,
    bool include_self_shielding
>
KOKKOS_INLINE_FUNCTION
void cooling_rate_lane(
    const int lane,               // 0 -> T, 1 -> 1.001*T
    double* rate_pair,
    const double aexp,
    const Element* elements,
    const CompactIonData& n_and_ion_fracs,
    const TabulatedData& tabData,
    const PhysicsFlags& flags,
    const double nCO,
    const double dust_to_gas_mass_ratio_over_mw,
    const double primary_cosmic_ray_ionization_rate,
    const double ss_factor,
    const double UV_background_G0,
    std::array<double, N_GROUPS>& N_phot_new,
    const SubcycleState& state)
{
    if (constant_temperature || !state.compute_temperature) return;

    rate_pair[lane] = all_cooling((lane == 0 ? 1.0 : 1.001) * state.TK, state.ne, aexp,
                                  n_and_ion_fracs, elements, N_phot_new,
                                  nCO, state.G0_local, UV_background_G0, state.f_shd_H2,
                                  dust_to_gas_mass_ratio_over_mw,
                                  state.xe, primary_cosmic_ray_ionization_rate,
                                  state.H2_cosmic_ray_ionization_rate,
                                  ss_factor, rt_advect,
                                  flags, tabData);
}


// Newton step of the cooling update from the pair produced by cooling_rate_lane.
// Single-threaded per cell (owns the writes to state).
template <
    bool constant_temperature,
    bool include_H2,
    bool include_CO,
    bool rt_advect,
    bool include_self_shielding
>
KOKKOS_INLINE_FUNCTION
void cooling_update(
    const double* rate_pair,
    const double ddt,
    const double loc_rho,
    SubcycleState& state)
{
    if (constant_temperature || !state.compute_temperature) return;

    double cooling_rate = rate_pair[0];
    double cooling_rate_prime = (rate_pair[1] - cooling_rate) / ((1.001*state.TK) - state.TK);


    double X_nHkb = 1.0 / (1.5 * loc_rho * 1.380649e-16);

    cooling_rate *= X_nHkb;
    cooling_rate_prime *= X_nHkb * state.mu;

    double dUU = fabs(fmax(T_MIN, state.Tmu_old + cooling_rate * ddt) - state.Tmu_old);
    state.Tmu_new = fmax(T_MIN, state.Tmu_old + cooling_rate * ddt/(1.0 - cooling_rate_prime * ddt));

    // ! New T2 value
    state.TK = state.Tmu_new * state.mu;

    double residual = fmax(dUU, fabs(state.Tmu_new - state.Tmu_old)) / (state.Tmu_old + T_MIN);
    residual = fmax(residual, fabs(cooling_rate_prime * ddt));

    // Convergence
    // X% rule

    // Max residual over all elements
    state.max_residual_all = fmax(state.max_residual_all, residual);

    if (residual > X_PCT_RULE){
        // Reset the success counter
        state.success_counter = 0;
        // Declare the x-percent rule was violated
        state.x_percent_rule = false;
        // Dont compute molecules
        state.compute_molecules = false;
        // Don't compute atoms
        state.compute_atoms = false;
    }
}


// Team variant: the two evaluations run on two team threads, then lane 0 does
// the Newton step. rate_pair must point to storage shared by the team. Used by
// the on-device straggler-tail loop, where a whole team already owns one cell.
template <
    bool constant_temperature,
    bool include_H2,
    bool include_CO,
    bool rt_advect,
    bool include_self_shielding,
    typename TeamMember
>
KOKKOS_INLINE_FUNCTION
void cooling_step_team(
    const TeamMember& team,
    double* rate_pair,
    const double ddt,
    const double aexp,
    const Element* elements,
    const CompactIonData& n_and_ion_fracs,
    const TabulatedData& tabData,
    const PhysicsFlags& flags,
    const double loc_rho,
    const double nCO,
    const double dust_to_gas_mass_ratio_over_mw,
    const double primary_cosmic_ray_ionization_rate,
    const double ss_factor,
    const double UV_background_G0,
    std::array<double, N_GROUPS>& N_phot_new,
    SubcycleState& state)
{
    if (constant_temperature || !state.compute_temperature) return;

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 2), [&](const int lane) {
        cooling_rate_lane<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            lane, rate_pair, aexp, elements, n_and_ion_fracs, tabData, flags,
            nCO, dust_to_gas_mass_ratio_over_mw, primary_cosmic_ray_ionization_rate,
            ss_factor, UV_background_G0, N_phot_new, state);
    });
    team.team_barrier();

    Kokkos::single(Kokkos::PerTeam(team), [&]() {
        cooling_update<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            rate_pair, ddt, loc_rho, state);
    });
}


/////////////////////////
//      Molecules      //
/////////////////////////
template <
    bool constant_temperature,
    bool include_H2,
    bool include_CO,
    bool rt_advect,
    bool include_self_shielding
>
KOKKOS_INLINE_FUNCTION
void molecular_step(
    const double ddt,
    const Element* elements,
    CompactIonData& n_and_ion_fracs,
    const TabulatedData& tabData,
    const double nCO,
    const double dust_to_gas_mass_ratio_over_mw,
    const double UV_background_G0,
    const std::array<double, N_GROUPS>& N_phot_new,
    SubcycleState& state)
{
    // Rates consumed by ions_step: reset every iteration, even when this
    // step is skipped, so ions_step sees zero molecular coupling.
    state.alpha_H2_loc = 0.0;
    state.de_H2 = 0.0;
    state.nCO_new = nCO;

    if (!state.compute_molecules) return;

    // Molecular Hydrogen
    if constexpr (include_H2) {

        // H2 fraction --> Note that we actually store 2 * xH2
        double xH2 = n_and_ion_fracs[1].ion_fracs_new[2]/2.0;

        double cr_H2 = 0.E0;

        // Creation //

        // Contains formation on dust and via the primordial channel
        state.alpha_H2_loc = alpha_H2(state.TK, dust_to_gas_mass_ratio_over_mw, state.xe, state.H2_cosmic_ray_ionization_rate,
                                state.G0_local*state.f_shd_H2 + UV_background_G0, n_and_ion_fracs[1].ion_fracs_new[0], n_and_ion_fracs[1].ion_fracs_new[1],
                                n_and_ion_fracs[1].n_element_new); //! [cm3 s-1]

        // We can only create H2 if there is any HI and HII
        cr_H2 += state.alpha_H2_loc;

        // Destruction //

        // Collisional destruction
        double beta_H2_loc = beta_H2_krome(state.TK, n_and_ion_fracs[1].ion_fracs_new[0] * n_and_ion_fracs[1].n_element_new,
                                    state.ne, xH2 * n_and_ion_fracs[1].n_element_new,
                                    n_and_ion_fracs[2].ion_fracs_new[0] * n_and_ion_fracs[2].n_element_new);
        state.de_H2 += beta_H2_loc;

        // Photodissociation UV background
        state.de_H2 += UV_background_G0 * 5.68E-11;

        // Photodissociation from local radiation field
        if constexpr (rt_advect) {
            for (int k = 0; k < N_GROUPS; k++) {
                if (tabData.group_energy(k) > 11.2 && tabData.group_energy(k) < 13.6)
                    // self-shield the LW band
                    state.de_H2 += tabData.cs_ph(elem_row(1), 2, k, 0) * N_phot_new[k] * state.f_shd_H2;
                else
                    state.de_H2 += tabData.cs_ph(elem_row(1), 2, k, 0) * N_phot_new[k];
            }
        }

        // Cosmic ray destruction
        state.de_H2 += state.H2_cosmic_ray_ionization_rate;

        // Update xH2
        xH2 = (cr_H2*ddt + xH2)/(1.+state.de_H2*ddt);

        // Store in the struct
        n_and_ion_fracs[1].ion_fracs_new[2] = 2.0 * fmin(fmax(xH2,MIN_XION),0.5);

        // Convergence
        // X% rule
        double residual = fabs(n_and_ion_fracs[1].ion_fracs_new[2] - n_and_ion_fracs[1].ion_fracs[2]) / (n_and_ion_fracs[1].ion_fracs[2] + X_FM);

        // Max residual over all elements
        state.max_residual_all = fmax(state.max_residual_all,residual);

        if (residual > X_PCT_RULE && n_and_ion_fracs[1].ion_fracs[2] > 1E-10){
            // Reset the success counter
            state.success_counter = 0;
            // Declare the x-percent rule was violated
            state.x_percent_rule = false;
            // Break the loop over ions
            state.compute_atoms = false;
        } else {
            //  Update mu and T
            state.mu = get_mu(state.ne, elements, n_and_ion_fracs, true, include_H2, nCO);
            state.TK = state.Tmu_new * state.mu;
            if (constant_temperature) state.TK = state.Tmu_old;
        }
    }

    // CO
    if constexpr (include_CO) {
        if (state.compute_atoms) {
            // Check if there is enough C, O, and CO to warrant computing the chemistry
            if ((n_and_ion_fracs[6].n_element_new + n_and_ion_fracs[8].n_element_new + nCO)/n_and_ion_fracs[1].n_element_new > 1e-10) {

                double nH   = n_and_ion_fracs[1].n_element_new;
                double nCII = n_and_ion_fracs[6].n_element_new * n_and_ion_fracs[6].ion_fracs_new[1];
                double nOI  = n_and_ion_fracs[8].n_element_new * n_and_ion_fracs[8].ion_fracs_new[0];
                double nH2  = n_and_ion_fracs[1].n_element_new * n_and_ion_fracs[1].ion_fracs_new[2] / 2.0;
                double x_OI = nOI / nH;

                //!! Creation !!
                double cr_CO = alpha_CO(state.G0_local*state.f_shd_CO + UV_background_G0, state.H2_cosmic_ray_ionization_rate, nCII, nH2, x_OI, nH * n_and_ion_fracs[1].ion_fracs_new[0]);

                //!! Destruction !!
                double de_CO = beta_CO(state.G0_local*state.f_shd_CO + UV_background_G0, state.H2_cosmic_ray_ionization_rate);

                // Compute the initial guess of new nCO
                state.nCO_new = (nCO + cr_CO*ddt) / (1.0 + de_CO*ddt);

                // Tentative update
                double delta_CO = state.nCO_new - nCO;

                // Enforce positivity floors: don't destroy more CO than exists,
                // and don't create more than allowed by available CII and OI
                double max_delta_CO = fmax(fmin(nCII - 1e-30, nOI - 1e-30), 0.0);
                double min_delta_CO = fmin(-nCO + 1e-30, 0.0);

                // Clip delta_CO to enforce physical bounds
                delta_CO = fmax(fmin(delta_CO, max_delta_CO), min_delta_CO);

                // Apply updates
                state.nCO_new  = nCO + delta_CO;
                double nCII_new = nCII - delta_CO;
                double nOI_new  = nOI  - delta_CO;

                // Now update the ion fractions for C and O
                double tot_C = n_and_ion_fracs[6].n_element_new - delta_CO;
                for (int i=0; i<elements[6].n_ions; i++) {
                    if (i != 1) {
                        n_and_ion_fracs[6].ion_fracs_new[i] = n_and_ion_fracs[6].n_element_new * n_and_ion_fracs[6].ion_fracs_new[i] / tot_C;
                    } else {
                        n_and_ion_fracs[6].ion_fracs_new[i] = nCII_new / tot_C;
                    }
                }

                double tot_O = n_and_ion_fracs[8].n_element_new - delta_CO;
                for (int i=0; i<elements[8].n_ions; i++) {
                    if (i != 0) {
                        n_and_ion_fracs[8].ion_fracs_new[i] = n_and_ion_fracs[8].n_element_new * n_and_ion_fracs[8].ion_fracs_new[i] / tot_O;
                    } else {
                        n_and_ion_fracs[8].ion_fracs_new[i] = nOI_new / tot_O;
                    }
                }

                // Check for convergence
                double residual = fabs(state.nCO_new - nCO) / (nCO + X_FM);

                // Max residual over all elements
                state.max_residual_all = fmax(state.max_residual_all,residual);

                if (residual > X_PCT_RULE && nCO > 1E-10){
                    // Reset the success counter
                    state.success_counter = 0;
                    // Declare the x-percent rule was violated
                    state.x_percent_rule = false;
                    // Break the loop over ions
                    state.compute_atoms = false;
                } else {
                    // Update species number densities
                    n_and_ion_fracs[6].n_element_new -= delta_CO;
                    n_and_ion_fracs[8].n_element_new -= delta_CO;
                }
            }
        }
    }
}


/////////////////////////
//        Atoms        //
/////////////////////////
template <
    bool constant_temperature,
    bool include_H2,
    bool include_CO,
    bool rt_advect,
    bool include_self_shielding
>
KOKKOS_INLINE_FUNCTION
void ions_gs_element(
    const int i,
    const double ddt,
    const Element* elements,
    CompactIonData& n_and_ion_fracs,
    const TabulatedData& tabData,
    const PhysicsFlags& flags,
    const double dust_to_gas_mass_ratio_over_mw,
    const double UV_background_G0,
    const double primary_cosmic_ray_ionization_rate,
    const double ss_factor,
    const std::array<double, N_GROUPS>& N_phot_new,
    const double dust_effective_number_density,
    SubcycleState& state)
{
    // Gauss-Seidel update of one element's ionization stages followed by its
    // normalization — the body of ions_step's element loop, also called
    // serially for H/He by HandHe_ions_step. Returns with
    // state.x_percent_rule == false (and no normalization applied) if any
    // stage violates the X% rule.

        // Get the number of ions
        int n_ions = elements[i].n_ions;
        const int ei = elem_row(i);

        // Loop over the number of ions
        for (int j = 0; j < n_ions; j++) {

            /////////////////////////
            //       Creation      //
            /////////////////////////
            double cr = 0.E0;

            // Account for molecular hydrogen
            if ((i == 1) & (j == 0) & include_H2) { // select only HI
                // Note: no factor of 2 needed since is 2*xH2
                cr += state.de_H2 * n_and_ion_fracs[1].ion_fracs_new[2];
            }

            // Recombinations of the more excited ionization state
            if (j < (n_ions - 1)) cr += recombination(state.TK, j+1, i, tabData) * state.ne * n_and_ion_fracs[i].ion_fracs_new[j+1];

            // Collisional ionization of the less excited state
            if (flags.include_collisional_ionization) {
                if (j > 0) cr += collisional_ionization(state.TK, j-1, i, tabData) * state.ne * n_and_ion_fracs[i].ion_fracs_new[j-1];
            }

            // Photoionization of the less excited state
            if (flags.include_HM12_UVB) {
                if (j > 0) cr += tabData.HM12_UVB_z(ei, j-1, 0) * ss_factor * n_and_ion_fracs[i].ion_fracs_new[j-1];
            }

            // Photoionization by sub-ionizing ISRF --> only impacts lowest ionization states
            if (flags.include_photoionization) {
                if (j == 1) cr += UV_background_G0 * elements[i].G0_photo_rate * n_and_ion_fracs[i].ion_fracs_new[j-1];
            }

            // Cosmic ray ionization of the less excited state
            if (flags.include_cosmic_ray_ionization) {
                if (j > 0) cr += tabData.cosmic_ray_ionization_rates(ei, j-1) * state.total_cosmic_ray_ionization_rate * n_and_ion_fracs[i].ion_fracs_new[j-1];

                // Cosmic ray ionization of the less excited state from induced UV
                if (j == 1) cr += tabData.cosmic_ray_ionization_rates_induced_UV(ei) * state.cosmic_ray_scale_factor * n_and_ion_fracs[i].ion_fracs_new[j-1];
            }

            // Recombination on dust for the more excited state
            if (flags.include_dust_recombination) {
                if (j < (n_ions - 1)) cr += dust_recombination_rates(j+1, i, state.TK, UV_background_G0, state.ne, tabData.dust_rec_coefs) * dust_effective_number_density * n_and_ion_fracs[i].ion_fracs_new[j+1];
            }


            // Photoionization of the less excited state from local radiation field
            if (rt_advect) {
                if (j > 0) {
                    for (int k = 0; k < N_GROUPS; k++) { // Loop over photon groups
                        cr += N_phot_new[k] * tabData.cs_ph(ei, j-1, k, 0) * n_and_ion_fracs[i].ion_fracs_new[j-1];
                    } // End loop over photon groups
                }
            }

            /////////////////////////
            //     Destruction     //
            /////////////////////////
            double de = 0.E0;

            // Account for molecular hydrogen
            if ((i == 1) & (j == 0) & include_H2) { // select only HI
                de += 2.0 * state.alpha_H2_loc / (n_and_ion_fracs[1].ion_fracs_new[0] + 1e-40);
            }

            // Collisional ionization
            if (flags.include_collisional_ionization) {
                if (j < (n_ions - 1)) de += collisional_ionization(state.TK, j, i, tabData) * state.ne;
            }

            // Photoionization
            if (flags.include_HM12_UVB) {
                if (j < (n_ions - 1)) de += tabData.HM12_UVB_z(ei, j, 0) * ss_factor;
            }

            // Photoionization by sub-ionizing ISRF --> only impacts lowest ionization states
            if (flags.include_photoionization) {
                if (j == 0) de += UV_background_G0 * elements[i].G0_photo_rate;
            }

            // Recombination
            if (j > 0) de += recombination(state.TK, j, i, tabData) * state.ne;

            // Cosmic ray ionization
            if (flags.include_cosmic_ray_ionization) {
                if (j < (n_ions - 1)) de += tabData.cosmic_ray_ionization_rates(ei, j) * state.total_cosmic_ray_ionization_rate;

                // Cosmic ray ionization from induced UV
                if (j == 0) de += tabData.cosmic_ray_ionization_rates_induced_UV(ei) * state.cosmic_ray_scale_factor;
            }

            // Recombination on dust
            if (flags.include_dust_recombination) {
                if (j > 0) de += dust_recombination_rates(j, i, state.TK, UV_background_G0, state.ne, tabData.dust_rec_coefs) * dust_effective_number_density;
            }


            // Photoionization from local radiation field
            if (rt_advect) {
                if (j < (n_ions - 1)) {
                    for (int k = 0; k < N_GROUPS; k++) { // Loop over photon groups
                        de += N_phot_new[k] * tabData.cs_ph(ei, j, k, 0);
                    } // End loop over photon groups
                }
            }

            /////////////////////////
            //   Charge Transfer   //
            /////////////////////////
            // Note, this was split off due to cross species
            // coupling. Saves us an extra double loop
            // Charge exchange
            if (flags.include_charge_exchange) {
                if (i == 1) {  // If element is hydrogen
                    // Loop over all other elements
                    for (int ii = 2; ii < MAX_ELEMENTS; ii++){
                        if (n_and_ion_fracs.n_element[ii] == 0.0) continue;

                        // Loop over all other ionization states
                        for (int jj = 0; jj < elements[ii].n_ions; jj++) {
                            double paired_ion_number_density = n_and_ion_fracs[ii].ion_fracs_new[jj] * n_and_ion_fracs[ii].n_element_new;
                            if (j == 0) { // H
                                cr += charge_transfer_ionization(jj,ii,state.TK, tabData) * n_and_ion_fracs[i].ion_fracs_new[j+1] * paired_ion_number_density; //! Example:  O + H+ => O+ + H
                                de += charge_transfer_recombination(jj,ii,state.TK, tabData) * paired_ion_number_density; //! Example:  O+ + H => O + H+
                            } else { // H+
                                de += charge_transfer_ionization(jj,ii,state.TK, tabData) * paired_ion_number_density; //! Example:  O + H+ => O+ + H
                                cr += charge_transfer_recombination(jj,ii,state.TK, tabData) * n_and_ion_fracs[i].ion_fracs_new[j-1] * paired_ion_number_density; //! Example:  O+ + H => O + H+
                            }
                        } // end loop over ionization states
                    } // end loop over other elements
                } else { // All other elements
                    double HI_number_density = n_and_ion_fracs[1].ion_fracs_new[0] * n_and_ion_fracs[1].n_element_new;
                    double HII_number_density = n_and_ion_fracs[1].ion_fracs_new[1] * n_and_ion_fracs[1].n_element_new;

                    if (j > 0) {
                        // Ionization from less excited state
                        cr += charge_transfer_ionization(j-1,i,state.TK, tabData) * n_and_ion_fracs[i].ion_fracs_new[j-1] * HII_number_density; //! Example:  O + H+ => O+ + H

                        // Charge exchange recombination
                        de += charge_transfer_recombination(j,i,state.TK, tabData) * HI_number_density; //! Example:  O+ + H => O + H+
                    }

                    if (j < (n_ions - 1)) {
                        // Charge exchange ionization
                        de += charge_transfer_ionization(j,i,state.TK, tabData) * HII_number_density; //! Example:  O + H+ => O+ + H

                        // Charge exchange recombination from the more excited state
                        cr += charge_transfer_recombination(j+1,i,state.TK, tabData) * n_and_ion_fracs[i].ion_fracs_new[j+1] * HI_number_density; //! Example:  O+ + H => O + H+
                    }
                }
            }

            /////////////////////////
            //       Update        //
            /////////////////////////

            // The update
            n_and_ion_fracs[i].ion_fracs_new[j] = (cr*ddt + n_and_ion_fracs[i].ion_fracs_new[j])/(1.0 + de*ddt);
            n_and_ion_fracs[i].ion_fracs_new[j] = fmin(fmax(n_and_ion_fracs[i].ion_fracs_new[j],MIN_XION),1.0);

            // Get the new electron fraction
            state.ne += ((n_and_ion_fracs[i].n_element_new * n_and_ion_fracs[i].ion_fracs_new[j]) - (n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs[j])) * (double)j;
            state.xe = state.ne / n_and_ion_fracs[1].n_element_new;

            if (flags.include_cosmic_ray_ionization) {
                double phi_s = secondary_cr_rates(state.xe);
                state.total_cosmic_ray_ionization_rate = primary_cosmic_ray_ionization_rate * (1.0 + phi_s);
            }

            // Update mu and T --> only H and He (others don't matter)
            if (i < 3) {
                //  Update mu and T
                state.mu = get_mu(state.ne, elements, n_and_ion_fracs, true, include_H2, state.nCO_new);
                state.TK = state.Tmu_new * state.mu;
                if (constant_temperature) state.TK = state.Tmu_old;
            }

            /////////////////////////
            //     Convergence     //
            /////////////////////////
            // X% rule
            double residual = fabs(n_and_ion_fracs[i].ion_fracs_new[j] - n_and_ion_fracs[i].ion_fracs[j]) / (n_and_ion_fracs[i].ion_fracs[j] + X_FM);

            // electron density residual
            double ne_residual = fabs(state.ne - state.ne_initial) / (state.ne + X_FM);
            residual = fmax(residual,ne_residual);

            // Max residual over all elements
            state.max_residual_all = fmax(state.max_residual_all,residual);

            // Check if X percent rule is broken
            if (n_and_ion_fracs[i].ion_fracs[j] > 1E-10){
                if (residual > X_PCT_RULE){
                    // Reset the success counter
                    state.success_counter = 0;
                    // Declare the x-percent rule was violated
                    state.x_percent_rule = false;
                    // Break the loop over ions
                    break;
                }
            }
        } // End ion loop

        //If the X% rule is violated restart with shorter timestep
        if (!state.x_percent_rule){
            // Reset the success counter
            state.success_counter = 0;
            // Skip the normalization; the iteration will be discarded
            return;
        }

        // REDUCE SO THAT IONIZATION FRACTIONS SUM TO 1
        int size = elements[i].n_ions + elements[i].n_mol;

        if (i <= 2) {
            double current_mass_frac = 0.0;
            for (int i_tmp = 1; i_tmp < size; i_tmp++) {
                current_mass_frac += n_and_ion_fracs[i].ion_fracs_new[i_tmp];
            }

            n_and_ion_fracs[i].ion_fracs_new[0] = 0.0;

            if (current_mass_frac >= 0.0) {
                if (current_mass_frac <= 1.0) {
                    n_and_ion_fracs[i].ion_fracs_new[0] = 1.0 - current_mass_frac;
                } else {
                    for (int i_tmp = 1; i_tmp < size; i_tmp++) {
                        n_and_ion_fracs[i].ion_fracs_new[i_tmp] /= current_mass_frac;
                    }
                }
            } else {
                n_and_ion_fracs[i].ion_fracs_new[0] = 1.0;
                for (int i_tmp = 1; i_tmp < size; i_tmp++) {
                    n_and_ion_fracs[i].ion_fracs_new[i_tmp] = 0.0;
                }
            }
        } else {
            double current_mass_frac = 0.0;
            for (int i_tmp = 0; i_tmp < size; i_tmp++) {
                current_mass_frac += n_and_ion_fracs[i].ion_fracs_new[i_tmp];
            }

            for (int i_tmp = 0; i_tmp < size; i_tmp++) {
                n_and_ion_fracs[i].ion_fracs_new[i_tmp] /= current_mass_frac;
            }
        }
}


template <
    bool constant_temperature,
    bool include_H2,
    bool include_CO,
    bool rt_advect,
    bool include_self_shielding
>
KOKKOS_INLINE_FUNCTION
void ions_step(
    const double ddt,
    const Element* elements,
    CompactIonData& n_and_ion_fracs,
    const TabulatedData& tabData,
    const PhysicsFlags& flags,
    const double dust_to_gas_mass_ratio_over_mw,
    const double UV_background_G0,
    const double primary_cosmic_ray_ionization_rate,
    const double ss_factor,
    const std::array<double, N_GROUPS>& N_phot_new,
    SubcycleState& state)
{
    if (!state.compute_atoms) return;

    // Update the HI number density in case it has changed e.g. due to charge exchange
    // TODO(code) --> check this, tis is just H, not HI
    double dust_effective_number_density = n_and_ion_fracs[1].n_element_new * dust_to_gas_mass_ratio_over_mw;

    // Loop over all elements
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (n_and_ion_fracs.n_element[i] == 0.0) continue;

        ions_gs_element<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            i, ddt, elements, n_and_ion_fracs, tabData, flags,
            dust_to_gas_mass_ratio_over_mw, UV_background_G0,
            primary_cosmic_ray_ionization_rate, ss_factor, N_phot_new,
            dust_effective_number_density, state);

        //If the X% rule was violated restart with shorter timestep
        if (!state.x_percent_rule) break;
    } // End element loop
}


// Sequential (Gauss-Seidel) update of hydrogen then helium on one thread:
// exact in-place ne/xe/mu/TK updates and per-stage residual checks, as
// ions_step does element-by-element. The stiff H/He/electron coupling needs
// this sequential treatment; the (weakly coupled) metals are then swept in
// parallel by metal_ions_step.
template <
    bool constant_temperature,
    bool include_H2,
    bool include_CO,
    bool rt_advect,
    bool include_self_shielding
>
KOKKOS_INLINE_FUNCTION
void HandHe_ions_step(
    const double ddt,
    const Element* elements,
    CompactIonData& n_and_ion_fracs,
    const TabulatedData& tabData,
    const PhysicsFlags& flags,
    const double dust_to_gas_mass_ratio_over_mw,
    const double UV_background_G0,
    const double primary_cosmic_ray_ionization_rate,
    const double ss_factor,
    const std::array<double, N_GROUPS>& N_phot_new,
    SubcycleState& state)
{
    if (!state.compute_atoms) return;

    const double dust_effective_number_density = n_and_ion_fracs[1].n_element_new * dust_to_gas_mass_ratio_over_mw;

    for (int i = 1; i <= 2; i++) {
        if (n_and_ion_fracs.n_element[i] == 0.0) continue;

        ions_gs_element<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            i, ddt, elements, n_and_ion_fracs, tabData, flags,
            dust_to_gas_mass_ratio_over_mw, UV_background_G0,
            primary_cosmic_ray_ionization_rate, ss_factor, N_phot_new,
            dust_effective_number_density, state);

        if (!state.x_percent_rule) break;
    }
}


// Team wrapper for the fused tail kernel: runs the sequential H/He update on
// a single team thread, then a team barrier so the updated H/He fractions and
// state are visible to every lane before metal_ions_step.
template <
    bool constant_temperature,
    bool include_H2,
    bool include_CO,
    bool rt_advect,
    bool include_self_shielding,
    typename TeamMember
>
KOKKOS_INLINE_FUNCTION
void HandHe_ions_step(
    const TeamMember& team,
    const double ddt,
    const Element* elements,
    CompactIonData& n_and_ion_fracs,
    const TabulatedData& tabData,
    const PhysicsFlags& flags,
    const double dust_to_gas_mass_ratio_over_mw,
    const double UV_background_G0,
    const double primary_cosmic_ray_ionization_rate,
    const double ss_factor,
    const std::array<double, N_GROUPS>& N_phot_new,
    SubcycleState& state)
{
    if (!state.compute_atoms) return;

    Kokkos::single(Kokkos::PerTeam(team), [&]() {
        HandHe_ions_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            ddt, elements, n_and_ion_fracs, tabData, flags,
            dust_to_gas_mass_ratio_over_mw, UV_background_G0,
            primary_cosmic_ray_ionization_rate, ss_factor, N_phot_new, state);
    });
    team.team_barrier();
}


// Parallel (Jacobi) update of the metal ionization stages: one team thread per
// (element, stage) entry of ion_map, which holds metal stages only (Z >= 3; H
// and He were already updated by HandHe_ions_step). Every stage
// reads a frozen snapshot of the post-H/He ion fractions with ne, xe, mu, TK
// and the cosmic-ray rates frozen, so the metals see the sequential H/He
// result but update independently of each other. The electron-density residual
// is computed once from the summed new fractions. ne/xe/mu/TK are not written
// back: the driver recomputes them from committed fractions on commit/prep.
// Must be called after HandHe_ions_step. Scratch level 0 must provide
// MAX_TOTAL_IONS * sizeof(CompactIonData::xion_t) bytes per team.
template <
    bool constant_temperature,
    bool include_H2,
    bool include_CO,
    bool rt_advect,
    bool include_self_shielding,
    typename TeamMember
>
KOKKOS_INLINE_FUNCTION
void metal_ions_step(
    const TeamMember& team,
    const double ddt,
    const Element* elements,
    CompactIonData& n_and_ion_fracs,
    const TabulatedData& tabData,
    const PhysicsFlags& flags,
    const double dust_to_gas_mass_ratio_over_mw,
    const double UV_background_G0,
    const double primary_cosmic_ray_ionization_rate,
    const double ss_factor,
    const std::array<double, N_GROUPS>& N_phot_new,
    const uint16_t* ion_map,
    const int n_ion_work,
    SubcycleState& state)
{
    // Skip if an earlier step (or the H/He phase) already broke the X% rule
    if (!state.compute_atoms || !state.x_percent_rule) return;

    using xion_t = CompactIonData::xion_t;

    // Frozen snapshot of the post-H/He ion fractions: the Jacobi sweep reads
    // its own element's stages from here while writing updates to ion_fracs_new
    // (metals are untouched by the H/He phase, so the snapshot holds their
    // post-molecular values)
    xion_t* x_snap = (xion_t*) team.team_scratch(0).get_shmem(MAX_TOTAL_IONS * sizeof(xion_t));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, MAX_TOTAL_IONS), [&](const int t) {
        x_snap[t] = n_and_ion_fracs.ion_fracs_new_flat[t];
    });
    team.team_barrier();

    // See the TODO(code) note in ions_step
    const double dust_effective_number_density = n_and_ion_fracs[1].n_element_new * dust_to_gas_mass_ratio_over_mw;

    // Live H fractions (updated in place by HandHe_ions_step) for metal charge exchange
    const xion_t* x_H = n_and_ion_fracs.ion_fracs_new_flat + n_and_ion_fracs.offsets[1];

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_ion_work), [&](const int t) {
        const int i = ion_map[t] >> 8;
        const int j = ion_map[t] & 0xFF;
        if (n_and_ion_fracs.n_element[i] == 0.0) return;

        const int n_ions = elements[i].n_ions;
        const int ei = elem_row(i);
        const xion_t* xi_snap = x_snap + n_and_ion_fracs.offsets[i];

        /////////////////////////
        //       Creation      //
        /////////////////////////
        double cr = 0.E0;

        // Recombinations of the more excited ionization state
        if (j < (n_ions - 1)) cr += recombination(state.TK, j+1, i, tabData) * state.ne * xi_snap[j+1];

        // Collisional ionization of the less excited state
        if (flags.include_collisional_ionization) {
            if (j > 0) cr += collisional_ionization(state.TK, j-1, i, tabData) * state.ne * xi_snap[j-1];
        }

        // Photoionization of the less excited state
        if (flags.include_HM12_UVB) {
            if (j > 0) cr += tabData.HM12_UVB_z(ei, j-1, 0) * ss_factor * xi_snap[j-1];
        }

        // Photoionization by sub-ionizing ISRF --> only impacts lowest ionization states
        if (flags.include_photoionization) {
            if (j == 1) cr += UV_background_G0 * elements[i].G0_photo_rate * xi_snap[j-1];
        }

        // Cosmic ray ionization of the less excited state
        if (flags.include_cosmic_ray_ionization) {
            if (j > 0) cr += tabData.cosmic_ray_ionization_rates(ei, j-1) * state.total_cosmic_ray_ionization_rate * xi_snap[j-1];

            // Cosmic ray ionization of the less excited state from induced UV
            if (j == 1) cr += tabData.cosmic_ray_ionization_rates_induced_UV(ei) * state.cosmic_ray_scale_factor * xi_snap[j-1];
        }

        // Recombination on dust for the more excited state
        if (flags.include_dust_recombination) {
            if (j < (n_ions - 1)) cr += dust_recombination_rates(j+1, i, state.TK, UV_background_G0, state.ne, tabData.dust_rec_coefs) * dust_effective_number_density * xi_snap[j+1];
        }

        // Photoionization of the less excited state from local radiation field
        if (rt_advect) {
            if (j > 0) {
                for (int k = 0; k < N_GROUPS; k++) { // Loop over photon groups
                    cr += N_phot_new[k] * tabData.cs_ph(ei, j-1, k, 0) * xi_snap[j-1];
                } // End loop over photon groups
            }
        }

        /////////////////////////
        //     Destruction     //
        /////////////////////////
        double de = 0.E0;

        // Collisional ionization
        if (flags.include_collisional_ionization) {
            if (j < (n_ions - 1)) de += collisional_ionization(state.TK, j, i, tabData) * state.ne;
        }

        // Photoionization
        if (flags.include_HM12_UVB) {
            if (j < (n_ions - 1)) de += tabData.HM12_UVB_z(ei, j, 0) * ss_factor;
        }

        // Photoionization by sub-ionizing ISRF --> only impacts lowest ionization states
        if (flags.include_photoionization) {
            if (j == 0) de += UV_background_G0 * elements[i].G0_photo_rate;
        }

        // Recombination
        if (j > 0) de += recombination(state.TK, j, i, tabData) * state.ne;

        // Cosmic ray ionization
        if (flags.include_cosmic_ray_ionization) {
            if (j < (n_ions - 1)) de += tabData.cosmic_ray_ionization_rates(ei, j) * state.total_cosmic_ray_ionization_rate;

            // Cosmic ray ionization from induced UV
            if (j == 0) de += tabData.cosmic_ray_ionization_rates_induced_UV(ei) * state.cosmic_ray_scale_factor;
        }

        // Recombination on dust
        if (flags.include_dust_recombination) {
            if (j > 0) de += dust_recombination_rates(j, i, state.TK, UV_background_G0, state.ne, tabData.dust_rec_coefs) * dust_effective_number_density;
        }

        // Photoionization from local radiation field
        if (rt_advect) {
            if (j < (n_ions - 1)) {
                for (int k = 0; k < N_GROUPS; k++) { // Loop over photon groups
                    de += N_phot_new[k] * tabData.cs_ph(ei, j, k, 0);
                } // End loop over photon groups
            }
        }

        /////////////////////////
        //   Charge Transfer   //
        /////////////////////////
        // Charge exchange couples each metal to hydrogen via the live H fractions
        if (flags.include_charge_exchange) {
            double HI_number_density = x_H[0] * n_and_ion_fracs[1].n_element_new;
            double HII_number_density = x_H[1] * n_and_ion_fracs[1].n_element_new;

            if (j > 0) {
                // Ionization from less excited state
                cr += charge_transfer_ionization(j-1,i,state.TK, tabData) * xi_snap[j-1] * HII_number_density; //! Example:  O + H+ => O+ + H

                // Charge exchange recombination
                de += charge_transfer_recombination(j,i,state.TK, tabData) * HI_number_density; //! Example:  O+ + H => O + H+
            }

            if (j < (n_ions - 1)) {
                // Charge exchange ionization
                de += charge_transfer_ionization(j,i,state.TK, tabData) * HII_number_density; //! Example:  O + H+ => O+ + H

                // Charge exchange recombination from the more excited state
                cr += charge_transfer_recombination(j+1,i,state.TK, tabData) * xi_snap[j+1] * HI_number_density; //! Example:  O+ + H => O + H+
            }
        }

        /////////////////////////
        //       Update        //
        /////////////////////////

        // The update
        double x_new = (cr*ddt + xi_snap[j])/(1.0 + de*ddt);
        n_and_ion_fracs[i].ion_fracs_new[j] = fmin(fmax(x_new,MIN_XION),1.0);
    });
    // I think the residual reduces below redistribute indices across threads,
    // so they may read an ion_fracs_new[j] written by a different thread above.
    team.team_barrier();

    /////////////////////////
    //     Convergence     //
    /////////////////////////

    // X% rule on the metal ion fractions (residual vs the committed values);
    // H/He already folded theirs in HandHe_ions_step
    double max_stage_residual = -999.;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n_ion_work), [&](const int t, double& lmax) {
        const int i = ion_map[t] >> 8;
        const int j = ion_map[t] & 0xFF;
        if (n_and_ion_fracs.n_element[i] == 0.0) return;
        const double x_old = n_and_ion_fracs[i].ion_fracs[j];
        const double x_new = n_and_ion_fracs[i].ion_fracs_new[j];
        lmax = fmax(lmax, fabs(x_new - x_old) / (x_old + X_FM));
    }, Kokkos::Max<double>(max_stage_residual));

    int violated = 0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n_ion_work), [&](const int t, int& lviol) {
        const int i = ion_map[t] >> 8;
        const int j = ion_map[t] & 0xFF;
        if (n_and_ion_fracs.n_element[i] == 0.0) return;
        const double x_old = n_and_ion_fracs[i].ion_fracs[j];
        const double x_new = n_and_ion_fracs[i].ion_fracs_new[j];
        const double residual = fabs(x_new - x_old) / (x_old + X_FM);
        if (x_old > 1E-10 && residual > X_PCT_RULE) lviol = 1;
    }, Kokkos::Max<int>(violated));

    // Electron density from the swept metal fractions; state.ne already carries
    // the H/He contributions from HandHe_ions_step
    double delta_ne = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n_ion_work), [&](const int t, double& lsum) {
        const int i = ion_map[t] >> 8;
        const int j = ion_map[t] & 0xFF;
        if (n_and_ion_fracs.n_element[i] == 0.0) return;
        lsum += ((n_and_ion_fracs[i].n_element_new * n_and_ion_fracs[i].ion_fracs_new[j])
               - (n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs[j])) * (double)j;
    }, delta_ne);

    const double ne_new = state.ne + delta_ne;
    const double ne_residual = fabs(ne_new - state.ne_initial) / (ne_new + X_FM);

    Kokkos::single(Kokkos::PerTeam(team), [&]() {
        state.max_residual_all = fmax(state.max_residual_all, fmax(max_stage_residual, ne_residual));
        // The Max reduction identity is INT_MIN, so compare against > 0
        if (violated > 0 || ne_residual > X_PCT_RULE) {
            // Reset the success counter
            state.success_counter = 0;
            // Declare the x-percent rule was violated
            state.x_percent_rule = false;
        }
    });
    team.team_barrier();

    // REDUCE SO THAT THE METAL IONIZATION FRACTIONS SUM TO 1 (H/He were
    // normalized in HandHe_ions_step)
    if (state.x_percent_rule) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, MAX_ELEMENTS - 1), [&](const int t) {
            const int i = t + 1;
            if (i <= 2) return; // H/He normalized in HandHe_ions_step
            if (n_and_ion_fracs.n_element[i] == 0.0) return;

            int size = elements[i].n_ions + elements[i].n_mol;

            double current_mass_frac = 0.0;
            for (int i_tmp = 0; i_tmp < size; i_tmp++) {
                current_mass_frac += n_and_ion_fracs[i].ion_fracs_new[i_tmp];
            }

            for (int i_tmp = 0; i_tmp < size; i_tmp++) {
                n_and_ion_fracs[i].ion_fracs_new[i_tmp] /= current_mass_frac;
            }
        });
        team.team_barrier();
    }
}


//////////////////////////
//    Prep / Commit     //
//////////////////////////
// The per-iteration prep and commit/timestep-control blocks of
// subcycle_chemistry, split out so a driver can also launch them as separate
// kernels around the step functions (see CoolingUpdate_PRISM_GPU.cpp).

template <
    bool constant_temperature,
    bool include_H2,
    bool include_CO,
    bool rt_advect,
    bool include_self_shielding
>
KOKKOS_INLINE_FUNCTION
void subcycle_prep_iteration(
    const Element* elements,
    CompactIonData &n_and_ion_fracs,
    const double nCO,
    const double total_dt,
    const double total_time,
    const double primary_cosmic_ray_ionization_rate,
    double &ddt,
    SubcycleState &state)
{
    state.compute_atoms = true;
    state.compute_molecules = true;
    state.compute_temperature = true;

    //Loop over elements and set the ion_fracs new to
    //the old ion fracs
    //do the same for the number densities
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (n_and_ion_fracs.n_element[i] == 0.0) continue;

        n_and_ion_fracs[i].n_element_new = n_and_ion_fracs[i].n_element;

        int size = elements[i].n_ions + elements[i].n_mol;
        for (int j = 0; j < size; j++) {
            n_and_ion_fracs[i].ion_fracs_new[j] = n_and_ion_fracs[i].ion_fracs[j];
        }
    } // End element loop

    // First get the electron number density
    state.ne = state.ne_initial;
    // Update the electron fraction
    state.xe = state.ne / n_and_ion_fracs[1].n_element_new;
    // Measure mu
    state.mu = get_mu(state.ne, elements, n_and_ion_fracs, true, include_H2, nCO);
    // Get TK
    state.TK = state.Tmu_old * state.mu;
    if (constant_temperature) state.TK = state.Tmu_old;

    // Measure phi_s for secondary CR ionization
    double phi_s = secondary_cr_rates(state.xe);
    // Now calculate the total CR ionization rate
    state.total_cosmic_ray_ionization_rate = primary_cosmic_ray_ionization_rate * (1.0 + phi_s);
    state.H2_cosmic_ray_ionization_rate = 2.0 * primary_cosmic_ray_ionization_rate * (1.0 + phi_s);

    // Set the cosmic ray scale factor -> only needed for secondary UVs
    state.cosmic_ray_scale_factor = state.H2_cosmic_ray_ionization_rate / 1E-16;

    // Initialize the X percent rule
    state.x_percent_rule = true;

    // Convergence
    state.max_residual_all = -999.;

    // Make sure the timestep isn't larger than how long we
    // need to evolve for, otherwise shorten it
    if (total_dt > 0.0) {
        if ((total_time+ddt) > total_dt){
            ddt = total_dt - total_time;
        }
    }
}


// Timestep-control law shared by subcycle_commit_and_control and the
// group-synchronized driver in CoolingUpdate_PRISM_GPU.cpp (which applies it
// once per sync group, with the group-max residual already divided by
// X_PCT_RULE).
KOKKOS_INLINE_FUNCTION
void subcycle_timestep_control(
    const bool committed,
    const double max_residual_over_x_pct,
    double& ddt)
{
    if (!committed) {
        // If we broke the 10% rule, divide timestep by 2.0
        ddt /= 2.0;
    } else {
        // TODO(code): allow for choice of aggressive vs conservative timestep
        ddt = fmin(0.9 * ddt / pow(0.07+max_residual_over_x_pct,0.3),2.0 * ddt);
    }
    // TODO(code): Make this a parameter for the timestep limiter
    // double tlimit = -0.57142857 * log10(n_and_ion_fracs[1].n_element_new) + 10.85714286;
    double tlimit = 20.0;
    ddt = fmin(ddt,pow(10.0,tlimit));
}


// Commit the iteration if the X% rule held (returns whether it did), then
// update the subcycle timestep for the next iteration.
template <
    bool constant_temperature,
    bool include_H2,
    bool include_CO,
    bool rt_advect,
    bool include_self_shielding
>
KOKKOS_INLINE_FUNCTION
bool subcycle_commit_and_control(
    const Element* elements,
    const TabulatedData& tabData,
    CompactIonData &n_and_ion_fracs,
    double &nCO,
    std::array<double, N_GROUPS> &N_PHOT,
    Array2D_RT &F_PHOT,
    const std::array<double, N_GROUPS> &N_phot_new,
    const std::array<double, N_GROUPS> &phot_att,
    double &ddt,
    double &total_time,
    SubcycleState &state)
{
    const bool committed = state.x_percent_rule;

    // If all elements completed
    // Loop over elements and ions and
    // set old values to new values
    if (state.x_percent_rule) {
        if (!constant_temperature) {
            // Update temperature
            state.Tmu_old = fmin(fmax(state.Tmu_new,T_MIN),T_MAX);
        }
        // Update atomic data
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (n_and_ion_fracs.n_element[i] == 0.0) continue;

            n_and_ion_fracs[i].n_element = n_and_ion_fracs[i].n_element_new;

            int size = elements[i].n_ions + elements[i].n_mol;
            for (int j = 0; j < size; j++) {
                n_and_ion_fracs[i].ion_fracs[j] = n_and_ion_fracs[i].ion_fracs_new[j];
            } // End ion loop
        } // End element loop

        // Update CO if needed
        if (include_CO) {
            nCO = state.nCO_new;
        }

        if constexpr (rt_advect) {
            for (int k = 0; k < N_GROUPS; k++) { // Loop over photon groups
                // Update the photon properties
                N_PHOT[k] = N_phot_new[k];

                // Update flux for all dimensions
                double fred = 0.0;
                for (int dim = 0; dim < N_DIM; dim++) {
                    F_PHOT[k][dim] *= phot_att[k];
                    fred += F_PHOT[k][dim] * F_PHOT[k][dim];
                }
                fred = sqrt(fred) / (C_CGS * tabData.c_frac() * N_PHOT[k]);

                // Reduce the flux if needed
                if (fred > 1.0) {
                    for (int dim = 0; dim < N_DIM; dim++) {
                        F_PHOT[k][dim] /= fred;
                    }
                }
            } // End loop over photon groups
        }

        // Update the total time --> only if the run converged
        total_time += ddt;

        // Update the electron fraction
        state.ne_initial = get_ne(elements, n_and_ion_fracs, false);

        // Update the success counter
        state.success_counter += 1;

    }

    // Set new dt
    state.max_residual_all /= (X_PCT_RULE);
    subcycle_timestep_control(state.x_percent_rule, state.max_residual_all, ddt);

    return committed;
}


template <
    bool constant_temperature,
    bool include_H2=true,
    bool include_CO=false,
    bool rt_advect=false,
    bool include_self_shielding=false
>
KOKKOS_FUNCTION
double subcycle_chemistry(
                      const Element* elements,
                      double loc_rho,
                      CompactIonData &n_and_ion_fracs,
                      double &nCO, double T2,
                      double dx, double aexp, double total_dt, double UV_background_G0,
                      double primary_cosmic_ray_ionization_rate,
                      double dust_to_gas_mass_ratio_over_mw,
                      int conv_its, int max_its, int& total_iter_reached,
                      const TabulatedData& tabData,
                      const PhysicsFlags& flags,
                      std::array<double, N_GROUPS> &N_PHOT,
                      Array2D_RT &F_PHOT) {
    /*
    Computes chemical equilibrium

    if total_dt [s] is <=0 --> run to equilibrium otherwise stop at a fixed time
    */

    //Parameters
    double ddt = 10000.0 * 365.25 * 60. * 60.;
    // If we are not running to convergence, try to take one single step
    if (total_dt > 0) {
        ddt = total_dt;
    }

    // Convergence variables
    int total_iterations        = 0;
    bool model_converged        = false;
    double total_time           = 0.E0;

    // Per-cell state shared between the step functions
    SubcycleState state{};
    state.success_counter = 0;
    state.Tmu_old = T2; // T/mu
    state.Tmu_new = T2; // T/mu

    // Radiation
    std::array<double, N_GROUPS> N_phot_new{};
    std::array<double, N_GROUPS> phot_att{};

    // Self-shielding
    double dx_SS = dx; // [cm] --> set the self-shielding length to the cell size
    double ss_factor = 1.0;
    if (include_self_shielding) {
        ss_factor = exp(-1.0 * n_and_ion_fracs[1].n_element/1E-2);
    }

    // First get the electron number density
    state.ne_initial = get_ne(elements, n_and_ion_fracs, false);

    //# infinite loop
    while (1) {  // Infinite loop: TODO(code): put a max loop counter here and then call an error function
        total_iterations += 1;

        subcycle_prep_iteration<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            elements, n_and_ion_fracs, nCO, total_dt, total_time,
            primary_cosmic_ray_ionization_rate, ddt, state);

        radiation_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            ddt, elements, n_and_ion_fracs, tabData, nCO, dx_SS,
            dust_to_gas_mass_ratio_over_mw,
            N_PHOT, N_phot_new, phot_att, state);

        cooling_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            ddt, aexp, elements, n_and_ion_fracs, tabData, flags,
            loc_rho, nCO, dust_to_gas_mass_ratio_over_mw,
            primary_cosmic_ray_ionization_rate, ss_factor, UV_background_G0,
            N_phot_new, state);

        molecular_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            ddt, elements, n_and_ion_fracs, tabData, nCO,
            dust_to_gas_mass_ratio_over_mw, UV_background_G0,
            N_phot_new, state);

        ions_step<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            ddt, elements, n_and_ion_fracs, tabData, flags,
            dust_to_gas_mass_ratio_over_mw, UV_background_G0,
            primary_cosmic_ray_ionization_rate, ss_factor,
            N_phot_new, state);

        /////////////////////////
        //   More Convergence  //
        /////////////////////////

        subcycle_commit_and_control<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            elements, tabData, n_and_ion_fracs, nCO, N_PHOT, F_PHOT,
            N_phot_new, phot_att, ddt, total_time, state);

        // Check if we are converged
        if (state.success_counter > conv_its) {
            model_converged = true;
        }

        // Check if we are converged
        if (total_iterations > max_its) {
            model_converged = true;
        }

        // Break the infinite loop if the timestep has been reached
        if (total_dt > 0.0) {
            if (fabs(total_time-total_dt)/total_dt < 1e-6) break;
        }

        // Finish the calculation if model converged
        if (model_converged) break; //break from infinite loop

    } // End infinite loop

    total_iter_reached = total_iterations;

    // printf("%e %e %e %d\n",n_and_ion_fracs[1].n_element,n_and_ion_fracs[1].ion_fracs_new[0],Tmu_old,total_iterations);

    return state.Tmu_old;
}


template <bool constant_temperature,bool include_H2,bool include_CO,bool rt_advect,bool include_self_shielding>
RTZ<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>::RTZ(std::string path_to_data)
{
        // initialize data structures

        // Declare an array (list) of 3 elements (structs)
        std::array<Element, MAX_ELEMENTS> elements;
        // ParticleIonData n_and_ion_fracs[MAX_ELEMENTS];

        // Initialize the elements
        initialize_elements(elements, include_H2);

        // Copy elements to device view for use in solve_chemistry_and_cooling
        elements_d = Kokkos::View<Element*>("elements", MAX_ELEMENTS);
        auto elements_h = Kokkos::create_mirror_view(elements_d);
        for (int i = 0; i < MAX_ELEMENTS; i++) elements_h(i) = elements[i];
        Kokkos::deep_copy(elements_d, elements_h);

        // Initialize the UV background data
        load_UVB_data(path_to_data);      // First load the UVB data from files

        // Initialize cosmic ray rates
        initialize_cr_rates();

        // Initialize the charge transfer rates
        load_ct_rates(path_to_data);

        // Initialize high temperature cooling tables
        initialize_high_temperature_metal_cooling(path_to_data);

        // Initialize the low temperature cooling tables
        init_fine_structure_tables(path_to_data);

        // Initialize cross sections
        initialize_cross_sections(path_to_data, tabData);

        // Hardcode some stuff for now
        UV_background_G0 = 0.0070977; // <-- G0 used in PRISM paper
        generic_cosmic_ray_ionization_rate = 1E-16;
        printf("Using a G0 UV background of %f\nUsing a CR ionization rate of %f\n", UV_background_G0, generic_cosmic_ray_ionization_rate);

        // HM-related data
        copy_data_3D(HM12_UVB_z, tabData.HM12_UVB_z);
        copy_data_1D(G0_heating_rates, tabData.G0_heating_rates);

        // Cosmic ray and dust-related data
        copy_data_2D(cosmic_ray_ionization_rates, tabData.cosmic_ray_ionization_rates);
        copy_data_1D(cosmic_ray_ionization_rates_induced_UV, tabData.cosmic_ray_ionization_rates_induced_UV);
        copy_data_1D(cosmic_ray_ionization_rates_induced_UV_heat, tabData.cosmic_ray_ionization_rates_induced_UV_heat);
        copy_data_2D(dust_rec_coefs, tabData.dust_rec_coefs);

        // Charge transfer
        copy_data_3D(CTRecomb, tabData.CTRecomb);
        copy_data_3D(CTIon, tabData.CTIon);

        // Fine structure cooling
        copy_data_3D(fs_cool_tab, tabData.fs_cool_tab);

        // High-T cooling rates
        copy_data_3D(high_t_cooling_rates, tabData.high_t_cooling_rates);
        copy_data_2D(high_t_cooling_rates_tflag, tabData.high_t_cooling_rates_tflag);

        init_recombination_rates(tabData);
        init_cloudy_rad_rec_data(path_to_data, tabData);
        init_collisional_ionization(path_to_data, tabData);

        // H2 and CO cross sections
        if (include_CO) {
            copy_data_1D(sco_col1, tabData.sco_col1);
            copy_data_1D(sco_col2, tabData.sco_col2);
        }
        if (include_H2) {
            copy_data_1D(sh2_col1, tabData.sh2_col1);
            copy_data_1D(sh2_col2, tabData.sh2_col2);
        }
}

template <bool constant_temperature,bool include_H2,bool include_CO,bool rt_advect,bool include_self_shielding>
void RTZ<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>::need_to_update_UVB(double redshift)
{
    update_UVB(redshift);

    // Re-copy updated global array into tabData Kokkos view
    copy_data_3D(HM12_UVB_z, tabData.HM12_UVB_z);
}

template <bool constant_temperature,bool include_H2,bool include_CO,bool rt_advect,bool include_self_shielding>
void RTZ<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>::need_to_update_cross_sections(double T_blackbody)
{
    // Update cross sections
    update_cross_sections(T_blackbody, tabData);
}

template <bool constant_temperature,bool include_H2,bool include_CO,bool rt_advect,bool include_self_shielding>
void RTZ<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>::set_photon_groups(std::array<double, N_GROUPS> E_min, std::array<double, N_GROUPS> E_max) {
    auto E_min_h = Kokkos::create_mirror_view(tabData.group_E_min);
    auto E_max_h = Kokkos::create_mirror_view(tabData.group_E_max);
    for (size_t i = 0; i < N_GROUPS; i++) {
        E_min_h(i) = E_min[i];
        E_max_h(i) = E_max[i];
    }
    Kokkos::deep_copy(tabData.group_E_min, E_min_h);
    Kokkos::deep_copy(tabData.group_E_max, E_max_h);
}

template <bool constant_temperature,bool include_H2,bool include_CO,bool rt_advect,bool include_self_shielding>
void RTZ<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>::set_reduced_speed_of_light_factor(double factor) {
    auto c_frac_h = Kokkos::create_mirror_view(tabData.c_frac);
    c_frac_h() = factor;
    Kokkos::deep_copy(tabData.c_frac, c_frac_h);
}

template <bool constant_temperature,bool include_H2,bool include_CO,bool rt_advect,bool include_self_shielding>
void RTZ<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>::set_UV_background_G0(double G0) {
    UV_background_G0 = G0;
    printf("Overriding G0 UV background: now %f\n", UV_background_G0);
}

template <bool constant_temperature,bool include_H2,bool include_CO,bool rt_advect,bool include_self_shielding>
KOKKOS_FUNCTION
void RTZ<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>::solve_chemistry_and_cooling(
    double TK,
    const double dx,
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
    PhysicsFlags flags) const
{

        const Element* elements = elements_d.data();

        double dust_to_gas_mass_ratio_over_mw = get_dust_mass_and_depletion(
            elements, n_and_ion_fracs_loc.n_element[8], n_and_ion_fracs_loc.n_element[1], nCO
        );

        double loc_rho = get_rho(elements, n_and_ion_fracs_loc);

        apply_dust_depletion(elements, dust_to_gas_mass_ratio_over_mw, n_and_ion_fracs_loc, nCO);

        // get_chemical_eqm returns Tmu (= T/mu), matching the reference
        double Tmu_final = subcycle_chemistry<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            elements, loc_rho, n_and_ion_fracs_loc,
            nCO, TK,
            dx,
            aexp, total_dt, UV_background_G0,
            generic_cosmic_ray_ionization_rate,
            dust_to_gas_mass_ratio_over_mw,
            conv_its, max_its, total_iter_reached,
            tabData,
            flags,
            N_PHOT_loc, F_PHOT_loc
        );

        {
            double ne_final = get_ne(elements, n_and_ion_fracs_loc, false);
            double mu_final = get_mu(ne_final, elements, n_and_ion_fracs_loc, false, include_H2, nCO);
            out_mu = mu_final;
            out_Tmu = Tmu_final;
            TK = Tmu_final;
        }

        remove_dust_depletion(elements, dust_to_gas_mass_ratio_over_mw, n_and_ion_fracs_loc, nCO);
}


// Explicit template instantiations
template class RTZ<false,true,false,false,false>;
template class RTZ<false,false,false,false,false>;
template class RTZ<false,true,true,true,false>;
template class RTZ<false, true, false, true, false>;
template class RTZ<false, false, false, false, true>;
template class RTZ<false, true, true, false, false>;
template class RTZ<false, false, false, true, true>;
template class RTZ<false, false, false, true, false>;

template class RTZ<true,true,false,false,false>;
template class RTZ<true,false,false,false,false>;
template class RTZ<true,true,true,true,false>;
template class RTZ<true, true, false, true, false>;
template class RTZ<true, false, false, false, true>;

} // namespace prism_gpu
