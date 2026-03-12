#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>

#include "types.hpp"
#include "recombination.hpp"
#include "collisional_ionization.hpp"
#include "cosmic_ray_ionization.hpp"
#include "photoionization_UVB.hpp"
#include "charge_transfer.hpp"
#include "dust_recombination.hpp"
#include "molecules.hpp"
#include "cooling.hpp"
#include "dyablo_api.hpp"
#include "cross_sections.hpp"

#include <Kokkos_Core.hpp>

// MAX_ELEMENTS is already defined in types.hpp (included via dyablo_api.hpp)
const double MIN_XION = 1E-20;         // Minimum ion fraction
const double MIN_XION_FLOOR = 1E-10;   // Minimum ion fraction
const double CONV_ABS = 1E-4;          // Value required for convergence
const double CONV_ABS_H2_metal = 1E-4; // Value required for convergence
const double X_PCT_RULE = 1E-1;        // Value required for convergence (matches reference)
const double X_PCT_RULE_TEMP = 1E-1;   // Value required for convergence (matches reference)
const double X_PCT_RULE_metal = 1E-1;  // Value required for convergence (matches reference)
const double X_FM = 1E-7;              // Damping value for X_PCT_RULE
const double T_MIN = 2.725;
const double T_MAX = 1E9;

#define uint unsigned int


template<typename T, typename T2>
void copy_data_3D(T& src, Kokkos::View<T2> &dst) {
    auto dst_h = Kokkos::create_mirror_view(dst);
    for (uint i = 0; i < dst.extent(0); i++)
    {
        for (uint j = 0; j < dst.extent(1); j++)
        {
            for (uint k = 0; k < dst.extent(2); k++)
            {
                dst_h(i, j, k) = src[i][j][k];
            }
        }
    }
    Kokkos::deep_copy(dst, dst_h);
};

template<typename T, typename T2>
void copy_data_2D(T& src, Kokkos::View<T2> &dst) {
    auto dst_h = Kokkos::create_mirror_view(dst);
    for (uint i = 0; i < dst.extent(0); i++)
    {
        for (uint j = 0; j < dst.extent(1); j++)
        {
            dst_h(i, j) = src[i][j];
        }
    }
    Kokkos::deep_copy(dst, dst_h);
};

template<typename T, typename T2>
void copy_data_1D(T& src, Kokkos::View<T2> &dst) {
    auto dst_h = Kokkos::create_mirror_view(dst);
    for (uint i = 0; i < dst.extent(0); i++)
    {
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
    for (int i = 0; i < MAX_ELEMENTS; i++)
    {
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

// typedef struct
// {
//     double n_element;
//     double ion_fracs[MAX_ELEMENTS];
//     double ion_fracs_new[MAX_ELEMENTS];
// } ParticleIonData;

KOKKOS_FUNCTION
double get_dust_mass_and_depletion(
    std::array<Element, MAX_ELEMENTS> &elements, const double metallicity)
{
    // Parameters for the RR14 dust-to-gas mass ratio
    double a = 2.21;
    double aH = 1.00;
    double b = 0.96;
    double aL = 3.10;
    double xt = 8.10;
    double xs = 8.69;
    double x, y;

    // Compute loc_z = 12 + log10(O_solar * Z / H_solar), matching reference exactly
    double loc_z = 12.0 + log10(elements[8].z_solar * metallicity / (elements[1].z_solar + 1E-20));
    x = fmax(loc_z, 5.0);

    // Determine y (gas-to-dust mass ratio)
    if (loc_z > xt)
    {
        y = a + aH * (xs - x);
    }
    else
    {
        y = b + aL * (xs - x);
    }

    double G2D = pow(10.0, y);
    double G2D_sol = pow(10.0, a);

    double y_ratio = fmax(fmin(G2D_sol / G2D, 1.0), 0.0);

    for (int i = 1; i < MAX_ELEMENTS; i++)
    {
        if (elements[i].atomic_number < 1)
            continue;
        elements[i].depletion = 1.0 - ((1.0 - elements[i].depletion) * y_ratio);
    }

    return y_ratio;
}

// Structs for element properties
KOKKOS_FUNCTION
void initialize_ion_fracs(std::array<Element, MAX_ELEMENTS> &elements, std::array<ParticleIonData, MAX_ELEMENTS> &n_and_ion_fracs, double hdens, double metallicity, int itype)
{
    // Loop over all elements
    for (int i = 1; i < MAX_ELEMENTS; i++)
    {
        if (elements[i].atomic_number < 1)
            continue;

        // Only scale metals by metallicity
        if (i < 3)
        {
            n_and_ion_fracs[i].n_element = hdens * elements[i].z_solar * elements[i].depletion;
        }
        else
        {
            n_and_ion_fracs[i].n_element = hdens * elements[i].z_solar * metallicity * elements[i].depletion;
        }
        // n_and_ion_fracs[i].ion_fracs = (double *)malloc((elements[i].n_ions + elements[i].n_mol) * sizeof(double));
        // n_and_ion_fracs[i].ion_fracs_new = (double *)malloc((elements[i].n_ions + elements[i].n_mol) * sizeof(double));

        switch (itype)
        {
        case 1:
            // fully ionized
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++)
            {
                n_and_ion_fracs[i].ion_fracs[j] = MIN_XION;
                n_and_ion_fracs[i].ion_fracs_new[j] = MIN_XION;
            }
            n_and_ion_fracs[i].ion_fracs[elements[i].n_ions - 1] = 1.0;
            n_and_ion_fracs[i].ion_fracs_new[elements[i].n_ions - 1] = 1.0;
            break;
        case 2:
            // fully neutral
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++)
            {
                n_and_ion_fracs[i].ion_fracs[j] = MIN_XION;
                n_and_ion_fracs[i].ion_fracs_new[j] = MIN_XION;
            }
            n_and_ion_fracs[i].ion_fracs[0] = 1.0;
            n_and_ion_fracs[i].ion_fracs_new[0] = 1.0;
            break;
        case 3:
            // fully molecular or neutral
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++)
            {
                n_and_ion_fracs[i].ion_fracs[j] = MIN_XION;
                n_and_ion_fracs[i].ion_fracs_new[j] = MIN_XION;
            }
            if (elements[i].n_mol > 0)
            {
                n_and_ion_fracs[i].ion_fracs[elements[i].n_ions + elements[i].n_mol - 1] = 1.0;
                n_and_ion_fracs[i].ion_fracs_new[elements[i].n_ions + elements[i].n_mol - 1] = 1.0;
            }
            else
            {
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
double get_ne(const std::array<Element, MAX_ELEMENTS> &elements, const std::array<ParticleIonData, MAX_ELEMENTS> &n_and_ion_fracs, const bool use_new)
{
    /*
    Returns the electron number density by looping over all elements
    and summing their contributions
    */
    double ne = 0E0;

    // Loop over all elements
    for (int i = 1; i < MAX_ELEMENTS; i++)
    {
        if (elements[i].atomic_number < 1)
            continue;

        // Get the number of ions
        int n_ions = elements[i].n_ions;

        // Loop over all ions
        // Start loop at 1, no electrons in the ground state
        for (int j = 1; j < n_ions; j++)
        {
            if (use_new)
            {
                ne += n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs_new[j] * j;
            }
            else
            {
                double a = n_and_ion_fracs[i].n_element;
                double b = n_and_ion_fracs[i].ion_fracs[j];
                ne += n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs[j] * j;
            }
        } // End loop ion fracs
    } // End loop elements

    return ne;
}

KOKKOS_FUNCTION
double get_rho(const std::array<Element, MAX_ELEMENTS>& elements,
               const std::array<ParticleIonData, MAX_ELEMENTS>& n_and_ion_fracs)
{
    double rho = 0.0;
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;
        rho += n_and_ion_fracs[i].n_element * elements[i].atomic_mass / elements[i].depletion;
    }
    return rho;
}

KOKKOS_FUNCTION
double get_mu(double ne,
              const std::array<Element, MAX_ELEMENTS>& elements,
              const std::array<ParticleIonData, MAX_ELEMENTS>& n_and_ion_fracs,
              bool use_new, bool include_H2) {
    /*
    Calculated the mean molecular weight
    */
    double m_bar = 0.0;
    double n_hat = 0.0;

    // Loop over all elements
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;

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
    for (int i = 0; i < size; i++)
    {
        total_ion_frac += ion_fracs[i];
    }
    // Loop again to normalize
    for (int i = 0; i < size; i++)
    {
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
    for (int i = 0; i < size; i++)
    {
        ion_fracs[i] = fmax(ion_fracs[i], MIN_XION);
        total_ion_frac += ion_fracs[i];
        if (ion_fracs[i] > max_frac)
        {
            max_frac = ion_fracs[i];
            max_idx = i;
        }
    }

    // Now reset to 1
    ion_fracs[max_idx] -= (total_ion_frac - 1.0);
}

// void copyArray(double src[], double dest[], int size)
// {
//     for (int i = 0; i < size; i++)
//     {
//         dest[i] = src[i];
//     }
// }

template <
    bool constant_temperature,
    bool include_H2=true,
    bool include_CO=false,
    bool rt_advect=false,
    bool include_self_shielding=false
>
KOKKOS_FUNCTION
double subcycle_chemistry(
                      const std::array<Element, MAX_ELEMENTS> &elements,
                      std::array<ParticleIonData, MAX_ELEMENTS> &n_and_ion_fracs,
                      double nCO, double T2, 
                      double aexp, double total_dt, double UV_background_G0,
                      double primary_cosmic_ray_ionization_rate,
                      double dust_to_gas_mass_ratio_over_mw,
                      int conv_its, int max_its,
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
    int success_counter     = 0;
    int total_iterations    = 0;
    bool model_converged    = false;
    bool x_percent_rule     = true;
    bool compute_atoms      = true;
    bool compute_molecules  = true;
    double total_time       = 0.E0;
    double residual;  
    double ne_residual;  
    double max_residual_all;  

    //Physical variables
    double mu; //Mean molecular weight
    double ne; //Electron number density
    double ne_initial; //Electron number density for convergence
    double cr; //Creation rate
    double de; //Destruction rate
    double xe; //Electron fraction (n_electron/nH)
    double phi_s; // Secondary electron CR factor
    double Tmu_old = T2; // T/mu
    double Tmu_new = T2; // T/mu

    std::array<double, MAX_ELEMENTS> element_number_densities{};
    std::array<int, MAX_ELEMENTS> element_number_ions{};
    Array2D element_ion_fractions{};

    // Radiation
    std::array<double, N_GROUPS> N_phot_new{};
    Array2D_RT F_phot_new{};

    int n_ions; //Number of ions --> reset per element
    int atomic_number;

    // Self-shielding
    double ss_factor = 1.0;
    if (include_self_shielding) {
        ss_factor = exp(-1.0 * n_and_ion_fracs[0].n_element/1E-2);
    }

    // Cooling
    double cooling_rate = 0.0;
    double cooling_rate_prime = 0.0;

    // Compute effective rho
    double loc_rho = get_rho(elements, n_and_ion_fracs);

    // First get the electron number density
    ne_initial = get_ne(elements, n_and_ion_fracs, false);

    //# infinite loop
    while (1) {  // Infinite loop: TODO(code): put a max loop counter here and then call an error function
        printf("%e %e %e\n",ddt,total_dt,total_time);
        total_iterations += 1;
        compute_atoms = true;
        compute_molecules = true;

        //Loop over elements and set the ion_fracs new to
        //the old ion fracs
        //do the same for the number densities
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (elements[i].atomic_number < 1) continue;

            n_and_ion_fracs[i].n_element_new = n_and_ion_fracs[i].n_element;

            int size = elements[i].n_ions + elements[i].n_mol;
            for (int j = 0; j < size; j++) {
                n_and_ion_fracs[i].ion_fracs_new[j] = n_and_ion_fracs[i].ion_fracs[j];
            }
        } // End element loop

        // First get the electron number density
        ne = ne_initial;
        // Update the electron fraction
        xe = ne / n_and_ion_fracs[1].n_element_new;
        // Measure mu
        mu = get_mu(ne, elements, n_and_ion_fracs, true, include_H2);
        // Get TK
        double TK = Tmu_old * mu;
        if (constant_temperature) TK = Tmu_old;

        // Measure phi_s for secondary CR ionization
        phi_s = secondary_cr_rates(xe);
        // Now calculate the total CR ionization rate
        double total_cosmic_ray_ionization_rate = primary_cosmic_ray_ionization_rate * (1.0 + phi_s);
        double H2_cosmic_ray_ionization_rate = 2.0 * primary_cosmic_ray_ionization_rate * (1.0 + phi_s);

        // Set the cosmic ray scale factor -> only needed for secondary UVs
        double cosmic_ray_scale_factor = H2_cosmic_ray_ionization_rate / 1E-16;

        // Initialize the X percent rule
        x_percent_rule = true;

        // Convergence
        max_residual_all = -999.;

        // Make sure the timestep isn't larger than how long we
        // need to evolve for, otherwise shorten it
        if (total_dt > 0.0) {
            if ((total_time+ddt) > total_dt){
                ddt = total_dt - total_time;
            }
        }

        /////////////////////////
        //      Radiation      //
        /////////////////////////
        if constexpr (rt_advect) {
            double phAbs = 0.0;

            // First get the opacity
            for (int k = 0; k < N_GROUPS; k++) { // Loop over photon groups

                phAbs = 0.0; // zero out absorption

                for (int i = 1; i < MAX_ELEMENTS; i++) { // Loop over all elements
                    if (elements[i].atomic_number < 1) continue;

                    // Get the number of ions
                    n_ions = elements[i].n_ions;

                    for (int j = 0; j < n_ions - 1; j++) { // Loop over the number of ions --> no absorption by final ion
                        phAbs += (n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs_new[j] * tabData.cs_ph(i,j,k,0));
                    } // End loop over ions
                } // End loop over elements

                // Deal with molecular hydrogen separately if needed
                if constexpr (include_H2) {
                    // TODO(code): for better accuracy need to include self-shielding here
                    phAbs += (0.5 * n_and_ion_fracs[1].n_element * n_and_ion_fracs[1].ion_fracs_new[1] * tabData.cs_ph(1,2,k,0));
                }

                // Next deplete the photons from the group
                N_phot_new[k] = N_PHOT[k] / (1.0 + ddt * phAbs);
                // TODO(code): loop over dimensions here and update RT
                F_phot_new[k][0] = F_PHOT[k][0] / (1.0 + ddt * phAbs);
                F_phot_new[k][1] = F_PHOT[k][1] / (1.0 + ddt * phAbs);

                // Measure the redisual
                residual = fabs((N_phot_new[k] - N_PHOT[k]) / (N_PHOT[k] + 1E-13));
                max_residual_all = fmax(max_residual_all, residual);

                if (residual > X_PCT_RULE){
                    // printf("Broken photons %e %e %e %e %e %d %e %e %e %e %e\n",Tmu_new,Tmu_old,residual,mu,ddt,total_iterations,element_ion_fractions[1]
                    // Reset the success counter
                    success_counter = 0;
                    // Declare the x-percent rule was violated
                    x_percent_rule = false;
                    // // Don't compute temperature
                    // compute_temperature = false;
                    // Dont compute molecules
                    compute_molecules = false;
                    // Don't compute atoms
                    compute_atoms = false;
                }
            } // End loop over photon groups
        }
        double f_shd_H2 = 1.0;
        double f_shd_CO = 1.0;

        /////////////////////////
        //     Temperature     //
        /////////////////////////
        if (!constant_temperature) {

            // double *element_number_densities;
            // double *element_number_ions;
            // double **element_ion_fractions;

            // element_number_densities = (double *)malloc(MAX_ELEMENTS * sizeof(double));
            // element_number_ions = (double *)malloc(MAX_ELEMENTS * sizeof(double));
            // element_ion_fractions = (double **)malloc(MAX_ELEMENTS * sizeof(double *));

            // Loop over all elements
            for (int i = 1; i < MAX_ELEMENTS; i++) {
                // element_ion_fractions[i] = (double *)malloc(MAX_ELEMENTS * sizeof(double));
                element_number_densities[i] = n_and_ion_fracs[i].n_element_new;
                element_number_ions[i] = elements[i].n_ions + elements[i].n_mol;
                for (int j = 0; j < element_number_ions[i]; j++) {
                    element_ion_fractions[i][j] = n_and_ion_fracs[i].ion_fracs_new[j];
                }
            }

            cooling_rate = all_cooling(TK, ne, aexp,
                                    element_number_densities, 
                                    element_number_ions, 
                                    element_ion_fractions,
                                    N_phot_new,
                                    nCO,
                                    UV_background_G0, 
                                    dust_to_gas_mass_ratio_over_mw, 
                                    xe, 
                                    primary_cosmic_ray_ionization_rate, 
                                    H2_cosmic_ray_ionization_rate,
                                    ss_factor, 
                                    flags.include_photoionization,
                                    flags.include_cosmic_ray_ionization, 
                                    flags.include_HM12_UVB,
                                    flags.include_dust_recombination, 
                                    flags.include_charge_exchange,
                                    tabData);

            cooling_rate_prime = all_cooling(1.001 * TK, ne, aexp,
                                            element_number_densities, 
                                            element_number_ions, 
                                            element_ion_fractions,
                                            N_phot_new,
                                            nCO,
                                            UV_background_G0, 
                                            dust_to_gas_mass_ratio_over_mw, 
                                            xe, 
                                            primary_cosmic_ray_ionization_rate, 
                                            H2_cosmic_ray_ionization_rate,
                                            ss_factor, 
                                            flags.include_photoionization,
                                            flags.include_cosmic_ray_ionization, 
                                            flags.include_HM12_UVB,
                                            flags.include_dust_recombination, 
                                            flags.include_charge_exchange,
                                            tabData
                                        );

            cooling_rate_prime = (cooling_rate_prime - cooling_rate) / ((1.001*TK) - TK);

            double X_nHkb = 1.0 / (1.5 * loc_rho * 1.380649e-16);

            cooling_rate *= X_nHkb;
            cooling_rate_prime *= (-X_nHkb) * mu;

            double dUU = fabs(fmax(T_MIN, Tmu_old + cooling_rate * ddt) - Tmu_old);
            Tmu_new = fmax(T_MIN,Tmu_old + cooling_rate * ddt/(1.0 - cooling_rate_prime * ddt));

            // ! New T2 value
            TK = Tmu_new * mu;

            residual = fmax(dUU, fabs(Tmu_new-Tmu_old)) / (Tmu_old+T_MIN);
            
            // Convergence
            // X% rule

            // Max residual over all elements
            max_residual_all = fmax(max_residual_all,residual);

            if (residual > X_PCT_RULE){
                // printf("Broken temperature %e %e %e %e %e %d %e %e %e %e %e\n",Tmu_new,Tmu_old,residual,mu,ddt,total_iterations,element_ion_fractions[1][0],element_ion_fractions[1][1],element_ion_fractions[1][2],cooling_rate,cooling_rate_prime);
                // Reset the success counter
                success_counter = 0;
                // Declare the x-percent rule was violated
                x_percent_rule = false;
                // Dont compute molecules
                compute_molecules = false;
                // Don't compute atoms
                compute_atoms = false;
            }
        }
        
        /////////////////////////
        //      Molecules      //
        /////////////////////////
        double alpha_H2_loc = 0.0;
        double beta_H2_loc = 0.0;
        double cr_H2 = 0.E0;
        double de_H2 = 0.E0;

        double cr_CO = 0.E0;
        double de_CO = 0.E0;
        double nCO_new = 0.E0;
        if (compute_molecules) {

            // Molecular Hydrogen
            if (include_H2) {

                // H2 fraction --> Note that we actually store 2 * xH2
                double xH2 = n_and_ion_fracs[1].ion_fracs_new[2]/2.0;

                // Creation //

                // Contains formation on dust and via the primordial channel
                // TODO(code): update G0 to be the total G0
                alpha_H2_loc = alpha_H2(TK, dust_to_gas_mass_ratio_over_mw, xe, H2_cosmic_ray_ionization_rate, 
                                        UV_background_G0, n_and_ion_fracs[1].ion_fracs_new[0], n_and_ion_fracs[1].ion_fracs_new[1],
                                        n_and_ion_fracs[1].n_element_new); //! [cm3 s-1]

                // We can only create H2 if there is any HI and HII
                cr_H2 += alpha_H2_loc;   

                // Destruction //

                // Collisional destruction
                beta_H2_loc = beta_H2_krome(TK,     
                                        n_and_ion_fracs[1].ion_fracs_new[0] * n_and_ion_fracs[1].n_element_new,
                                        ne,
                                        xH2 * n_and_ion_fracs[1].n_element_new,
                                        n_and_ion_fracs[2].ion_fracs_new[0] * n_and_ion_fracs[2].n_element_new);
                de_H2 += beta_H2_loc;

                // Photodissociation
                de_H2 += UV_background_G0 * 5.68E-11;

                // Cosmic ray destruction
                de_H2 += H2_cosmic_ray_ionization_rate;

                // Update xH2
                xH2 = (cr_H2*ddt + xH2)/(1.+de_H2*ddt);

                // Store in the struct
                n_and_ion_fracs[1].ion_fracs_new[2] = 2.0 * fmin(fmax(xH2,MIN_XION),0.5);

                // Convergence
                // X% rule
                residual = fabs(n_and_ion_fracs[1].ion_fracs_new[2] - n_and_ion_fracs[1].ion_fracs[2]) / (n_and_ion_fracs[1].ion_fracs[2] + X_FM);

                // Max residual over all elements
                max_residual_all = fmax(max_residual_all,residual);

                if (residual > X_PCT_RULE){
                    // printf("Broken molecules %e %e %e %e\n",n_and_ion_fracs[1].ion_fracs_new[2],n_and_ion_fracs[1].ion_fracs[2],residual,ddt);
                    // Reset the success counter
                    success_counter = 0;
                    // Declare the x-percent rule was violated
                    x_percent_rule = false;
                    // Break the loop over ions
                    compute_atoms = false;
                } else {
                    //  Update mu and T
                    mu = get_mu(ne, elements, n_and_ion_fracs, true, include_H2);
                    TK = Tmu_new * mu;
                    if (constant_temperature) TK = Tmu_old;
                }
            }

            // CO
            if (include_CO && compute_atoms) {
                // Check if there is enough C, O, and CO to warrant computing the chemistru
                if ((n_and_ion_fracs[6].n_element_new + n_and_ion_fracs[8].n_element_new + nCO)/n_and_ion_fracs[1].n_element_new > 1e-10) {

                    double n_CII = n_and_ion_fracs[6].n_element_new * n_and_ion_fracs[6].ion_fracs_new[1];
                    double n_OI  = n_and_ion_fracs[8].n_element_new * n_and_ion_fracs[8].ion_fracs_new[0];
                    double n_H2  = n_and_ion_fracs[1].n_element_new * n_and_ion_fracs[1].ion_fracs_new[2] / 2.0;
                    double x_OI  = n_OI / n_and_ion_fracs[1].n_element_new;

                    //!! Creation !!
                    cr_CO = alpha_CO(UV_background_G0, H2_cosmic_ray_ionization_rate, n_CII, n_H2, x_OI);

                    //!! Destruction !!
                    de_CO = beta_CO(UV_background_G0*f_shd_CO, H2_cosmic_ray_ionization_rate);

                    // Compute the initial guess of new nCO
                    nCO_new = (nCO + cr_CO*ddt) / (1.0 + de_CO*ddt);

                    // Tentative update
                    double delta_CO = nCO_new - nCO;

                    // Enforce positivity floors: don't destroy more CO than exists,
                    // and don't create more than allowed by available CII and OI
                    double max_delta_CO = fmin(n_CII - 1e-30, n_OI - 1e-30);
                    double min_delta_CO = -nCO + 1e-30;       

                    // Clip delta_CO to enforce physical bounds
                    delta_CO = fmax(fmin(delta_CO, max_delta_CO), min_delta_CO);

                    // Apply updates
                    nCO_new  = nCO + delta_CO;
                    double nCII_new = n_CII - delta_CO;
                    double nOI_new  = n_OI  - delta_CO;

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
                            n_and_ion_fracs[8].ion_fracs[i] = n_and_ion_fracs[8].n_element_new * n_and_ion_fracs[8].ion_fracs_new[i] / tot_O;
                        } else {
                            n_and_ion_fracs[8].ion_fracs[i] = nOI_new / tot_O;
                        }
                    }

                    // Check for convergence
                    residual = fabs(nCO_new - nCO) / (nCO + X_FM);

                    // Max residual over all elements
                    max_residual_all = fmax(max_residual_all,residual);

                    if (residual > X_PCT_RULE){
                        // Reset the success counter
                        success_counter = 0;
                        // Declare the x-percent rule was violated
                        x_percent_rule = false;
                        // Break the loop over ions
                        compute_atoms = false;
                    } else {
                        // Update species number densities
                        n_and_ion_fracs[6].n_element_new -= delta_CO;
                        n_and_ion_fracs[8].n_element_new -= delta_CO;
                    }
                }
            }
        }

        /////////////////////////
        //        Atoms        //
        /////////////////////////
        // Update the HI number density in case it has changed e.g. due to charge exchange
        // TODO(code) --> check this, tis is just H, not HI
        double dust_effective_number_density = n_and_ion_fracs[1].n_element_new * dust_to_gas_mass_ratio_over_mw;

        if (compute_atoms) {
            // Loop over all elements
            for (int i = 1; i < MAX_ELEMENTS; i++) {
                if (elements[i].atomic_number < 1) continue;

                // Get the atomic number
                atomic_number = elements[i].atomic_number;

                // Get the number of ions
                n_ions = elements[i].n_ions;

                // Loop over the number of ions
                for (int j = 0; j < n_ions; j++) {

                    /////////////////////////
                    //       Creation      //
                    /////////////////////////
                    cr = 0.E0;

                    // Account for molecular hydrogen
                    if ((i == 1) & (j == 0) & include_H2) { // select only HI
                        // Note: no factor of 2 needed since is 2*xH2
                        cr += de_H2 * n_and_ion_fracs[1].ion_fracs_new[2];
                    }

                    // Recombinations of the more excited ionization state
                    if (j < (n_ions - 1)) cr += recombination(TK, j+1, i, tabData) * ne * n_and_ion_fracs[i].ion_fracs_new[j+1]; 

                    // Collisional ionization of the less excited state
                    if (flags.include_collisional_ionization) {
                        if (j > 0) cr += collisional_ionization(TK, j-1, i, tabData) * ne * n_and_ion_fracs[i].ion_fracs_new[j-1];
                    }

                    // Photoionization of the less excited state
                    if (flags.include_HM12_UVB) {
                        if (j > 0) cr += tabData.HM12_UVB_z(i, j-1, 0) * ss_factor * n_and_ion_fracs[i].ion_fracs_new[j-1];
                    }
                    
                    // Photoionization by sub-ionizing ISRF --> only impacts lowest ionization states
                    if (flags.include_photoionization) {
                        if (j == 1) cr += UV_background_G0 * elements[i].G0_photo_rate * n_and_ion_fracs[i].ion_fracs_new[j-1];
                    }

                    // Cosmic ray ionization of the less excited state
                    if (flags.include_cosmic_ray_ionization) {
                        if (j > 0) cr += tabData.cosmic_ray_ionization_rates(i, j-1) * total_cosmic_ray_ionization_rate * n_and_ion_fracs[i].ion_fracs_new[j-1];

                        // Cosmic ray ionization of the less excited state from induced UV
                        if (j == 1) cr += tabData.cosmic_ray_ionization_rates_induced_UV(i) * cosmic_ray_scale_factor * n_and_ion_fracs[i].ion_fracs_new[j-1];
                    }

                    // Recombination on dust for the more excited state
                    if (flags.include_dust_recombination) {
                        if (j < (n_ions - 1)) cr += dust_recombination_rates(j+1, i, TK, UV_background_G0, ne, tabData.dust_rec_coefs) * dust_effective_number_density * n_and_ion_fracs[i].ion_fracs_new[j+1];
                    }

                    /////////////////////////
                    //     Destruction     //
                    /////////////////////////
                    de = 0.E0;

                    // Account for molecular hydrogen
                    if ((i == 1) & (j == 0) & include_H2) { // select only HI
                        de += alpha_H2_loc;
                    }

                    // Collisional ionization 
                    if (flags.include_collisional_ionization) {
                        if (j < (n_ions - 1)) de += collisional_ionization(TK, j, i, tabData) * ne;
                    }

                    // Photoionization
                    if (flags.include_HM12_UVB) {
                        if (j < (n_ions - 1)) de += tabData.HM12_UVB_z(i, j, 0) * ss_factor;
                    }

                    // Photoionization by sub-ionizing ISRF --> only impacts lowest ionization states
                    if (flags.include_photoionization) {
                        if (j == 0) de += UV_background_G0 * elements[i].G0_photo_rate;
                    }

                    // Recombination
                    if (j > 0) de += recombination(TK, j, i, tabData) * ne;

                    // Cosmic ray ionization
                    if (flags.include_cosmic_ray_ionization) {
                        if (j < (n_ions - 1)) de += tabData.cosmic_ray_ionization_rates(i, j) * total_cosmic_ray_ionization_rate;

                        // Cosmic ray ionization from induced UV
                        if (j == 0) de += tabData.cosmic_ray_ionization_rates_induced_UV(i) * cosmic_ray_scale_factor;
                    }

                    // Recombination on dust
                    if (flags.include_dust_recombination) {
                        if (j > 0) de += dust_recombination_rates(j, i, TK, UV_background_G0, ne, tabData.dust_rec_coefs) * dust_effective_number_density;
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
                                if (elements[ii].atomic_number < 1) continue;

                                // Loop over all other ionization states
                                for (int jj = 0; jj < elements[ii].n_ions; jj++) {
                                    double paired_ion_number_density = n_and_ion_fracs[ii].ion_fracs_new[jj] * n_and_ion_fracs[ii].n_element_new;
                                    if (j == 0) { // H
                                        cr += charge_transfer_ionization(jj,ii,TK, tabData) * n_and_ion_fracs[i].ion_fracs_new[j+1] * paired_ion_number_density; //! Example:  O + H+ => O+ + H
                                        de += charge_transfer_recombination(jj,ii,TK, tabData) * paired_ion_number_density; //! Example:  O+ + H => O + H+
                                    } else { // H+
                                        de += charge_transfer_ionization(jj,ii,TK, tabData) * paired_ion_number_density; //! Example:  O + H+ => O+ + H
                                        cr += charge_transfer_recombination(jj,ii,TK, tabData) * n_and_ion_fracs[i].ion_fracs_new[j-1] * paired_ion_number_density; //! Example:  O+ + H => O + H+
                                    }
                                } // end loop over ionization states
                            } // end loop over other elements
                        } else { // All other elements
                            double HI_number_density = n_and_ion_fracs[1].ion_fracs_new[0] * n_and_ion_fracs[1].n_element_new;
                            double HII_number_density = n_and_ion_fracs[1].ion_fracs_new[1] * n_and_ion_fracs[1].n_element_new;

                            if (j > 0) {
                                // Ionization from less excited state
                                cr += charge_transfer_ionization(j-1,i,TK, tabData) * n_and_ion_fracs[i].ion_fracs_new[j-1] * HII_number_density; //! Example:  O + H+ => O+ + H   

                                // Charge exchange recombination 
                                de += charge_transfer_recombination(j,i,TK, tabData) * HI_number_density; //! Example:  O+ + H => O + H+
                            }    

                            if (j < (n_ions - 1)) {
                                // Charge exchange ionization
                                de += charge_transfer_ionization(j,i,TK, tabData) * HII_number_density; //! Example:  O + H+ => O+ + H   

                                // Charge exchange recombination from the more excited state
                                cr += charge_transfer_recombination(j+1,i,TK, tabData) * n_and_ion_fracs[i].ion_fracs_new[j+1] * HI_number_density; //! Example:  O+ + H => O + H+
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
                    ne += n_and_ion_fracs[i].n_element_new * (n_and_ion_fracs[i].ion_fracs_new[j] - n_and_ion_fracs[i].ion_fracs[j]) * (double)j;
                    xe = ne / n_and_ion_fracs[1].n_element_new;

                    if (flags.include_cosmic_ray_ionization) {
                        phi_s = secondary_cr_rates(xe);
                        total_cosmic_ray_ionization_rate = primary_cosmic_ray_ionization_rate * (1.0 + phi_s);
                    }

                    // Update mu and T --> only H and He (others don't matter)
                    if (i < 3) {
                        //  Update mu and T
                        mu = get_mu(ne, elements, n_and_ion_fracs, true, include_H2);
                        TK = Tmu_new * mu;
                        if (constant_temperature) TK = Tmu_old;
                    }

                    /////////////////////////
                    //     Convergence     //
                    /////////////////////////
                    // X% rule
                    residual = fabs(n_and_ion_fracs[i].ion_fracs_new[j] - n_and_ion_fracs[i].ion_fracs[j]) / (n_and_ion_fracs[i].ion_fracs[j] + X_FM);

                    // electron density residual
                    ne_residual = fabs(ne - ne_initial) / (ne + X_FM);
                    residual = fmax(residual,ne_residual);

                    // Max residual over all elements
                    max_residual_all = fmax(max_residual_all,residual);

                    // Check if X percent rule is broken
                    if (residual > X_PCT_RULE){
                        // printf("Broken element %d %d, %e %e %e %e\n",i,j,n_and_ion_fracs[i].ion_fracs_new[j],n_and_ion_fracs[i].ion_fracs[j],residual,ddt);
                        // Reset the success counter
                        success_counter = 0;
                        // Declare the x-percent rule was violated
                        x_percent_rule = false;
                        // Break the loop over ions
                        break;
                    }
                } // End ion loop

                //If the X% rule is violated restart with shorter timestep
                if (!x_percent_rule){
                    // Reset the success counter
                    success_counter = 0;
                    // Break out of the element loop
                    break;
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

            } // End element loop
        } // End compute atoms

        /////////////////////////
        //   More Convergence  //
        /////////////////////////
        
        // If all elements completed
        // Loop over elements and ions and 
        // set old values to new values
        if (x_percent_rule) {
            if (!constant_temperature) {
                // Update temperature
                Tmu_old = fmin(fmax(Tmu_new,T_MIN),T_MAX);
            }
            // Update atomic data
            for (int i = 1; i < MAX_ELEMENTS; i++) {
                if (elements[i].atomic_number < 1) continue;

                n_and_ion_fracs[i].n_element = n_and_ion_fracs[i].n_element_new;

                int size = elements[i].n_ions + elements[i].n_mol;
                for (int j = 0; j < size; j++) {
                    n_and_ion_fracs[i].ion_fracs[j] = n_and_ion_fracs[i].ion_fracs_new[j];
                } // End ion loop
            } // End element loop

            // Update CO if needed
            if (include_CO) {
                nCO = nCO_new;
            }

            if constexpr (rt_advect) {
                for (int k = 0; k < N_GROUPS; k++) { // Loop over photon groups
                    // Update the photon properties
                    N_PHOT[k] = fmax(N_phot_new[k],1E-20);
                    // TODO(code): update for NDIM
                    F_PHOT[k][0] = F_phot_new[k][0];
                    F_PHOT[k][1] = F_phot_new[k][1];
                    // Reduce the flux if needed
                    // TODO(code): update for NDIM
                    double fred = sqrt(pow(F_PHOT[k][0],2.0) + pow(F_PHOT[k][1],2.0)) / (C_CGS * N_PHOT[k]);
                    if (fred > 1.0) {
                        // TODO(code): update for NDIM
                        F_PHOT[k][0] /= fred;
                        F_PHOT[k][1] /= fred;
                    }
                } // End loop over photon groups
            }

            // Update the total time --> only if the run converged
            total_time += ddt;

            // Update the electron fraction
            ne_initial = ne;

            // Update the success counter
            success_counter += 1;

        }

        // Set new dt
        max_residual_all /= (X_PCT_RULE);
        if (!x_percent_rule) {
            // If we broke the 10% rule, divide timestep by 2.0
            ddt /= 2.0; 
        } else {
            // TODO(code): allow for choice of aggressive vs conservative timestep
            ddt = fmin(0.9 * ddt / pow(0.07+max_residual_all,0.3),2.0 * ddt);
        }
        // TODO(code): Make this a parameter for the timestep limiter
        double tlimit = -0.57142857 * log10(n_and_ion_fracs[1].n_element_new) + 10.85714286;
        ddt = fmin(ddt,pow(10.0,tlimit));

        // Check if we are converged
        if (success_counter > conv_its) {
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

    printf("%e %e %e %e %d %e %e %e %e %e\n",n_and_ion_fracs[1].n_element,Tmu_old*mu,Tmu_old,mu,total_iterations,n_and_ion_fracs[1].ion_fracs_new[0],n_and_ion_fracs[1].ion_fracs_new[1],n_and_ion_fracs[1].ion_fracs_new[2],ddt,total_time);
    // printf("%e %e %e %e %d %e %e %e",n_and_ion_fracs[1].n_element,Tmu_old*mu,Tmu_old,mu,total_iterations,n_and_ion_fracs[6].n_element,nCO,ddt);

    // printf("%e ",Tmu_old);
    // for (int i=0; i<27; i++){
    //     printf("%e ",n_and_ion_fracs[26].ion_fracs_new[i]);
    // }

    // printf("%e ",Tmu_old);
    // for (int i=0; i<2; i++){
    //     printf("%e ",n_and_ion_fracs[1].ion_fracs_new[i]);
    // }

    return Tmu_old;
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
        UV_background_G0 = 1e-40;
        generic_cosmic_ray_ionization_rate = 1E-40;

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
KOKKOS_FUNCTION
void RTZ<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>::solve_chemistry_and_cooling(
    double TK,
    const double metallicity,
    const double aexp,
    const double total_dt,
    std::array<double, MAX_ELEMENTS> &nelements_loc,
    std::array<double, MAX_ELEMENTS * MAX_ELEMENTS> &xions_loc,
    double& nCO,
    std::array<double, N_GROUPS> &N_PHOT_loc,
    Array2D_RT &F_PHOT_loc,
    double& out_Tmu,
    double& out_mu,
    int conv_its,
    int max_its,
    PhysicsFlags flags) const
{

        std::array<Element, MAX_ELEMENTS> elements_loc;
        initialize_elements(elements_loc, include_H2);

        // solve for ionization balance and cooling rates
        std::array<ParticleIonData, MAX_ELEMENTS> n_and_ion_fracs_loc{};

        int iion = 0;
        for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
            // Deactivate elements with zero abundance to save time in the chemistry solver
            if (nelements_loc[ii] == 0.0) elements_loc[ii].atomic_number = -1;

            if (elements_loc[ii].atomic_number < 1) continue;

            n_and_ion_fracs_loc[ii].n_element = nelements_loc[ii];

            for (int jj = 0; jj < elements_loc[ii].n_ions + elements_loc[ii].n_mol; jj++)
            {
                n_and_ion_fracs_loc[ii].ion_fracs[jj] = xions_loc[iion];
                iion++;
            }
        }

        // Get the dust to gas mass ratio and the depletion factors
        double dust_to_gas_mass_ratio_over_mw = get_dust_mass_and_depletion(elements_loc, metallicity);

        // get_chemical_eqm returns Tmu (= T/mu), matching the reference
        double Tmu_final = subcycle_chemistry<constant_temperature, include_H2, include_CO, rt_advect, include_self_shielding>(
            elements_loc, n_and_ion_fracs_loc,
            nCO, TK,
            aexp, total_dt, UV_background_G0,
            generic_cosmic_ray_ionization_rate,
            dust_to_gas_mass_ratio_over_mw,
            conv_its, max_its,
            tabData,
            flags,
            N_PHOT_loc, F_PHOT_loc
        );

        {
            double ne_final = get_ne(elements_loc, n_and_ion_fracs_loc, false);
            double mu_final = get_mu(ne_final, elements_loc, n_and_ion_fracs_loc, false, true);
            out_mu = mu_final;
            out_Tmu = Tmu_final;
            TK = Tmu_final;
        }

        // printf("Final temperature: %e\n", TK);

        // Loop over all elements
        iion = 0;
        for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
            if (elements_loc[ii].atomic_number < 1) continue;

            nelements_loc[ii] = n_and_ion_fracs_loc[ii].n_element;

            for (int jj = 0; jj < elements_loc[ii].n_ions + elements_loc[ii].n_mol; jj++)
            {
                xions_loc[iion] = n_and_ion_fracs_loc[ii].ion_fracs[jj];
                iion++;
            }
        }
}


// Explicit template instantiations
template class RTZ<false,true,false,false,false>;
template class RTZ<false,false,false,false,false>;
template class RTZ<false,true,true,true,false>;

template class RTZ<true,true,false,false,false>;
template class RTZ<true,false,false,false,false>;
