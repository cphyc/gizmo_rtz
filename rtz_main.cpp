#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "recombination.hpp"
#include "collisional_ionization.hpp"
#include "cosmic_ray_ionization.hpp"
#include "photoionization_UVB.hpp"
#include "charge_transfer.hpp"
#include "dust_recombination.hpp"
#include "molecules.hpp"
#include "cooling.hpp"

#include <Kokkos_Core.hpp>

#define MAX_ELEMENTS 27
const double MIN_XION = 1E-20;         // Minimum ion fraction
const double MIN_XION_FLOOR = 1E-10;   // Minimum ion fraction
const double CONV_ABS = 1E-4;          // Value required for convergence
const double CONV_ABS_H2_metal = 1E-4; // Value required for convergence
const double X_PCT_RULE = 1E-2;        // Value required for convergence
const double X_PCT_RULE_TEMP = 1E-2;   // Value required for convergence
const double X_PCT_RULE_metal = 1E-2;  // Value required for convergence
const double T_MIN = 2.725;
const double T_MAX = 1E9;

// Define the struct
typedef struct
{
    int atomic_number;
    int n_ions;
    int n_mol;
    double atomic_mass;
    double z_solar;
    double G0_photo_rate;
    double depletion;
} Element;

// Structs for element properties
void initialize_elements(Element *elements)
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
    elements[1].n_mol = 1;
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

typedef struct
{
    double n_element;
    double *ion_fracs;
    double *ion_fracs_new;
} ParticleIonData;

double get_dust_mass_and_depletion(
    Element *elements, double metallicity)
{
    // Parameters for the RR14 dust-to-gas mass ratio
    double a = 2.21;
    double aH = 1.00;
    double b = 0.96;
    double aL = 3.10;
    double xt = 8.10;
    double xs = 8.69;
    double x, y;

    // Compute x (metallicity proxy)
    x = 8.69 + log10(metallicity);

    // Determine y (gas-to-dust mass ratio)
    if (x > xt)
    {
        y = a + (aH * (xs - x));
    }
    else
    {
        y = b + (aL * (xs - x));
    }

    y = pow(10.0, y); // Convert to actual ratio

    // Set the depletion factors
    double y_ratio = fmin(1.0, 162.0 / y);

    for (int i = 1; i < MAX_ELEMENTS; i++)
    {
        if (elements[i].atomic_number < 1)
            continue;
        elements[i].depletion = 1.0 - ((1.0 - elements[i].depletion) * y_ratio);
    }

    return y_ratio;
}

// Structs for element properties
void initialize_ion_fracs(Element *elements, ParticleIonData *n_and_ion_fracs, double hdens, double metallicity, int itype)
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
        n_and_ion_fracs[i].ion_fracs = (double *)malloc((elements[i].n_ions + elements[i].n_mol) * sizeof(double));
        n_and_ion_fracs[i].ion_fracs_new = (double *)malloc((elements[i].n_ions + elements[i].n_mol) * sizeof(double));

        switch (itype)
        {
        case 1:
            // fully ionized
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++)
            {
                n_and_ion_fracs[i].ion_fracs[j] = 0.0;
                n_and_ion_fracs[i].ion_fracs_new[j] = 0.0;
            }
            n_and_ion_fracs[i].ion_fracs[elements[i].n_ions - 1] = 1.0;
            n_and_ion_fracs[i].ion_fracs_new[elements[i].n_ions - 1] = 1.0;
            break;
        case 2:
            // fully neutral
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++)
            {
                n_and_ion_fracs[i].ion_fracs[j] = 0.0;
                n_and_ion_fracs[i].ion_fracs_new[j] = 0.0;
            }
            n_and_ion_fracs[i].ion_fracs[0] = 1.0;
            n_and_ion_fracs[i].ion_fracs_new[0] = 1.0;
            break;
        case 3:
            // fully molecular or neutral
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++)
            {
                n_and_ion_fracs[i].ion_fracs[j] = 0.0;
                n_and_ion_fracs[i].ion_fracs_new[j] = 0.0;
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

double get_ne(Element *elements, ParticleIonData *n_and_ion_fracs, bool use_new)
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
                ne += n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs[j] * j;
            }
        } // End loop ion fracs
    } // End loop elements

    return fmax(ne, MIN_XION);
}

double get_mu(double xHI, double xHII, double xHeII, double xHeIII, double nH, double nHe)
{
    /*
    Calculated the mean molecular weight
    --> here we ignore metals as their contribution is minimal
    but we do include molecular hydrogen
    */

    double X = nH / (nH + 4.0 * nHe);          // Hydrogen mass fraction
    double Y = (4.0 * nHe) / (nH + 4.0 * nHe); // Helium mass fraction

    double mu = 1. / (X * (0.5 + 0.5 * xHI + 1.5 * xHII) + 0.25 * Y * (1. + xHeII + 2. * xHeIII));

    return mu;
}

void reduce_xion(double ion_fracs[], int size)
{
    /*
    Makes sure that the ion fractions sum to one and that
    the ion fractions don't go below some minimum value
    */
    double total_ion_frac = 0.0;

    // Loop over all ions and get the total
    for (int i = 0; i < size; i++)
    {
        ion_fracs[i] = fmax(ion_fracs[i], MIN_XION);
        total_ion_frac += ion_fracs[i];
        // total_ion_frac += fmax(ion_fracs[i],MIN_XION);
    }
    // Loop again to normalize
    for (int i = 0; i < size; i++)
    {
        ion_fracs[i] /= total_ion_frac;
    }
}

void reduce_xion_max(double ion_fracs[], int size)
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

void copyArray(double src[], double dest[], int size)
{
    for (int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
}

KOKKOS_FUNCTION
double get_chemical_eqm(Element *elements, ParticleIonData *n_and_ion_fracs, double TK,
                        double aexp, double total_dt, double UV_background_G0,
                        double total_cosmic_ray_ionization_rate,
                        double dust_to_gas_mass_ratio_over_mw,
                        bool include_H2, bool include_CO,
                        bool ramses_rt_T_scheme, bool rosenbrock_T_scheme,
                        bool constant_temperature)
{
    /*
    Computes chemical equilibrium

    if total_dt [s] is <=0 --> run to equilibrium otherwise stop at a fixed time
    */

    // Parameters
    //  double ddt = 3.15E16; // Initial timestep [s]
    double ddt = 10000.0 * 365.25 * 60. * 60.;
    // If we are not running to convergence, try to take one single step
    if (total_dt > 0)
    {
        ddt = total_dt;
    }

    // Convergence variables
    int success_counter = 0;
    int convergence_counter = 0;
    int total_iterations = 0;
    bool model_converged = false;
    bool convergence_bool = true;
    bool x_percent_rule = true;
    bool compute_atoms = true;
    bool compute_molecules = true;
    double total_time = 0.E0;
    double residual;
    double ne_residual;
    double max_residual;
    double max_residual_all;
    double max_residual_abs;

    // Physical variables
    double mu; // Mean molecular weight
    // double TK; //Actual temperature (not T/mu)
    double ne;                      // Electron number density
    double ne_initial;              // Electron number density for convergence
    double cr;                      // Creation rate
    double de;                      // Destruction rate
    double xe;                      // Electron fraction (n_electron/nH)
    double phi_s;                   // Secondary electron CR factor
    double cosmic_ray_scale_factor; // Most rates normalized to 1e-16

    int n_ions; // Number of ions --> reset per element
    int atomic_number;

    // Self-shielding
    double ss_factor = exp(-1.0 * n_and_ion_fracs[0].n_element / 1E-2);

    // Cooling
    double cooling_rate = 0.0;
    double cooling_rate_prime = 0.0;
    double cooling_rate_2 = 0.0;
    double cooling_rate_prime_2 = 0.0;

    double element_number_densities[MAX_ELEMENTS];
    int element_number_ions[MAX_ELEMENTS];
    Array2D element_ion_fractions;

    for (int i = 1; i < MAX_ELEMENTS; i++) {
        element_number_densities[i] = n_and_ion_fracs[i].n_element;
        element_number_ions[i] = elements[i].n_ions + elements[i].n_mol;
    }

    // # infinite loop
    while (1)
    { // Infinite loop: TODO(code): put a max loop counter here and then call an error function
        total_iterations += 1;
        compute_atoms = true;
        compute_molecules = true;

        // Loop over elements and set the ion_fracs new to
        // the old ion fracs
        for (int i = 1; i < MAX_ELEMENTS; i++)
        {
            if (elements[i].atomic_number < 1)
                continue;

            int size = elements[i].n_ions + elements[i].n_mol;
            for (int j = 0; j < size; j++)
            {
                n_and_ion_fracs[i].ion_fracs_new[j] = n_and_ion_fracs[i].ion_fracs[j];
            }
        } // End element loop

        // First get the electron number density
        ne = get_ne(elements, n_and_ion_fracs, false);
        // Update the electron fraction
        xe = ne / n_and_ion_fracs[1].n_element;
        // Measure mu
        mu = get_mu(
            n_and_ion_fracs[1].ion_fracs_new[0], n_and_ion_fracs[1].ion_fracs_new[1],
            n_and_ion_fracs[2].ion_fracs_new[1], n_and_ion_fracs[2].ion_fracs_new[2],
            n_and_ion_fracs[1].n_element, n_and_ion_fracs[2].n_element);
        // Measure phi_s for secondary CR ionization
        phi_s = secondary_cr_rates(xe);
        // Now calculate the total CR ionization rate
        double primary_cosmic_ray_ionization_rate = total_cosmic_ray_ionization_rate / (1.0 + phi_s);
        double H2_cosmic_ray_ionization_rate = 2.0 * primary_cosmic_ray_ionization_rate * (1.0 + phi_s);

        // Set the cosmic ray scale factor -> only needed for secondary UVs
        cosmic_ray_scale_factor = total_cosmic_ray_ionization_rate / 1E-16;

        // Initialize the X percent rule
        x_percent_rule = true;

        // Convergence
        convergence_bool = true;
        max_residual_all = -999.;
        max_residual = -999.;
        max_residual_abs = -999.;

        // Make sure the timestep isn't larger than how long we
        // need to evolve for, otherwise shorten it
        if (total_dt > 0.0)
        {
            if ((total_time + ddt) > total_dt)
            {
                ddt = total_dt - total_time;
            }
        }

        /////////////////////////
        //      Radiation      //
        /////////////////////////
        // TODO(code): need to add radiation variable
        // This is a place holder for now as I will deal with it
        // later

        /////////////////////////
        //     Temperature     //
        /////////////////////////

        // Loop over all elements
        for (int i = 1; i < MAX_ELEMENTS; i++)
        {
            element_number_densities[i] = n_and_ion_fracs[i].n_element;
            element_number_ions[i] = elements[i].n_ions + elements[i].n_mol;
            for (int j = 0; j < element_number_ions[i]; j++)
            {
                element_ion_fractions[i][j] = n_and_ion_fracs[i].ion_fracs_new[j];
            }
        }

        // Is we use the ramses-rt scheme for the temperature update
        // then we perform the update first
        double nH = n_and_ion_fracs[1].n_element / 0.76;      // TODO(code): hard coded at 0.76 -- need to calculate
        double X_nHkb = 1.0 / (1.5 * nH * 1.380649e-16 * mu); // TODO(code): double check mu goes here
        double TK_new = TK;
        if (!constant_temperature)
        {
            if (ramses_rt_T_scheme)
            {
                cooling_rate = all_cooling(TK, ne, aexp,
                                           element_number_densities,
                                           element_number_ions,
                                           element_ion_fractions,
                                           UV_background_G0,
                                           dust_to_gas_mass_ratio_over_mw,
                                           xe,
                                           primary_cosmic_ray_ionization_rate,
                                           2.0 * primary_cosmic_ray_ionization_rate,
                                           ss_factor);

                cooling_rate_prime = all_cooling(1.001 * TK, ne, aexp,
                                                 element_number_densities,
                                                 element_number_ions,
                                                 element_ion_fractions,
                                                 UV_background_G0,
                                                 dust_to_gas_mass_ratio_over_mw,
                                                 xe,
                                                 primary_cosmic_ray_ionization_rate,
                                                 2.0 * primary_cosmic_ray_ionization_rate,
                                                 ss_factor);

                cooling_rate_prime = (cooling_rate - cooling_rate_prime) / (TK - (1.001 * TK));

                cooling_rate *= X_nHkb;
                cooling_rate_prime *= (-X_nHkb);

                double dUU = fabs(fmax(T_MIN, TK + cooling_rate * ddt) - TK);

                // ! New T2 value
                TK_new = fmax(T_MIN, TK + cooling_rate * ddt / (1. - cooling_rate_prime * ddt));
                residual = fmax(dUU, fabs(TK_new - TK)) / (TK + T_MIN);

                // Convergence
                // X% rule
                // residual = dTK / TK;

                // Update the max residual
                max_residual = fmax(max_residual, residual);

                // Max residual over all elements
                max_residual_all = fmax(max_residual_all, max_residual);

                double max_frac_diff_temp = X_PCT_RULE_TEMP;
                if (residual > max_frac_diff_temp)
                {
                    // printf("Broken temperature %e %e\n",TK,residual);
                    // Declare the x-percent rule was violated
                    x_percent_rule = false;
                    // Dont compute molecules
                    compute_molecules = false;
                    // Don't compute atoms
                    compute_atoms = false;
                }
            }
        }

        /////////////////////////
        //      Molecules      //
        /////////////////////////
        double alpha_H2_loc = 0.0;
        double beta_H2_loc = 0.0;
        double cr_H2 = 0.E0;
        double de_H2 = 0.E0;
        if (compute_molecules)
        {

            // Molecular Hydrogen
            if (include_H2)
            {

                // H2 fraction --> Note that we actually store 2 * xH2
                double xH2 = n_and_ion_fracs[1].ion_fracs_new[2] / 2.0;

                // Creation //

                // Contains formation on dust and via the primordial channel
                // TODO(code): update G0 to be the total G0
                alpha_H2_loc = alpha_H2(TK, dust_to_gas_mass_ratio_over_mw, xe, H2_cosmic_ray_ionization_rate,
                                        UV_background_G0, n_and_ion_fracs[1].ion_fracs_new[0], n_and_ion_fracs[1].ion_fracs_new[1],
                                        n_and_ion_fracs[1].n_element); //! [cm3 s-1]

                // We can only create H2 if there is any HI and HII
                cr_H2 += alpha_H2_loc;

                // Destruction //

                // Collisional destruction
                // beta_H2_loc = beta_H2(TK,
                //                     n_and_ion_fracs[1].n_element,
                //                     n_and_ion_fracs[1].ion_fracs_new[0], // xHI
                //                     xH2, // xH2
                //                     n_and_ion_fracs[2].ion_fracs_new[0], // xHeI
                //                     ne,
                //                     n_and_ion_fracs[1].ion_fracs_new[0] * n_and_ion_fracs[1].n_element,   // nHI
                //                     xH2 * n_and_ion_fracs[1].n_element,   // nH2
                //                     n_and_ion_fracs[2].ion_fracs_new[0] * n_and_ion_fracs[2].n_element);  // nHeI

                beta_H2_loc = beta_H2_umist(TK,
                                            n_and_ion_fracs[1].ion_fracs_new[0] * n_and_ion_fracs[1].n_element,
                                            ne,
                                            xH2 * n_and_ion_fracs[1].n_element);

                de_H2 += beta_H2_loc;

                // Photodissociation
                de_H2 += UV_background_G0 * 5.68E-11;

                // Cosmic ray destruction
                de_H2 += H2_cosmic_ray_ionization_rate;

                // Update xH2
                xH2 = (cr_H2 * ddt + xH2) / (1. + de_H2 * ddt);

                // Store in the struct
                n_and_ion_fracs[1].ion_fracs_new[2] = 2.0 * fmin(fmax(xH2, MIN_XION), 0.5);

                // Make sure ions sum to 1
                int size = elements[1].n_ions + elements[1].n_mol;
                reduce_xion(n_and_ion_fracs[1].ion_fracs_new, size);

                // Update electron fraction
                ne = get_ne(elements, n_and_ion_fracs, true);

                // Convergence
                // X% rule
                residual = fabs(n_and_ion_fracs[1].ion_fracs_new[2] - n_and_ion_fracs[1].ion_fracs[2]) / fmax(n_and_ion_fracs[1].ion_fracs[2], MIN_XION_FLOOR);

                // Update the max residual
                max_residual = fmax(max_residual, residual);

                // Note that this functional format actually helps with convergence at high densities
                if (fabs(n_and_ion_fracs[1].ion_fracs_new[2] - n_and_ion_fracs[1].ion_fracs[2]) > CONV_ABS_H2_metal)
                {
                    convergence_bool = false;
                }

                // Max residual over all elements
                max_residual_all = fmax(max_residual_all, max_residual);
                max_residual_abs = fmax(max_residual_abs, fabs(n_and_ion_fracs[1].ion_fracs_new[2] - n_and_ion_fracs[1].ion_fracs[2]) / CONV_ABS);

                if (residual > X_PCT_RULE)
                {
                    // Declare the x-percent rule was violated
                    x_percent_rule = false;
                    // Break the loop over ions
                    compute_atoms = false;
                }
            }
        }

        // CO
        // TODO(code)

        /////////////////////////
        //        Atoms        //
        /////////////////////////
        if (compute_atoms)
        {
            // Loop over all elements
            for (int i = 1; i < MAX_ELEMENTS; i++)
            {
                if (elements[i].atomic_number < 1)
                    continue;

                // Get the atomic number
                atomic_number = elements[i].atomic_number;

                // Get the number of ions
                n_ions = elements[i].n_ions;

                // Initialize the max residual
                max_residual = 0.E0;

                // Loop over the number of ions
                for (int j = 0; j < n_ions; j++)
                {

                    // Update the HI number density in case it has changed e.g. due to charge exchange
                    double dust_effective_number_density = n_and_ion_fracs[1].n_element * dust_to_gas_mass_ratio_over_mw;

                    /////////////////////////
                    //       Creation      //
                    /////////////////////////
                    cr = 0.E0;

                    // Account for molecular hydrogen
                    if ((i == 1) & (j == 0) & include_H2)
                    { // select only HI
                        // Note: no factor of 2 needed since is 2*xH2
                        cr += de_H2 * n_and_ion_fracs[1].ion_fracs_new[2];
                    }

                    // Recombinations of the more excited ionization state
                    if (j < (n_ions - 1))
                        cr += recombination(TK, j + 1, i) * ne * n_and_ion_fracs[i].ion_fracs_new[j + 1];

                    // Collisional ionization of the less excited state
                    if (j > 0)
                        cr += collisional_ionization(TK, j - 1, i) * ne * n_and_ion_fracs[i].ion_fracs_new[j - 1];

                    // Photoionization of the less excited state
                    if (j > 0)
                        cr += HM12_UVB_z[i][j - 1][0] * ss_factor * n_and_ion_fracs[i].ion_fracs_new[j - 1];

                    // Photoionization by sub-ionizing ISRF --> only impacts lowest ionization states
                    if (j == 1)
                        cr += UV_background_G0 * elements[i].G0_photo_rate * n_and_ion_fracs[i].ion_fracs_new[j - 1];

                    // Cosmic ray ionization of the less excited state
                    if (j > 0)
                        cr += cosmic_ray_ionization_rates[i][j - 1] * total_cosmic_ray_ionization_rate * n_and_ion_fracs[i].ion_fracs_new[j - 1];

                    // Cosmic ray ionization of the less excited state from induced UV
                    if (j == 1)
                        cr += cosmic_ray_ionization_rates_induced_UV[i] * cosmic_ray_scale_factor * n_and_ion_fracs[i].ion_fracs_new[j - 1];

                    // Recombination on dust for the more excited state
                    if (j < (n_ions - 1))
                        cr += dust_recombination_rates(j + 1, i, TK, UV_background_G0, ne) * dust_effective_number_density * n_and_ion_fracs[i].ion_fracs_new[j + 1];

                    /////////////////////////
                    //     Destruction     //
                    /////////////////////////
                    de = 0.E0;

                    // Account for molecular hydrogen
                    if ((i == 1) & (j == 0) & include_H2)
                    { // select only HI
                        de += alpha_H2_loc;
                    }

                    // Collisional ionization
                    if (j < (n_ions - 1))
                        de += collisional_ionization(TK, j, i) * ne;

                    // Photoionization
                    if (j < (n_ions - 1))
                        de += HM12_UVB_z[i][j][0] * ss_factor;

                    // Photoionization by sub-ionizing ISRF --> only impacts lowest ionization states
                    if (j == 0)
                        de += UV_background_G0 * elements[i].G0_photo_rate;

                    // Recombination
                    if (j > 0)
                        de += recombination(TK, j, i) * ne;

                    // Cosmic ray ionization
                    if (j < (n_ions - 1))
                        de += cosmic_ray_ionization_rates[i][j] * total_cosmic_ray_ionization_rate;

                    // Cosmic ray ionization from induced UV
                    if (j == 0)
                        de += cosmic_ray_ionization_rates_induced_UV[i] * cosmic_ray_scale_factor;

                    // Recombination on dust
                    if (j > 0)
                        de += dust_recombination_rates(j, i, TK, UV_background_G0, ne) * dust_effective_number_density;

                    /////////////////////////
                    //   Charge Transfer   //
                    /////////////////////////
                    // Note, this was split off due to cross species
                    // coupling. Saves us an extra double loop
                    // Charge exchange
                    if (i == 1)
                    { // If element is hydrogen
                        // Loop over all other elements
                        for (int ii = 2; ii < MAX_ELEMENTS; ii++)
                        {
                            if (elements[ii].atomic_number < 1)
                                continue;

                            // Loop over all other ionization states
                            for (int jj = 0; jj < elements[ii].n_ions; jj++)
                            {
                                double paired_ion_number_density = n_and_ion_fracs[ii].ion_fracs_new[jj] * n_and_ion_fracs[ii].n_element;
                                if (j == 0)
                                {                                                                                                                       // H
                                    cr += charge_transfer_ionization(jj, ii, TK) * n_and_ion_fracs[i].ion_fracs_new[j + 1] * paired_ion_number_density; //! Example:  O + H+ => O+ + H
                                    de += charge_transfer_recombination(jj, ii, TK) * paired_ion_number_density;                                        //! Example:  O+ + H => O + H+
                                }
                                else
                                {                                                                                                                          // H+
                                    de += charge_transfer_ionization(jj, ii, TK) * paired_ion_number_density;                                              //! Example:  O + H+ => O+ + H
                                    cr += charge_transfer_recombination(jj, ii, TK) * n_and_ion_fracs[i].ion_fracs_new[j - 1] * paired_ion_number_density; //! Example:  O+ + H => O + H+
                                }
                            } // end loop over ionization states
                        } // end loop over other elements
                    }
                    else
                    { // All other elements
                        double HI_number_density = n_and_ion_fracs[1].ion_fracs_new[0] * n_and_ion_fracs[1].n_element;
                        double HII_number_density = n_and_ion_fracs[1].ion_fracs_new[1] * n_and_ion_fracs[1].n_element;

                        if (j > 0)
                        {
                            // Ionization from less excited state
                            cr += charge_transfer_ionization(j - 1, i, TK) * n_and_ion_fracs[i].ion_fracs_new[j - 1] * HII_number_density; //! Example:  O + H+ => O+ + H

                            // Charge exchange recombination
                            de += charge_transfer_recombination(j, i, TK) * HI_number_density; //! Example:  O+ + H => O + H+
                        }

                        if (j < (n_ions - 1))
                        {
                            // Charge exchange ionization
                            de += charge_transfer_ionization(j, i, TK) * HII_number_density; //! Example:  O + H+ => O+ + H

                            // Charge exchange recombination from the more excited state
                            cr += charge_transfer_recombination(j + 1, i, TK) * n_and_ion_fracs[i].ion_fracs_new[j + 1] * HI_number_density; //! Example:  O+ + H => O + H+
                        }
                    }

                    /////////////////////////
                    //       Update        //
                    /////////////////////////
                    // The update
                    n_and_ion_fracs[i].ion_fracs_new[j] = (cr * ddt + n_and_ion_fracs[i].ion_fracs_new[j]) / (1.0 + de * ddt);
                    n_and_ion_fracs[i].ion_fracs_new[j] = fmin(fmax(n_and_ion_fracs[i].ion_fracs_new[j], MIN_XION), 1.0);

                    int size = elements[i].n_ions + elements[i].n_mol;
                    reduce_xion(n_and_ion_fracs[i].ion_fracs_new, size);

                    // Get the new electron fraction
                    ne_initial = ne;
                    ne = get_ne(elements, n_and_ion_fracs, true);
                    xe = ne / n_and_ion_fracs[1].n_element;
                    phi_s = secondary_cr_rates(xe);
                    primary_cosmic_ray_ionization_rate = total_cosmic_ray_ionization_rate / (1.0 + phi_s);

                    /////////////////////////
                    //     Convergence     //
                    /////////////////////////
                    // X% rule
                    residual = fabs(n_and_ion_fracs[i].ion_fracs_new[j] - n_and_ion_fracs[i].ion_fracs[j]) / fmax(n_and_ion_fracs[i].ion_fracs[j], MIN_XION_FLOOR);

                    // electron density residual
                    ne_residual = fabs(ne - ne_initial) / fmax(ne, MIN_XION_FLOOR);
                    residual = fmax(residual, ne_residual);

                    // residual of derivative
                    double dy_dt = cr - (de * n_and_ion_fracs[i].ion_fracs[j]);
                    double dx_dy_dt = -1.0 * de;
                    double first_order_residual = fabs(dx_dy_dt / fmax(dy_dt, 1e-10));
                    // residual = fmax(residual,first_order_residual);

                    // Check if convergence rule is broken
                    double CONV_ABS_loc = CONV_ABS;
                    if (i > 2)
                    {
                        CONV_ABS_loc = CONV_ABS_H2_metal;
                    }

                    // Check if the model has converged
                    if (fabs(n_and_ion_fracs[i].ion_fracs_new[j] - n_and_ion_fracs[i].ion_fracs[j]) > CONV_ABS_loc)
                    {
                        convergence_bool = false;
                    }

                    // Update the max residual
                    max_residual = fmax(max_residual, residual);

                    // Max residual over all elements
                    max_residual_all = fmax(max_residual_all, max_residual);
                    max_residual_abs = fmax(max_residual_abs, fabs(n_and_ion_fracs[i].ion_fracs_new[j] - n_and_ion_fracs[i].ion_fracs[j]) / CONV_ABS_loc);

                    // Check if X percent rule is broken
                    double loc_X_PCT_RULE = X_PCT_RULE;
                    if (i > 2)
                    {
                        loc_X_PCT_RULE = X_PCT_RULE_metal;
                    }

                    if (residual > loc_X_PCT_RULE)
                    {
                        // Declare the x-percent rule was violated
                        x_percent_rule = false;
                        // Break the loop over ions
                        break;
                    }
                } // End ion loop

                // Make sure ions sum to 1
                int size = elements[i].n_ions + elements[i].n_mol;
                reduce_xion(n_and_ion_fracs[i].ion_fracs_new, size);

                // If the X% rule is violated restart with shorter timestep
                if (!x_percent_rule)
                {
                    // Reset the success counter
                    success_counter = 0;
                    // Break out of the element loop
                    break;
                }

            } // End element loop
        }

        // Update temeprature afterwards if using the rosenbrock T scheme
        if (!constant_temperature)
        {
            if (rosenbrock_T_scheme)
            {

                // Update mu
                mu = get_mu(
                    n_and_ion_fracs[1].ion_fracs_new[0], n_and_ion_fracs[1].ion_fracs_new[1],
                    n_and_ion_fracs[2].ion_fracs_new[1], n_and_ion_fracs[2].ion_fracs_new[2],
                    n_and_ion_fracs[1].n_element, n_and_ion_fracs[2].n_element);
                // Update X_nHkb
                X_nHkb = 1.0 / (1.5 * nH * 1.380649e-16 * mu); // TODO(code): double check mu goes here

                double rosenbrock_gamma = 0.5;
                double rosenbrock_h = ddt;
                double rosenbrock_time = 0.0;

                while (rosenbrock_time < ddt)
                {
                    cooling_rate = all_cooling(TK, ne, aexp,
                                               element_number_densities,
                                               element_number_ions,
                                               element_ion_fractions,
                                               UV_background_G0,
                                               dust_to_gas_mass_ratio_over_mw,
                                               xe,
                                               primary_cosmic_ray_ionization_rate,
                                               2.0 * primary_cosmic_ray_ionization_rate,
                                               ss_factor);

                    cooling_rate_prime = all_cooling(1.001 * TK, ne, aexp,
                                                     element_number_densities,
                                                     element_number_ions,
                                                     element_ion_fractions,
                                                     UV_background_G0,
                                                     dust_to_gas_mass_ratio_over_mw,
                                                     xe,
                                                     primary_cosmic_ray_ionization_rate,
                                                     2.0 * primary_cosmic_ray_ionization_rate,
                                                     ss_factor);

                    cooling_rate_prime = (cooling_rate - cooling_rate_prime) / (TK - (1.001 * TK));

                    cooling_rate *= X_nHkb;
                    cooling_rate_prime *= (-X_nHkb); // This is now the Jacobian

                    double rosenbrock_M = 1.0 - (rosenbrock_gamma * rosenbrock_h * cooling_rate_prime);
                    double rosenbrock_k1 = cooling_rate / rosenbrock_M;

                    double T_h_k1 = TK + (rosenbrock_h * rosenbrock_k1);

                    cooling_rate_2 = all_cooling(T_h_k1, ne, aexp,
                                                 element_number_densities,
                                                 element_number_ions,
                                                 element_ion_fractions,
                                                 UV_background_G0,
                                                 dust_to_gas_mass_ratio_over_mw,
                                                 xe,
                                                 primary_cosmic_ray_ionization_rate,
                                                 2.0 * primary_cosmic_ray_ionization_rate,
                                                 ss_factor);

                    cooling_rate_2 *= X_nHkb;

                    double rosenbrock_k2 = (cooling_rate_2 - (-1.0 * (rosenbrock_M - 1.0) * rosenbrock_k1)) / rosenbrock_M;
                    TK_new = TK + rosenbrock_h * (rosenbrock_k1 + rosenbrock_k2);
                    TK_new = fmin(fmax(TK_new, T_MIN), T_MAX);

                    // 10% rule
                    residual = fabs(TK_new - TK) / (TK + T_MIN);

                    if (residual > 0.05)
                    {
                        // Reduce the timestep if needed
                        // Can be more clever here
                        rosenbrock_h /= 2.0;
                    }
                    else
                    {
                        // Update the time
                        rosenbrock_time += rosenbrock_h;

                        // Don't overshoot the timestep
                        if ((rosenbrock_time + rosenbrock_h) > ddt)
                        {
                            rosenbrock_h = ddt - rosenbrock_time;
                        }
                        // Update the temperature
                        TK = TK_new;
                    }
                }
            }
        }

        /////////////////////////
        //   More Convergence  //
        /////////////////////////

        // Set new dt --> Strategy from Nick Gnedin
        max_residual_all /= (X_PCT_RULE);
        double max_residual_ddt = fmax(max_residual_all, max_residual_abs * CONV_ABS / 1e-3);
        if (max_residual_ddt > 1.0)
        {
            ddt = 0.9 * ddt / sqrt(2. + max_residual_ddt);
        }
        else
        {
            ddt = 0.9 * ddt / pow(0.07 + max_residual_ddt, 0.3);
        }
        ddt = fmin(ddt, 1E11);

        // Check if the model has converged
        if (convergence_bool)
        {
            // If yes, increment counter
            convergence_counter += 1;
        }
        else
        {
            // Else reset the counter
            convergence_counter = 0;
        }

        // If all elements completed
        // Loop over elements and ions and
        // set old values to new values
        if (x_percent_rule)
        {
            // Update temperature
            TK = fmin(fmax(TK_new, T_MIN), T_MAX);
            // Update atomic data
            for (int i = 1; i < MAX_ELEMENTS; i++)
            {
                if (elements[i].atomic_number < 1)
                    continue;

                int size = elements[i].n_ions + elements[i].n_mol;
                for (int j = 0; j < size; j++)
                {
                    n_and_ion_fracs[i].ion_fracs[j] = n_and_ion_fracs[i].ion_fracs_new[j];
                } // End ion loop
            } // End element loop
        }

        // Update the success counter
        success_counter += 1;

        // Check if we are converged
        if (convergence_counter > 5000)
        {
            // printf("#Converged\n");
            model_converged = true;
        }
        if (success_counter > 1000)
        {
            // printf("#Success\n");
            model_converged = true;
        }

        // Update the total time
        total_time += ddt;

        // Break the infinite loop if the timestep has been reached
        if (total_dt > 0.0)
        {
            if (fabs(total_time - total_dt) / total_dt < 1e-6)
                break;
        }

        // Finish the calculation if model converged
        if (model_converged)
            break; // break from infinite loop

    } // End infinite loop

    // Free the memory
    // for (int i = 1; i < MAX_ELEMENTS; i++)
    //     free(element_ion_fractions[i]);

    return TK;
}

int main()
{
    double redshift = 1000.0;                          // Redshift of the calculation
    double UV_background_G0 = 1.0;                     // G0 (sub ionizing radiaiton field)
    double generic_cosmic_ray_ionization_rate = 1E-16; // Background CR ionization rate (primary + secondary) --> H
    double metallicity = 1E-5;                         // Gas phase metallicity (solar units)
    bool include_H2 = true;                            // Whether to include molecular hydrogen
    bool include_CO = false;                           // Whether to include CO
    bool ramses_rt_T_scheme = true;                    // Whether to use the ramses-rt temperature update
    bool rosenbrock_T_scheme = false;                  // Whether to use the rosenbrocat temperature update
    bool constant_temperature = false;                 // Whether to compute at constant temperature

    // Declare an array (list) of 3 elements (structs)
    Element elements[MAX_ELEMENTS];
    ParticleIonData n_and_ion_fracs[MAX_ELEMENTS];

    // Initialize the elements
    initialize_elements(elements);

    // Get the dust to gas mass ratio and the depletion factors
    double dust_to_gas_mass_ratio_over_mw = get_dust_mass_and_depletion(elements, metallicity);

    // Initialize the UV background data
    load_UVB_data();      // First load the UVB data from files
    update_UVB(redshift); // Interpolate to the correct redshift

    // Initialize cosmic ray rates
    initialize_cr_rates();

    // Initialize the charge transfer rates
    load_ct_rates();

    // Initialize high temperature cooling tables
    initialize_high_temperature_metal_cooling();

    // Initialize the low temperature cooling tables
    init_fine_structure_tables();

    Kokkos::initialize();

    {
        int NT = 48;
        int Nions = 113;
        Kokkos::View<double **> xions("xions", NT, Nions);

        Kokkos::parallel_for("Solve rtz", NT, KOKKOS_LAMBDA(const int i) {
            const double TK = 1e6;
            double rho = pow(10.0,-2+((double)i*0.1));

            Element elements_loc[MAX_ELEMENTS];
            ParticleIonData n_and_ion_fracs_loc[MAX_ELEMENTS];

            // Initialize the elements
            initialize_elements(elements_loc);

            // Initialize the ion fractions
            initialize_ion_fracs(elements_loc,n_and_ion_fracs_loc,rho,metallicity,1);

            get_chemical_eqm(elements_loc, n_and_ion_fracs_loc, TK, (1.0/redshift) - 1.0, -1.0, UV_background_G0,
                              generic_cosmic_ray_ionization_rate, dust_to_gas_mass_ratio_over_mw,
                              include_H2, include_CO, ramses_rt_T_scheme, rosenbrock_T_scheme, constant_temperature);

            // Loop over all elements
            int iion = 0;
            for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
                if (elements[ii].atomic_number < 1) continue;

                for (int jj = 0; jj < elements[ii].n_ions + elements[ii].n_mol; jj++)
                {
                    xions(i, iion) = n_and_ion_fracs_loc[ii].ion_fracs[jj];
                    iion++;
                }
                
            } });

        // Copy the data back to the host
        Kokkos::View<double **>::HostMirror xions_h = Kokkos::create_mirror_view(xions);
        Kokkos::deep_copy(xions_h, xions);

        // Print the data
        for (int i = 0; i < NT; i++)
        {
            printf("%e %e ", log10(pow(10.0, -2 + ((double)i * 0.1))), log10(1e6));
            for (int j = 0; j < Nions; j++)
            {
                printf("%e ", xions_h(i, j));
            }
            printf("\n");
        }
    }

    Kokkos::finalize();
    return 0;
}
