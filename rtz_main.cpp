#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>

#include "dyablo_api.hpp"

#include <Kokkos_Core.hpp>

int main()
{
    double redshift = 1000.0;                          // Redshift of the calculation
    double UV_background_G0 = 1.0;                     // G0 (sub ionizing radiaiton field)
    double generic_cosmic_ray_ionization_rate = 1E-16; // Background CR ionization rate (primary + secondary) --> H
    double metallicity = 1E-5;                         // Gas phase metallicity (solar units)
    constexpr bool include_H2 = true;                            // Whether to include molecular hydrogen
    constexpr bool include_CO = false;                           // Whether to include CO
    constexpr bool ramses_rt_T_scheme = true;                    // Whether to use the ramses-rt temperature update
    constexpr bool rosenbrock_T_scheme = false;                  // Whether to use the rosenbrocat temperature update
    constexpr bool constant_temperature = false;                 // Whether to compute at constant temperature

    // // Declare an array (list) of 3 elements (structs)
    // Element elements[MAX_ELEMENTS];
    // // ParticleIonData n_and_ion_fracs[MAX_ELEMENTS];

    // // Initialize the elements
    // initialize_elements(elements);

    // // Get the dust to gas mass ratio and the depletion factors
    // double dust_to_gas_mass_ratio_over_mw = get_dust_mass_and_depletion(elements, metallicity);

    // // Initialize the UV background data
    // load_UVB_data();      // First load the UVB data from files
    // update_UVB(redshift); // Interpolate to the correct redshift

    // // Initialize cosmic ray rates
    // initialize_cr_rates();

    // // Initialize the charge transfer rates
    // load_ct_rates();

    // // Initialize high temperature cooling tables
    // initialize_high_temperature_metal_cooling();

    // // Initialize the low temperature cooling tables
    // init_fine_structure_tables();

    Kokkos::initialize();

    {
        RTZ<constant_temperature, ramses_rt_T_scheme, rosenbrock_T_scheme, include_H2, include_CO> rtz_solver;

        int NT = 48;
        int Nions = 113;
        Kokkos::View<double **> xions("xions", NT, Nions);
        Kokkos::View<double **> nelements("nelements", NT, MAX_ELEMENTS);

        std::cout << "Starting chemistry and cooling calculations..." << std::endl;

        // TabulatedData tabData;

        // // HM-related data
        // copy_data_3D(HM12_UVB_z, tabData.HM12_UVB_z);
        // copy_data_1D(G0_heating_rates, tabData.G0_heating_rates);

        // // Cosmic ray and dust-related data
        // copy_data_2D(cosmic_ray_ionization_rates, tabData.cosmic_ray_ionization_rates);
        // copy_data_1D(cosmic_ray_ionization_rates_induced_UV, tabData.cosmic_ray_ionization_rates_induced_UV);
        // copy_data_1D(cosmic_ray_ionization_rates_induced_UV_heat, tabData.cosmic_ray_ionization_rates_induced_UV_heat);
        // copy_data_2D(dust_rec_coefs, tabData.dust_rec_coefs);

        // // Charge transfer
        // copy_data_3D(CTRecomb, tabData.CTRecomb);
        // copy_data_3D(CTIon, tabData.CTIon);

        // // Fine structure cooling
        // copy_data_3D(fs_cool_tab, tabData.fs_cool_tab);

        // // High-T cooling rates
        // copy_data_3D(high_t_cooling_rates, tabData.high_t_cooling_rates);
        // copy_data_2D(high_t_cooling_rates_tflag, tabData.high_t_cooling_rates_tflag);

        // init_recombination_rates(tabData);
        // init_collisional_ionization(tabData);


        Kokkos::parallel_for("Solve rtz", NT, KOKKOS_LAMBDA(const int i) {
            double TK = 1e6;
            double rho = pow(10.0, -2 + 4.8 * i / double(NT));
            // double rho = pow(10.0,-2+((double)i*0.1));

            ParticleIonData n_and_ion_fracs_loc[MAX_ELEMENTS];

            Element elements_loc[MAX_ELEMENTS];
            // Initialize the elements
            initialize_elements(elements_loc);

            // Initialize the ion fractions
            initialize_ion_fracs(elements_loc,n_and_ion_fracs_loc,rho,metallicity,1);

            // get_chemical_eqm<constant_temperature, ramses_rt_T_scheme, rosenbrock_T_scheme, include_H2, include_CO>(elements_loc, n_and_ion_fracs_loc, TK, (1.0/redshift) - 1.0, -1.0, UV_background_G0,
            //                 generic_cosmic_ray_ionization_rate, dust_to_gas_mass_ratio_over_mw, tabData);

            std::cout << "Making Kokkos Views" << std::endl;
            // Loop over all elements
            double xions_loc[MAX_ELEMENTS * MAX_ELEMENTS];
            double nelements_loc[MAX_ELEMENTS];
            std::cout << "Copying data to Kokkos Views" << std::endl;
            int iion = 0;
            for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
                if (elements_loc[ii].atomic_number < 1) continue;

                nelements_loc[ii] = n_and_ion_fracs_loc[ii].n_element;

                for (int jj = 0; jj < elements_loc[ii].n_ions + elements_loc[ii].n_mol; jj++)
                {
                    xions_loc[iion] = n_and_ion_fracs_loc[ii].ion_fracs[jj];
                    iion++;
                }
            }

            std::cout << "Density: " << rho << " Temperature: " << TK << std::endl;

            rtz_solver.solve_chemistry_and_cooling(TK, rho, metallicity, (1.0/redshift) - 1.0, nelements_loc, xions_loc);

            std::cout << "Density: " << rho << " Temperature: " << TK << std::endl;

        });

        // Copy the data back to the host
        Kokkos::View<double **>::HostMirror xions_h = Kokkos::create_mirror_view(xions);
        Kokkos::deep_copy(xions_h, xions);
        Kokkos::View<double **>::HostMirror nelements_h = Kokkos::create_mirror_view(nelements);
        Kokkos::deep_copy(nelements_h, nelements);

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
