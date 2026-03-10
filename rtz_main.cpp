#include <stdio.h>
#include <math.h>

#include "dyablo_api.hpp"
#include <Kokkos_Core.hpp>

int main()
{
    double redshift = 0.0;
    double aexp = 1.0 / (1.0 + redshift);
    double metallicity = 1E-5;

    constexpr bool include_H2 = true;
    constexpr bool include_CO = false;
    constexpr bool ramses_rt_T_scheme = true;
    constexpr bool constant_temperature = false;

    int NT = 300;

    Kokkos::initialize();
    {
        RTZ<constant_temperature, ramses_rt_T_scheme, include_H2, include_CO> rtz_solver;
        // Zero out the UVB (matches reference include_UVB=false)
        rtz_solver.need_to_update_UVB(1000.0);

        std::array<Element, MAX_ELEMENTS> elements;
        initialize_elements(elements);

        // Update depletion factors based on dust-to-gas ratio BEFORE computing n_element
        // (matches reference ordering: at low metallicity, depletion → 1.0)
        {
            double loc_z = 12.0 + log10(elements[8].z_solar * metallicity / (elements[1].z_solar + 1E-20));
            double x = fmax(loc_z, 5.0);
            double y;
            if (loc_z > 8.10)
                y = 2.21 + 1.00 * (8.69 - x);
            else
                y = 0.96 + 3.10 * (8.69 - x);
            double G2D = pow(10.0, y);
            double G2D_sol = pow(10.0, 2.21);
            double y_ratio = fmax(fmin(G2D_sol / G2D, 1.0), 0.0);
            for (int i = 1; i < MAX_ELEMENTS; i++) {
                if (elements[i].atomic_number < 1) continue;
                elements[i].depletion = 1.0 - ((1.0 - elements[i].depletion) * y_ratio);
            }
        }

        // Initial density for first step
        double rho_old = pow(10.0, -2.0 + 0.0 * 7.0 / (NT - 1));
        std::array<ParticleIonData, MAX_ELEMENTS> n_and_ion_fracs{};
        initialize_ion_fracs(elements, n_and_ion_fracs, rho_old, metallicity, 2);

        // Flat ion-fraction array persists across steps (carry-over like reference)
        std::array<double, MAX_ELEMENTS> nelements_loc{};
        std::array<double, MAX_ELEMENTS * MAX_ELEMENTS> xions_loc{};
        {
            int iion = 0;
            for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
                if (elements[ii].atomic_number < 1) continue;
                nelements_loc[ii] = n_and_ion_fracs[ii].n_element;
                for (int jj = 0; jj < elements[ii].n_ions + elements[ii].n_mol; jj++)
                    xions_loc[iion++] = n_and_ion_fracs[ii].ion_fracs[jj];
            }
        }

        double TK = 1e5;   // carried over between steps

        for (int i = 0; i < NT; i++) {
            double rho = pow(10.0, -2.0 + (double)i * 7.0 / (NT - 1));

            // Scale n_element by rho/rho_old (keep ion fracs unchanged — reference behaviour)
            for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
                if (elements[ii].atomic_number < 1) continue;
                nelements_loc[ii] *= rho / rho_old;
            }
            rho_old = rho;

            double out_Tmu, out_mu;
            int out_its;
            double out_ddt;

            rtz_solver.solve_chemistry_and_cooling(TK, metallicity, aexp,
                                                   nelements_loc, xions_loc,
                                                   &out_Tmu, &out_mu,
                                                   &out_its, &out_ddt,
                                                   20000, 40000);

            double T_actual = TK * out_mu;
            fprintf(stderr, "step %d/%d rho=%e TK=%e its=%d\n", i+1, NT, rho, T_actual, out_its);

            // xHI, xHII, xH2 are the first 3 entries (H is element 1, always first)
            printf("%e %e %e %e %d %e %e %e %e\n",
                   nelements_loc[1], T_actual, out_Tmu, out_mu, out_its,
                   xions_loc[0], xions_loc[1], xions_loc[2], out_ddt);
        }
    }
    Kokkos::finalize();
    return 0;
}
