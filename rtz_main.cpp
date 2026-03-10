#include <stdio.h>
#include <math.h>
#include <string.h>

#include "dyablo_api.hpp"
#include <Kokkos_Core.hpp>

// Common initialization: elements + depletion + ion fractions
static void setup_elements(std::array<Element, MAX_ELEMENTS> &elements,
                           std::array<double, MAX_ELEMENTS> &nelements_loc,
                           std::array<double, MAX_ELEMENTS * MAX_ELEMENTS> &xions_loc,
                           double rho, double metallicity)
{
    initialize_elements(elements);

    // Update depletion factors based on dust-to-gas ratio BEFORE computing n_element
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

    std::array<ParticleIonData, MAX_ELEMENTS> n_and_ion_fracs{};
    initialize_ion_fracs(elements, n_and_ion_fracs, rho, metallicity, 2);

    int iion = 0;
    for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
        if (elements[ii].atomic_number < 1) continue;
        nelements_loc[ii] = n_and_ion_fracs[ii].n_element;
        for (int jj = 0; jj < elements[ii].n_ions + elements[ii].n_mol; jj++)
            xions_loc[iion++] = n_and_ion_fracs[ii].ion_fracs[jj];
    }
}

static int run_ism_test()
{
    double redshift = 0.0;
    double aexp = 1.0 / (1.0 + redshift);
    double metallicity = 1E-5;
    int NT = 300;

    RTZ<false, true, true, false> rtz_solver;
    rtz_solver.need_to_update_UVB(1000.0);

    PhysicsFlags flags;
    flags.include_HM12_UVB = false;

    std::array<Element, MAX_ELEMENTS> elements;
    std::array<double, MAX_ELEMENTS> nelements_loc{};
    std::array<double, MAX_ELEMENTS * MAX_ELEMENTS> xions_loc{};

    double rho_old = pow(10.0, -2.0);
    setup_elements(elements, nelements_loc, xions_loc, rho_old, metallicity);

    double TK = 1e5;

    for (int i = 0; i < NT; i++) {
        double rho = pow(10.0, -2.0 + (double)i * 7.0 / (NT - 1));

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
                                               20000, 40000, flags);

        double T_actual = TK * out_mu;
        fprintf(stderr, "step %d/%d rho=%e TK=%e its=%d\n", i+1, NT, rho, T_actual, out_its);

        printf("%e %e %e %e %d %e %e %e %e\n",
               nelements_loc[1], T_actual, out_Tmu, out_mu, out_its,
               xions_loc[0], xions_loc[1], xions_loc[2], out_ddt);
    }
    return 0;
}

static int run_cie_test()
{
    double redshift = 0.0;
    double aexp = 1.0 / (1.0 + redshift);
    double metallicity = 1E0;
    int NT = 296;  // match golden reference (truncated at step 296)

    RTZ<true, true, false, false> rtz_solver;
    rtz_solver.need_to_update_UVB(1000.0);

    PhysicsFlags flags;
    flags.include_collisional_ionization = true;
    flags.include_photoionization = false;
    flags.include_cosmic_ray_ionization = false;
    flags.include_HM12_UVB = false;
    flags.include_dust_recombination = false;
    flags.include_charge_exchange = false;

    std::array<Element, MAX_ELEMENTS> elements;
    std::array<double, MAX_ELEMENTS> nelements_loc{};
    std::array<double, MAX_ELEMENTS * MAX_ELEMENTS> xions_loc{};

    double rho = 1e-2;
    setup_elements(elements, nelements_loc, xions_loc, rho, metallicity);

    double TK = 1e5;

    for (int i = 0; i < NT; i++) {
        TK = pow(10.0, ((double)i / 499.0) * (8.0 - 3.0) + 3.0);

        double out_Tmu, out_mu;
        int out_its;
        double out_ddt;

        rtz_solver.solve_chemistry_and_cooling(TK, metallicity, aexp,
                                               nelements_loc, xions_loc,
                                               &out_Tmu, &out_mu,
                                               &out_its, &out_ddt,
                                               20000, 40000, flags);

        double T_actual = TK * out_mu;
        fprintf(stderr, "step %d/%d TK=%e its=%d\n", i+1, NT, T_actual, out_its);

        printf("%e %e %e %e %d %e %e %e %e\n",
               nelements_loc[1], T_actual, out_Tmu, out_mu, out_its,
               xions_loc[0], xions_loc[1], xions_loc[2], out_ddt);
    }
    return 0;
}

static void print_usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <test_name>\n", prog);
    fprintf(stderr, "  test_name: ISM or CIE\n");
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *test_name = argv[1];
    int rc = 0;

    Kokkos::initialize(argc, argv);
    {
        if (strcmp(test_name, "ISM") == 0) {
            rc = run_ism_test();
        } else if (strcmp(test_name, "CIE") == 0) {
            rc = run_cie_test();
        } else {
            fprintf(stderr, "Unknown test: %s\n", test_name);
            print_usage(argv[0]);
            rc = 1;
        }
    }
    Kokkos::finalize();
    return rc;
}
