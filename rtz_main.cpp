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

struct StepResult {
    double nH, T_actual, Tmu, mu, ddt;
    double xHI, xHII, xH2;
    double n_C, nCO;
    int its;
};

template <typename SolverType>
void solve_one_step(const SolverType &rtz_solver,
                    Kokkos::View<double[MAX_ELEMENTS]> &d_nel,
                    Kokkos::View<double[MAX_ELEMENTS * MAX_ELEMENTS]> &d_xions,
                    Kokkos::View<double[1]> &d_TK,
                    Kokkos::View<double[1]> &d_nCO,
                    double metallicity, double aexp, PhysicsFlags flags,
                    int conv_its, int max_its,
                    Kokkos::View<StepResult[1]> &d_result)
{
    Kokkos::parallel_for("solve_step", 1, KOKKOS_LAMBDA(int) {
        // Copy views into local arrays
        std::array<double, MAX_ELEMENTS> nel_loc{};
        std::array<double, MAX_ELEMENTS * MAX_ELEMENTS> xions_loc{};
        for (int i = 0; i < MAX_ELEMENTS; i++) nel_loc[i] = d_nel(i);
        for (int i = 0; i < MAX_ELEMENTS * MAX_ELEMENTS; i++) xions_loc[i] = d_xions(i);
        double TK = d_TK(0);
        double nCO = d_nCO(0);

        double out_Tmu, out_mu;
        int out_its;
        double out_ddt;

        rtz_solver.solve_chemistry_and_cooling(TK, metallicity, aexp, -1.0,
                                               nel_loc, xions_loc, nCO,
                                               out_Tmu, out_mu,
                                               out_its, out_ddt,
                                               conv_its, max_its, flags);

        // Write results back to views
        d_TK(0) = TK;
        d_nCO(0) = nCO;
        for (int i = 0; i < MAX_ELEMENTS; i++) d_nel(i) = nel_loc[i];
        for (int i = 0; i < MAX_ELEMENTS * MAX_ELEMENTS; i++) d_xions(i) = xions_loc[i];

        d_result(0).nH = nel_loc[1];
        d_result(0).T_actual = TK * out_mu;
        d_result(0).Tmu = out_Tmu;
        d_result(0).mu = out_mu;
        d_result(0).its = out_its;
        d_result(0).ddt = out_ddt;
        d_result(0).xHI = xions_loc[0];
        d_result(0).xHII = xions_loc[1];
        d_result(0).xH2 = xions_loc[2];
        d_result(0).n_C = nel_loc[6];
        d_result(0).nCO = nCO;
    });
    Kokkos::fence();
}

static int run_ism_test()
{
    double redshift = 0.0;
    double aexp = 1.0 / (1.0 + redshift);
    double metallicity = 1E0;
    int NT = 300;

    RTZ<false, true, true, true> rtz_solver("./data");
    rtz_solver.need_to_update_UVB(redshift);

    PhysicsFlags flags;
    flags.include_HM12_UVB = false;

    std::array<Element, MAX_ELEMENTS> elements;
    std::array<double, MAX_ELEMENTS> nelements_loc{};
    std::array<double, MAX_ELEMENTS * MAX_ELEMENTS> xions_loc{};

    double rho_old = pow(10.0, -2.0);
    setup_elements(elements, nelements_loc, xions_loc, rho_old, metallicity);

    // Device views for mutable state
    Kokkos::View<double[MAX_ELEMENTS]> d_nel("nel");
    Kokkos::View<double[MAX_ELEMENTS * MAX_ELEMENTS]> d_xions("xions");
    Kokkos::View<double[1]> d_TK("TK");
    Kokkos::View<double[1]> d_nCO("nCO");
    Kokkos::View<StepResult[1]> d_result("result");

    // Host mirrors
    auto h_nel = Kokkos::create_mirror_view(d_nel);
    auto h_xions = Kokkos::create_mirror_view(d_xions);
    auto h_TK = Kokkos::create_mirror_view(d_TK);
    auto h_nCO = Kokkos::create_mirror_view(d_nCO);
    auto h_result = Kokkos::create_mirror_view(d_result);

    for (int i = 0; i < MAX_ELEMENTS; i++) h_nel(i) = nelements_loc[i];
    for (int i = 0; i < MAX_ELEMENTS * MAX_ELEMENTS; i++) h_xions(i) = xions_loc[i];
    h_TK(0) = 1e5;
    h_nCO(0) = 0.0;
    Kokkos::deep_copy(d_nel, h_nel);
    Kokkos::deep_copy(d_xions, h_xions);
    Kokkos::deep_copy(d_TK, h_TK);
    Kokkos::deep_copy(d_nCO, h_nCO);

    for (int i = 0; i < NT; i++) {
        double rho = pow(10.0, -2.0 + (double)i * 7.0 / (NT - 1));

        // Recompute element densities from scratch at each density step
        Kokkos::deep_copy(h_nel, d_nel);
        for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
            if (elements[ii].atomic_number < 1) continue;
            if (ii < 3) {
                h_nel(ii) = rho * elements[ii].z_solar * elements[ii].depletion;
            } else {
                h_nel(ii) = rho * elements[ii].z_solar * metallicity * elements[ii].depletion;
            }
        }
        rho_old = rho;
        Kokkos::deep_copy(d_nel, h_nel);

        // Reset nCO at each density step
        h_nCO(0) = 0.0;
        Kokkos::deep_copy(d_nCO, h_nCO);

        solve_one_step(rtz_solver, d_nel, d_xions, d_TK, d_nCO,
                       metallicity, aexp, flags, 20000, 40000, d_result);

        Kokkos::deep_copy(h_result, d_result);
        auto &r = h_result(0);
        fprintf(stderr, "step %d/%d rho=%e TK=%e its=%d\n", i+1, NT, rho, r.T_actual, r.its);
        printf("%.15e %.15e %.15e %.15e %d %.15e %.15e %.15e\n",
               r.nH, r.T_actual, r.Tmu, r.mu, r.its,
               r.n_C, r.nCO, r.ddt);
    }
    return 0;
}

static int run_cie_test()
{
    double redshift = 0.0;
    double aexp = 1.0 / (1.0 + redshift);
    double metallicity = 1E0;
    int NT = 296;

    RTZ<true, true, false, false> rtz_solver("./data");
    rtz_solver.need_to_update_UVB(redshift);

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

    // Device views
    Kokkos::View<double[MAX_ELEMENTS]> d_nel("nel");
    Kokkos::View<double[MAX_ELEMENTS * MAX_ELEMENTS]> d_xions("xions");
    Kokkos::View<double[1]> d_TK("TK");
    Kokkos::View<double[1]> d_nCO("nCO");
    Kokkos::View<StepResult[1]> d_result("result");

    auto h_nel = Kokkos::create_mirror_view(d_nel);
    auto h_xions = Kokkos::create_mirror_view(d_xions);
    auto h_TK = Kokkos::create_mirror_view(d_TK);
    auto h_nCO = Kokkos::create_mirror_view(d_nCO);
    auto h_result = Kokkos::create_mirror_view(d_result);

    for (int i = 0; i < MAX_ELEMENTS; i++) h_nel(i) = nelements_loc[i];
    for (int i = 0; i < MAX_ELEMENTS * MAX_ELEMENTS; i++) h_xions(i) = xions_loc[i];
    h_nCO(0) = 0.0;
    Kokkos::deep_copy(d_nel, h_nel);
    Kokkos::deep_copy(d_xions, h_xions);
    Kokkos::deep_copy(d_nCO, h_nCO);

    for (int i = 0; i < NT; i++) {
        h_TK(0) = pow(10.0, ((double)i / 499.0) * (8.0 - 3.0) + 3.0);
        Kokkos::deep_copy(d_TK, h_TK);

        solve_one_step(rtz_solver, d_nel, d_xions, d_TK, d_nCO,
                       metallicity, aexp, flags, 20000, 40000, d_result);

        Kokkos::deep_copy(h_result, d_result);
        auto &r = h_result(0);
        fprintf(stderr, "step %d/%d TK=%e its=%d\n", i+1, NT, r.T_actual, r.its);
        printf("%e %e %e %e %d %e %e %e %e\n",
               r.nH, r.T_actual, r.Tmu, r.mu, r.its,
               r.xHI, r.xHII, r.xH2, r.ddt);
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
