#include <stdio.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <array>

#include "dyablo_api.hpp"
#include <Kokkos_Core.hpp>

// Common initialization: elements + depletion + ion fractions
static void setup_elements(std::array<Element, MAX_ELEMENTS> &elements,
                           std::array<double, MAX_ELEMENTS> &nelements_loc,
                           double* xions_loc,
                           double rho,
                           double metallicity,
                           bool include_H2)
{
    initialize_elements(elements, include_H2);

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
    double nH, T_actual, Tmu, mu;
    double xHI, xHII, xH2;
    double xHeI;
    double n_C, nCO;
    double total_iter;
};

template <typename SolverType>
void solve_one_step(const SolverType &rtz_solver,
                    Kokkos::View<double[MAX_ELEMENTS]> &d_nel,
                    Kokkos::View<double[MAX_TOTAL_IONS]> &d_xions,
                    Kokkos::View<double[1]> &d_TK,
                    Kokkos::View<double[1]> &d_nCO,
                    double metallicity, double aexp, PhysicsFlags flags,
                    int conv_its, int max_its,
                    // int n_total_ions,
                    Kokkos::View<StepResult[1]> &d_result)
{
    Kokkos::parallel_for("solve_step", 1, KOKKOS_LAMBDA(int) {
        // Build CompactIonData from views
        const Element* elements = rtz_solver.elements_d.data();
        CompactIonData n_and_ion_fracs{};
        n_and_ion_fracs.init_offsets(elements);

        int iion = 0;
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (elements[i].atomic_number < 1) continue;
            double nel = d_nel(i);
            // if (nel == 0.0) continue;
            n_and_ion_fracs.n_element[i] = nel;
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++) {
                n_and_ion_fracs[i].ion_fracs[j] = d_xions(iion);
                iion++;
            }
        }

        double TK = d_TK(0);
        double nCO = d_nCO(0);
        double out_Tmu, out_mu;
        std::array<double, N_GROUPS> N_PHOT{};
        Array2D_RT F_PHOT{};

        int total_iter_reached = 0;

        rtz_solver.solve_chemistry_and_cooling(TK, metallicity, aexp, -1.0,
                                               n_and_ion_fracs, nCO,
                                               N_PHOT, F_PHOT,
                                               out_Tmu, out_mu,
                                               conv_its, max_its, total_iter_reached, flags);

        // Write results back to views
        d_nCO(0) = nCO;
        iion = 0;
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (elements[i].atomic_number < 1) continue;
            if (n_and_ion_fracs.n_element[i] == 0.0) continue;
            d_nel(i) = n_and_ion_fracs.n_element[i];
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++) {
                d_xions(iion) = n_and_ion_fracs[i].ion_fracs[j];
                iion++;
            }
        }

        d_result(0).nH = n_and_ion_fracs.n_element[1];
        d_result(0).T_actual = out_Tmu * out_mu;
        d_result(0).Tmu = out_Tmu;
        d_result(0).mu = out_mu;
        d_result(0).xHI = n_and_ion_fracs[1].ion_fracs[0];
        d_result(0).xHII = n_and_ion_fracs[1].ion_fracs[1];
        d_result(0).xH2 = (elements[1].n_mol > 0) ? n_and_ion_fracs[1].ion_fracs[2] : 0.0;
        d_result(0).xHeI = n_and_ion_fracs[2].ion_fracs[0];
        d_result(0).n_C = n_and_ion_fracs.n_element[6];
        d_result(0).nCO = nCO;
        d_result(0).total_iter = (double)total_iter_reached;
    });
}

// One coupled photoionization+photoheating sub-step with a FIXED incident photon
// field (constant illumination). Advances T and the ion fractions over total_dt_s.
// The incident photon density d_Nphot is left unchanged (the cell is continuously
// re-supplied by the source), so we test the steady photoheating, not a one-shot
// reservoir. d_result(0).total_iter is repurposed to report the solver iteration count.
template <typename SolverType>
void solve_photoheat_step(const SolverType &rtz_solver,
                          Kokkos::View<double[MAX_ELEMENTS]> &d_nel,
                          Kokkos::View<double[MAX_TOTAL_IONS]> &d_xions,
                          Kokkos::View<double[1]> &d_TK,
                          Kokkos::View<double[N_GROUPS]> &d_Nphot,
                          double metallicity, double aexp, double total_dt_s,
                          PhysicsFlags flags,
                          int conv_its, int max_its,
                          Kokkos::View<StepResult[1]> &d_result)
{
    Kokkos::parallel_for("photoheat_step", 1, KOKKOS_LAMBDA(int) {
        const Element* elements = rtz_solver.elements_d.data();
        CompactIonData n_and_ion_fracs{};
        n_and_ion_fracs.init_offsets(elements);

        int iion = 0;
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (elements[i].atomic_number < 1) continue;
            n_and_ion_fracs.n_element[i] = d_nel(i);
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++)
                n_and_ion_fracs[i].ion_fracs[j] = d_xions(iion++);
        }

        double TK = d_TK(0);
        double nCO = 0.0;
        double out_Tmu, out_mu;
        std::array<double, N_GROUPS> N_PHOT;
        for (int g = 0; g < N_GROUPS; g++) N_PHOT[g] = d_Nphot(g);  // fixed incident field
        Array2D_RT F_PHOT{};
        int total_iter_reached = 0;

        rtz_solver.solve_chemistry_and_cooling(TK, metallicity, aexp, total_dt_s,
                                               n_and_ion_fracs, nCO,
                                               N_PHOT, F_PHOT,
                                               out_Tmu, out_mu,
                                               conv_its, max_its, total_iter_reached, flags);

        // Advance the state. NOTE: the solver's temperature I/O is T/mu (Tmu), NOT T,
        // so we must carry Tmu across steps (feed back out_Tmu, not out_Tmu*out_mu).
        // Feeding back the physical T would re-multiply by mu every step -> spurious
        // ~mu^n "heating" independent of rt_advect.
        d_TK(0) = out_Tmu;
        iion = 0;
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (elements[i].atomic_number < 1) continue;
            d_nel(i) = n_and_ion_fracs.n_element[i];
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++)
                d_xions(iion++) = n_and_ion_fracs[i].ion_fracs[j];
        }

        d_result(0).nH = n_and_ion_fracs.n_element[1];
        d_result(0).T_actual = out_Tmu * out_mu;
        d_result(0).Tmu = out_Tmu;
        d_result(0).mu = out_mu;
        d_result(0).xHI = n_and_ion_fracs[1].ion_fracs[0];
        d_result(0).xHII = n_and_ion_fracs[1].ion_fracs[1];
        d_result(0).xHeI = n_and_ion_fracs[2].ion_fracs[0];
        d_result(0).nCO = nCO;
        d_result(0).total_iter = (double)total_iter_reached;  // report solver iterations
    });
}

// Photoheating test: a single neutral H+He cell at T=100 K is held under a fixed
// incident ionizing photon field and integrated forward in time. We watch T(t) to
// see whether it settles near photoionization equilibrium (~1-2e4 K) or runs away.
static int run_photoheating_test()
{
    const double redshift = 0.0;
    const double aexp = 1.0;
    const double metallicity = 0.0;        // pure H+He (Stromgren-like)
    const bool   include_H2 = false;

    // constant_temperature=false, include_H2=false, include_CO=false,
    // rt_advect=true, include_self_shielding=true  (matches the PRISM config)
    RTZ<false, false, false, true, false> rtz_solver("../data");
    rtz_solver.need_to_update_UVB(redshift);
    rtz_solver.set_reduced_speed_of_light_factor(1.0);   // full speed of light

    // Photon groups: 13.6-15.2 eV, 15.2-24.5 eV, 24.5-54.4 eV, 54.4-500 eV
    std::array<double, N_GROUPS> E_min, E_max;
    std::array<double, 5> lower = {13.6, 15.2, 24.5, 54.4, 500.0};
    for (int g = 0; g < N_GROUPS; g++) {
        E_min[g] = lower[g];
        E_max[g] = lower[g + 1];
    }
    rtz_solver.set_photon_groups(E_min, E_max);
    const double T_blackbody = 1.0e5;
    rtz_solver.need_to_update_cross_sections(T_blackbody);

    PhysicsFlags flags;
    flags.include_collisional_ionization = true;
    flags.include_photoionization        = true;
    flags.include_cosmic_ray_ionization  = false;
    flags.include_HM12_UVB               = false;
    flags.include_dust_recombination     = false;
    flags.include_charge_exchange        = false;

    // ---------------- test-cell parameters (tweak these) ----------------
    const double rho          = 1.0e-2;     // n_H ~ 100 cm^-3 (dense, like the hot cell)
    const double T_init       = 5e3;     // K, cold neutral gas
    const double dt_step_s    = 3e15;     // s per step
    const int    NT           = 1000;
    // Incident photon number density [cm^-3]. Default: all in the softest group.
    std::array<double, N_GROUPS> Nphot0{};
    Nphot0[0] = 1e3;                        // ionization parameter U = Nphot/n_H ~ 1e-2
    Nphot0[1] = 0;
    Nphot0[2] = 0;
    // --------------------------------------------------------------------

    std::array<Element, MAX_ELEMENTS> elements;
    std::array<double, MAX_ELEMENTS> nelements_loc{};
    double xions_loc[MAX_TOTAL_IONS] = {};
    setup_elements(elements, nelements_loc, xions_loc, rho, metallicity, include_H2);

    // Force a fully neutral initial state (ground state of every element).
    int n_total_ions = 0, iion = 0;
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;
        for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++)
            // Nearly neutral, but with a tiny ionization seed so n_e > 0.
            // A *fully* neutral start (n_e = 0) leaves the solver unable to
            // bootstrap, and it maxes out the iteration count without converging.
            xions_loc[iion++] = (j == 0) ? (1.0 - 1e-8) : (j == 1 ? 1e-8 : 0.0);
        n_total_ions += elements[i].n_ions + elements[i].n_mol;
    }

    Kokkos::View<double[MAX_ELEMENTS]> d_nel("nel");
    Kokkos::View<double[MAX_TOTAL_IONS]> d_xions("xions");
    Kokkos::View<double[1]> d_TK("TK");
    Kokkos::View<double[N_GROUPS]> d_Nphot("Nphot");
    Kokkos::View<StepResult[1]> d_result("result");

    auto h_nel = Kokkos::create_mirror_view(d_nel);
    auto h_xions = Kokkos::create_mirror_view(d_xions);
    auto h_TK = Kokkos::create_mirror_view(d_TK);
    auto h_Nphot = Kokkos::create_mirror_view(d_Nphot);
    auto h_result = Kokkos::create_mirror_view(d_result);

    for (int i = 0; i < MAX_ELEMENTS; i++) h_nel(i) = nelements_loc[i];
    for (int i = 0; i < n_total_ions; i++) h_xions(i) = xions_loc[i];
    // d_TK carries T/mu (Tmu), matching the solver's convention. Convert the desired
    // physical T_init using the neutral H+He mean molecular weight (mu ~ 1.24).
    const double mu_neutral = 1.24;
    h_TK(0) = T_init / mu_neutral;
    for (int g = 0; g < N_GROUPS; g++) h_Nphot(g) = Nphot0[g];
    Kokkos::deep_copy(d_nel, h_nel);
    Kokkos::deep_copy(d_xions, h_xions);
    Kokkos::deep_copy(d_TK, h_TK);
    Kokkos::deep_copy(d_Nphot, h_Nphot);

    printf("# photoheating test: nH=%.3e cm^-3  T_init=%.1f K  U=Nphot/nH=%.3e\n",
           rho, T_init, Nphot0[0] / rho);
    printf("# bands [eV]:");
    for (int g = 0; g < N_GROUPS; g++) printf(" [%.1f-%.1f]", E_min[g], E_max[g]);
    printf("\n# step  time_yr  nH  T_K  xHI  xHII  n_iter\n");

    double t = 0.0;
    for (int i = 0; i < NT; i++) {
        solve_photoheat_step(rtz_solver, d_nel, d_xions, d_TK, d_Nphot,
                             metallicity, aexp, dt_step_s,
                             flags, 200000, 4000000, d_result);
        t += dt_step_s * 3.17e-8;  // convert to yr
        Kokkos::deep_copy(h_result, d_result);
        auto &r = h_result(0);
        printf("%d %.0f %.1e %.6e %.6e %.6e %.0f\n", i, t, r.nH, r.T_actual, r.xHI, r.xHeI, r.total_iter);
    }
    return 0;
}

static int run_ism_test()
{
    double redshift = 0.0;
    double aexp = 1.0 / (1.0 + redshift);
    double metallicity = 1E-1;
    int NT = 101;

    const bool include_H2 = false;

    RTZ<false, include_H2, false, false, false> rtz_solver("../data");
    rtz_solver.need_to_update_UVB(redshift);

    PhysicsFlags flags;
    flags.include_HM12_UVB = false;

    std::array<Element, MAX_ELEMENTS> elements;
    std::array<double, MAX_ELEMENTS> nelements_loc{};
    double xions_loc[MAX_TOTAL_IONS] = {};

    double rho_old = pow(10.0, -3.0);
    setup_elements(elements, nelements_loc, xions_loc, rho_old, metallicity, include_H2);

    // Compute total ion count
    int n_total_ions = 0;
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;
        n_total_ions += elements[i].n_ions + elements[i].n_mol;
    }

    // Device views for mutable state
    Kokkos::View<double[MAX_ELEMENTS]> d_nel("nel");
    Kokkos::View<double[MAX_TOTAL_IONS]> d_xions("xions");
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
    for (int i = 0; i < n_total_ions; i++) h_xions(i) = xions_loc[i];
    h_TK(0) = 1e5;
    h_nCO(0) = 0.0;
    Kokkos::deep_copy(d_nel, h_nel);
    Kokkos::deep_copy(d_xions, h_xions);
    Kokkos::deep_copy(d_TK, h_TK);
    Kokkos::deep_copy(d_nCO, h_nCO);

    for (int i = 0; i < NT; i++) {
        double rho = pow(10.0, -3.0 + (double)i * 7.0 / (NT - 1));

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
        //fprintf(stderr, "step %d/%d rho=%e TK=%e\n", i+1, NT, rho, r.T_actual);
        printf("%.15e %.15e %.15e %.15e %.15e %.15e\n",
               r.nH, r.T_actual, r.Tmu, r.mu,
               r.n_C, r.nCO);
    }
    return 0;
}

static int run_cie_test()
{
    double redshift = 0.0;
    double aexp = 1.0 / (1.0 + redshift);
    double metallicity = 1E0;
    int NT = 101;
    bool write_output = false;

    RTZ<true, false, false, false, false> rtz_solver("../data");
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
    double xions_loc[MAX_TOTAL_IONS] = {};

    double rho = 1e-2;
    setup_elements(elements, nelements_loc, xions_loc, rho, metallicity, false);

    // Compute total ion count
    int n_total_ions = 0;
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;
        n_total_ions += elements[i].n_ions + elements[i].n_mol;
    }

    // Device views
    Kokkos::View<double[MAX_ELEMENTS]> d_nel("nel");
    Kokkos::View<double[MAX_TOTAL_IONS]> d_xions("xions");
    Kokkos::View<double[1]> d_TK("TK");
    Kokkos::View<double[1]> d_nCO("nCO");
    Kokkos::View<StepResult[1]> d_result("result");

    auto h_nel = Kokkos::create_mirror_view(d_nel);
    auto h_xions = Kokkos::create_mirror_view(d_xions);
    auto h_TK = Kokkos::create_mirror_view(d_TK);
    auto h_nCO = Kokkos::create_mirror_view(d_nCO);
    auto h_result = Kokkos::create_mirror_view(d_result);

    for (int i = 0; i < MAX_ELEMENTS; i++) h_nel(i) = nelements_loc[i];
    for (int i = 0; i < n_total_ions; i++) h_xions(i) = xions_loc[i];
    h_nCO(0) = 0.0;
    Kokkos::deep_copy(d_nel, h_nel);
    Kokkos::deep_copy(d_xions, h_xions);
    Kokkos::deep_copy(d_nCO, h_nCO);

    auto element_name = [](int Z) -> const char* {
        switch (Z) {
            case 1:  return "H";
            case 2:  return "He";
            case 6:  return "C";
            case 7:  return "N";
            case 8:  return "O";
            case 10: return "Ne";
            case 12: return "Mg";
            case 14: return "Si";
            case 16: return "S";
            case 26: return "Fe";
            default: return "X";
        }
    };

    // One file per active element; columns are ionization states (+ H2 for H).
    FILE* ion_files[MAX_ELEMENTS] = {};
    if (write_output) {
        for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
            if (elements[ii].atomic_number < 1) continue;
            const char* name = element_name(elements[ii].atomic_number);
            char fname[256];
            snprintf(fname, sizeof(fname), "../data/CIE_init/cie_%s.dat", name);
            ion_files[ii] = fopen(fname, "w");
            fprintf(ion_files[ii], "# T_K");
            for (int j = 0; j < elements[ii].n_ions; j++)
                fprintf(ion_files[ii], " %s%d", name, j + 1);
            if (ii == 1 && elements[1].n_mol > 0)
                fprintf(ion_files[ii], " H2");
            fprintf(ion_files[ii], "\n");
        }
    }

    for (int i = 0; i < NT; i++) {
        // go from 1e3 to 1e9 K
        h_TK(0) = pow(10.0, (((double)i / (double)(NT - 1)) * (9.0 - 3.0)) + 3.0);
        Kokkos::deep_copy(d_TK, h_TK);

        auto t_start = std::chrono::high_resolution_clock::now();
        solve_one_step(rtz_solver, d_nel, d_xions, d_TK, d_nCO,
                       metallicity, aexp, flags, 20000, 40000, d_result);
        auto t_end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

        Kokkos::deep_copy(h_result, d_result);
        Kokkos::deep_copy(h_xions, d_xions);

        fprintf(stderr, "step %d/%d Tmu=%e solver_time=%ld ms\n", i+1, NT, h_TK(0), elapsed);
        
        if (write_output) {
            int counter = 0;
            for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
                if (elements[ii].atomic_number < 1) continue;
                fprintf(ion_files[ii], "%.15e", h_TK(0));
                for (int j = 0; j < elements[ii].n_ions + elements[ii].n_mol; j++) {
                    fprintf(ion_files[ii], " %.15e", h_xions(counter));
                    counter++;
                }
                fprintf(ion_files[ii], "\n");
            }
        }
    }

    if (write_output) {
        for (int ii = 1; ii < MAX_ELEMENTS; ii++) {
            if (ion_files[ii]) fclose(ion_files[ii]);
        }
    }
    return 0;
}

static void print_usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <test_name>\n", prog);
    fprintf(stderr, "  test_name: ISM, CIE, or PHOTOHEAT\n");
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
        } else if (strcmp(test_name, "PHOTOHEAT") == 0) {
            rc = run_photoheating_test();
        } else {
            fprintf(stderr, "Unknown test: %s\n", test_name);
            print_usage(argv[0]);
            rc = 1;
        }
    }
    Kokkos::finalize();
    return rc;
}
