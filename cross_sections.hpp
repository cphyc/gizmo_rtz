// cross_sections.h
#pragma once
#include <array>
#include <fstream>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "constants.hpp"
#include "types.hpp"

constexpr int n_int_points = 10000;
constexpr double group_E_min[N_GROUPS] = { 13.6 };
constexpr double group_E_max[N_GROUPS] = { 500.0 };
// -------------------------------------------------------------------------
// Initialization Routine (Runs on Host)
// -------------------------------------------------------------------------
void initialize_cross_sections(const std::string& path_to_data, TabulatedData &tabData) {
    // 1. Create Host Mirrors
    auto h_tot = Kokkos::create_mirror_view(tabData.photoionization_cross_sections_tot);
    auto h_nl  = Kokkos::create_mirror_view(tabData.photoionization_cross_sections_nl);

    // 2. Initialize to large negative numbers
    Kokkos::parallel_for("Cross sections",
        Kokkos::MDRangePolicy({0, 0}, {MAX_ELEMENTS, MAX_ELEMENTS}), KOKKOS_LAMBDA(const int i, const int j)
    {
        h_tot(i, j) = {-999, -999, -999, -999, -999, -999, -999, -999, -999};
        for (auto k = 0; k < 7; ++k) h_nl(i, j, k) = {-999, -999, -999, -999, -999, -999, -1};
    });

    // 3. Load Total Cross Sections
    std::ifstream file_tot(path_to_data + "/verner_cross_sections/phfit_tot.dat");
    int i1, i2;
    double E_th, E_max, E_0, sig_0, y_a, P, y_w, y_0, y_1;

    while (file_tot >> i1 >> i2 >> E_th >> E_max >> E_0 >> sig_0 >> y_a >> P >> y_w >> y_0 >> y_1) {
        if (i1 < 27) {
            h_tot(i1, i1 - i2 + 1) = {
                .E_th  = E_th,
                .E_max = E_max,
                .E_0   = E_0,
                .sig_0 = sig_0,
                .y_a   = y_a,
                .P     = P,
                .y_w   = y_w,
                .y_0   = y_0,
                .y_1   = y_1
            };
        }
    }
    file_tot.close();

    // 4. Load n,l-resolved Cross Sections
    // TODO(code): generalize location of data files
    std::ifstream file_nl(path_to_data + "/verner_cross_sections/phfit.dat");
    int a, b, c, d;

    while (file_nl >> a >> b >> c >> d >> E_th >> E_0 >> sig_0 >> y_a >> P >> y_w) {
        if (a < 27) {
            int nl_state = 0;
            if      (c == 1 && d == 0) nl_state = 1; // 1s
            else if (c == 2 && d == 0) nl_state = 2; // 2s
            else if (c == 2 && d == 1) nl_state = 3; // 2p
            else if (c == 3 && d == 0) nl_state = 4; // 3s
            else if (c == 3 && d == 1) nl_state = 5; // 3p
            else if (c == 3 && d == 2) nl_state = 6; // 3d
            else if (c == 4 && d == 0) nl_state = 7; // 4s

            if (nl_state != 0) {
                h_nl(a, a - b + 1, nl_state) = {
                    .E_th  = E_th,
                    .E_0   = E_0,
                    .sig_0 = sig_0,
                    .y_a   = y_a,
                    .P     = P,
                    .y_w   = y_w,
                    .l     = d
                };
            }
        }
    }
    file_nl.close();

    // 5. Deep copy from Host memory to Device memory
    Kokkos::deep_copy(tabData.photoionization_cross_sections_tot, h_tot);
    Kokkos::deep_copy(tabData.photoionization_cross_sections_nl, h_nl);
}

// -------------------------------------------------------------------------
// Math Functions (Callable on Device)
// -------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
double F_of_y(double y, double yw, double ya, double P, double Q) {
    return (pow(y - 1.0, 2.0) + pow(yw, 2.0)) * pow(y, -Q) * pow(1.0 + sqrt(y / ya), -P);
}

KOKKOS_INLINE_FUNCTION
double verner_cs_nl(double E, double sig_0, double E_0, double yw, double ya, double P, double l) {
    double y = E / E_0;
    double Q = 5.5 + l - (0.5 * P);
    return sig_0 * F_of_y(y, yw, ya, P, Q);
}

// -------------------------------------------------------------------------
// Main Cross Section Calculator (Callable on Device)
// -------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
double getCrosssection_rtz(double lambda, int element, int ion, const TabulatedData& tabData) {
    double cross_sec = 0.0;

    // Convert lambda into ev
    double E = H_PLANCK * C_CGS / (lambda * 1.0e-8) / EV_2_ERG;

    // Deal with molecular hydrogen separately
    if (element == 1 && ion == 3) {
        if (E > 11.20 && E < 13.59) return 2.47e-18;
        if (E >= 13.59 && E <= 15.21) return 0.0;
        if (E > 15.21 && E <= 15.45) return 0.09e-18;
        if (E > 15.70 && E <= 15.95) return 1.15e-18;
        if (E > 15.95 && E <= 16.20) return 3.00e-18;
        if (E > 16.20 && E <= 16.40) return 5.00e-18;
        if (E > 16.40 && E <= 16.65) return 6.75e-18;
        if (E > 16.65 && E <= 16.85) return 8.00e-18;
        if (E > 16.85 && E <= 17.00) return 9.00e-18;
        if (E > 17.00 && E <= 17.20) return 9.50e-18;
        if (E > 17.20 && E <= 17.65) return 9.80e-18;
        if (E > 17.65 && E <= 18.10) return 10.10e-18;
        if (E > 18.10) return (10.10e-18) * pow(18.10 / E, 3.0);
        return 0.0;
    }

    // Access global Kokkos views directly
    auto &[E_th, E_max, E_0_tot, sig_0_tot, y_a, P_tot, y_w, y_0, y_1] = tabData.photoionization_cross_sections_tot(element, ion);

    if (E < E_th) {
        return 0.0;
    }

    double x = (E / E_0_tot) - y_0;
    double y = sqrt((x * x) + (y_1 * y_1));

    double F = pow(x - 1.0, 2.0) + pow(y_w, 2.0);
    F *= pow(y, (0.5 * P_tot) - 5.5);
    F *= pow(1.0 + sqrt(y / y_a), -P_tot);

    cross_sec = sig_0_tot * F * 1.0e-18;

    // Above E_max, use the n,l resolved cross sections
    if (E > E_max) {
        cross_sec = 0.0;
        for (int i = 1; i <= 7; ++i) {
            double l = tabData.photoionization_cross_sections_nl(element, ion, i).l;
            if (l >= 0.0) {
                double E_0_nl = tabData.photoionization_cross_sections_nl(element, ion, i).E_0;
                double sig_0_nl = tabData.photoionization_cross_sections_nl(element, ion, i).sig_0;
                double ya_nl = tabData.photoionization_cross_sections_nl(element, ion, i).y_a;
                double P_nl = tabData.photoionization_cross_sections_nl(element, ion, i).P;
                double yw_nl = tabData.photoionization_cross_sections_nl(element, ion, i).y_w;
                
                cross_sec += verner_cs_nl(E, sig_0_nl, E_0_nl, yw_nl, ya_nl, P_nl, l);
            }
        }
        cross_sec *= 1.0e-18; // Convert to cm^-2
    }

    return cross_sec;
}

KOKKOS_INLINE_FUNCTION
double blackbody_nu(double T, double nu){
    // Blackbody function B_lam
    // T --> temeprature [K]
    // ni --> frequency [Hz] 

    // now compute B_lam
    double B_nu = 2.0 * H_PLANCK * pow(nu,3.0) / pow(C_CGS,2.0);
    B_nu = B_nu * (1.0 / (exp(H_PLANCK * nu / (KB * T)) - 1.0));

    return B_nu;
}

KOKKOS_INLINE_FUNCTION
double sigma_N_num(double nu, double T, int element, int ion, const TabulatedData& tabData){
    return getCrosssection_rtz((C_CGS / nu)*1e8, element, ion, tabData) * blackbody_nu(T, nu) / (H_PLANCK * nu);
}

KOKKOS_INLINE_FUNCTION

double sigma_N_den(double nu, double T, int element, int ion){
    return blackbody_nu(T, nu) / (H_PLANCK * nu);
}

KOKKOS_INLINE_FUNCTION
double sigma_N(double E0, double E1, double T, int element, int ion, const TabulatedData& tabData){
    // Photon number-weighted cross section
    // E0 lower energy of the bin in eV
    // E1 upper energy of the bin in eV

    double a = (E0 * EV_2_ERG) / H_PLANCK;
    double b = (E1 * EV_2_ERG) / H_PLANCK;
    double h = (b - a) / n_int_points;

    double sum = 0.0;

    // Simpson's rule for sigma * B / hnu
    sum = sigma_N_num(a, T, element, ion, tabData);
    sum += sigma_N_num(b, T, element, ion, tabData);
    for (int i = 1; i < n_int_points; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            sum += 2 * sigma_N_num(x, T, element, ion, tabData);
        } else {
            sum += 4 * sigma_N_num(x, T, element, ion, tabData);
        }
    }
    double numerator = sum * h / 3.0;

    // Simpson's rule for B / hnu
    sum = sigma_N_den(a, T, element, ion);
    sum += sigma_N_den(b, T, element, ion);
    for (int i = 1; i < n_int_points; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            sum += 2 * sigma_N_den(x, T, element, ion);
        } else {
            sum += 4 * sigma_N_den(x, T, element, ion);
        }
    }
    double denominator = sum * h / 3.0;

    return numerator / denominator;
}

KOKKOS_INLINE_FUNCTION
double sigma_E_num(double nu, double T, int element, int ion, const TabulatedData& tabData){
    return getCrosssection_rtz((C_CGS / nu)*1e8, element, ion, tabData) * blackbody_nu(T, nu);
}

KOKKOS_INLINE_FUNCTION
double sigma_E_den(double nu, double T, int element, int ion){
    return blackbody_nu(T, nu);
}

KOKKOS_INLINE_FUNCTION
double sigma_E(double E0, double E1, double T, int element, int ion, const TabulatedData& tabData){
    // Photon energy-weighted cross section
    // E0 lower energy of the bin in eV
    // E1 upper energy of the bin in eV

    double a = (E0 * EV_2_ERG) / H_PLANCK;
    double b = (E1 * EV_2_ERG) / H_PLANCK;
    double h = (b - a) / n_int_points;

    double sum = 0.0;

    // Simpson's rule for sigma * B / hnu
    sum = sigma_E_num(a, T, element, ion, tabData);
    sum += sigma_E_num(b, T, element, ion, tabData);
    for (int i = 1; i < n_int_points; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            sum += 2 * sigma_E_num(x, T, element, ion, tabData);
        } else {
            sum += 4 * sigma_E_num(x, T, element, ion, tabData);
        }
    }
    double numerator = sum * h / 3.0;

    // Simpson's rule for B / hnu
    sum = sigma_E_den(a, T, element, ion);
    sum += sigma_E_den(b, T, element, ion);
    for (int i = 1; i < n_int_points; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            sum += 2 * sigma_E_den(x, T, element, ion);
        } else {
            sum += 4 * sigma_E_den(x, T, element, ion);
        }
    }
    double denominator = sum * h / 3.0;

    return numerator / denominator;
}

KOKKOS_INLINE_FUNCTION
double e_bar_num(double nu, double T){
    return blackbody_nu(T, nu);
}

KOKKOS_INLINE_FUNCTION
double e_bar_den(double nu, double T){
    return blackbody_nu(T, nu) / (H_PLANCK * nu);
}

KOKKOS_INLINE_FUNCTION
double e_bar(double E0, double E1, double T){
    // Mean energy in the bin
    // E0 lower energy of the bin in eV
    // E1 upper energy of the bin in eV

    double a = (E0 * EV_2_ERG) / H_PLANCK;
    double b = (E1 * EV_2_ERG) / H_PLANCK;
    double h = (b - a) / n_int_points;

    double sum = 0.0;

    // Simpson's rule for sigma * B / hnu
    sum = e_bar_num(a, T);
    sum += e_bar_num(b, T);
    for (int i = 1; i < n_int_points; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            sum += 2 * e_bar_num(x, T);
        } else {
            sum += 4 * e_bar_num(x, T);
        }
    }
    double numerator = sum * h / 3.0;

    // Simpson's rule for B / hnu
    sum = e_bar_den(a, T);
    sum += e_bar_den(b, T);
    for (int i = 1; i < n_int_points; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            sum += 2 * e_bar_den(x, T);
        } else {
            sum += 4 * e_bar_den(x, T);
        }
    }
    double denominator = sum * h / 3.0;

    return (numerator / denominator) / EV_2_ERG;
}

// TODO(code): Declate the 4D array and then copy to the tabdata
void update_cross_sections(double T, TabulatedData &tabData) {
    // Updates the cross sections for each element and group
    // TODO(code): update C_CGS with the reduced speed of light if needed
    printf("Updating cross sections to a %e K blackbody\n",T);

    auto &cs_ph = tabData.cs_ph;
    const auto &verner_cross_sections = tabData.photoionization_cross_sections_tot;
    const auto &group_E_min = tabData.group_E_min;
    const auto &group_E_max = tabData.group_E_max;
    
    auto grp_element_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_GROUPS, 27});
    Kokkos::parallel_for("update_cross_sections", grp_element_policy, KOKKOS_LAMBDA(const int i, const int j) {
        // First get the group energy in eV
        double group_energy = e_bar(group_E_min(i), group_E_max(i), T);

        // Initialize everything to 0 first
        for (int k = 0; k < 27; k++){
            cs_ph(j, k, i, 0) = 0.0;
            cs_ph(j, k, i, 1) = 0.0;
            cs_ph(j, k, i, 2) = 0.0;
        }

        // Now calculate cross sections
        for (int k = 0; k < j; k++){ // Loop over ions
            // Update number-weighted cross section
            cs_ph(j, k, i, 0) = sigma_N(group_E_min(i), group_E_max(i), T, j, k, tabData) * C_CGS; // cm^3 / s

            // Update energy-weighted cross section
            cs_ph(j, k, i, 1) = sigma_E(group_E_min(i), group_E_max(i), T, j, k, tabData) * C_CGS; // cm^3 / s

            // Update photoheating rates
            cs_ph(j, k, i, 2) = (cs_ph(j, k, i, 1) * group_energy) - (cs_ph(j, k, i, 0) * verner_cross_sections(j, k).E_th);
            cs_ph(j, k, i, 2) = fmax(cs_ph(j, k, i, 2),0.0) * EV_2_ERG; // Don't let photoheating go below 0.0
        } // End loop over elements

        if (i == 0) {
            // Deal with H2 separately
            cs_ph(1, 2, i, 0) = sigma_N(group_E_min(i), group_E_max(i), T, 1, 2, tabData) * C_CGS; // cm^3 / s
            cs_ph(1, 2, i, 1) = sigma_E(group_E_min(i), group_E_max(i), T, 1, 2, tabData) * C_CGS; // cm^3 / s
            cs_ph(1, 2, i, 2) = (cs_ph(1, 2, i, 1) * group_energy) - (cs_ph(1, 2, i, 0) * verner_cross_sections(1, 2).E_th);
            cs_ph(1, 2, i, 2) = fmax(cs_ph(1, 2, i, 2),0.0) * EV_2_ERG; // Don't let photoheating go below 0.0
        }
    });
}