// cross_sections.h
#pragma once
#include <array>
#include <fstream>
#include <cmath>
#include <vector>
#include <Kokkos_Core.hpp>

#include "constants.hpp"
#include "types.hpp"

constexpr int n_int_points = 10000;
constexpr double group_E_min[N_GROUPS] = { 13.6 };
constexpr double group_E_max[N_GROUPS] = { 500.0 };
constexpr int n_photo_elements = 27;
constexpr int n_photo_pairs = (n_photo_elements * (n_photo_elements - 1)) / 2;

// -------------------------------------------------------------------------
// Load the Zubko+2004 BARE-GR-S dust absorption cross sections (Runs on Host)
// -------------------------------------------------------------------------
void load_bare_gr_s_dust(const std::string& path_to_data, auto &tabData) {
    auto h_wav      = Kokkos::create_mirror_view(tabData.dust_wav);
    auto h_cabs     = Kokkos::create_mirror_view(tabData.dust_cabs);
    auto h_log_wav  = Kokkos::create_mirror_view(tabData.dust_log_wav);
    auto h_log_cabs = Kokkos::create_mirror_view(tabData.dust_log_cabs);

    const std::string filepath = path_to_data + "/zubko2004_bare_gr_s_cross_sections.dat";
    std::ifstream file(filepath);
    if (!file.is_open()) {
        Kokkos::abort(("ERROR: could not open " + filepath).c_str());
    }

    std::string line;
    int n = 0;
    while (std::getline(file, line)) {
        // Skip comment lines (first non-whitespace character is '!') and blanks
        const size_t p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos || line[p] == '!') continue;
        if (n >= N_DUST_WAV) break;

        std::istringstream iss(line);
        double wav, cabs, csca, csrp; // columns: wav, C_abs, C_sca, (1-g)C_sca
        if (!(iss >> wav >> cabs >> csca >> csrp)) continue;

        h_wav(n)      = wav;
        h_cabs(n)     = cabs;
        h_log_wav(n)  = log10(wav);
        h_log_cabs(n) = (cabs > 0.0) ? log10(cabs) : -50.0;
        ++n;
    }
    file.close();

    if (n != N_DUST_WAV) {
        Kokkos::abort("ERROR: unexpected row count in zubko2004_bare_gr_s_cross_sections.dat");
    }

    Kokkos::deep_copy(tabData.dust_wav,      h_wav);
    Kokkos::deep_copy(tabData.dust_cabs,     h_cabs);
    Kokkos::deep_copy(tabData.dust_log_wav,  h_log_wav);
    Kokkos::deep_copy(tabData.dust_log_cabs, h_log_cabs);
}

// -------------------------------------------------------------------------
// Initialization Routine (Runs on Host)
// -------------------------------------------------------------------------
void initialize_cross_sections(const std::string& path_to_data, auto &tabData) {
    // 1. Create Host Mirrors
    auto h_tot = Kokkos::create_mirror_view(tabData.photoionization_cross_sections_tot);
    auto h_nl  = Kokkos::create_mirror_view(tabData.photoionization_cross_sections_nl);

    // 2. Initialize to large negative numbers
    for (int i = 0; i < MAX_ELEMENTS; ++i) {
        for (int j = 0; j < MAX_ELEMENTS; ++j) {
            h_tot(i, j) = {-999, -999, -999, -999, -999, -999, -999, -999, -999};
            for (int k = 0; k < 8; ++k) h_nl(i, j, k) = {-999, -999, -999, -999, -999, -999, -1};
        }
    }

    // 3. Load Total Cross Sections
    std::ifstream file_tot(path_to_data + "/verner_cross_sections/phfit_tot.dat");
    int i1, i2;
    double E_th, E_max, E_0, sig_0, y_a, P, y_w, y_0, y_1;

    while (file_tot >> i1 >> i2 >> E_th >> E_max >> E_0 >> sig_0 >> y_a >> P >> y_w >> y_0 >> y_1) {
        if (i1 < 27) {
            h_tot(i1, i1 - i2 + 1) = {
                .E_th  = static_cast<Real>(E_th),
                .E_max = static_cast<Real>(E_max),
                .E_0   = static_cast<Real>(E_0),
                .sig_0 = static_cast<Real>(sig_0),
                .y_a   = static_cast<Real>(y_a),
                .P     = static_cast<Real>(P),
                .y_w   = static_cast<Real>(y_w),
                .y_0   = static_cast<Real>(y_0),
                .y_1   = static_cast<Real>(y_1)
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
                    .E_th  = static_cast<Real>(E_th),
                    .E_0   = static_cast<Real>(E_0),
                    .sig_0 = static_cast<Real>(sig_0),
                    .y_a   = static_cast<Real>(y_a),
                    .P     = static_cast<Real>(P),
                    .y_w   = static_cast<Real>(y_w),
                    .l     = d
                };
            }
        }
    }
    file_nl.close();

    // 5. Deep copy from Host memory to Device memory
    Kokkos::deep_copy(tabData.photoionization_cross_sections_tot, h_tot);
    Kokkos::deep_copy(tabData.photoionization_cross_sections_nl, h_nl);

    // 6. Load the tabulated dust (BARE-GR-S) absorption cross sections
    load_bare_gr_s_dust(path_to_data, tabData);
}

// -------------------------------------------------------------------------
// Math Functions (Callable on Device)
// -------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
Real F_of_y(Real y, Real yw, Real ya, Real P, Real Q) {
    return (Kokkos::pow(y - Real(1.0), Real(2.0)) + Kokkos::pow(yw, Real(2.0))) * Kokkos::pow(y, -Q) * Kokkos::pow(Real(1.0) + Kokkos::sqrt(y / ya), -P);
}

KOKKOS_INLINE_FUNCTION
Real verner_cs_nl(Real E, Real sig_0, Real E_0, Real yw, Real ya, Real P, Real l) {
    Real y = E / E_0;
    Real Q = Real(5.5) + l - (Real(0.5) * P);
    return sig_0 * F_of_y(y, yw, ya, P, Q);
}

// -------------------------------------------------------------------------
// Main Cross Section Calculator (Callable on Device)
// -------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
Real getCrosssection_rtz(Real lambda, int element, int ion, const auto& tabData) {
    Real cross_sec = Real(0.0);

    // Convert lambda into ev
    Real E = Real(H_PLANCK) * Real(C_CGS) / (lambda * Real(1.0e-8)) / Real(EV_2_ERG);

    // Deal with molecular hydrogen separately
    if (element == 1 && ion == 3) {
        if (E > Real(11.20) && E < Real(13.59)) return Real(2.47e-18);
        if (E >= Real(13.59) && E <= Real(15.21)) return Real(0.0);
        if (E > Real(15.21) && E <= Real(15.45)) return Real(0.09e-18);
        if (E > Real(15.70) && E <= Real(15.95)) return Real(1.15e-18);
        if (E > Real(15.95) && E <= Real(16.20)) return Real(3.00e-18);
        if (E > Real(16.20) && E <= Real(16.40)) return Real(5.00e-18);
        if (E > Real(16.40) && E <= Real(16.65)) return Real(6.75e-18);
        if (E > Real(16.65) && E <= Real(16.85)) return Real(8.00e-18);
        if (E > Real(16.85) && E <= Real(17.00)) return Real(9.00e-18);
        if (E > Real(17.00) && E <= Real(17.20)) return Real(9.50e-18);
        if (E > Real(17.20) && E <= Real(17.65)) return Real(9.80e-18);
        if (E > Real(17.65) && E <= Real(18.10)) return Real(10.10e-18);
        if (E > Real(18.10)) return (Real(10.10e-18)) * Kokkos::pow(Real(18.10) / E, Real(3.0));
        return Real(0.0);
    }

    // Access global Kokkos views directly
    const auto& fit = tabData.photoionization_cross_sections_tot(element, ion);
    const Real E_th = fit.E_th;
    const Real E_max = fit.E_max;
    const Real E_0_tot = fit.E_0;
    const Real sig_0_tot = fit.sig_0;
    const Real y_a = fit.y_a;
    const Real P_tot = fit.P;
    const Real y_w = fit.y_w;
    const Real y_0 = fit.y_0;
    const Real y_1 = fit.y_1;


    if (E < E_th) {
        return Real(0.0);
    }

    Real x = (E / E_0_tot) - y_0;
    Real y = Kokkos::sqrt((x * x) + (y_1 * y_1));

    Real F = Kokkos::pow(x - Real(1.0), Real(2.0)) + Kokkos::pow(y_w, Real(2.0));
    F *= Kokkos::pow(y, (Real(0.5) * P_tot) - Real(5.5));
    F *= Kokkos::pow(Real(1.0) + Kokkos::sqrt(y / y_a), -P_tot);

    cross_sec = sig_0_tot * F * Real(1.0e-18);

    // Above E_max, use the n,l resolved cross sections
    if (E > E_max) {
        cross_sec = Real(0.0);
        for (int i = 1; i < 8; ++i) {
            Real l = tabData.photoionization_cross_sections_nl(element, ion, i).l;
            if (l >= Real(0.0)) {
                Real E_0_nl = tabData.photoionization_cross_sections_nl(element, ion, i).E_0;
                Real sig_0_nl = tabData.photoionization_cross_sections_nl(element, ion, i).sig_0;
                Real ya_nl = tabData.photoionization_cross_sections_nl(element, ion, i).y_a;
                Real P_nl = tabData.photoionization_cross_sections_nl(element, ion, i).P;
                Real yw_nl = tabData.photoionization_cross_sections_nl(element, ion, i).y_w;
                
                cross_sec += verner_cs_nl(E, sig_0_nl, E_0_nl, yw_nl, ya_nl, P_nl, l);
            }
        }
        cross_sec *= Real(1.0e-18); // Convert to cm^-2
    }

    return cross_sec;
}

// -------------------------------------------------------------------------
// Dust absorption cross section (Callable on Device)
// -------------------------------------------------------------------------
// Effective per-H dust absorption cross section [cm^2/H] for the BARE-GR-S model
KOKKOS_INLINE_FUNCTION
Real getCrosssection_dust(Real lambda, const auto& tabData) {
    const auto& wav      = tabData.dust_wav;
    const auto& cabs     = tabData.dust_cabs;
    const auto& log_wav  = tabData.dust_log_wav;
    const auto& log_cabs = tabData.dust_log_cabs;

    if (lambda <= wav(0)) {
        return cabs(0);
    } else if (lambda >= wav(N_DUST_WAV - 1)) {
        return cabs(N_DUST_WAV - 1);
    }

    // Binary search for the bracketing wavelength interval [low, low+1]
    int low = 0;
    int high = N_DUST_WAV - 1;
    while (high - low > 1) {
        const int mid = (low + high) / 2;
        if (wav(mid) <= lambda) low = mid;
        else high = mid;
    }

    const Real t = (Kokkos::log10(lambda) - log_wav(low)) / (log_wav(low + 1) - log_wav(low));
    const Real log_val = log_cabs(low) + t * (log_cabs(low + 1) - log_cabs(low));
    return Kokkos::pow(Real(10.0), log_val);
}

KOKKOS_INLINE_FUNCTION
Real blackbody_nu(Real T, Real nu){
    // Blackbody function B_lam
    // T --> temeprature [K]
    // ni --> frequency [Hz] 

    // now compute B_lam
#ifdef RTZ_FLOAT32_CROSS_SECTIONS
    Real B_nu = ((Real(2) * Real(H_PLANCK) / Real(C_CGS)) * nu) * ((nu / Real(C_CGS)) * nu);
#else
    Real B_nu = Real(2.0) * Real(H_PLANCK) * Kokkos::pow(nu,Real(3.0)) / Kokkos::pow(Real(C_CGS),Real(2.0));
#endif
    B_nu = B_nu * (Real(1.0) / (Kokkos::exp(Real(H_PLANCK) * nu / (Real(KB) * T)) - Real(1.0)));

    return B_nu;
}

KOKKOS_INLINE_FUNCTION
Real spectrum_nu(Real T, Real nu,
                   const auto& sed_frequency,
                   const auto& sed_fnu) {
    const int n = static_cast<int>(sed_frequency.extent(0));
    if (n == 0)
        return blackbody_nu(T, nu);

    int low = 0;
    int high = n - 1;
    while (high - low > 1) {
        const int mid = (low + high) / 2;
        if (Real(sed_frequency(mid)) <= nu)
            low = mid;
        else
            high = mid;
    }

    const Real t = (nu - Real(sed_frequency(low))) /
                     (Real(sed_frequency(high)) - Real(sed_frequency(low)));
    return Real(sed_fnu(low)) + t * (Real(sed_fnu(high)) - Real(sed_fnu(low)));
}

KOKKOS_INLINE_FUNCTION
Real simpson_weight(int i) {
    if (i == 0 || i == n_int_points) {
        return Real(1.0);
    }

    return (i % 2 == 0) ? Real(2.0) : Real(4.0);
}

KOKKOS_INLINE_FUNCTION
void integration_bounds(Real E0, Real E1, Real &a, Real &b, Real &h) {
    a = (E0 * Real(EV_2_ERG)) / Real(H_PLANCK);
    b = (E1 * Real(EV_2_ERG)) / Real(H_PLANCK);
    h = (b - a) / n_int_points;
}

KOKKOS_INLINE_FUNCTION
void compute_spectrum_bin_integrals(Real E0, Real E1, Real T,
                                    const auto& sed_frequency,
                                    const auto& sed_fnu,
                                    Real &number_denominator,
                                    Real &energy_denominator,
                                    Real &group_energy) {
    Real a, b, h;
    integration_bounds(E0, E1, a, b, h);

    Real weighted_number_sum = Real(0.0);
    Real weighted_energy_sum = Real(0.0);

    for (int i = 0; i <= n_int_points; ++i) {
        const Real nu = a + (i * h);
        const Real weight = simpson_weight(i);
        const Real B_nu = spectrum_nu(T, nu, sed_frequency, sed_fnu);

        weighted_number_sum += weight * B_nu / (Real(H_PLANCK) * nu);
        weighted_energy_sum += weight * B_nu;
    }

    number_denominator = weighted_number_sum * h / Real(3.0);
    energy_denominator = weighted_energy_sum * h / Real(3.0);
    group_energy = (energy_denominator / number_denominator) / Real(EV_2_ERG);
}

KOKKOS_INLINE_FUNCTION
void compute_weighted_cross_sections(Real E0, Real E1, Real T,
                                     int element, int ion,
                                     const auto& tabData,
                                     const auto& sed_frequency,
                                     const auto& sed_fnu,
                                     Real number_denominator,
                                     Real energy_denominator,
                                     Real &sigma_number,
                                     Real &sigma_energy) {
    Real a, b, h;
    integration_bounds(E0, E1, a, b, h);

    Real weighted_number_sum = Real(0.0);
    Real weighted_energy_sum = Real(0.0);

    for (int i = 0; i <= n_int_points; ++i) {
        const Real nu = a + (i * h);
        const Real weight = simpson_weight(i);
        const Real B_nu = spectrum_nu(T, nu, sed_frequency, sed_fnu);
        const Real sigma = getCrosssection_rtz((Real(C_CGS) / nu) * Real(1e8), element, ion, tabData);

        weighted_number_sum += weight * sigma * B_nu / (Real(H_PLANCK) * nu);
        weighted_energy_sum += weight * sigma * B_nu;
    }

    sigma_number = (weighted_number_sum * h / Real(3.0)) / number_denominator;
    sigma_energy = (weighted_energy_sum * h / Real(3.0)) / energy_denominator;
}

KOKKOS_INLINE_FUNCTION
Real compute_weighted_dust_cross_section(Real E0, Real E1, Real T,
                                           const auto& tabData,
                                           const auto& sed_frequency,
                                           const auto& sed_fnu,
                                           Real number_denominator) {
    Real a, b, h;
    integration_bounds(E0, E1, a, b, h);

    Real weighted_number_sum = Real(0.0);
    for (int i = 0; i <= n_int_points; ++i) {
        const Real nu = a + (i * h);
        const Real weight = simpson_weight(i);
        const Real B_nu = spectrum_nu(T, nu, sed_frequency, sed_fnu);
        const Real sigma = getCrosssection_dust((Real(C_CGS) / nu) * Real(1e8), tabData);

        weighted_number_sum += weight * sigma * B_nu / (Real(H_PLANCK) * nu);
    }

    return (weighted_number_sum * h / Real(3.0)) / number_denominator;
}

KOKKOS_INLINE_FUNCTION
Real sigma_N(Real E0, Real E1, Real T, int element, int ion, const auto& tabData,
               const auto& sed_frequency,
               const auto& sed_fnu){
    // Photon number-weighted cross section
    // E0 lower energy of the bin in eV
    // E1 upper energy of the bin in eV

    Real number_denominator, energy_denominator, group_energy;
    Real sigma_number, sigma_energy;

    compute_spectrum_bin_integrals(E0, E1, T, sed_frequency, sed_fnu,
                                   number_denominator, energy_denominator, group_energy);
    compute_weighted_cross_sections(E0, E1, T, element, ion, tabData,
                                    sed_frequency, sed_fnu,
                                    number_denominator, energy_denominator,
                                    sigma_number, sigma_energy);

    return sigma_number;
}

KOKKOS_INLINE_FUNCTION
Real sigma_E(Real E0, Real E1, Real T, int element, int ion, const auto& tabData,
               const auto& sed_frequency,
               const auto& sed_fnu){
    // Photon energy-weighted cross section
    // E0 lower energy of the bin in eV
    // E1 upper energy of the bin in eV

    Real number_denominator, energy_denominator, group_energy;
    Real sigma_number, sigma_energy;

    compute_spectrum_bin_integrals(E0, E1, T, sed_frequency, sed_fnu,
                                   number_denominator, energy_denominator, group_energy);
    compute_weighted_cross_sections(E0, E1, T, element, ion, tabData,
                                    sed_frequency, sed_fnu,
                                    number_denominator, energy_denominator,
                                    sigma_number, sigma_energy);

    return sigma_energy;
}

KOKKOS_INLINE_FUNCTION
Real e_bar(Real E0, Real E1, Real T,
             const auto& sed_frequency,
             const auto& sed_fnu){
    // Mean energy in the bin
    // E0 lower energy of the bin in eV
    // E1 upper energy of the bin in eV

    Real number_denominator, energy_denominator, group_energy;
    compute_spectrum_bin_integrals(E0, E1, T, sed_frequency, sed_fnu,
                                   number_denominator, energy_denominator, group_energy);

    return group_energy;
}

// TODO(code): Declate the 4D array and then copy to the tabdata
template<class Data>
void update_cross_sections(Real T, Data &tabData,
                           const std::vector<double>& sed_frequency,
                           const std::vector<double>& sed_fnu) {
    using Space = typename Data::memory_space;
    using Execution = typename Data::execution_space;
    // Updates the cross sections for each element and group

    auto &cs_ph = tabData.cs_ph;
    const auto &verner_cross_sections = tabData.photoionization_cross_sections_tot;
    const auto &group_E_min = tabData.group_E_min;
    const auto &group_E_max = tabData.group_E_max;
    const auto &c_frac = tabData.c_frac;
    Kokkos::View<Real[N_GROUPS][3], Space> group_bin_data {"group_bin_data"};
    Kokkos::View<Real*, Space> sed_frequency_d;
    Kokkos::View<Real*, Space> sed_fnu_d;
    if (!sed_frequency.empty()) {
        sed_frequency_d = Kokkos::View<Real*, Space>("sed_frequency", sed_frequency.size());
        sed_fnu_d = Kokkos::View<Real*, Space>("sed_fnu", sed_fnu.size());
        auto sed_frequency_h = Kokkos::create_mirror_view(sed_frequency_d);
        auto sed_fnu_h = Kokkos::create_mirror_view(sed_fnu_d);
        for (std::size_t i = 0; i < sed_frequency.size(); ++i) {
            sed_frequency_h(i) = sed_frequency[i];
            sed_fnu_h(i) = sed_fnu[i];
        }
        Kokkos::deep_copy(sed_frequency_d, sed_frequency_h);
        Kokkos::deep_copy(sed_fnu_d, sed_fnu_h);
    }

    Kokkos::deep_copy(cs_ph, Real(0.0));

    Kokkos::parallel_for("prepare_cross_section_groups", Kokkos::RangePolicy<Execution>(0, N_GROUPS), KOKKOS_LAMBDA(const int i) {
        Real number_denominator = Real(0.0);
        Real energy_denominator = Real(0.0);
        Real group_energy = Real(0.0);

        compute_spectrum_bin_integrals(group_E_min(i), group_E_max(i), T,
                                       sed_frequency_d, sed_fnu_d,
                                       number_denominator, energy_denominator, group_energy);

        group_bin_data(i, 0) = number_denominator;
        group_bin_data(i, 1) = energy_denominator;
        group_bin_data(i, 2) = group_energy;

        // Persist the mean photon energy [eV] of this group for the G0 update
        tabData.group_energy(i) = group_energy;

        // Blackbody-averaged dust absorption rate constant [cm^3/s]
        Real sigma_dust = Real(0.0);
        if (number_denominator > Real(0.0)) {
            sigma_dust = compute_weighted_dust_cross_section(
                group_E_min(i), group_E_max(i), T, tabData, sed_frequency_d, sed_fnu_d,
                number_denominator);
        }
        tabData.cs_dust(i) = sigma_dust * Real(C_CGS) * Real(c_frac());
    });

    auto grp_pair_policy = Kokkos::MDRangePolicy<Execution, Kokkos::Rank<2>>({0, 0}, {N_GROUPS, n_photo_pairs});
    Kokkos::parallel_for("update_cross_sections", grp_pair_policy, KOKKOS_LAMBDA(const int i, const int pair_index) {
        const Real number_denominator = group_bin_data(i, 0);
        const Real energy_denominator = group_bin_data(i, 1);
        const Real group_energy = group_bin_data(i, 2);
        const int j = (int)((Real(1.0) + Kokkos::sqrt(Real(1.0) + Real(8.0) * pair_index)) / Real(2.0));
        const int k = pair_index - j * (j - 1) / 2;
        const int erow = elem_row(j);
        if (erow < 0) return;

        Real sigma_number = Real(0.0);
        Real sigma_energy = Real(0.0);

        compute_weighted_cross_sections(group_E_min(i), group_E_max(i), T, j, k+1, tabData,
                                        sed_frequency_d, sed_fnu_d,
                                        number_denominator, energy_denominator,
                                        sigma_number, sigma_energy);

        // Update number-weighted cross section
        const Real number_rate = sigma_number * Real(C_CGS) * Real(c_frac());
        const Real energy_rate = sigma_energy * Real(C_CGS) * Real(c_frac());
        cs_ph(erow, k, i, 0) = number_rate; // cm^3 / s

        // Update energy-weighted cross section
        cs_ph(erow, k, i, 1) = energy_rate; // cm^3 / s

        // Update photoheating rates
        cs_ph(erow, k, i, 2) = (energy_rate * group_energy) - (number_rate * Real(verner_cross_sections(j, k + 1).E_th));
        cs_ph(erow, k, i, 2) = Kokkos::fmax(Real(cs_ph(erow, k, i, 2)),Real(0.0)) * Real(EV_2_ERG); // Don't let photoheating go below 0.0
    });

    Kokkos::parallel_for("update_cross_sections_h2", Kokkos::RangePolicy<Execution>(0, N_GROUPS), KOKKOS_LAMBDA(const int i) {
        const Real number_denominator = group_bin_data(i, 0);
        const Real energy_denominator = group_bin_data(i, 1);
        const Real group_energy = group_bin_data(i, 2);

        Real sigma_number = Real(0.0);
        Real sigma_energy = Real(0.0);

        compute_weighted_cross_sections(group_E_min(i), group_E_max(i), T, 1, 2+1, tabData,
                                        sed_frequency_d, sed_fnu_d,
                                        number_denominator, energy_denominator,
                                        sigma_number, sigma_energy);

        constexpr int erow_H = elem_row(1);
        const Real number_rate = sigma_number * Real(C_CGS) * Real(c_frac());
        const Real energy_rate = sigma_energy * Real(C_CGS) * Real(c_frac());
        cs_ph(erow_H, 2, i, 0) = number_rate; // cm^3 / s
        cs_ph(erow_H, 2, i, 1) = energy_rate; // cm^3 / s
        cs_ph(erow_H, 2, i, 2) = energy_rate * group_energy;
        cs_ph(erow_H, 2, i, 2) = Kokkos::fmax(Real(cs_ph(erow_H, 2, i, 2)),Real(0.0)) * Real(EV_2_ERG); // Don't let photoheating go below 0.0
    });
}