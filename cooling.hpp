// cooling.h
#pragma once
#include <array>
#include <Kokkos_Core.hpp>

#include "constants.hpp"
#include "types.hpp"
#include "molecules.hpp"

// nvcc never constant-folds pow(), even for a literal integer exponent and a
// compile-time-constant base: every call emits __internal_accurate_pow (a ~194
// instruction subroutine). Writing the small integer powers as multiplications
// removes the call and, where the base is built from literal species constants,
// lets the whole expression fold to a literal at compile time.
#define SQR(x)  ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))


// Constants
const Real MIN_METAL_COOL_DENS = Real(1E-10); //minimum number density to calc cooling
const Real MIN_COOL_ION = Real(1.E-10); //minimum number density to calculate cooling for a particular ion

// Initialize the relevant arrays
const int N_HIGH_T_COOLING_TEMP = 121;
double high_t_cooling_temp[N_HIGH_T_COOLING_TEMP];
double high_t_cooling_rates[N_HIGH_T_COOLING_TEMP][27][27];
bool high_t_cooling_rates_tflag[27][27];
double fs_cool_tab[27][160][8];  // Array for fine structure cooling rates


double G0_heating_rates[27] = {
    0.0, // 0
    0.0, // 1 - Hydrogen
    0.0, // 2 - Helium
    0.0, // 3
    0.0, // 4
    0.0, // 5
    4.6185e-22, // 6 - Carbon
    0.0, // 7 - Nitrogen
    0.0, // 8 - Oxygen
    0.0, // 9
    0.0, // 10 - Neon
    9.0362e-23, // 11 - Sodium
    1.2635e-22, // 12 - Magnesium
    1.0445e-20, // 13 - Aluminum
    9.8347e-21, // 14 - Silicon
    0.0, // 15
    2.1308e-21, // 16 - Sulfur
    0.0, // 17 - Chlorine
    0.0, // 18 - Argon
    0.0, // 19
    7.6333e-22, // 20 - Calcium
    0.0, // 21
    0.0, // 22
    0.0, // 23
    0.0, // 24
    0.0, // 25
    1.4517e-21 // 26 - Iron
};


KOKKOS_INLINE_FUNCTION
Real collisional_ionization_cooling_HI(Real T){
    // From Cen 1992
    Real term_1 = Real(1.27E-21) * Kokkos::sqrt(T);
    Real term_2 = Real(1.0) /(Real(1.0) + Kokkos::sqrt(T/Real(1E5)));
    Real term_3 = Kokkos::exp(-Real(157809.1)/T);
    Real cooling_rate = term_1 * term_2 * term_3;
    return Kokkos::fmax(cooling_rate, Real(1e-100));
}

KOKKOS_INLINE_FUNCTION
Real collisional_ionization_cooling_HeI(Real T){
    // From Cen 1992
    Real term_1 = Real(9.38E-22) * Kokkos::sqrt(T);
    Real term_2 = Real(1.0) /(Real(1.0) + Kokkos::sqrt(T/Real(1E5)));
    Real term_3 = Kokkos::exp(-Real(285335.4)/T);
    Real cooling_rate = term_1 * term_2 * term_3;
    return Kokkos::fmax(cooling_rate, Real(1e-100));
}

KOKKOS_INLINE_FUNCTION
Real collisional_ionization_cooling_HeII(Real T){
    // From Cen 1992
    Real term_1 = Real(4.95E-22) * Kokkos::sqrt(T);
    Real term_2 = Real(1.0) /(Real(1.0) + Kokkos::sqrt(T/Real(1E5)));
    Real term_3 = Kokkos::exp(-Real(631515.0)/T);
    Real cooling_rate = term_1 * term_2 * term_3;
    return Kokkos::fmax(cooling_rate, Real(1e-100));
}

KOKKOS_INLINE_FUNCTION
Real recombination_cooling_case_B_HII(Real T){
    // From Hui & Gnedin 1997
    Real lam_HI = Real(315614.0) / T;
    Real term_1 = Real(3.435E-30) * T;
    Real term_2 = Kokkos::pow(lam_HI,Real(1.97));
    Real term_3 = Kokkos::pow(Real(1.0) + Kokkos::pow(lam_HI/Real(2.25),Real(0.376)),-Real(3.72));
    Real cooling_rate = term_1 * term_2 * term_3;
    return Kokkos::fmax(cooling_rate, Real(1e-100));
}

KOKKOS_INLINE_FUNCTION
Real recombination_cooling_case_B_HeII(Real T){
    // From Hui & Gnedin 1997
    Real lam_HeI = Real(570670.0) / T;
    Real term_1 = Real(KB) * T * Real(1.26E-14);
    Real term_2 = Kokkos::pow(lam_HeI,Real(0.75));
    Real cooling_rate = term_1 * term_2;
    return Kokkos::fmax(cooling_rate, Real(1e-100));
}

KOKKOS_INLINE_FUNCTION
Real recombination_cooling_case_B_HeIII(Real T){
    // From Hui & Gnedin 1997
    Real lam_HeII = Real(1263030.0) / T;
    Real term_1 = Real(8.0) * Real(3.435E-30) * T;
    Real term_2 = Kokkos::pow(lam_HeII,Real(1.97));
    Real term_3 = Kokkos::pow(Real(1.0) + Kokkos::pow(lam_HeII/Real(2.25),Real(0.376)),-Real(3.72));
    Real cooling_rate = term_1 * term_2 * term_3;
    return Kokkos::fmax(cooling_rate, Real(1e-100));
}


KOKKOS_INLINE_FUNCTION
Real dielectronic_recombination_cooling_HeII(Real T){
    // From Black 1981
    Real term_1 = Real(1.24E-13) * Kokkos::pow(T,-Real(1.5));
    Real term_2 = Kokkos::exp(-Real(470000.0)/T);
    Real term_3 = Real(1.0) + (Real(0.3) * Kokkos::exp(-Real(94000.0)/T));
    Real cooling_rate = term_1 * term_2 * term_3;
    return Kokkos::fmax(cooling_rate, Real(1e-100));
}


KOKKOS_INLINE_FUNCTION
Real collisional_excitation_cooling_HI(Real T){
    // From Seon 2020
    Real cooling_rate = Real(4.13e-19) * Kokkos::exp(-Real(117744.0)/T);
    return Kokkos::fmax(cooling_rate, Real(1e-100));
}

// double collisional_excitation_cooling_HI(double T){
//     // From Cen 1992
//     double term_1 = 7.5e-19;
//     double term_2 = 1.0 / (1.0 + sqrt(T/1e5));
//     double term_3 = exp(-118348.0/T);
//     double cooling_rate = term_1 * term_2 * term_3;
//     return fmax(cooling_rate,1e-100);
// }

KOKKOS_INLINE_FUNCTION
Real collisional_excitation_cooling_HeII(Real T){
    // From Cen 1992
    Real term_1 = Real(5.54E-17) * Kokkos::pow(T,-Real(0.397));
    Real term_2 = Real(1.0) /(Real(1.0) + Kokkos::sqrt(T/Real(1E5)));
    Real term_3 = Kokkos::exp(-Real(473638.0)/T);
    Real cooling_rate = term_1 * term_2 * term_3;
    return Kokkos::fmax(cooling_rate, Real(1e-100));
}

KOKKOS_INLINE_FUNCTION
Real bremmstrahlung(Real T){
    // Osterbrock and Ferland 2006
    Real cooling_rate = Real(1.42E-27) * Kokkos::sqrt(T);
    return Kokkos::fmax(cooling_rate, Real(1e-100));
}

KOKKOS_INLINE_FUNCTION
Real compton_cooling(Real T, Real a){

    Real Tcmb_over_a = Real(2.727)/a;

    // Haiman+ 1996
    Real term_1 = Real(1.017E-37);
    Real term_2 = SQR(SQR(Tcmb_over_a));
    Real term_3 = T - Tcmb_over_a;
    Real cooling_rate = term_1 * term_2 * term_3;
    return cooling_rate;
}

KOKKOS_INLINE_FUNCTION
Real H2_cooling(Real nH, Real nH2, Real T){
    // From Moseley 2021
    Real T3 = Real(1E-3) * T;
    Real n1 = Real(50.0);
    Real n2 = Real(450.0);
    Real n3 = Real(25.0);
    Real n4 = Real(900.0);

    Real x1 = nH + (Real(5.0) * nH2);
    Real x2 = nH + (Real(4.5) * nH2);
    Real x3 = nH + (Real(0.75) * nH2);
    Real x4 = nH + (Real(0.05) * nH2);

    Real f1 = Real(1.1E-25) * Kokkos::sqrt(T3) * Kokkos::exp(-Real(0.51) / T3);
    f1 = f1 * (((Real(0.7) * x1) / (Real(1.0) + (x1/n1))) + ((Real(0.3) * x1)/(Real(1.0) + (x1/(Real(10.0)*n1)))));

    Real f2 = Real(2.0E-25) * T3 * Kokkos::exp(-Real(1.0) / T3);
    f2 = f2 * (((Real(0.35) * x2) / (Real(1.0) + (x2/n2))) + ((Real(0.65) * x2)/(Real(1.0) + (x2/(Real(10.0)*n2)))));

    Real f3 = Real(2.4E-24) * (Kokkos::pow(T3,Real(1.50))) * Kokkos::exp(-Real(2.0) / T3);
    f3 = f3 * (x3 / (Real(1.0) + (x3/n3)));

    Real f4 = Real(1.7E-23) * (Kokkos::pow(T3,Real(1.50))) * Kokkos::exp(-Real(4.0) / T3);
    f4 = f4 * (((Real(0.45) * x4) / (Real(1.0) + (x4/n4))) + ((Real(0.55) * x4)/(Real(1.0) + (x4/(Real(10.0)*n4)))));

    return Kokkos::fmax(nH2 * (f1 + f2 + f3 + f4), Real(1e-100));
}

void initialize_high_temperature_metal_cooling(std::string path_to_data){
    // Cloudy tables of metal line cooling
    // which are valid at high temperature
    FILE *file;

    printf("Initializing high temperature cooling tables\n");

    // Temperatures
    file = fopen((path_to_data + "/high_T_cooling/temperatures.dat").c_str(), "r");
    if (file == NULL) {
        printf("Error: Could not open high temperature cooling temperatures file\n");
    }

    // Reading data from the file into the array
    for (int i = 0; i < N_HIGH_T_COOLING_TEMP; i++) {
        fscanf(file, "%lf", &high_t_cooling_temp[i]);
    }
    // Close the file
    fclose(file);

    // Initialize high-temeprature cooling table to 0
    // We use -50 because it's log
    for (int i = 0; i < N_HIGH_T_COOLING_TEMP; i++) {
        for (int j = 0; j < 27; j++) {
            for (int k = 0; k < 27; k++) {
                high_t_cooling_rates[i][j][k] = -50.0;
            }
        }
    }

    // Carbon
    file = fopen((path_to_data + "/high_T_cooling/CARBON/all_cool.dat").c_str(), "r");
    if (file == NULL) {
        printf("Error: Could not open high temperature cooling carbon rates\n");
    }

    // Reading data from the file into the 2D array
    for (int j = 0; j < 7; j++) {
        for (int i = 0; i < N_HIGH_T_COOLING_TEMP; i++) {
            fscanf(file, "%lf", &high_t_cooling_rates[i][6][j]);
        }
    }
    // Close the file
    fclose(file);

    // Nitrogen
    file = fopen((path_to_data + "/high_T_cooling/NITROGEN/all_cool.dat").c_str(), "r");
    if (file == NULL) {
        printf("Error: Could not open high temperature cooling nitrogen rates\n");
    }

    // Reading data from the file into the 2D array
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < N_HIGH_T_COOLING_TEMP; i++) {
            fscanf(file, "%lf", &high_t_cooling_rates[i][7][j]);
        }
    }
    // Close the file
    fclose(file);

    // Oxygen
    file = fopen((path_to_data + "/high_T_cooling/OXYGEN/all_cool.dat").c_str(), "r");
    if (file == NULL) {
        printf("Error: Could not open high temperature cooling oxygen rates\n");
    }

    // Reading data from the file into the 2D array
    for (int j = 0; j < 9; j++) {
        for (int i = 0; i < N_HIGH_T_COOLING_TEMP; i++) {
            fscanf(file, "%lf", &high_t_cooling_rates[i][8][j]);
        }
    }
    // Close the file
    fclose(file);

    // Neon
    file = fopen((path_to_data + "/high_T_cooling/NEON/all_cool.dat").c_str(), "r");
    if (file == NULL) {
        printf("Error: Could not open high temperature cooling neon rates\n");
    }

    // Reading data from the file into the 2D array
    for (int j = 0; j < 11; j++) {
        for (int i = 0; i < N_HIGH_T_COOLING_TEMP; i++) {
            fscanf(file, "%lf", &high_t_cooling_rates[i][10][j]);
        }
    }
    // Close the file
    fclose(file);

    // Magnesium
    file = fopen((path_to_data + "/high_T_cooling/MAGNESIUM/all_cool.dat").c_str(), "r");
    if (file == NULL) {
        printf("Error: Could not open high temperature cooling magnesium rates\n");
    }

    // Reading data from the file into the 2D array
    for (int j = 0; j < 13; j++) {
        for (int i = 0; i < N_HIGH_T_COOLING_TEMP; i++) {
            fscanf(file, "%lf", &high_t_cooling_rates[i][12][j]);
        }
    }
    // Close the file
    fclose(file);

    // Silicon
    file = fopen((path_to_data + "/high_T_cooling/SILICON/all_cool.dat").c_str(), "r");
    if (file == NULL) {
        printf("Error: Could not open high temperature cooling silicon rates\n");
    }

    // Reading data from the file into the 2D array
    for (int j = 0; j < 15; j++) {
        for (int i = 0; i < N_HIGH_T_COOLING_TEMP; i++) {
            fscanf(file, "%lf", &high_t_cooling_rates[i][14][j]);
        }
    }
    // Close the file
    fclose(file);

    // Sulfur
    file = fopen((path_to_data + "/high_T_cooling/SULPHUR/all_cool.dat").c_str(), "r");
    if (file == NULL) {
        printf("Error: Could not open high temperature cooling sulphur rates\n");
    }

    // Reading data from the file into the 2D array
    for (int j = 0; j < 17; j++) {
        for (int i = 0; i < N_HIGH_T_COOLING_TEMP; i++) {
            fscanf(file, "%lf", &high_t_cooling_rates[i][16][j]);
        }
    }
    // Close the file
    fclose(file);

    // IRON
    file = fopen((path_to_data + "/high_T_cooling/IRON/all_cool.dat").c_str(), "r");
    if (file == NULL) {
        printf("Error: Could not open high temperature cooling iron rates\n");
    }

    // Reading data from the file into the 2D array
    for (int j = 0; j < 27; j++) {
        for (int i = 0; i < N_HIGH_T_COOLING_TEMP; i++) {
            fscanf(file, "%lf", &high_t_cooling_rates[i][26][j]);
        }
    }
    // Close the file
    fclose(file);

    // Initialize the Tflag
    // this tells us which lines we also compute fine structure cooling for
    // and we need to smoothly interpolate between the temperature regimes
    for (int j = 0; j < 27; j++) {
        for (int i = 0; i < 27; i++) {
            // First initialize everything to False
            high_t_cooling_rates_tflag[j][i] = false;
        }
    }

    // Set all of the fine structure lines to true
    // prob a better way of doing this but fine for now

    // CI
    high_t_cooling_rates_tflag[6][0] = true;
    // CII
    high_t_cooling_rates_tflag[6][1] = true;
    // NII
    high_t_cooling_rates_tflag[7][1] = true;
    // OI
    high_t_cooling_rates_tflag[8][0] = true;
    // OIII
    high_t_cooling_rates_tflag[8][2] = true;
    // NeII
    high_t_cooling_rates_tflag[10][1] = true;
    // SiI
    high_t_cooling_rates_tflag[14][0] = true;
    // SiII
    high_t_cooling_rates_tflag[14][1] = true;
    // SiI
    high_t_cooling_rates_tflag[16][0] = true;
    // FeI
    high_t_cooling_rates_tflag[26][0] = true;
    // FeII
    high_t_cooling_rates_tflag[26][1] = true;

}

template<typename Table>
KOKKOS_FUNCTION
Real get_high_t_cooling_rates(Real T, Real ne,
                                const CompactIonData& n_and_ion_fracs,
                                const Element* elements,
                                Real temp_smooth,
                                const Table& tabData){
    // Computes the high-temperature cooling rates
    // derived from Harley's custom cloudy models

    Real loc_T = Kokkos::fmax(T,Real(1000.0));
    Real log_T = Kokkos::log10(loc_T);
    Real t_min = Real(3.0);
    Real t_max = Real(9.0);
    Real dt = Real(0.05);
    Real total_metal_cooling_rate = Real(0.0);
    Real two_e4_overT = Real(2000.0)/loc_T;
    Real t_scale_fac = Kokkos::exp(-Real(1.0) * Kokkos::pow(two_e4_overT, 5));

    // bounds for temperature --> no cooling
    if (log_T < t_min){
        return total_metal_cooling_rate;
    }

    // Still cool above 10^9 K but set a bound
    log_T = Kokkos::fmin(log_T, t_max);

    // Prepare for 1D interpolation
    int idx_low = (int)Kokkos::floor((log_T - t_min)/dt);

	// Uniform grid: high_t_cooling_temp[k] = t_min + k * dt
    Real frac_high = (log_T - (t_min + idx_low * dt)) / dt;
    Real frac_low = Real(1.0) - frac_high;

    // Loop over the tabulated elements -- not including H and He (rows 0 and 1)
    for (int erow = 2; erow < N_TAB_ELEMENTS; erow++) {
        const int i = row_elem(erow);
        if (Real(n_and_ion_fracs.n_element_new[i]) > MIN_METAL_COOL_DENS){
            // Get the element number density
            Real n_element = Real(n_and_ion_fracs.n_element_new[i]);
            for (int j = 0; j < (elements[i].n_ions + elements[i].n_mol); j++) {
                // Get the element ion fraction
                Real n_ion = Real(n_and_ion_fracs[i].ion_fracs_new[j]);

                // Interpolate the cooling rate for the ion
                Real loc_cooling_rate = (frac_low * Real(tabData.high_t_cooling_rates(idx_low, erow, j)));
                loc_cooling_rate += (frac_high * Real(tabData.high_t_cooling_rates(idx_low+1, erow, j)));
                loc_cooling_rate = Kokkos::pow(10, loc_cooling_rate);

                // Smooth with temperature if necessary
                Real loc_temp_smooth = Real(1.0);
                if (Real(tabData.high_t_cooling_rates_tflag(erow,j))) {
                    loc_temp_smooth = temp_smooth;
                }

                // Multiply cooling rate by ion and electron number densities
                total_metal_cooling_rate += (n_element * n_ion * ne * loc_cooling_rate * loc_temp_smooth);
            }
        }
    }

    return Kokkos::fmax(total_metal_cooling_rate * t_scale_fac,Real(1e-100));
}

void init_fine_structure_tables(std::string path_to_data){
    // Initialization for fine structure cooling tables
    printf("Initializing fine structure cooling tables\n");

    const int N_LINES = 27;
    const char* file_names[N_LINES] = {
        "CII_158um_rates.dat", "CI_609um_rates.dat", "CI_230um_rates.dat", "CI_370um_rates.dat",
        "NII_205um_rates.dat", "NII_76um_rates.dat", "NII_122um_rates.dat", "OI_63um_rates.dat",
        "OI_44um_rates.dat", "OI_145um_rates.dat", "NeII_13um_rates.dat", "SiII_35um_rates.dat",
        "OIII_88um_rates.dat", "OIII_33um_rates.dat", "OIII_52um_rates.dat", "FeI_14um_rates.dat",
        "FeI_24um_rates.dat", "FeI_35um_rates.dat", "FeII_26um_rates.dat", "FeII_15um_rates.dat",
        "FeII_35um_rates.dat", "SiI_130um_rates.dat", "SiI_45um_rates.dat", "SiI_68um_rates.dat",
        "SI_17um_rates.dat", "SI_25um_rates.dat", "SI_56um_rates.dat"
    };

    std::string fs_cooling_dir = path_to_data + "/fine_structure_data/";

    FILE* file;
    bool ok;

    // Check that all files exist and concatenate directory and filename
    for (int i = 0; i < N_LINES; i++) {
        std::string file_path = fs_cooling_dir + file_names[i];
        file = fopen(file_path.c_str(), "r");
        ok = (file != NULL);
        if (!ok) {
            printf("Cannot access fine structure cooling file %s\n", file_path.c_str());
        }
        fclose(file);
    }

    // Read in the fine structure data
    for (int j = 0; j < N_LINES; j++) {
        std::string file_path = fs_cooling_dir + file_names[j];
        file = fopen(file_path.c_str(), "r");
        if (!file) {
            printf("Failed to open file %s\n", file_path.c_str());
        }

        // Loop over temperatures
        for (int i = 0; i < 160; i++) {
            for (int k = 0; k < 8; k++) {
                if (fscanf(file, "%lf", &fs_cool_tab[j][i][k]) != 1) {
                    printf("Error reading data from file %s\n", file_path.c_str());
                    fclose(file);
                }
            }
        }

        fclose(file);
    }
}

// Form the Planck prefactor without cubing a frequency (~1e13 Hz):
// nu^3 can overflow float before multiplication by Planck's constant.
KOKKOS_INLINE_FUNCTION
Real line_planck_spectrum(Real nu, Real temperature) {
#if defined(RTZ_FLOAT32_ARITHMETIC) || defined(RTZ_FLOAT32_COOLING)
    const Real nu_over_c = nu / Real(C_CGS);
    const Real prefactor = (Real(2) * Real(H_PLANCK) * nu) * nu_over_c * nu_over_c;
#else
    const Real prefactor = Real(2) * Real(H_PLANCK) * CUBE(nu) / (Real(C_CGS) * Real(C_CGS));
#endif
    return prefactor / (Kokkos::exp(Real(H_PLANCK) * nu / (Real(KB) * temperature)) - Real(1));
}

template<typename Table>
KOKKOS_FUNCTION
Real three_level(Real g_0, Real g_1, Real g_2,
                   Real lam_10, Real lam_20, Real lam_21,
                   Real A_10, Real A_20, Real A_21,
                   Real z, Real T, Real n_ion,
                   Real ne, Real nH, Real nHp,
                   Real nHe, Real nHep, Real nHepp,
                   Real nH2,
                   int idx_0, int idx_1, int idx_2,
                   const Table& tabData
                ){
    // Harley's three-level ion solver

    // ! Initialize the result
    Real three_level_cooling = Real(1E-50);

    Real T_cmb = Real(2.725) * (Real(1.0) + z); //! CMB temperature

    // ! Get log T and enforce bounds
    Real logT = Kokkos::log10(T);

    // Enforce bounds on temperature
    Real tmin = Real(1.0);
    Real tmax = Real(4.975);
    Real delta_temp = Real(0.025);
    logT = Kokkos::fmax(logT,Real(1.0));
    logT = Kokkos::fmin(logT,tmax);
    Real lam_10_minus4 = lam_10 * Real(1E-4);
    Real lam_20_minus4 = lam_20 * Real(1E-4);
    Real lam_21_minus4 = lam_21 * Real(1E-4);

    Real nu_10 = Real(C_CGS) / lam_10_minus4; //! Hz
    Real nu_20 = Real(C_CGS) / lam_20_minus4; //! Hz
    Real nu_21 = Real(C_CGS) / lam_21_minus4; //! Hz

    Real E_10 = (Real(H_PLANCK) * Real(C_CGS) / lam_10_minus4) / Real(KB); //! E/K (K)
    Real E_20 = (Real(H_PLANCK) * Real(C_CGS) / lam_20_minus4) / Real(KB); //! E/K (K)
    Real E_21 = (Real(H_PLANCK) * Real(C_CGS) / lam_21_minus4) / Real(KB); //! E/K (K)

    Real B_01 = A_10 * (CUBE(lam_10_minus4)) * (g_1 / g_0) / (Real(2.0) * Real(H_PLANCK) * Real(C_CGS));
    Real B_02 = A_20 * (CUBE(lam_20_minus4)) * (g_2 / g_0) / (Real(2.0) * Real(H_PLANCK) * Real(C_CGS));
    Real B_12 = A_21 * (CUBE(lam_21_minus4)) * (g_2 / g_1) / (Real(2.0) * Real(H_PLANCK) * Real(C_CGS));

    Real B_10 = (g_0 / g_1) * B_01;
    Real B_20 = (g_0 / g_2) * B_02;
    Real B_21 = (g_1 / g_2) * B_12;

    //! CMB black body spectrum
    Real B_nu_10 = line_planck_spectrum(nu_10, T_cmb);
    Real B_nu_20 = line_planck_spectrum(nu_20, T_cmb);
    Real B_nu_21 = line_planck_spectrum(nu_21, T_cmb);

    // ! Find temperature index.
    int itemp_low = (int)Kokkos::floor((logT - tmin)/delta_temp);
    int itemp_high = itemp_low + 1;
    Real frac_high = (logT - (tmin + ((Real)itemp_low * delta_temp))) / delta_temp;
    Real frac_low = Real(1.0) - frac_high;

    // ! All collison strengths are in cm^3 s^-1
    // ! For collisions with o-H2
    Real q10_oH2 = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 6)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 6)) * frac_high);
    Real q20_oH2 = (Real(tabData.fs_cool_tab(idx_1, itemp_low, 6)) * frac_low) + (Real(tabData.fs_cool_tab(idx_1, itemp_high, 6)) * frac_high);
    Real q21_oH2 = (Real(tabData.fs_cool_tab(idx_2, itemp_low, 6)) * frac_low) + (Real(tabData.fs_cool_tab(idx_2, itemp_high, 6)) * frac_high);

    // ! For collisions with p-H2
    Real q10_pH2 = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 7)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 7)) * frac_high);
    Real q20_pH2 = (Real(tabData.fs_cool_tab(idx_1, itemp_low, 7)) * frac_low) + (Real(tabData.fs_cool_tab(idx_1, itemp_high, 7)) * frac_high);
    Real q21_pH2 = (Real(tabData.fs_cool_tab(idx_2, itemp_low, 7)) * frac_low) + (Real(tabData.fs_cool_tab(idx_2, itemp_high, 7)) * frac_high);

    // ! For collisions with H
    Real q10_H = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 4)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 4)) * frac_high);
    Real q20_H = (Real(tabData.fs_cool_tab(idx_1, itemp_low, 4)) * frac_low) + (Real(tabData.fs_cool_tab(idx_1, itemp_high, 4)) * frac_high);
    Real q21_H = (Real(tabData.fs_cool_tab(idx_2, itemp_low, 4)) * frac_low) + (Real(tabData.fs_cool_tab(idx_2, itemp_high, 4)) * frac_high);

    // ! For collisions with H+
    Real q10_Hp = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 1)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 1)) * frac_high);
    Real q20_Hp = (Real(tabData.fs_cool_tab(idx_1, itemp_low, 1)) * frac_low) + (Real(tabData.fs_cool_tab(idx_1, itemp_high, 1)) * frac_high);
    Real q21_Hp = (Real(tabData.fs_cool_tab(idx_2, itemp_low, 1)) * frac_low) + (Real(tabData.fs_cool_tab(idx_2, itemp_high, 1)) * frac_high);

    // ! For collisions with e
    Real q10_e = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 0)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 0)) * frac_high);
    Real q20_e = (Real(tabData.fs_cool_tab(idx_1, itemp_low, 0)) * frac_low) + (Real(tabData.fs_cool_tab(idx_1, itemp_high, 0)) * frac_high);
    Real q21_e = (Real(tabData.fs_cool_tab(idx_2, itemp_low, 0)) * frac_low) + (Real(tabData.fs_cool_tab(idx_2, itemp_high, 0)) * frac_high);

    // ! For collisions with He
    Real q10_He = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 5)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 5)) * frac_high);
    Real q20_He = (Real(tabData.fs_cool_tab(idx_1, itemp_low, 5)) * frac_low) + (Real(tabData.fs_cool_tab(idx_1, itemp_high, 5)) * frac_high);
    Real q21_He = (Real(tabData.fs_cool_tab(idx_2, itemp_low, 5)) * frac_low) + (Real(tabData.fs_cool_tab(idx_2, itemp_high, 5)) * frac_high);

    // ! For collisions with He+
    Real q10_Hep = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 2)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 2)) * frac_high);
    Real q20_Hep = (Real(tabData.fs_cool_tab(idx_1, itemp_low, 2)) * frac_low) + (Real(tabData.fs_cool_tab(idx_1, itemp_high, 2)) * frac_high);
    Real q21_Hep = (Real(tabData.fs_cool_tab(idx_2, itemp_low, 2)) * frac_low) + (Real(tabData.fs_cool_tab(idx_2, itemp_high, 2)) * frac_high);

    // ! For collisions with He++
    Real q10_Hepp = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 3)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 3)) * frac_high);
    Real q20_Hepp = (Real(tabData.fs_cool_tab(idx_1, itemp_low, 3)) * frac_low) + (Real(tabData.fs_cool_tab(idx_1, itemp_high, 3)) * frac_high);
    Real q21_Hepp = (Real(tabData.fs_cool_tab(idx_2, itemp_low, 3)) * frac_low) + (Real(tabData.fs_cool_tab(idx_2, itemp_high, 3)) * frac_high);

    // ! Net collision strengths note the 75-25 ortho-para H2
    Real C_10 = (q10_e * ne) + (q10_H * nH) + (q10_Hp * nHp) + (q10_oH2 * Real(0.75) * nH2) + (q10_pH2 * Real(0.25) * nH2) + (q10_He * nHe) + (q10_Hep * nHep) + (q10_Hepp * nHepp);
    Real C_20 = (q20_e * ne) + (q20_H * nH) + (q20_Hp * nHp) + (q20_oH2 * Real(0.75) * nH2) + (q20_pH2 * Real(0.25) * nH2) + (q20_He * nHe) + (q20_Hep * nHep) + (q20_Hepp * nHepp);
    Real C_21 = (q21_e * ne) + (q21_H * nH) + (q21_Hp * nHp) + (q21_oH2 * Real(0.75) * nH2) + (q21_pH2 * Real(0.25) * nH2) + (q21_He * nHe) + (q21_Hep * nHep) + (q21_Hepp * nHepp);

    Real C_01 = C_10 * (g_1/g_0) * Kokkos::exp(-Real(1.0) * E_10 / T);
    Real C_02 = C_20 * (g_2/g_0) * Kokkos::exp(-Real(1.0) * E_20 / T);
    Real C_12 = C_21 * (g_2/g_1) * Kokkos::exp(-Real(1.0) * E_21 / T);

    C_01 += (B_01*B_nu_10);
    C_02 += (B_02*B_nu_20);
    C_12 += (B_12*B_nu_21);

    C_10 += (B_10*B_nu_10);
    C_20 += (B_20*B_nu_20);
    C_21 += (B_21*B_nu_21);

    // ! Analytic solution to the 3 level system (see paul goldsmith papers)
    // ! Done this way to avoid numerical errors
    Real n2_n1_top = (C_12 * (C_01 + C_02)) + (C_02 * (A_10 + C_10));
    Real n2_n1_bot = ((A_21 + C_21 + C_20) * (C_01 + C_02)) - (C_20 * C_02);
    Real n2_n1 = n2_n1_top / n2_n1_bot;

    Real n1_n0_top = ((A_21 + C_21 + C_20) * (C_01 + C_02)) - (C_20*C_02);
    Real n1_n0_bot = ((A_21 + C_21 + C_20) * (A_10 + C_10)) + (C_20*C_12);
    Real n1_n0 = n1_n0_top / n1_n0_bot;

    Real n_0 = Real(1.0) / (Real(1.0) + n1_n0 + (n2_n1 * n1_n0));
    Real n_1 = Real(1.0) / (Real(1.0) + n2_n1 + (Real(1.0)/n1_n0));
    Real n_2 = Real(1.0) / (Real(1.0) + (Real(1.0)/n2_n1) + ((Real(1.0)/n2_n1)*(Real(1.0)/n1_n0)));

    // ! Cooling and heating rates
    Real cool_0 = (A_10 + (B_10 * B_nu_10)) * E_10 * Real(KB) * n_1 * n_ion;
    Real cool_1 = (A_20 + (B_20 * B_nu_20)) * E_20 * Real(KB) * n_2 * n_ion;
    Real cool_2 = (A_21 + (B_21 * B_nu_21)) * E_21 * Real(KB) * n_2 * n_ion;

    Real heat_0 = B_01 * B_nu_10 * E_10 * Real(KB) * n_0 * n_ion;
    Real heat_1 = B_02 * B_nu_20 * E_20 * Real(KB) * n_0 * n_ion;
    Real heat_2 = B_12 * B_nu_21 * E_21 * Real(KB) * n_1 * n_ion;

    // ! Total cooling rate
    three_level_cooling = (cool_0 + cool_1 + cool_2) - (heat_0 + heat_1 + heat_2);

    return three_level_cooling;
}

template<typename Table>
KOKKOS_FUNCTION
Real two_level(Real g_0, Real g_1, Real lam_10, Real A_10,
                Real z, Real T, Real n_ion, Real ne,
                Real nH, Real nHp, Real nHe, Real nHep,
                Real nHepp, Real nH2, int idx_0, const Table& tabData){
    // Harley's two-level ion solver

    // ! Initialize the result
    Real two_level_cooling = Real(1E-50);

    Real T_cmb = Real(2.725) * (Real(1.0) + z); //! CMB temperature

    // ! Get log T and enforce bounds
    Real logT = Kokkos::log10(T);

    // Enforce bounds on temperature
    Real tmin = Real(1.0);
    Real tmax = Real(4.975);
    Real delta_temp = Real(0.025);
    logT = Kokkos::fmax(logT,tmin);
    logT = Kokkos::fmin(logT,tmax);
    Real lam_10_minus4 = lam_10 * Real(1E-4);

    Real nu_10 = Real(C_CGS) / lam_10_minus4; //! Hz

    Real E_10 = (Real(H_PLANCK) * Real(C_CGS) / lam_10_minus4) / Real(KB); //! E/K (K)

    Real B_01 = A_10 * CUBE(lam_10_minus4) * (g_1 / g_0) / (Real(2.0) * Real(H_PLANCK) * Real(C_CGS));

    Real B_10 = (g_0 / g_1) * B_01;

    //! CMB black body spectrum
    Real B_nu_10 = line_planck_spectrum(nu_10, T_cmb);

    // ! Find temperature index.
    int itemp_low = (int)Kokkos::floor((logT - tmin)/delta_temp);
    int itemp_high = itemp_low + 1;
    Real frac_high = (logT - (tmin + ((Real)itemp_low * delta_temp))) / delta_temp;
    Real frac_low = Real(1.0) - frac_high;

    // ! All collison strengths are in cm^3 s^-1
    // ! For collisions with o-H2
    Real q10_oH2 = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 6)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 6)) * frac_high);

    // ! For collisions with p-H2
    Real q10_pH2 = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 7)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 7)) * frac_high);

    // ! For collisions with H
    Real q10_H = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 4)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 4)) * frac_high);

    // ! For collisions with H+
    Real q10_Hp = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 1)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 1)) * frac_high);

    // ! For collisions with e
    Real q10_e = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 0)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 0)) * frac_high);

    // ! For collisions with He
    Real q10_He = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 5)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 5)) * frac_high);

    // ! For collisions with He+
    Real q10_Hep = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 2)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 2)) * frac_high);

    // ! For collisions with He++
    Real q10_Hepp = (Real(tabData.fs_cool_tab(idx_0, itemp_low, 3)) * frac_low) + (Real(tabData.fs_cool_tab(idx_0, itemp_high, 3)) * frac_high);

    // ! Net collision strengths note the 75-25 ortho-para H2
    Real C_10 = (q10_e * ne) + (q10_H * nH) + (q10_Hp * nHp) + (q10_oH2 * Real(0.75) * nH2) + (q10_pH2 * Real(0.25) * nH2) + (q10_He * nHe) + (q10_Hep * nHep) + (q10_Hepp * nHepp);

    Real C_01 = C_10 * (g_1/g_0) * Kokkos::exp(-Real(1.0) * E_10 / T);

    // ! Analytic solution to the 2 level system (modified version of paul goldsmith papers)
    // ! Done this way to avoid numerical errors
    Real t1 = ( (B_01*B_nu_10) + C_01 );
    Real t2 = ( A_10 + (B_10*B_nu_10) + C_10 );

    Real nu_over_nl = t1 / t2;
    Real n_0 = Real(1.0) / (nu_over_nl + Real(1.0));
    Real n_1 = Real(1.0) / ((Real(1.0)/nu_over_nl) + Real(1.0));

    // ! Cooling and heating rates
    Real cool_0 = (A_10 + (B_10 * B_nu_10)) * E_10 * Real(KB) * n_1 * n_ion;

    Real heat_0 = B_01 * B_nu_10 * E_10 * Real(KB) * n_0 * n_ion;

    // ! Total cooling rate
    two_level_cooling = (cool_0 - heat_0);

    return two_level_cooling;
}

/* OI fine structure calculation function */
template<typename Table>
KOKKOS_INLINE_FUNCTION
Real OI_fine_structure(Real T, Real n_ion, Real nH, Real nHp,
                         Real ne, Real nH2, Real nHe, Real nHep,
                         Real nHepp, Real z, const Table& tabData){
    /* Degeneracy variables */
    Real g_0 = Real(5.0);
    Real g_1 = Real(3.0);
    Real g_2 = Real(1.0);

    /* Transition rates (s^-1) */
    Real A_10 = Real(8.910E-5);
    Real A_20 = Real(1.340E-10);
    Real A_21 = Real(1.750E-5);

    /* Wavelengths (microns) */
    Real lam_10 = Real(63.1679);
    Real lam_20 = Real(44.0453);
    Real lam_21 = Real(145.495);

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      7, 8, 9, tabData);
}

/* OIII fine structure calculation function */
template<typename Table>
KOKKOS_INLINE_FUNCTION
Real OIII_fine_structure(Real T, Real n_ion, Real nH, Real nHp,
                           Real ne, Real nH2, Real nHe, Real nHep,
                           Real nHepp, Real z, const Table& tabData){
    /* Degeneracy variables */
    Real g_0 = Real(1.0);
    Real g_1 = Real(3.0);
    Real g_2 = Real(5.0);

    /* Transition rates (s^-1) */
    Real A_10 = Real(2.597e-5);
    Real A_20 = Real(3.170e-11);
    Real A_21 = Real(9.760e-5);

    /* Wavelengths (microns) */
    Real lam_10 = Real(88.3323);
    Real lam_20 = Real(32.6523);
    Real lam_21 = Real(51.8004);

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      12, 13, 14, tabData);
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
Real CI_fine_structure(Real T, Real n_ion, Real nH, Real nHp,
                           Real ne, Real nH2, Real nHe, Real nHep,
                           Real nHepp, Real z, const Table& tabData){
    /* Degeneracy variables */
    Real g_0 = Real(1.0);
    Real g_1 = Real(3.0);
    Real g_2 = Real(5.0);

    /* Transition rates (s^-1) */
    Real A_10 = Real(7.930E-8);
    Real A_20 = Real(1.000E-30);
    Real A_21 = Real(2.650E-7);

    /* Wavelengths (microns) */
    Real lam_10 = Real(609.590);
    Real lam_20 = Real(230.352);
    Real lam_21 = Real(370.269);

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      1, 2, 3, tabData);
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
Real CII_fine_structure(Real T, Real n_ion, Real nH, Real nHp,
                          Real ne, Real nH2, Real nHe, Real nHep,
                          Real nHepp, Real z, const Table& tabData){
    /* Degeneracy variables */
    Real g_0 = Real(2.0);
    Real g_1 = Real(4.0);

    /* Transition rates (s^-1) */
    Real A_10 = Real(2.290E-6);

    /* Wavelengths (microns) */
    Real lam_10 = Real(157.636);

    return two_level(g_0, g_1,
                    lam_10,
                    A_10,
                    z, T,
                    n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                    0,
                    tabData);
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
Real NII_fine_structure(Real T, Real n_ion, Real nH, Real nHp,
                          Real ne, Real nH2, Real nHe, Real nHep,
                          Real nHepp, Real z, const Table& tabData){
    /* Degeneracy variables */
    Real g_0 = Real(1.0);
    Real g_1 = Real(3.0);
    Real g_2 = Real(5.0);

    /* Transition rates (s^-1) */
    Real A_10 = Real(2.080E-06);
    Real A_20 = Real(1.000E-30);
    Real A_21 = Real(7.460E-06);

    /* Wavelengths (microns) */
    Real lam_10 = Real(205.244);
    Real lam_20 = Real(76.4318);
    Real lam_21 = Real(121.767);

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      4, 5, 6, tabData);
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
Real SiI_fine_structure(Real T, Real n_ion, Real nH, Real nHp,
                          Real ne, Real nH2, Real nHe, Real nHep,
                          Real nHepp, Real z, const Table& tabData){
    /* Degeneracy variables */
    Real g_0 = Real(1.0);
    Real g_1 = Real(3.0);
    Real g_2 = Real(5.0);

    /* Transition rates (s^-1) */
    Real A_10 = Real(8.250E-6);
    Real A_20 = Real(3.490E-10);
    Real A_21 = Real(4.210E-5);

    /* Wavelengths (microns) */
    Real lam_10 = Real(129.641);
    Real lam_20 = Real(44.7993);
    Real lam_21 = Real(68.4548);

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      21, 22, 23, tabData);
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
Real SiII_fine_structure(Real T, Real n_ion, Real nH, Real nHp,
                           Real ne, Real nH2, Real nHe, Real nHep,
                           Real nHepp, Real z, const Table& tabData){
    /* Degeneracy variables */
    Real g_0 = Real(2.0);
    Real g_1 = Real(4.0);

    /* Transition rates (s^-1) */
    Real A_10 = Real(2.131E-4);

    /* Wavelengths (microns) */
    Real lam_10 = Real(34.8046);

    return two_level(g_0, g_1,
                    lam_10,
                    A_10,
                    z, T,
                    n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                    11, tabData);
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
Real NeII_fine_structure(Real T, Real n_ion, Real nH, Real nHp,
                           Real ne, Real nH2, Real nHe, Real nHep,
                           Real nHepp, Real z, const Table& tabData){
    /* Degeneracy variables */
    Real g_0 = Real(4.0);
    Real g_1 = Real(2.0);

    /* Transition rates (s^-1) */
    Real A_10 = Real(8.590E-3);

    /* Wavelengths (microns) */
    Real lam_10 = Real(12.8101);

    return two_level(g_0, g_1,
                    lam_10,
                    A_10,
                    z, T,
                    n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                    10, tabData);
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
Real FeI_fine_structure(Real T, Real n_ion, Real nH, Real nHp,
                          Real ne, Real nH2, Real nHe, Real nHep,
                          Real nHepp, Real z, const Table& tabData){
    /* Degeneracy variables */
    Real g_0 = Real(9.0);
    Real g_1 = Real(7.0);
    Real g_2 = Real(5.0);

    /* Transition rates (s^-1) */
    Real A_10 = Real(2.510E-3);
    Real A_20 = Real(1.000E-30);
    Real A_21 = Real(1.560E-3);

    /* Wavelengths (microns) */
    Real lam_10 = Real(14.2005);
    Real lam_20 = Real(24.0358);
    Real lam_21 = Real(34.7038);

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      15, 16, 17, tabData);
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
Real FeII_fine_structure(Real T, Real n_ion, Real nH, Real nHp,
                           Real ne, Real nH2, Real nHe, Real nHep,
                           Real nHepp, Real z, const Table& tabData){
    /* Degeneracy variables */
    Real g_0 = Real(10.0);
    Real g_1 = Real(8.0);
    Real g_2 = Real(6.0);

    /* Transition rates (s^-1) */
    Real A_10 = Real(2.050E-3);
    Real A_20 = Real(1.000E-30);
    Real A_21 = Real(1.560E-3);

    /* Wavelengths (microns) */
    Real lam_10 = Real(25.9811);
    Real lam_20 = Real(14.9731);
    Real lam_21 = Real(35.3394);

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      18, 19, 20, tabData);
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
Real SI_fine_structure(Real T, Real n_ion, Real nH, Real nHp,
                         Real ne, Real nH2, Real nHe, Real nHep,
                         Real nHepp, Real z, const Table& tabData){
    /* Degeneracy variables */
    Real g_0 = Real(5.0);
    Real g_1 = Real(3.0);
    Real g_2 = Real(1.0);

    /* Transition rates (s^-1) */
    Real A_10 = Real(1.400E-3);
    Real A_20 = Real(7.050E-8);
    Real A_21 = Real(3.020E-4);

    /* Wavelengths (microns) */
    Real lam_10 = Real(17.4278);
    Real lam_20 = Real(25.2421);
    Real lam_21 = Real(56.2957);

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      24, 25, 26, tabData);
}

KOKKOS_INLINE_FUNCTION
Real dust_recombination_cooling(Real T, Real G0, Real ne, Real f_dg, Real nH){
    // Dust recombination cooling
    Real beta_drc = Real(0.74) / Kokkos::pow(T,Real(0.068));
    Real dust_recomb_cool = (Real(1.5) * Real(4.65E-30)) * Kokkos::pow(T,Real(0.94)) * Kokkos::pow(G0 * Kokkos::sqrt(T) / (Real(0.5)*ne),beta_drc) * ne * Real(0.5) * f_dg * nH;

    return dust_recomb_cool;
}

KOKKOS_INLINE_FUNCTION
Real dust_gas_collisional_cooling(Real T, Real G0, Real xH2, Real aexp, Real nH, Real f_dg){
    Real dust_hc_const = Real(1.0E-33); //! For atomic dominated regions
    dust_hc_const += (Real(3.8E-33) - Real(1.0E-33)) * Kokkos::fmax(Kokkos::fmin(xH2,Real(1.0)),Real(0.0));

    Real T_dust = Real(16.4) * Kokkos::pow(Real(1.7) * G0,Real(1.0)/Real(6.0));
    T_dust = Kokkos::fmax( T_dust, Real(2.725)/aexp ); // ! Limit dust temp minimum to CMB temp

    Real dust_coll_cool = dust_hc_const * Kokkos::sqrt(T) * (T - T_dust) * ( Real(1.0) - ( Real(0.80) * Kokkos::exp(-Real(75.0)/T) ) );
    return dust_coll_cool * nH * nH * f_dg;
}

KOKKOS_INLINE_FUNCTION
Real PE_efficiency(Real G0, Real T, Real ne){
    Real phi_pah = Real(0.5); //! wolfire+(03)
    Real fact = G0 * Kokkos::sqrt(T) / (ne * phi_pah);
    Real PE_efficiency = (Real(4.9E-2)/(Real(1.0) + Real(4E-3) * Kokkos::pow(fact,Real(0.73)))) + (Real(3.7E-2) * Kokkos::pow(T/Real(1E4),Real(0.7)) / (Real(1.0) + Real(2E-4) * fact));
    return Kokkos::fmax(PE_efficiency, Real(1e-100));
}

KOKKOS_INLINE_FUNCTION
Real photoelectric_heating(Real T, Real G0, Real ne, Real f_dg, Real nH){
    Real eps_PE = PE_efficiency(G0, T, ne);
    Real HratePE = Real(1.50) * Real(1.3E-24) * eps_PE * G0 * f_dg * nH; //! [erg/cm3/s]
    return Kokkos::fmax(HratePE, Real(1e-100));
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
Real cosmic_ray_heating(Real xe, Real n_HI, Real n_HeI, Real n_H2,
                          Real ne, Real xi_h_cr,
                          const CompactIonData& n_and_ion_fracs,
                          const Element* elements,
                          const Table& tabData
                        ){
    Real q_cr = Real(6.43) * ( Real(1.0) + Real(4.06) * Kokkos::sqrt( xe / (Real(0.07) + xe) ) ) * Real(EV_2_ERG); //! Bialy 2019

    Real phi_s = secondary_cr_rates(xe);
    Real total_cosmic_ray_ionization_rate = xi_h_cr * (Real(1.0) + phi_s);

    Real HI_cosmic_ray_ionization_rate_primary = Real(1.0) * xi_h_cr;
    Real He_cosmic_ray_ionization_rate_primary = Real(1.1) * xi_h_cr;
    Real H2_cosmic_ray_ionization_rate_primary = Real(2.0) * xi_h_cr;

    Real cr_heating_tot = Real(0.0);

    // HI
    cr_heating_tot +=  ( n_HI * HI_cosmic_ray_ionization_rate_primary * q_cr ); // erg/s/cm^3

    // HeI
    cr_heating_tot +=  ( n_HeI * He_cosmic_ray_ionization_rate_primary * q_cr ); // erg/s/cm^3

    // H2
    cr_heating_tot += ( n_H2 * H2_cosmic_ray_ionization_rate_primary * q_cr ); // erg/s/cm^3

    // Direct heating of the electrons from cosmic rays
    cr_heating_tot += (ne * xi_h_cr * Real(287.0) * Real(EV_2_ERG)); // erg/s/cm^3

    // Cosmic ray heating on metals
    // --> Here we skip H and He as it's done separately above
    Real induced_UV_heat_scale_fac = H2_cosmic_ray_ionization_rate_primary / Real(1E-16);
    for (int erow = 2; erow < N_TAB_ELEMENTS; erow++) { // Tabulated elements from row 2 on: H and He are rows 0 and 1
        const int i = row_elem(erow);
        Real n_element = Real(n_and_ion_fracs.n_element_new[i]);
        if (n_element < MIN_COOL_ION) continue;

        int n_ions = (elements[i].n_ions + elements[i].n_mol);

        for (int j = 0; j < n_ions - 1; j++) { // This comes from ionization --> no heating for final ion
            Real x_ion = Real(n_and_ion_fracs[i].ion_fracs_new[j]);

            // This is the primary heating term from ionization
            cr_heating_tot += (n_element * x_ion * q_cr * Real(tabData.cosmic_ray_ionization_rates(erow, j)) * total_cosmic_ray_ionization_rate); // erg/s/cm^3

            // This is the secondary heating term from induced UV emission
            // Only happend for the ground state
            if (j == 0){
                cr_heating_tot += ( n_element * x_ion
                                    * Real(tabData.cosmic_ray_ionization_rates_induced_UV_heat(erow))
                                    * Real(tabData.cosmic_ray_ionization_rates_induced_UV(erow)) * Real(EV_2_ERG)
                                    * induced_UV_heat_scale_fac ); // erg/s/cm^3
            }
        }
    }

    return cr_heating_tot;
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
Real photoheating_UVB(const CompactIonData& n_and_ion_fracs,
                        const Element* elements,
                        const Table& tabData){
    // Photoheating contribution from a UVB

    Real heating_rate = Real(0.0);

    // Loop over the tabulated elements
    for (int erow = 0; erow < N_TAB_ELEMENTS; erow++) {
        const int i = row_elem(erow);
        Real n_element = Real(n_and_ion_fracs.n_element_new[i]);
        if (n_element < MIN_COOL_ION) continue;

        int n_ions = (elements[i].n_ions + elements[i].n_mol);
        for (int j = 0; j < n_ions - 1; j++) { // This comes from ionization --> no heating for final ion
            heating_rate += (n_element * Real(n_and_ion_fracs[i].ion_fracs_new[j]) * Real(tabData.HM12_UVB_z(erow, j, 1))); // erg/s/cm^3
        }
    }

    return heating_rate;
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
Real photoheating_UVB_G0(Real G0,
                           const CompactIonData& n_and_ion_fracs,
                           const Table& tabData){
    // Photoheating from the G0 background
    // Note that this only impacts the ground state

    Real heating_rate = Real(0.0);

    // Loop over the tabulated elements (row 2 onwards: no need for H and He here)
    for (int erow = 2; erow < N_TAB_ELEMENTS; erow++) {
        const int i = row_elem(erow);
        Real n_element = Real(n_and_ion_fracs.n_element_new[i]);
        if (n_element < MIN_COOL_ION) continue;

        heating_rate += n_element * Real(n_and_ion_fracs[i].ion_fracs_new[0]) * Real(tabData.G0_heating_rates(erow)) * G0; // erg/s/cm^3
    }

    return heating_rate;
}

KOKKOS_INLINE_FUNCTION
Real Epump(Real nH, Real T, Real xH2, Real xHI){
    //! Energy converted to heat from UV pumping
    //! see Appendix A http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1990ApJ...365..620B&amp;data_type=PDF_HIGH&amp;whole_paper=YES&amp;type=PRINTER&amp;filetype=.pdf
    Real Crad = Real(2.0E-7); //     !radiation de-excitation rate Burton 1990
    Real Cdex = (Real(1.0E-12))*((Real(1.4)*xH2*Kokkos::exp(-Real(18100.0)/(T + Real(1200.0)))) + (Real(1.0)*xHI*Kokkos::exp(-Real(1000.0)/T)))*Kokkos::sqrt(T)*nH; // !collisional de-excitation rate Burton 1990
    Real Cfrac = Cdex / (Cdex + Crad);
    Real Epump_loc = Real(2.0) * Cfrac * Real(EV_2_ERG);  //!ergs
    return Epump_loc;
}

KOKKOS_INLINE_FUNCTION
Real H2_heating_bialy(Real G0, Real nH2, Real nH, Real T, Real xHI,
                        Real xHII, Real xe, Real f_dg, Real xi_h2_cr){

    // Heating from H2 formation and destruction following Bialy 2018
    Real H2_heating = Real(0.0);

    Real D0 = Real(5.68E-11);
    Real I_UV = Real(1.7) * G0;

    Real E_pump = Real(1.12) * Real(EV_2_ERG);
    Real n_crit = Real(1.1E5) / Kokkos::sqrt(T/Real(1000.0));
    Real ncrit_factor = Real(1.0) / (Real(1.0) + (n_crit/nH));

    // heating from H2 pumping
    H2_heating += Real(9.0) * D0 * I_UV * E_pump * nH2 * ncrit_factor; // erg/s/cm^3

    // heating from H2 photodissociation
    Real E_pd = Real(0.4) * Real(EV_2_ERG);
    H2_heating += D0 * I_UV * E_pd * nH2; // erg/s/cm^3

    // heating from H2 formation --> dust channel
    Real E_form1_dust = Real(0.2) * Real(EV_2_ERG);
    Real E_form2_dust = Real(4.48) * Real(EV_2_ERG) * ncrit_factor;
    Real H2_formation_rate_dust = alpha_H2_dust(T, f_dg);

    H2_heating += H2_formation_rate_dust * (E_form1_dust + E_form2_dust) * nH * nH * xHI; // erg/s/cm^3

    // heating from H2 formation --> primordial channel
    Real E_form1_prim = Real(0.6) * Real(EV_2_ERG);
    Real E_form2_prim = Real(3.13) * Real(EV_2_ERG) * ncrit_factor;
    Real H2_formation_rate_prim = alpha_H2_prim(T, xe, xi_h2_cr, G0, xHI, xHII, nH);

    H2_heating += H2_formation_rate_prim * (E_form1_prim + E_form2_prim) * nH * nH * xHI; // erg/s/cm^3

    return Kokkos::fmax(H2_heating, Real(1e-100));
}

struct CTReaction { bool is_ionization; int stage; int Z; Real dE; };

template<typename Table>
KOKKOS_INLINE_FUNCTION
Real CT_heat_cool(Real T, const CompactIonData& n_and_ion_fracs,
                    const Table& tabData){
    // Heating and cooling from charge exchange reactions

    Real ct_hc_rate = Real(0.0);

    if (T > Real(1E5)){
        return ct_hc_rate;
    }


    Real loc_nHI = Real(n_and_ion_fracs.n_element_new[1]) * Real(n_and_ion_fracs[1].ion_fracs_new[0]);
    Real loc_nHII = Real(n_and_ion_fracs.n_element_new[1]) * Real(n_and_ion_fracs[1].ion_fracs_new[1]);

    // Charge transfer reactions, one table entry per reaction:
    //   is_ionization : true  --> H+ + X(n)  -> X(n+1) + H,  rate scales with nHII
    //                   false --> H  + X(n)  -> X(n-1) + H+, rate scales with nHI
    //   stage         : ionization stage of X, which is also its ion_fracs_new index
    //   Z             : atomic number of X
    //   dE            : energy released by the reaction [eV], negative if absorbed
    static constexpr CTReaction ct_reactions[] = {
        // Helium
        {false, 1,  2,  Real(10.99)}, // H + He+ -> He + H+

        // Carbon
        {true,  0,  6,   Real(2.34)}, // H+ + C -> C+ + H
        {false, 1,  6,  -Real(2.34)}, // H + C+ -> C + H+
        {false, 2,  6,   Real(4.01)}, // H + C++ -> C+ + H+
        {false, 3,  6,   Real(5.73)}, // H + C+++ -> C++ + H+
        {false, 4,  6,  Real(11.3) }, // H + C++++ -> C+++ + H+

        // Nitrogen
        {true,  0,  7,  -Real(0.94)}, // H+ + N -> N+ + H
        {false, 1,  7,   Real(0.94)}, // H + N+ -> N + H+
        {false, 2,  7,   Real(4.56)}, // H + N++ -> N+ + H+
        {false, 3,  7,   Real(6.40)}, // H + N+++ -> N++ + H+
        {false, 4,  7,  Real(11.0) }, // H + N++++ -> N+++ + H+

        // Oxygen
        {true,  0,  8,  -Real(0.02)}, // H+ + O -> O+ + H
        {false, 1,  8,   Real(0.02)}, // H + O+ -> O + H+
        {false, 2,  8,   Real(6.65)}, // H + O++ -> O+ + H+
        {false, 3,  8,   Real(5.00)}, // H + O+++ -> O++ + H+
        {false, 4,  8,   Real(8.47)}, // H + O++++ -> O+++ + H+

        // Neon
        {false, 3, 10,   Real(5.82)}, // H + Ne+++ -> Ne++ + H+
        {false, 4, 10,   Real(8.60)}, // H + Ne++++ -> Ne+++ + H+

        // Magnesium
        {true,  0, 12,   Real(1.52)}, // H+ + Mg -> Mg+ + H
        {true,  1, 12,  -Real(1.44)}, // H+ + Mg+ -> Mg++ + H+
        {false, 2, 12,   Real(1.44)}, // H + Mg++ -> Mg+ + H+
        {false, 3, 12,   Real(5.73)}, // H + Mg+++ -> Mg++ + H+
        {false, 4, 12,   Real(8.60)}, // H + Mg++++ -> Mg+++ + H+

        // Silicon
        {true,  0, 14,   Real(0.12)}, // H+ + Si -> Si+ + H
        {true,  1, 14,  -Real(2.72)}, // H+ + Si+ -> Si++ + H+
        {false, 2, 14,   Real(2.72)}, // H + Si++ -> Si+ + H+
        {false, 3, 14,   Real(4.23)}, // H + Si+++ -> Si++ + H+
        {false, 4, 14,   Real(7.49)}, // H + Si++++ -> Si+++ + H+

        // Sulfur
        {false, 1, 16,  -Real(3.24)}, // H + S+ -> S + H+
        {false, 3, 16,   Real(5.73)}, // H + S+++ -> S++ + H+
        {false, 4, 16,   Real(8.60)}, // H + S++++ -> S+++ + H+

        // Iron
        {true,  1, 26,  -Real(2.56)}, // H+ + Fe+ -> Fe++ + H+
        {false, 2, 26,   Real(2.56)}, // H + Fe++ -> Fe+ + H+
        {false, 3, 26,   Real(6.30)}, // H + Fe+++ -> Fe++ + H+
        {false, 4, 26,  Real(10.0) }, // H + Fe++++ -> Fe+++ + H+
    };

    // Don't unroll to save on instructions (saves ~30% of SASS instructions for cooling kernel)
#if defined(__clang__)      // clang: HIP/ROCm, SYCL, clang-CUDA (host and device)
#pragma clang loop unroll(disable)
#elif defined(__CUDA_ARCH__)  // nvcc device pass; its host pass accepts neither spelling
#pragma unroll 1
#endif
    for (const CTReaction& r : ct_reactions) {
        Real ct_rate = r.is_ionization ? charge_transfer_ionization(r.stage, r.Z, T, tabData)
                                         : charge_transfer_recombination(r.stage, r.Z, T, tabData);
        Real n_H_partner = r.is_ionization ? loc_nHII : loc_nHI;

        ct_hc_rate += ct_rate * n_H_partner * Real(n_and_ion_fracs.n_element_new[r.Z]) * Real(n_and_ion_fracs[r.Z].ion_fracs_new[r.stage]) * r.dE * Real(EV_2_ERG);
    }

    return ct_hc_rate;
}


template<typename Table>
KOKKOS_FUNCTION
Real local_photoheating(std::array<Real, N_GROUPS>& dNp,
                          const CompactIonData& n_and_ion_fracs,
                          const Element* elements,
                          const Table& tabData) {
    // Heading due to the local radiation field

    Real heating_rate = Real(0.0);

    // looping over number of elements being used (active species)
    // refactor: remove if-statement

    for (int erow = 0; erow < N_TAB_ELEMENTS; erow++) { // Loop over tabulated elements
        const int i = row_elem(erow);
        Real n_element = Real(n_and_ion_fracs.n_element_new[i]);
        if (n_element < MIN_COOL_ION) continue;

        int n_ions = (elements[i].n_ions + elements[i].n_mol);

        for (int j = 0; j < n_ions - 1; j++) { // Loop over ions --> no heating for final ion
            Real n_ion = n_element * Real(n_and_ion_fracs[i].ion_fracs_new[j]);
            for (int k = 0; k < N_GROUPS; k++) { // Loop over photon groups
                heating_rate += (Real(dNp[k]) * n_ion * Real(tabData.cs_ph(erow, j, k, 2)));
            } // End loop over photon groups
        } // End loop over ions
    } // End loop over elements
    return Kokkos::fmax(heating_rate, Real(1e-100));
}

KOKKOS_INLINE_FUNCTION
Real CO_cooling_koyama_00(Real nH, Real nH2, Real nHI, Real nCO, Real T) {
    // CO cooling
    Real rate = Real(0.0);
    Real T_pivot = Real(3080.0);
    Real term_a = (Real(3.3e6) * Kokkos::pow(T / Real(1.0e3), Real(0.75))) / nH;

    // CO Rotational cooling
    Real rot_numerator = nCO * Real(4.0) * SQR(Real(KB) * T) * Real(9.7e-8);
    Real rot_denominator = Real(2.76) * Real(KB) * (Real(1.0) + term_a + Real(1.5) * Kokkos::sqrt(term_a));
    rate += rot_numerator / rot_denominator;

    // CO Vibrational cooling (H2 collisions)
    rate += nH2 * nCO * T_pivot * Real(KB) * Real(4.3e-14) * T * Kokkos::exp(-Kokkos::pow(Real(3.14e5) / T, Real(0.333))) * Kokkos::exp(-T_pivot / T);

    // CO Vibrational cooling (HI collisions)
    rate += nHI * nCO * T_pivot * Real(KB) * Real(3.0e-12) * Kokkos::sqrt(T) * Kokkos::exp(-Kokkos::pow(Real(2000.0) / T, Real(3.43))) * Kokkos::exp(-T_pivot / T);

    return Kokkos::fmax(rate, Real(1e-100));
}

template<typename Table>
KOKKOS_FUNCTION
Real all_cooling(Real T, Real ne, Real aexp, const CompactIonData& n_and_ion_fracs,
                   const Element* elements,
                   std::array<Real, N_GROUPS>& dNp,
                   Real nCO,
                   Real G0_local, Real UV_background_G0, Real f_shd_H2,
                   Real f_dg, Real xe, Real xi_h_cr, Real xi_h2_cr,
                   Real ss_factor,
                   bool rt_advect,
                   const PhysicsFlags& flags,
                   const Table& tabData){
    // Main cooling driver
    /*
    T --> Temperature [K]
    ne --> Electron number density [cm^-3]
    aexp --> Scale factor
    n_and_ion_fracs --> compact ion data with element densities and ion fractions
    elements --> element metadata (n_ions, n_mol, etc.)
    dNp --> photon number densities in each group
    G0 --> habing band radiation field (MW units)
    f_dg --> dust to gas mass ratio normalized by MW value
    xe --> electron fraction i.e. ne / (rho/mH)
    xi_h_cr --> cosmic ray ionization rate
    */

    Real net_cooling = Real(0.0); // heating terms are positive, cooling terms are negative

    Real nH    = Real(n_and_ion_fracs.n_element_new[1]);
    Real xHI   = Real(n_and_ion_fracs[1].ion_fracs_new[0]);
    Real xHII  = Real(n_and_ion_fracs[1].ion_fracs_new[1]);
    Real xH2   = (elements[1].n_mol > 0) ? Real(n_and_ion_fracs[1].ion_fracs_new[2]) : Real(0.0);
    Real nH_I  = nH * xHI;
    Real nH_II = nH * xHII;
    Real nH2   = nH * xH2 * Real(0.5);

    Real nHe_I   = Real(n_and_ion_fracs.n_element_new[2]) * Real(n_and_ion_fracs[2].ion_fracs_new[0]);
    Real nHe_II  = Real(n_and_ion_fracs.n_element_new[2]) * Real(n_and_ion_fracs[2].ion_fracs_new[1]);
    Real nHe_III = Real(n_and_ion_fracs.n_element_new[2]) * Real(n_and_ion_fracs[2].ion_fracs_new[2]);

    // metal cool smoothing parameters
    Real metal_cool_smooth_f1 = Real(0.5) * (Kokkos::tanh( (Real(5E-3)) * ( T - Real(1.E4) ) ) + Real(1.0) );
    Real metal_cool_smooth_f2 = Real(0.5) * (Kokkos::tanh( (Real(5E-3)) * ( (-Real(1.0) * T) + Real(1.E4) ) ) + Real(1.0) );

    // Cooling from primordial species
    net_cooling -= (collisional_ionization_cooling_HI(T) + collisional_excitation_cooling_HI(T)) * ne * nH_I; // cooling_HI
    net_cooling -= recombination_cooling_case_B_HII(T) * ne * nH_II; // cooling_HII
    net_cooling -= collisional_ionization_cooling_HeI(T) * ne * nHe_I; // cooling_HeI
    net_cooling -= (collisional_ionization_cooling_HeII(T) + collisional_excitation_cooling_HeII(T) +
                                 recombination_cooling_case_B_HeII(T) + dielectronic_recombination_cooling_HeII(T)) * ne * nHe_II; // cooling_HeII
    net_cooling -= recombination_cooling_case_B_HeIII(T) * ne * nHe_III; // cooling_HeIII
    net_cooling -= bremmstrahlung(T) * ne * (nH_II + nHe_II + (Real(4.0) * nHe_III)); // cooling_bremmstrahlung
    net_cooling -= compton_cooling(T,aexp) * ne; // cooling_compton
    net_cooling -= H2_cooling(nH_I, nH2, T); // cooling_H2

    // Metal line cooling -- high temperature
    net_cooling -= get_high_t_cooling_rates(T, ne,
                                            n_and_ion_fracs, elements,
                                            metal_cool_smooth_f1,
                                            tabData);

    // Metal line cooling -- low temperature (fine structure)
    Real n_ion_fs;
    Real z = (Real(1.0) / aexp) - Real(1.0);
    Real total_fine_structure = Real(0.0);
    if (T < Real(1.1E4)) {
        // CI cooling
        n_ion_fs = Real(n_and_ion_fracs.n_element_new[6]) * Real(n_and_ion_fracs[6].ion_fracs_new[0]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += CI_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                    nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // CII cooling
        n_ion_fs = Real(n_and_ion_fracs.n_element_new[6]) * Real(n_and_ion_fracs[6].ion_fracs_new[1]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += CII_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                     nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // NII cooling
        n_ion_fs = Real(n_and_ion_fracs.n_element_new[7]) * Real(n_and_ion_fracs[7].ion_fracs_new[1]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += NII_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                     nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // OI cooling
        n_ion_fs = Real(n_and_ion_fracs.n_element_new[8]) * Real(n_and_ion_fracs[8].ion_fracs_new[0]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += OI_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                    nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // OIII cooling
        n_ion_fs = Real(n_and_ion_fracs.n_element_new[8]) * Real(n_and_ion_fracs[8].ion_fracs_new[2]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += OIII_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                      nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // NeII cooling
        n_ion_fs = Real(n_and_ion_fracs.n_element_new[10]) * Real(n_and_ion_fracs[10].ion_fracs_new[1]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += NeII_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                      nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // SiI cooling
        n_ion_fs = Real(n_and_ion_fracs.n_element_new[14]) * Real(n_and_ion_fracs[14].ion_fracs_new[0]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += SiI_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                     nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // SiII cooling
        n_ion_fs = Real(n_and_ion_fracs.n_element_new[14]) * Real(n_and_ion_fracs[14].ion_fracs_new[1]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += SiII_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                      nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // SI cooling
        n_ion_fs = Real(n_and_ion_fracs.n_element_new[16]) * Real(n_and_ion_fracs[16].ion_fracs_new[0]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += SI_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                    nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // FeI cooling
        n_ion_fs = Real(n_and_ion_fracs.n_element_new[26]) * Real(n_and_ion_fracs[26].ion_fracs_new[0]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += FeI_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                     nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // FeII cooling
        n_ion_fs = Real(n_and_ion_fracs.n_element_new[26]) * Real(n_and_ion_fracs[26].ion_fracs_new[1]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += FeII_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                      nH2, nHe_I, nHe_II, nHe_III, z, tabData);
    }
    // Smooth the fine structure cooling with temeprature if needed
    net_cooling -= total_fine_structure * metal_cool_smooth_f2;

    // Dust cooling
    net_cooling -= dust_recombination_cooling(T, G0_local + UV_background_G0, ne, f_dg, nH);
    net_cooling -= dust_gas_collisional_cooling(T, G0_local + UV_background_G0, xH2*Real(2.0), aexp, nH, f_dg);

    // CO cooling
    net_cooling -= CO_cooling_koyama_00(nH, nH2, nH_I, nCO, T);

    // Photoelectric heating --> note factor of 1.7 is because IUV
    net_cooling += photoelectric_heating(T, G0_local + UV_background_G0, ne, f_dg, nH);

    // Cosmic ray heating
    if (flags.include_cosmic_ray_ionization)
		net_cooling += cosmic_ray_heating(xe, nH_I, nHe_I, nH2, ne, xi_h_cr,
                                             n_and_ion_fracs, elements, tabData);

    // Photoheating from the UV background
    if (flags.include_HM12_UVB)
        net_cooling += photoheating_UVB(n_and_ion_fracs, elements, tabData) * ss_factor; // Account for self-shielding

    // Photoheating from the G0 FUV background
    net_cooling += photoheating_UVB_G0(UV_background_G0, n_and_ion_fracs,  tabData);

    // Heating from H2 formation and destruction
    if (elements[1].n_mol > 0)
        net_cooling += H2_heating_bialy(G0_local*f_shd_H2 + UV_background_G0, nH2, nH, T, xHI, xHII, xe, f_dg, xi_h2_cr);

    // Heating and cooling from charge transfer reactions
    if (flags.include_charge_exchange)
		net_cooling += CT_heat_cool(T, n_and_ion_fracs, tabData);

    if (rt_advect)
        net_cooling += local_photoheating(dNp, n_and_ion_fracs, elements, tabData);

    return net_cooling;
}
