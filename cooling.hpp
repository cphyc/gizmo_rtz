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


#ifdef RTZ_FLOAT32_COOLING
using CoolingReal = float;
namespace cooling_math = rtz_float_math;
#else
using CoolingReal = Real;
namespace cooling_math = rtz_math;
#endif

// Constants
const CoolingReal MIN_METAL_COOL_DENS = CoolingReal(1E-10); //minimum number density to calc cooling
const CoolingReal MIN_COOL_ION = CoolingReal(1.E-10); //minimum number density to calculate cooling for a particular ion

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
CoolingReal collisional_ionization_cooling_HI(CoolingReal T){
    // From Cen 1992
    CoolingReal term_1 = CoolingReal(1.27E-21) * cooling_math::sqrt(T);
    CoolingReal term_2 = CoolingReal(1.0) /(CoolingReal(1.0) + cooling_math::sqrt(T/CoolingReal(1E5)));
    CoolingReal term_3 = cooling_math::exp(-CoolingReal(157809.1)/T);
    CoolingReal cooling_rate = term_1 * term_2 * term_3;
    return cooling_math::fmax(cooling_rate, CoolingReal(1e-100));
}

KOKKOS_INLINE_FUNCTION
CoolingReal collisional_ionization_cooling_HeI(CoolingReal T){
    // From Cen 1992
    CoolingReal term_1 = CoolingReal(9.38E-22) * cooling_math::sqrt(T);
    CoolingReal term_2 = CoolingReal(1.0) /(CoolingReal(1.0) + cooling_math::sqrt(T/CoolingReal(1E5)));
    CoolingReal term_3 = cooling_math::exp(-CoolingReal(285335.4)/T);
    CoolingReal cooling_rate = term_1 * term_2 * term_3;
    return cooling_math::fmax(cooling_rate, CoolingReal(1e-100));
}

KOKKOS_INLINE_FUNCTION
CoolingReal collisional_ionization_cooling_HeII(CoolingReal T){
    // From Cen 1992
    CoolingReal term_1 = CoolingReal(4.95E-22) * cooling_math::sqrt(T);
    CoolingReal term_2 = CoolingReal(1.0) /(CoolingReal(1.0) + cooling_math::sqrt(T/CoolingReal(1E5)));
    CoolingReal term_3 = cooling_math::exp(-CoolingReal(631515.0)/T);
    CoolingReal cooling_rate = term_1 * term_2 * term_3;
    return cooling_math::fmax(cooling_rate, CoolingReal(1e-100));
}

KOKKOS_INLINE_FUNCTION
CoolingReal recombination_cooling_case_B_HII(CoolingReal T){
    // From Hui & Gnedin 1997
    CoolingReal lam_HI = CoolingReal(315614.0) / T;
    CoolingReal term_1 = CoolingReal(3.435E-30) * T;
    CoolingReal term_2 = cooling_math::pow(lam_HI,CoolingReal(1.97));
    CoolingReal term_3 = cooling_math::pow(CoolingReal(1.0) + cooling_math::pow(lam_HI/CoolingReal(2.25),CoolingReal(0.376)),-CoolingReal(3.72));
    CoolingReal cooling_rate = term_1 * term_2 * term_3;
    return cooling_math::fmax(cooling_rate, CoolingReal(1e-100));
}

KOKKOS_INLINE_FUNCTION
CoolingReal recombination_cooling_case_B_HeII(CoolingReal T){
    // From Hui & Gnedin 1997
    CoolingReal lam_HeI = CoolingReal(570670.0) / T;
    CoolingReal term_1 = CoolingReal(KB) * T * CoolingReal(1.26E-14);
    CoolingReal term_2 = cooling_math::pow(lam_HeI,CoolingReal(0.75));
    CoolingReal cooling_rate = term_1 * term_2;
    return cooling_math::fmax(cooling_rate, CoolingReal(1e-100));
}

KOKKOS_INLINE_FUNCTION
CoolingReal recombination_cooling_case_B_HeIII(CoolingReal T){
    // From Hui & Gnedin 1997
    CoolingReal lam_HeII = CoolingReal(1263030.0) / T;
    CoolingReal term_1 = CoolingReal(8.0) * CoolingReal(3.435E-30) * T;
    CoolingReal term_2 = cooling_math::pow(lam_HeII,CoolingReal(1.97));
    CoolingReal term_3 = cooling_math::pow(CoolingReal(1.0) + cooling_math::pow(lam_HeII/CoolingReal(2.25),CoolingReal(0.376)),-CoolingReal(3.72));
    CoolingReal cooling_rate = term_1 * term_2 * term_3;
    return cooling_math::fmax(cooling_rate, CoolingReal(1e-100));
}


KOKKOS_INLINE_FUNCTION
CoolingReal dielectronic_recombination_cooling_HeII(CoolingReal T){
    // From Black 1981
    CoolingReal term_1 = CoolingReal(1.24E-13) * cooling_math::pow(T,-CoolingReal(1.5));
    CoolingReal term_2 = cooling_math::exp(-CoolingReal(470000.0)/T);
    CoolingReal term_3 = CoolingReal(1.0) + (CoolingReal(0.3) * cooling_math::exp(-CoolingReal(94000.0)/T));
    CoolingReal cooling_rate = term_1 * term_2 * term_3;
    return cooling_math::fmax(cooling_rate, CoolingReal(1e-100));
}


KOKKOS_INLINE_FUNCTION
CoolingReal collisional_excitation_cooling_HI(CoolingReal T){
    // From Seon 2020
    CoolingReal cooling_rate = CoolingReal(4.13e-19) * cooling_math::exp(-CoolingReal(117744.0)/T);
    return cooling_math::fmax(cooling_rate, CoolingReal(1e-100));
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
CoolingReal collisional_excitation_cooling_HeII(CoolingReal T){
    // From Cen 1992
    CoolingReal term_1 = CoolingReal(5.54E-17) * cooling_math::pow(T,-CoolingReal(0.397));
    CoolingReal term_2 = CoolingReal(1.0) /(CoolingReal(1.0) + cooling_math::sqrt(T/CoolingReal(1E5)));
    CoolingReal term_3 = cooling_math::exp(-CoolingReal(473638.0)/T);
    CoolingReal cooling_rate = term_1 * term_2 * term_3;
    return cooling_math::fmax(cooling_rate, CoolingReal(1e-100));
}

KOKKOS_INLINE_FUNCTION
CoolingReal bremmstrahlung(CoolingReal T){
    // Osterbrock and Ferland 2006
    CoolingReal cooling_rate = CoolingReal(1.42E-27) * cooling_math::sqrt(T);
    return cooling_math::fmax(cooling_rate, CoolingReal(1e-100));
}

KOKKOS_INLINE_FUNCTION
CoolingReal compton_cooling(CoolingReal T, CoolingReal a){

    CoolingReal Tcmb_over_a = CoolingReal(2.727)/a;

    // Haiman+ 1996
    CoolingReal term_1 = CoolingReal(1.017E-37);
    CoolingReal term_2 = SQR(SQR(Tcmb_over_a));
    CoolingReal term_3 = T - Tcmb_over_a;
    CoolingReal cooling_rate = term_1 * term_2 * term_3;
    return cooling_rate;
}

KOKKOS_INLINE_FUNCTION
CoolingReal H2_cooling(CoolingReal nH, CoolingReal nH2, CoolingReal T){
    // From Moseley 2021
    CoolingReal T3 = CoolingReal(1E-3) * T;
    CoolingReal n1 = CoolingReal(50.0);
    CoolingReal n2 = CoolingReal(450.0);
    CoolingReal n3 = CoolingReal(25.0);
    CoolingReal n4 = CoolingReal(900.0);

    CoolingReal x1 = nH + (CoolingReal(5.0) * nH2);
    CoolingReal x2 = nH + (CoolingReal(4.5) * nH2);
    CoolingReal x3 = nH + (CoolingReal(0.75) * nH2);
    CoolingReal x4 = nH + (CoolingReal(0.05) * nH2);

    CoolingReal f1 = CoolingReal(1.1E-25) * cooling_math::sqrt(T3) * cooling_math::exp(-CoolingReal(0.51) / T3);
    f1 = f1 * (((CoolingReal(0.7) * x1) / (CoolingReal(1.0) + (x1/n1))) + ((CoolingReal(0.3) * x1)/(CoolingReal(1.0) + (x1/(CoolingReal(10.0)*n1)))));

    CoolingReal f2 = CoolingReal(2.0E-25) * T3 * cooling_math::exp(-CoolingReal(1.0) / T3);
    f2 = f2 * (((CoolingReal(0.35) * x2) / (CoolingReal(1.0) + (x2/n2))) + ((CoolingReal(0.65) * x2)/(CoolingReal(1.0) + (x2/(CoolingReal(10.0)*n2)))));

    CoolingReal f3 = CoolingReal(2.4E-24) * (cooling_math::pow(T3,CoolingReal(1.50))) * cooling_math::exp(-CoolingReal(2.0) / T3);
    f3 = f3 * (x3 / (CoolingReal(1.0) + (x3/n3)));

    CoolingReal f4 = CoolingReal(1.7E-23) * (cooling_math::pow(T3,CoolingReal(1.50))) * cooling_math::exp(-CoolingReal(4.0) / T3);
    f4 = f4 * (((CoolingReal(0.45) * x4) / (CoolingReal(1.0) + (x4/n4))) + ((CoolingReal(0.55) * x4)/(CoolingReal(1.0) + (x4/(CoolingReal(10.0)*n4)))));

    return cooling_math::fmax(nH2 * (f1 + f2 + f3 + f4), CoolingReal(1e-100));
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
CoolingReal get_high_t_cooling_rates(CoolingReal T, CoolingReal ne,
                                const CompactIonData& n_and_ion_fracs,
                                const Element* elements,
                                CoolingReal temp_smooth,
                                const Table& tabData){
    // Computes the high-temperature cooling rates
    // derived from Harley's custom cloudy models

    CoolingReal loc_T = cooling_math::fmax(T,CoolingReal(1000.0));
    CoolingReal log_T = cooling_math::log10(loc_T);
    CoolingReal t_min = CoolingReal(3.0);
    CoolingReal t_max = CoolingReal(9.0);
    CoolingReal dt = CoolingReal(0.05);
    CoolingReal total_metal_cooling_rate = CoolingReal(0.0);
    CoolingReal two_e4_overT = CoolingReal(2000.0)/loc_T;
    CoolingReal t_scale_fac = cooling_math::exp(-CoolingReal(1.0) * cooling_math::pow(two_e4_overT, 5));

    // bounds for temperature --> no cooling
    if (log_T < t_min){
        return total_metal_cooling_rate;
    }

    // Still cool above 10^9 K but set a bound
    log_T = cooling_math::fmin(log_T, t_max);

    // Prepare for 1D interpolation
    int idx_low = (int)cooling_math::floor((log_T - t_min)/dt);

	// Uniform grid: high_t_cooling_temp[k] = t_min + k * dt
    CoolingReal frac_high = (log_T - (t_min + idx_low * dt)) / dt;
    CoolingReal frac_low = CoolingReal(1.0) - frac_high;

    // Loop over the tabulated elements -- not including H and He (rows 0 and 1)
    for (int erow = 2; erow < N_TAB_ELEMENTS; erow++) {
        const int i = row_elem(erow);
        if (CoolingReal(n_and_ion_fracs.n_element_new[i]) > MIN_METAL_COOL_DENS){
            // Get the element number density
            CoolingReal n_element = CoolingReal(n_and_ion_fracs.n_element_new[i]);
            for (int j = 0; j < (elements[i].n_ions + elements[i].n_mol); j++) {
                // Get the element ion fraction
                CoolingReal n_ion = CoolingReal(n_and_ion_fracs[i].ion_fracs_new[j]);

                // Interpolate the cooling rate for the ion
                CoolingReal loc_cooling_rate = (frac_low * CoolingReal(tabData.high_t_cooling_rates(idx_low, erow, j)));
                loc_cooling_rate += (frac_high * CoolingReal(tabData.high_t_cooling_rates(idx_low+1, erow, j)));
                loc_cooling_rate = cooling_math::exp10(loc_cooling_rate);

                // Smooth with temperature if necessary
                CoolingReal loc_temp_smooth = CoolingReal(1.0);
                if (CoolingReal(tabData.high_t_cooling_rates_tflag(erow,j))) {
                    loc_temp_smooth = temp_smooth;
                }

                // Multiply cooling rate by ion and electron number densities
                total_metal_cooling_rate += (n_element * n_ion * ne * loc_cooling_rate * loc_temp_smooth);
            }
        }
    }

    return cooling_math::fmax(total_metal_cooling_rate * t_scale_fac,CoolingReal(1e-100));
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
CoolingReal line_planck_spectrum(CoolingReal nu, CoolingReal temperature) {
#if defined(RTZ_FLOAT32_ARITHMETIC) || defined(RTZ_FLOAT32_COOLING)
    const CoolingReal nu_over_c = nu / CoolingReal(C_CGS);
    const CoolingReal prefactor = (CoolingReal(2) * CoolingReal(H_PLANCK) * nu) * nu_over_c * nu_over_c;
#else
    const CoolingReal prefactor = CoolingReal(2) * CoolingReal(H_PLANCK) * CUBE(nu) / (CoolingReal(C_CGS) * CoolingReal(C_CGS));
#endif
    return prefactor / (cooling_math::exp(CoolingReal(H_PLANCK) * nu / (CoolingReal(KB) * temperature)) - CoolingReal(1));
}

template<typename Table>
KOKKOS_FUNCTION
CoolingReal three_level(CoolingReal g_0, CoolingReal g_1, CoolingReal g_2,
                   CoolingReal lam_10, CoolingReal lam_20, CoolingReal lam_21,
                   CoolingReal A_10, CoolingReal A_20, CoolingReal A_21,
                   CoolingReal z, CoolingReal T, CoolingReal n_ion,
                   CoolingReal ne, CoolingReal nH, CoolingReal nHp,
                   CoolingReal nHe, CoolingReal nHep, CoolingReal nHepp,
                   CoolingReal nH2,
                   int idx_0, int idx_1, int idx_2,
                   const Table& tabData
                ){
    // Harley's three-level ion solver

    // ! Initialize the result
    CoolingReal three_level_cooling = CoolingReal(1E-50);

    CoolingReal T_cmb = CoolingReal(2.725) * (CoolingReal(1.0) + z); //! CMB temperature

    // ! Get log T and enforce bounds
    CoolingReal logT = cooling_math::log10(T);

    // Enforce bounds on temperature
    CoolingReal tmin = CoolingReal(1.0);
    CoolingReal tmax = CoolingReal(4.975);
    CoolingReal delta_temp = CoolingReal(0.025);
    logT = cooling_math::fmax(logT,CoolingReal(1.0));
    logT = cooling_math::fmin(logT,tmax);
    CoolingReal lam_10_minus4 = lam_10 * CoolingReal(1E-4);
    CoolingReal lam_20_minus4 = lam_20 * CoolingReal(1E-4);
    CoolingReal lam_21_minus4 = lam_21 * CoolingReal(1E-4);

    CoolingReal nu_10 = CoolingReal(C_CGS) / lam_10_minus4; //! Hz
    CoolingReal nu_20 = CoolingReal(C_CGS) / lam_20_minus4; //! Hz
    CoolingReal nu_21 = CoolingReal(C_CGS) / lam_21_minus4; //! Hz

    CoolingReal E_10 = (CoolingReal(H_PLANCK) * CoolingReal(C_CGS) / lam_10_minus4) / CoolingReal(KB); //! E/K (K)
    CoolingReal E_20 = (CoolingReal(H_PLANCK) * CoolingReal(C_CGS) / lam_20_minus4) / CoolingReal(KB); //! E/K (K)
    CoolingReal E_21 = (CoolingReal(H_PLANCK) * CoolingReal(C_CGS) / lam_21_minus4) / CoolingReal(KB); //! E/K (K)

    CoolingReal B_01 = A_10 * (CUBE(lam_10_minus4)) * (g_1 / g_0) / (CoolingReal(2.0) * CoolingReal(H_PLANCK) * CoolingReal(C_CGS));
    CoolingReal B_02 = A_20 * (CUBE(lam_20_minus4)) * (g_2 / g_0) / (CoolingReal(2.0) * CoolingReal(H_PLANCK) * CoolingReal(C_CGS));
    CoolingReal B_12 = A_21 * (CUBE(lam_21_minus4)) * (g_2 / g_1) / (CoolingReal(2.0) * CoolingReal(H_PLANCK) * CoolingReal(C_CGS));

    CoolingReal B_10 = (g_0 / g_1) * B_01;
    CoolingReal B_20 = (g_0 / g_2) * B_02;
    CoolingReal B_21 = (g_1 / g_2) * B_12;

    //! CMB black body spectrum
    CoolingReal B_nu_10 = line_planck_spectrum(nu_10, T_cmb);
    CoolingReal B_nu_20 = line_planck_spectrum(nu_20, T_cmb);
    CoolingReal B_nu_21 = line_planck_spectrum(nu_21, T_cmb);

    // ! Find temperature index.
    int itemp_low = (int)cooling_math::floor((logT - tmin)/delta_temp);
    int itemp_high = itemp_low + 1;
    CoolingReal frac_high = (logT - (tmin + ((CoolingReal)itemp_low * delta_temp))) / delta_temp;
    CoolingReal frac_low = CoolingReal(1.0) - frac_high;

    // ! All collison strengths are in cm^3 s^-1
    // ! For collisions with o-H2
    CoolingReal q10_oH2 = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 6)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 6)) * frac_high);
    CoolingReal q20_oH2 = (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_low, 6)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_high, 6)) * frac_high);
    CoolingReal q21_oH2 = (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_low, 6)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_high, 6)) * frac_high);

    // ! For collisions with p-H2
    CoolingReal q10_pH2 = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 7)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 7)) * frac_high);
    CoolingReal q20_pH2 = (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_low, 7)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_high, 7)) * frac_high);
    CoolingReal q21_pH2 = (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_low, 7)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_high, 7)) * frac_high);

    // ! For collisions with H
    CoolingReal q10_H = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 4)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 4)) * frac_high);
    CoolingReal q20_H = (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_low, 4)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_high, 4)) * frac_high);
    CoolingReal q21_H = (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_low, 4)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_high, 4)) * frac_high);

    // ! For collisions with H+
    CoolingReal q10_Hp = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 1)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 1)) * frac_high);
    CoolingReal q20_Hp = (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_low, 1)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_high, 1)) * frac_high);
    CoolingReal q21_Hp = (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_low, 1)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_high, 1)) * frac_high);

    // ! For collisions with e
    CoolingReal q10_e = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 0)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 0)) * frac_high);
    CoolingReal q20_e = (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_low, 0)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_high, 0)) * frac_high);
    CoolingReal q21_e = (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_low, 0)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_high, 0)) * frac_high);

    // ! For collisions with He
    CoolingReal q10_He = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 5)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 5)) * frac_high);
    CoolingReal q20_He = (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_low, 5)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_high, 5)) * frac_high);
    CoolingReal q21_He = (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_low, 5)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_high, 5)) * frac_high);

    // ! For collisions with He+
    CoolingReal q10_Hep = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 2)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 2)) * frac_high);
    CoolingReal q20_Hep = (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_low, 2)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_high, 2)) * frac_high);
    CoolingReal q21_Hep = (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_low, 2)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_high, 2)) * frac_high);

    // ! For collisions with He++
    CoolingReal q10_Hepp = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 3)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 3)) * frac_high);
    CoolingReal q20_Hepp = (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_low, 3)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_1, itemp_high, 3)) * frac_high);
    CoolingReal q21_Hepp = (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_low, 3)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_2, itemp_high, 3)) * frac_high);

    // ! Net collision strengths note the 75-25 ortho-para H2
    CoolingReal C_10 = (q10_e * ne) + (q10_H * nH) + (q10_Hp * nHp) + (q10_oH2 * CoolingReal(0.75) * nH2) + (q10_pH2 * CoolingReal(0.25) * nH2) + (q10_He * nHe) + (q10_Hep * nHep) + (q10_Hepp * nHepp);
    CoolingReal C_20 = (q20_e * ne) + (q20_H * nH) + (q20_Hp * nHp) + (q20_oH2 * CoolingReal(0.75) * nH2) + (q20_pH2 * CoolingReal(0.25) * nH2) + (q20_He * nHe) + (q20_Hep * nHep) + (q20_Hepp * nHepp);
    CoolingReal C_21 = (q21_e * ne) + (q21_H * nH) + (q21_Hp * nHp) + (q21_oH2 * CoolingReal(0.75) * nH2) + (q21_pH2 * CoolingReal(0.25) * nH2) + (q21_He * nHe) + (q21_Hep * nHep) + (q21_Hepp * nHepp);

    CoolingReal C_01 = C_10 * (g_1/g_0) * cooling_math::exp(-CoolingReal(1.0) * E_10 / T);
    CoolingReal C_02 = C_20 * (g_2/g_0) * cooling_math::exp(-CoolingReal(1.0) * E_20 / T);
    CoolingReal C_12 = C_21 * (g_2/g_1) * cooling_math::exp(-CoolingReal(1.0) * E_21 / T);

    C_01 += (B_01*B_nu_10);
    C_02 += (B_02*B_nu_20);
    C_12 += (B_12*B_nu_21);

    C_10 += (B_10*B_nu_10);
    C_20 += (B_20*B_nu_20);
    C_21 += (B_21*B_nu_21);

    // ! Analytic solution to the 3 level system (see paul goldsmith papers)
    // ! Done this way to avoid numerical errors
    CoolingReal n2_n1_top = (C_12 * (C_01 + C_02)) + (C_02 * (A_10 + C_10));
    CoolingReal n2_n1_bot = ((A_21 + C_21 + C_20) * (C_01 + C_02)) - (C_20 * C_02);
    CoolingReal n2_n1 = n2_n1_top / n2_n1_bot;

    CoolingReal n1_n0_top = ((A_21 + C_21 + C_20) * (C_01 + C_02)) - (C_20*C_02);
    CoolingReal n1_n0_bot = ((A_21 + C_21 + C_20) * (A_10 + C_10)) + (C_20*C_12);
    CoolingReal n1_n0 = n1_n0_top / n1_n0_bot;

    CoolingReal n_0 = CoolingReal(1.0) / (CoolingReal(1.0) + n1_n0 + (n2_n1 * n1_n0));
    CoolingReal n_1 = CoolingReal(1.0) / (CoolingReal(1.0) + n2_n1 + (CoolingReal(1.0)/n1_n0));
    CoolingReal n_2 = CoolingReal(1.0) / (CoolingReal(1.0) + (CoolingReal(1.0)/n2_n1) + ((CoolingReal(1.0)/n2_n1)*(CoolingReal(1.0)/n1_n0)));

    // ! Cooling and heating rates
    CoolingReal cool_0 = (A_10 + (B_10 * B_nu_10)) * E_10 * CoolingReal(KB) * n_1 * n_ion;
    CoolingReal cool_1 = (A_20 + (B_20 * B_nu_20)) * E_20 * CoolingReal(KB) * n_2 * n_ion;
    CoolingReal cool_2 = (A_21 + (B_21 * B_nu_21)) * E_21 * CoolingReal(KB) * n_2 * n_ion;

    CoolingReal heat_0 = B_01 * B_nu_10 * E_10 * CoolingReal(KB) * n_0 * n_ion;
    CoolingReal heat_1 = B_02 * B_nu_20 * E_20 * CoolingReal(KB) * n_0 * n_ion;
    CoolingReal heat_2 = B_12 * B_nu_21 * E_21 * CoolingReal(KB) * n_1 * n_ion;

    // ! Total cooling rate
    three_level_cooling = (cool_0 + cool_1 + cool_2) - (heat_0 + heat_1 + heat_2);

    return three_level_cooling;
}

template<typename Table>
KOKKOS_FUNCTION
CoolingReal two_level(CoolingReal g_0, CoolingReal g_1, CoolingReal lam_10, CoolingReal A_10,
                CoolingReal z, CoolingReal T, CoolingReal n_ion, CoolingReal ne,
                CoolingReal nH, CoolingReal nHp, CoolingReal nHe, CoolingReal nHep,
                CoolingReal nHepp, CoolingReal nH2, int idx_0, const Table& tabData){
    // Harley's two-level ion solver

    // ! Initialize the result
    CoolingReal two_level_cooling = CoolingReal(1E-50);

    CoolingReal T_cmb = CoolingReal(2.725) * (CoolingReal(1.0) + z); //! CMB temperature

    // ! Get log T and enforce bounds
    CoolingReal logT = cooling_math::log10(T);

    // Enforce bounds on temperature
    CoolingReal tmin = CoolingReal(1.0);
    CoolingReal tmax = CoolingReal(4.975);
    CoolingReal delta_temp = CoolingReal(0.025);
    logT = cooling_math::fmax(logT,tmin);
    logT = cooling_math::fmin(logT,tmax);
    CoolingReal lam_10_minus4 = lam_10 * CoolingReal(1E-4);

    CoolingReal nu_10 = CoolingReal(C_CGS) / lam_10_minus4; //! Hz

    CoolingReal E_10 = (CoolingReal(H_PLANCK) * CoolingReal(C_CGS) / lam_10_minus4) / CoolingReal(KB); //! E/K (K)

    CoolingReal B_01 = A_10 * CUBE(lam_10_minus4) * (g_1 / g_0) / (CoolingReal(2.0) * CoolingReal(H_PLANCK) * CoolingReal(C_CGS));

    CoolingReal B_10 = (g_0 / g_1) * B_01;

    //! CMB black body spectrum
    CoolingReal B_nu_10 = line_planck_spectrum(nu_10, T_cmb);

    // ! Find temperature index.
    int itemp_low = (int)cooling_math::floor((logT - tmin)/delta_temp);
    int itemp_high = itemp_low + 1;
    CoolingReal frac_high = (logT - (tmin + ((CoolingReal)itemp_low * delta_temp))) / delta_temp;
    CoolingReal frac_low = CoolingReal(1.0) - frac_high;

    // ! All collison strengths are in cm^3 s^-1
    // ! For collisions with o-H2
    CoolingReal q10_oH2 = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 6)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 6)) * frac_high);

    // ! For collisions with p-H2
    CoolingReal q10_pH2 = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 7)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 7)) * frac_high);

    // ! For collisions with H
    CoolingReal q10_H = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 4)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 4)) * frac_high);

    // ! For collisions with H+
    CoolingReal q10_Hp = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 1)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 1)) * frac_high);

    // ! For collisions with e
    CoolingReal q10_e = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 0)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 0)) * frac_high);

    // ! For collisions with He
    CoolingReal q10_He = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 5)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 5)) * frac_high);

    // ! For collisions with He+
    CoolingReal q10_Hep = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 2)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 2)) * frac_high);

    // ! For collisions with He++
    CoolingReal q10_Hepp = (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_low, 3)) * frac_low) + (CoolingReal(tabData.fs_cool_tab(idx_0, itemp_high, 3)) * frac_high);

    // ! Net collision strengths note the 75-25 ortho-para H2
    CoolingReal C_10 = (q10_e * ne) + (q10_H * nH) + (q10_Hp * nHp) + (q10_oH2 * CoolingReal(0.75) * nH2) + (q10_pH2 * CoolingReal(0.25) * nH2) + (q10_He * nHe) + (q10_Hep * nHep) + (q10_Hepp * nHepp);

    CoolingReal C_01 = C_10 * (g_1/g_0) * cooling_math::exp(-CoolingReal(1.0) * E_10 / T);

    // ! Analytic solution to the 2 level system (modified version of paul goldsmith papers)
    // ! Done this way to avoid numerical errors
    CoolingReal t1 = ( (B_01*B_nu_10) + C_01 );
    CoolingReal t2 = ( A_10 + (B_10*B_nu_10) + C_10 );

    CoolingReal nu_over_nl = t1 / t2;
    CoolingReal n_0 = CoolingReal(1.0) / (nu_over_nl + CoolingReal(1.0));
    CoolingReal n_1 = CoolingReal(1.0) / ((CoolingReal(1.0)/nu_over_nl) + CoolingReal(1.0));

    // ! Cooling and heating rates
    CoolingReal cool_0 = (A_10 + (B_10 * B_nu_10)) * E_10 * CoolingReal(KB) * n_1 * n_ion;

    CoolingReal heat_0 = B_01 * B_nu_10 * E_10 * CoolingReal(KB) * n_0 * n_ion;

    // ! Total cooling rate
    two_level_cooling = (cool_0 - heat_0);

    return two_level_cooling;
}

/* OI fine structure calculation function */
template<typename Table>
KOKKOS_INLINE_FUNCTION
CoolingReal OI_fine_structure(CoolingReal T, CoolingReal n_ion, CoolingReal nH, CoolingReal nHp,
                         CoolingReal ne, CoolingReal nH2, CoolingReal nHe, CoolingReal nHep,
                         CoolingReal nHepp, CoolingReal z, const Table& tabData){
    /* Degeneracy variables */
    CoolingReal g_0 = CoolingReal(5.0);
    CoolingReal g_1 = CoolingReal(3.0);
    CoolingReal g_2 = CoolingReal(1.0);

    /* Transition rates (s^-1) */
    CoolingReal A_10 = CoolingReal(8.910E-5);
    CoolingReal A_20 = CoolingReal(1.340E-10);
    CoolingReal A_21 = CoolingReal(1.750E-5);

    /* Wavelengths (microns) */
    CoolingReal lam_10 = CoolingReal(63.1679);
    CoolingReal lam_20 = CoolingReal(44.0453);
    CoolingReal lam_21 = CoolingReal(145.495);

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
CoolingReal OIII_fine_structure(CoolingReal T, CoolingReal n_ion, CoolingReal nH, CoolingReal nHp,
                           CoolingReal ne, CoolingReal nH2, CoolingReal nHe, CoolingReal nHep,
                           CoolingReal nHepp, CoolingReal z, const Table& tabData){
    /* Degeneracy variables */
    CoolingReal g_0 = CoolingReal(1.0);
    CoolingReal g_1 = CoolingReal(3.0);
    CoolingReal g_2 = CoolingReal(5.0);

    /* Transition rates (s^-1) */
    CoolingReal A_10 = CoolingReal(2.597e-5);
    CoolingReal A_20 = CoolingReal(3.170e-11);
    CoolingReal A_21 = CoolingReal(9.760e-5);

    /* Wavelengths (microns) */
    CoolingReal lam_10 = CoolingReal(88.3323);
    CoolingReal lam_20 = CoolingReal(32.6523);
    CoolingReal lam_21 = CoolingReal(51.8004);

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
CoolingReal CI_fine_structure(CoolingReal T, CoolingReal n_ion, CoolingReal nH, CoolingReal nHp,
                           CoolingReal ne, CoolingReal nH2, CoolingReal nHe, CoolingReal nHep,
                           CoolingReal nHepp, CoolingReal z, const Table& tabData){
    /* Degeneracy variables */
    CoolingReal g_0 = CoolingReal(1.0);
    CoolingReal g_1 = CoolingReal(3.0);
    CoolingReal g_2 = CoolingReal(5.0);

    /* Transition rates (s^-1) */
    CoolingReal A_10 = CoolingReal(7.930E-8);
    CoolingReal A_20 = CoolingReal(1.000E-30);
    CoolingReal A_21 = CoolingReal(2.650E-7);

    /* Wavelengths (microns) */
    CoolingReal lam_10 = CoolingReal(609.590);
    CoolingReal lam_20 = CoolingReal(230.352);
    CoolingReal lam_21 = CoolingReal(370.269);

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
CoolingReal CII_fine_structure(CoolingReal T, CoolingReal n_ion, CoolingReal nH, CoolingReal nHp,
                          CoolingReal ne, CoolingReal nH2, CoolingReal nHe, CoolingReal nHep,
                          CoolingReal nHepp, CoolingReal z, const Table& tabData){
    /* Degeneracy variables */
    CoolingReal g_0 = CoolingReal(2.0);
    CoolingReal g_1 = CoolingReal(4.0);

    /* Transition rates (s^-1) */
    CoolingReal A_10 = CoolingReal(2.290E-6);

    /* Wavelengths (microns) */
    CoolingReal lam_10 = CoolingReal(157.636);

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
CoolingReal NII_fine_structure(CoolingReal T, CoolingReal n_ion, CoolingReal nH, CoolingReal nHp,
                          CoolingReal ne, CoolingReal nH2, CoolingReal nHe, CoolingReal nHep,
                          CoolingReal nHepp, CoolingReal z, const Table& tabData){
    /* Degeneracy variables */
    CoolingReal g_0 = CoolingReal(1.0);
    CoolingReal g_1 = CoolingReal(3.0);
    CoolingReal g_2 = CoolingReal(5.0);

    /* Transition rates (s^-1) */
    CoolingReal A_10 = CoolingReal(2.080E-06);
    CoolingReal A_20 = CoolingReal(1.000E-30);
    CoolingReal A_21 = CoolingReal(7.460E-06);

    /* Wavelengths (microns) */
    CoolingReal lam_10 = CoolingReal(205.244);
    CoolingReal lam_20 = CoolingReal(76.4318);
    CoolingReal lam_21 = CoolingReal(121.767);

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
CoolingReal SiI_fine_structure(CoolingReal T, CoolingReal n_ion, CoolingReal nH, CoolingReal nHp,
                          CoolingReal ne, CoolingReal nH2, CoolingReal nHe, CoolingReal nHep,
                          CoolingReal nHepp, CoolingReal z, const Table& tabData){
    /* Degeneracy variables */
    CoolingReal g_0 = CoolingReal(1.0);
    CoolingReal g_1 = CoolingReal(3.0);
    CoolingReal g_2 = CoolingReal(5.0);

    /* Transition rates (s^-1) */
    CoolingReal A_10 = CoolingReal(8.250E-6);
    CoolingReal A_20 = CoolingReal(3.490E-10);
    CoolingReal A_21 = CoolingReal(4.210E-5);

    /* Wavelengths (microns) */
    CoolingReal lam_10 = CoolingReal(129.641);
    CoolingReal lam_20 = CoolingReal(44.7993);
    CoolingReal lam_21 = CoolingReal(68.4548);

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
CoolingReal SiII_fine_structure(CoolingReal T, CoolingReal n_ion, CoolingReal nH, CoolingReal nHp,
                           CoolingReal ne, CoolingReal nH2, CoolingReal nHe, CoolingReal nHep,
                           CoolingReal nHepp, CoolingReal z, const Table& tabData){
    /* Degeneracy variables */
    CoolingReal g_0 = CoolingReal(2.0);
    CoolingReal g_1 = CoolingReal(4.0);

    /* Transition rates (s^-1) */
    CoolingReal A_10 = CoolingReal(2.131E-4);

    /* Wavelengths (microns) */
    CoolingReal lam_10 = CoolingReal(34.8046);

    return two_level(g_0, g_1,
                    lam_10,
                    A_10,
                    z, T,
                    n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                    11, tabData);
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
CoolingReal NeII_fine_structure(CoolingReal T, CoolingReal n_ion, CoolingReal nH, CoolingReal nHp,
                           CoolingReal ne, CoolingReal nH2, CoolingReal nHe, CoolingReal nHep,
                           CoolingReal nHepp, CoolingReal z, const Table& tabData){
    /* Degeneracy variables */
    CoolingReal g_0 = CoolingReal(4.0);
    CoolingReal g_1 = CoolingReal(2.0);

    /* Transition rates (s^-1) */
    CoolingReal A_10 = CoolingReal(8.590E-3);

    /* Wavelengths (microns) */
    CoolingReal lam_10 = CoolingReal(12.8101);

    return two_level(g_0, g_1,
                    lam_10,
                    A_10,
                    z, T,
                    n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                    10, tabData);
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
CoolingReal FeI_fine_structure(CoolingReal T, CoolingReal n_ion, CoolingReal nH, CoolingReal nHp,
                          CoolingReal ne, CoolingReal nH2, CoolingReal nHe, CoolingReal nHep,
                          CoolingReal nHepp, CoolingReal z, const Table& tabData){
    /* Degeneracy variables */
    CoolingReal g_0 = CoolingReal(9.0);
    CoolingReal g_1 = CoolingReal(7.0);
    CoolingReal g_2 = CoolingReal(5.0);

    /* Transition rates (s^-1) */
    CoolingReal A_10 = CoolingReal(2.510E-3);
    CoolingReal A_20 = CoolingReal(1.000E-30);
    CoolingReal A_21 = CoolingReal(1.560E-3);

    /* Wavelengths (microns) */
    CoolingReal lam_10 = CoolingReal(14.2005);
    CoolingReal lam_20 = CoolingReal(24.0358);
    CoolingReal lam_21 = CoolingReal(34.7038);

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
CoolingReal FeII_fine_structure(CoolingReal T, CoolingReal n_ion, CoolingReal nH, CoolingReal nHp,
                           CoolingReal ne, CoolingReal nH2, CoolingReal nHe, CoolingReal nHep,
                           CoolingReal nHepp, CoolingReal z, const Table& tabData){
    /* Degeneracy variables */
    CoolingReal g_0 = CoolingReal(10.0);
    CoolingReal g_1 = CoolingReal(8.0);
    CoolingReal g_2 = CoolingReal(6.0);

    /* Transition rates (s^-1) */
    CoolingReal A_10 = CoolingReal(2.050E-3);
    CoolingReal A_20 = CoolingReal(1.000E-30);
    CoolingReal A_21 = CoolingReal(1.560E-3);

    /* Wavelengths (microns) */
    CoolingReal lam_10 = CoolingReal(25.9811);
    CoolingReal lam_20 = CoolingReal(14.9731);
    CoolingReal lam_21 = CoolingReal(35.3394);

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
CoolingReal SI_fine_structure(CoolingReal T, CoolingReal n_ion, CoolingReal nH, CoolingReal nHp,
                         CoolingReal ne, CoolingReal nH2, CoolingReal nHe, CoolingReal nHep,
                         CoolingReal nHepp, CoolingReal z, const Table& tabData){
    /* Degeneracy variables */
    CoolingReal g_0 = CoolingReal(5.0);
    CoolingReal g_1 = CoolingReal(3.0);
    CoolingReal g_2 = CoolingReal(1.0);

    /* Transition rates (s^-1) */
    CoolingReal A_10 = CoolingReal(1.400E-3);
    CoolingReal A_20 = CoolingReal(7.050E-8);
    CoolingReal A_21 = CoolingReal(3.020E-4);

    /* Wavelengths (microns) */
    CoolingReal lam_10 = CoolingReal(17.4278);
    CoolingReal lam_20 = CoolingReal(25.2421);
    CoolingReal lam_21 = CoolingReal(56.2957);

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      24, 25, 26, tabData);
}

KOKKOS_INLINE_FUNCTION
CoolingReal dust_recombination_cooling(CoolingReal T, CoolingReal G0, CoolingReal ne, CoolingReal f_dg, CoolingReal nH){
    // Dust recombination cooling
    CoolingReal beta_drc = CoolingReal(0.74) / cooling_math::pow(T,CoolingReal(0.068));
    CoolingReal dust_recomb_cool = (CoolingReal(1.5) * CoolingReal(4.65E-30)) * cooling_math::pow(T,CoolingReal(0.94)) * cooling_math::pow(G0 * cooling_math::sqrt(T) / (CoolingReal(0.5)*ne),beta_drc) * ne * CoolingReal(0.5) * f_dg * nH;

    return dust_recomb_cool;
}

KOKKOS_INLINE_FUNCTION
CoolingReal dust_gas_collisional_cooling(CoolingReal T, CoolingReal G0, CoolingReal xH2, CoolingReal aexp, CoolingReal nH, CoolingReal f_dg){
    CoolingReal dust_hc_const = CoolingReal(1.0E-33); //! For atomic dominated regions
    dust_hc_const += (CoolingReal(3.8E-33) - CoolingReal(1.0E-33)) * cooling_math::fmax(cooling_math::fmin(xH2,CoolingReal(1.0)),CoolingReal(0.0));

    CoolingReal T_dust = CoolingReal(16.4) * cooling_math::pow(CoolingReal(1.7) * G0,CoolingReal(1.0)/CoolingReal(6.0));
    T_dust = cooling_math::fmax( T_dust, CoolingReal(2.725)/aexp ); // ! Limit dust temp minimum to CMB temp

    CoolingReal dust_coll_cool = dust_hc_const * cooling_math::sqrt(T) * (T - T_dust) * ( CoolingReal(1.0) - ( CoolingReal(0.80) * cooling_math::exp(-CoolingReal(75.0)/T) ) );
    return dust_coll_cool * nH * nH * f_dg;
}

KOKKOS_INLINE_FUNCTION
CoolingReal PE_efficiency(CoolingReal G0, CoolingReal T, CoolingReal ne){
    CoolingReal phi_pah = CoolingReal(0.5); //! wolfire+(03)
    CoolingReal fact = G0 * cooling_math::sqrt(T) / (ne * phi_pah);
    CoolingReal PE_efficiency = (CoolingReal(4.9E-2)/(CoolingReal(1.0) + CoolingReal(4E-3) * cooling_math::pow(fact,CoolingReal(0.73)))) + (CoolingReal(3.7E-2) * cooling_math::pow(T/CoolingReal(1E4),CoolingReal(0.7)) / (CoolingReal(1.0) + CoolingReal(2E-4) * fact));
    return cooling_math::fmax(PE_efficiency, CoolingReal(1e-100));
}

KOKKOS_INLINE_FUNCTION
CoolingReal photoelectric_heating(CoolingReal T, CoolingReal G0, CoolingReal ne, CoolingReal f_dg, CoolingReal nH){
    CoolingReal eps_PE = PE_efficiency(G0, T, ne);
    CoolingReal HratePE = CoolingReal(1.50) * CoolingReal(1.3E-24) * eps_PE * G0 * f_dg * nH; //! [erg/cm3/s]
    return cooling_math::fmax(HratePE, CoolingReal(1e-100));
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
CoolingReal cosmic_ray_heating(CoolingReal xe, CoolingReal n_HI, CoolingReal n_HeI, CoolingReal n_H2,
                          CoolingReal ne, CoolingReal xi_h_cr,
                          const CompactIonData& n_and_ion_fracs,
                          const Element* elements,
                          const Table& tabData
                        ){
    CoolingReal q_cr = CoolingReal(6.43) * ( CoolingReal(1.0) + CoolingReal(4.06) * cooling_math::sqrt( xe / (CoolingReal(0.07) + xe) ) ) * CoolingReal(EV_2_ERG); //! Bialy 2019

    CoolingReal phi_s = secondary_cr_rates(xe);
    CoolingReal total_cosmic_ray_ionization_rate = xi_h_cr * (CoolingReal(1.0) + phi_s);

    CoolingReal HI_cosmic_ray_ionization_rate_primary = CoolingReal(1.0) * xi_h_cr;
    CoolingReal He_cosmic_ray_ionization_rate_primary = CoolingReal(1.1) * xi_h_cr;
    CoolingReal H2_cosmic_ray_ionization_rate_primary = CoolingReal(2.0) * xi_h_cr;

    CoolingReal cr_heating_tot = CoolingReal(0.0);

    // HI
    cr_heating_tot +=  ( n_HI * HI_cosmic_ray_ionization_rate_primary * q_cr ); // erg/s/cm^3

    // HeI
    cr_heating_tot +=  ( n_HeI * He_cosmic_ray_ionization_rate_primary * q_cr ); // erg/s/cm^3

    // H2
    cr_heating_tot += ( n_H2 * H2_cosmic_ray_ionization_rate_primary * q_cr ); // erg/s/cm^3

    // Direct heating of the electrons from cosmic rays
    cr_heating_tot += (ne * xi_h_cr * CoolingReal(287.0) * CoolingReal(EV_2_ERG)); // erg/s/cm^3

    // Cosmic ray heating on metals
    // --> Here we skip H and He as it's done separately above
    CoolingReal induced_UV_heat_scale_fac = H2_cosmic_ray_ionization_rate_primary / CoolingReal(1E-16);
    for (int erow = 2; erow < N_TAB_ELEMENTS; erow++) { // Tabulated elements from row 2 on: H and He are rows 0 and 1
        const int i = row_elem(erow);
        CoolingReal n_element = CoolingReal(n_and_ion_fracs.n_element_new[i]);
        if (n_element < MIN_COOL_ION) continue;

        int n_ions = (elements[i].n_ions + elements[i].n_mol);

        for (int j = 0; j < n_ions - 1; j++) { // This comes from ionization --> no heating for final ion
            CoolingReal x_ion = CoolingReal(n_and_ion_fracs[i].ion_fracs_new[j]);

            // This is the primary heating term from ionization
            cr_heating_tot += (n_element * x_ion * q_cr * CoolingReal(tabData.cosmic_ray_ionization_rates(erow, j)) * total_cosmic_ray_ionization_rate); // erg/s/cm^3

            // This is the secondary heating term from induced UV emission
            // Only happend for the ground state
            if (j == 0){
                cr_heating_tot += ( n_element * x_ion
                                    * CoolingReal(tabData.cosmic_ray_ionization_rates_induced_UV_heat(erow))
                                    * CoolingReal(tabData.cosmic_ray_ionization_rates_induced_UV(erow)) * CoolingReal(EV_2_ERG)
                                    * induced_UV_heat_scale_fac ); // erg/s/cm^3
            }
        }
    }

    return cr_heating_tot;
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
CoolingReal photoheating_UVB(const CompactIonData& n_and_ion_fracs,
                        const Element* elements,
                        const Table& tabData){
    // Photoheating contribution from a UVB

    CoolingReal heating_rate = CoolingReal(0.0);

    // Loop over the tabulated elements
    for (int erow = 0; erow < N_TAB_ELEMENTS; erow++) {
        const int i = row_elem(erow);
        CoolingReal n_element = CoolingReal(n_and_ion_fracs.n_element_new[i]);
        if (n_element < MIN_COOL_ION) continue;

        int n_ions = (elements[i].n_ions + elements[i].n_mol);
        for (int j = 0; j < n_ions - 1; j++) { // This comes from ionization --> no heating for final ion
            heating_rate += (n_element * CoolingReal(n_and_ion_fracs[i].ion_fracs_new[j]) * CoolingReal(tabData.HM12_UVB_z(erow, j, 1))); // erg/s/cm^3
        }
    }

    return heating_rate;
}

template<typename Table>
KOKKOS_INLINE_FUNCTION
CoolingReal photoheating_UVB_G0(CoolingReal G0,
                           const CompactIonData& n_and_ion_fracs,
                           const Table& tabData){
    // Photoheating from the G0 background
    // Note that this only impacts the ground state

    CoolingReal heating_rate = CoolingReal(0.0);

    // Loop over the tabulated elements (row 2 onwards: no need for H and He here)
    for (int erow = 2; erow < N_TAB_ELEMENTS; erow++) {
        const int i = row_elem(erow);
        CoolingReal n_element = CoolingReal(n_and_ion_fracs.n_element_new[i]);
        if (n_element < MIN_COOL_ION) continue;

        heating_rate += n_element * CoolingReal(n_and_ion_fracs[i].ion_fracs_new[0]) * CoolingReal(tabData.G0_heating_rates(erow)) * G0; // erg/s/cm^3
    }

    return heating_rate;
}

KOKKOS_INLINE_FUNCTION
CoolingReal Epump(CoolingReal nH, CoolingReal T, CoolingReal xH2, CoolingReal xHI){
    //! Energy converted to heat from UV pumping
    //! see Appendix A http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1990ApJ...365..620B&amp;data_type=PDF_HIGH&amp;whole_paper=YES&amp;type=PRINTER&amp;filetype=.pdf
    CoolingReal Crad = CoolingReal(2.0E-7); //     !radiation de-excitation rate Burton 1990
    CoolingReal Cdex = (CoolingReal(1.0E-12))*((CoolingReal(1.4)*xH2*cooling_math::exp(-CoolingReal(18100.0)/(T + CoolingReal(1200.0)))) + (CoolingReal(1.0)*xHI*cooling_math::exp(-CoolingReal(1000.0)/T)))*cooling_math::sqrt(T)*nH; // !collisional de-excitation rate Burton 1990
    CoolingReal Cfrac = Cdex / (Cdex + Crad);
    CoolingReal Epump_loc = CoolingReal(2.0) * Cfrac * CoolingReal(EV_2_ERG);  //!ergs
    return Epump_loc;
}

KOKKOS_INLINE_FUNCTION
CoolingReal H2_heating_bialy(CoolingReal G0, CoolingReal nH2, CoolingReal nH, CoolingReal T, CoolingReal xHI,
                        CoolingReal xHII, CoolingReal xe, CoolingReal f_dg, CoolingReal xi_h2_cr){

    // Heating from H2 formation and destruction following Bialy 2018
    CoolingReal H2_heating = CoolingReal(0.0);

    CoolingReal D0 = CoolingReal(5.68E-11);
    CoolingReal I_UV = CoolingReal(1.7) * G0;

    CoolingReal E_pump = CoolingReal(1.12) * CoolingReal(EV_2_ERG);
    CoolingReal n_crit = CoolingReal(1.1E5) / cooling_math::sqrt(T/CoolingReal(1000.0));
    CoolingReal ncrit_factor = CoolingReal(1.0) / (CoolingReal(1.0) + (n_crit/nH));

    // heating from H2 pumping
    H2_heating += CoolingReal(9.0) * D0 * I_UV * E_pump * nH2 * ncrit_factor; // erg/s/cm^3

    // heating from H2 photodissociation
    CoolingReal E_pd = CoolingReal(0.4) * CoolingReal(EV_2_ERG);
    H2_heating += D0 * I_UV * E_pd * nH2; // erg/s/cm^3

    // heating from H2 formation --> dust channel
    CoolingReal E_form1_dust = CoolingReal(0.2) * CoolingReal(EV_2_ERG);
    CoolingReal E_form2_dust = CoolingReal(4.48) * CoolingReal(EV_2_ERG) * ncrit_factor;
    CoolingReal H2_formation_rate_dust = alpha_H2_dust(T, f_dg);

    H2_heating += H2_formation_rate_dust * (E_form1_dust + E_form2_dust) * nH * nH * xHI; // erg/s/cm^3

    // heating from H2 formation --> primordial channel
    CoolingReal E_form1_prim = CoolingReal(0.6) * CoolingReal(EV_2_ERG);
    CoolingReal E_form2_prim = CoolingReal(3.13) * CoolingReal(EV_2_ERG) * ncrit_factor;
    CoolingReal H2_formation_rate_prim = alpha_H2_prim(T, xe, xi_h2_cr, G0, xHI, xHII, nH);

    H2_heating += H2_formation_rate_prim * (E_form1_prim + E_form2_prim) * nH * nH * xHI; // erg/s/cm^3

    return cooling_math::fmax(H2_heating, CoolingReal(1e-100));
}

struct CTReaction { bool is_ionization; int stage; int Z; CoolingReal dE; };

template<typename Table>
KOKKOS_INLINE_FUNCTION
CoolingReal CT_heat_cool(CoolingReal T, const CompactIonData& n_and_ion_fracs,
                    const Table& tabData){
    // Heating and cooling from charge exchange reactions

    CoolingReal ct_hc_rate = CoolingReal(0.0);

    if (T > CoolingReal(1E5)){
        return ct_hc_rate;
    }


    CoolingReal loc_nHI = CoolingReal(n_and_ion_fracs.n_element_new[1]) * CoolingReal(n_and_ion_fracs[1].ion_fracs_new[0]);
    CoolingReal loc_nHII = CoolingReal(n_and_ion_fracs.n_element_new[1]) * CoolingReal(n_and_ion_fracs[1].ion_fracs_new[1]);

    // Charge transfer reactions, one table entry per reaction:
    //   is_ionization : true  --> H+ + X(n)  -> X(n+1) + H,  rate scales with nHII
    //                   false --> H  + X(n)  -> X(n-1) + H+, rate scales with nHI
    //   stage         : ionization stage of X, which is also its ion_fracs_new index
    //   Z             : atomic number of X
    //   dE            : energy released by the reaction [eV], negative if absorbed
    static constexpr CTReaction ct_reactions[] = {
        // Helium
        {false, 1,  2,  CoolingReal(10.99)}, // H + He+ -> He + H+

        // Carbon
        {true,  0,  6,   CoolingReal(2.34)}, // H+ + C -> C+ + H
        {false, 1,  6,  -CoolingReal(2.34)}, // H + C+ -> C + H+
        {false, 2,  6,   CoolingReal(4.01)}, // H + C++ -> C+ + H+
        {false, 3,  6,   CoolingReal(5.73)}, // H + C+++ -> C++ + H+
        {false, 4,  6,  CoolingReal(11.3) }, // H + C++++ -> C+++ + H+

        // Nitrogen
        {true,  0,  7,  -CoolingReal(0.94)}, // H+ + N -> N+ + H
        {false, 1,  7,   CoolingReal(0.94)}, // H + N+ -> N + H+
        {false, 2,  7,   CoolingReal(4.56)}, // H + N++ -> N+ + H+
        {false, 3,  7,   CoolingReal(6.40)}, // H + N+++ -> N++ + H+
        {false, 4,  7,  CoolingReal(11.0) }, // H + N++++ -> N+++ + H+

        // Oxygen
        {true,  0,  8,  -CoolingReal(0.02)}, // H+ + O -> O+ + H
        {false, 1,  8,   CoolingReal(0.02)}, // H + O+ -> O + H+
        {false, 2,  8,   CoolingReal(6.65)}, // H + O++ -> O+ + H+
        {false, 3,  8,   CoolingReal(5.00)}, // H + O+++ -> O++ + H+
        {false, 4,  8,   CoolingReal(8.47)}, // H + O++++ -> O+++ + H+

        // Neon
        {false, 3, 10,   CoolingReal(5.82)}, // H + Ne+++ -> Ne++ + H+
        {false, 4, 10,   CoolingReal(8.60)}, // H + Ne++++ -> Ne+++ + H+

        // Magnesium
        {true,  0, 12,   CoolingReal(1.52)}, // H+ + Mg -> Mg+ + H
        {true,  1, 12,  -CoolingReal(1.44)}, // H+ + Mg+ -> Mg++ + H+
        {false, 2, 12,   CoolingReal(1.44)}, // H + Mg++ -> Mg+ + H+
        {false, 3, 12,   CoolingReal(5.73)}, // H + Mg+++ -> Mg++ + H+
        {false, 4, 12,   CoolingReal(8.60)}, // H + Mg++++ -> Mg+++ + H+

        // Silicon
        {true,  0, 14,   CoolingReal(0.12)}, // H+ + Si -> Si+ + H
        {true,  1, 14,  -CoolingReal(2.72)}, // H+ + Si+ -> Si++ + H+
        {false, 2, 14,   CoolingReal(2.72)}, // H + Si++ -> Si+ + H+
        {false, 3, 14,   CoolingReal(4.23)}, // H + Si+++ -> Si++ + H+
        {false, 4, 14,   CoolingReal(7.49)}, // H + Si++++ -> Si+++ + H+

        // Sulfur
        {false, 1, 16,  -CoolingReal(3.24)}, // H + S+ -> S + H+
        {false, 3, 16,   CoolingReal(5.73)}, // H + S+++ -> S++ + H+
        {false, 4, 16,   CoolingReal(8.60)}, // H + S++++ -> S+++ + H+

        // Iron
        {true,  1, 26,  -CoolingReal(2.56)}, // H+ + Fe+ -> Fe++ + H+
        {false, 2, 26,   CoolingReal(2.56)}, // H + Fe++ -> Fe+ + H+
        {false, 3, 26,   CoolingReal(6.30)}, // H + Fe+++ -> Fe++ + H+
        {false, 4, 26,  CoolingReal(10.0) }, // H + Fe++++ -> Fe+++ + H+
    };

    // Don't unroll to save on instructions (saves ~30% of SASS instructions for cooling kernel)
#if defined(__clang__)      // clang: HIP/ROCm, SYCL, clang-CUDA (host and device)
#pragma clang loop unroll(disable)
#elif defined(__CUDA_ARCH__)  // nvcc device pass; its host pass accepts neither spelling
#pragma unroll 1
#endif
    for (const CTReaction& r : ct_reactions) {
        CoolingReal ct_rate = r.is_ionization ? charge_transfer_ionization(r.stage, r.Z, T, tabData)
                                         : charge_transfer_recombination(r.stage, r.Z, T, tabData);
        CoolingReal n_H_partner = r.is_ionization ? loc_nHII : loc_nHI;

        ct_hc_rate += ct_rate * n_H_partner * CoolingReal(n_and_ion_fracs.n_element_new[r.Z]) * CoolingReal(n_and_ion_fracs[r.Z].ion_fracs_new[r.stage]) * r.dE * CoolingReal(EV_2_ERG);
    }

    return ct_hc_rate;
}


template<typename Table>
KOKKOS_FUNCTION
CoolingReal local_photoheating(std::array<Real, N_GROUPS>& dNp,
                          const CompactIonData& n_and_ion_fracs,
                          const Element* elements,
                          const Table& tabData) {
    // Heading due to the local radiation field

    CoolingReal heating_rate = CoolingReal(0.0);

    // looping over number of elements being used (active species)
    // refactor: remove if-statement

    for (int erow = 0; erow < N_TAB_ELEMENTS; erow++) { // Loop over tabulated elements
        const int i = row_elem(erow);
        CoolingReal n_element = CoolingReal(n_and_ion_fracs.n_element_new[i]);
        if (n_element < MIN_COOL_ION) continue;

        int n_ions = (elements[i].n_ions + elements[i].n_mol);

        for (int j = 0; j < n_ions - 1; j++) { // Loop over ions --> no heating for final ion
            CoolingReal n_ion = n_element * CoolingReal(n_and_ion_fracs[i].ion_fracs_new[j]);
            for (int k = 0; k < N_GROUPS; k++) { // Loop over photon groups
                heating_rate += (CoolingReal(dNp[k]) * n_ion * CoolingReal(tabData.cs_ph(erow, j, k, 2)));
            } // End loop over photon groups
        } // End loop over ions
    } // End loop over elements
    return cooling_math::fmax(heating_rate, CoolingReal(1e-100));
}

KOKKOS_INLINE_FUNCTION
CoolingReal CO_cooling_koyama_00(CoolingReal nH, CoolingReal nH2, CoolingReal nHI, CoolingReal nCO, CoolingReal T) {
    // CO cooling
    CoolingReal rate = CoolingReal(0.0);
    CoolingReal T_pivot = CoolingReal(3080.0);
    CoolingReal term_a = (CoolingReal(3.3e6) * cooling_math::pow(T / CoolingReal(1.0e3), CoolingReal(0.75))) / nH;

    // CO Rotational cooling
    CoolingReal rot_numerator = nCO * CoolingReal(4.0) * SQR(CoolingReal(KB) * T) * CoolingReal(9.7e-8);
    CoolingReal rot_denominator = CoolingReal(2.76) * CoolingReal(KB) * (CoolingReal(1.0) + term_a + CoolingReal(1.5) * cooling_math::sqrt(term_a));
    rate += rot_numerator / rot_denominator;

    // CO Vibrational cooling (H2 collisions)
    rate += nH2 * nCO * T_pivot * CoolingReal(KB) * CoolingReal(4.3e-14) * T * cooling_math::exp(-cooling_math::pow(CoolingReal(3.14e5) / T, CoolingReal(0.333))) * cooling_math::exp(-T_pivot / T);

    // CO Vibrational cooling (HI collisions)
    rate += nHI * nCO * T_pivot * CoolingReal(KB) * CoolingReal(3.0e-12) * cooling_math::sqrt(T) * cooling_math::exp(-cooling_math::pow(CoolingReal(2000.0) / T, CoolingReal(3.43))) * cooling_math::exp(-T_pivot / T);

    return cooling_math::fmax(rate, CoolingReal(1e-100));
}

template<typename Table>
KOKKOS_FUNCTION
CoolingReal all_cooling(CoolingReal T, CoolingReal ne, CoolingReal aexp, const CompactIonData& n_and_ion_fracs,
                   const Element* elements,
                   std::array<Real, N_GROUPS>& dNp,
                   CoolingReal nCO,
                   CoolingReal G0_local, CoolingReal UV_background_G0, CoolingReal f_shd_H2,
                   CoolingReal f_dg, CoolingReal xe, CoolingReal xi_h_cr, CoolingReal xi_h2_cr,
                   CoolingReal ss_factor,
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

    CoolingReal net_cooling = CoolingReal(0.0); // heating terms are positive, cooling terms are negative

    CoolingReal nH    = CoolingReal(n_and_ion_fracs.n_element_new[1]);
    CoolingReal xHI   = CoolingReal(n_and_ion_fracs[1].ion_fracs_new[0]);
    CoolingReal xHII  = CoolingReal(n_and_ion_fracs[1].ion_fracs_new[1]);
    CoolingReal xH2   = (elements[1].n_mol > 0) ? CoolingReal(n_and_ion_fracs[1].ion_fracs_new[2]) : CoolingReal(0.0);
    CoolingReal nH_I  = nH * xHI;
    CoolingReal nH_II = nH * xHII;
    CoolingReal nH2   = nH * xH2 * CoolingReal(0.5);

    CoolingReal nHe_I   = CoolingReal(n_and_ion_fracs.n_element_new[2]) * CoolingReal(n_and_ion_fracs[2].ion_fracs_new[0]);
    CoolingReal nHe_II  = CoolingReal(n_and_ion_fracs.n_element_new[2]) * CoolingReal(n_and_ion_fracs[2].ion_fracs_new[1]);
    CoolingReal nHe_III = CoolingReal(n_and_ion_fracs.n_element_new[2]) * CoolingReal(n_and_ion_fracs[2].ion_fracs_new[2]);

    // metal cool smoothing parameters
    CoolingReal metal_cool_smooth_f1 = CoolingReal(0.5) * (cooling_math::tanh( (CoolingReal(5E-3)) * ( T - CoolingReal(1.E4) ) ) + CoolingReal(1.0) );
    CoolingReal metal_cool_smooth_f2 = CoolingReal(0.5) * (cooling_math::tanh( (CoolingReal(5E-3)) * ( (-CoolingReal(1.0) * T) + CoolingReal(1.E4) ) ) + CoolingReal(1.0) );

    // Cooling from primordial species
    net_cooling -= (collisional_ionization_cooling_HI(T) + collisional_excitation_cooling_HI(T)) * ne * nH_I; // cooling_HI
    net_cooling -= recombination_cooling_case_B_HII(T) * ne * nH_II; // cooling_HII
    net_cooling -= collisional_ionization_cooling_HeI(T) * ne * nHe_I; // cooling_HeI
    net_cooling -= (collisional_ionization_cooling_HeII(T) + collisional_excitation_cooling_HeII(T) +
                                 recombination_cooling_case_B_HeII(T) + dielectronic_recombination_cooling_HeII(T)) * ne * nHe_II; // cooling_HeII
    net_cooling -= recombination_cooling_case_B_HeIII(T) * ne * nHe_III; // cooling_HeIII
    net_cooling -= bremmstrahlung(T) * ne * (nH_II + nHe_II + (CoolingReal(4.0) * nHe_III)); // cooling_bremmstrahlung
    net_cooling -= compton_cooling(T,aexp) * ne; // cooling_compton
    net_cooling -= H2_cooling(nH_I, nH2, T); // cooling_H2

    // Metal line cooling -- high temperature
    net_cooling -= get_high_t_cooling_rates(T, ne,
                                            n_and_ion_fracs, elements,
                                            metal_cool_smooth_f1,
                                            tabData);

    // Metal line cooling -- low temperature (fine structure)
    CoolingReal n_ion_fs;
    CoolingReal z = (CoolingReal(1.0) / aexp) - CoolingReal(1.0);
    CoolingReal total_fine_structure = CoolingReal(0.0);
    if (T < CoolingReal(1.1E4)) {
        // CI cooling
        n_ion_fs = CoolingReal(n_and_ion_fracs.n_element_new[6]) * CoolingReal(n_and_ion_fracs[6].ion_fracs_new[0]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += CI_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                    nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // CII cooling
        n_ion_fs = CoolingReal(n_and_ion_fracs.n_element_new[6]) * CoolingReal(n_and_ion_fracs[6].ion_fracs_new[1]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += CII_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                     nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // NII cooling
        n_ion_fs = CoolingReal(n_and_ion_fracs.n_element_new[7]) * CoolingReal(n_and_ion_fracs[7].ion_fracs_new[1]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += NII_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                     nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // OI cooling
        n_ion_fs = CoolingReal(n_and_ion_fracs.n_element_new[8]) * CoolingReal(n_and_ion_fracs[8].ion_fracs_new[0]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += OI_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                    nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // OIII cooling
        n_ion_fs = CoolingReal(n_and_ion_fracs.n_element_new[8]) * CoolingReal(n_and_ion_fracs[8].ion_fracs_new[2]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += OIII_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                      nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // NeII cooling
        n_ion_fs = CoolingReal(n_and_ion_fracs.n_element_new[10]) * CoolingReal(n_and_ion_fracs[10].ion_fracs_new[1]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += NeII_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                      nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // SiI cooling
        n_ion_fs = CoolingReal(n_and_ion_fracs.n_element_new[14]) * CoolingReal(n_and_ion_fracs[14].ion_fracs_new[0]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += SiI_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                     nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // SiII cooling
        n_ion_fs = CoolingReal(n_and_ion_fracs.n_element_new[14]) * CoolingReal(n_and_ion_fracs[14].ion_fracs_new[1]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += SiII_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                      nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // SI cooling
        n_ion_fs = CoolingReal(n_and_ion_fracs.n_element_new[16]) * CoolingReal(n_and_ion_fracs[16].ion_fracs_new[0]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += SI_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                    nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // FeI cooling
        n_ion_fs = CoolingReal(n_and_ion_fracs.n_element_new[26]) * CoolingReal(n_and_ion_fracs[26].ion_fracs_new[0]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += FeI_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                     nH2, nHe_I, nHe_II, nHe_III, z, tabData);

        // FeII cooling
        n_ion_fs = CoolingReal(n_and_ion_fracs.n_element_new[26]) * CoolingReal(n_and_ion_fracs[26].ion_fracs_new[1]);
        if (n_ion_fs > MIN_COOL_ION)
            total_fine_structure += FeII_fine_structure(T, n_ion_fs, nH_I, nH_II, ne,
                                                      nH2, nHe_I, nHe_II, nHe_III, z, tabData);
    }
    // Smooth the fine structure cooling with temeprature if needed
    net_cooling -= total_fine_structure * metal_cool_smooth_f2;

    // Dust cooling
    net_cooling -= dust_recombination_cooling(T, G0_local + UV_background_G0, ne, f_dg, nH);
    net_cooling -= dust_gas_collisional_cooling(T, G0_local + UV_background_G0, xH2*CoolingReal(2.0), aexp, nH, f_dg);

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
