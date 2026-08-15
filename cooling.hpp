// cooling.h
#pragma once
#include <array>
#include <Kokkos_Core.hpp>

#include "constants.hpp"
#include "types.hpp"
#include "molecules.hpp"

// Constants
const double MIN_METAL_COOL_DENS = 1E-10; //minimum number density to calc cooling
const double MIN_COOL_ION = 1.E-10; //minimum number density to calculate cooling for a particular ion

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
double collisional_ionization_cooling_HI(double T){
    // From Cen 1992
    double term_1 = 1.27E-21 * sqrt(T);
    double term_2 = 1.0 /(1.0 + sqrt(T/1E5));
    double term_3 = exp(-157809.1/T);
    double cooling_rate = term_1 * term_2 * term_3;
    return fmax(cooling_rate, 1e-100);
}

KOKKOS_INLINE_FUNCTION
double collisional_ionization_cooling_HeI(double T){
    // From Cen 1992
    double term_1 = 9.38E-22 * sqrt(T);
    double term_2 = 1.0 /(1.0 + sqrt(T/1E5));
    double term_3 = exp(-285335.4/T);
    double cooling_rate = term_1 * term_2 * term_3;
    return fmax(cooling_rate, 1e-100);
}

KOKKOS_INLINE_FUNCTION
double collisional_ionization_cooling_HeII(double T){
    // From Cen 1992
    double term_1 = 4.95E-22 * sqrt(T);
    double term_2 = 1.0 /(1.0 + sqrt(T/1E5));
    double term_3 = exp(-631515.0/T);
    double cooling_rate = term_1 * term_2 * term_3;
    return fmax(cooling_rate, 1e-100);
}

KOKKOS_INLINE_FUNCTION
double recombination_cooling_case_B_HII(double T){
    // From Hui & Gnedin 1997
    double lam_HI = 315614.0 / T;
    double term_1 = 3.435E-30 * T;
    double term_2 = pow(lam_HI,1.97);
    double term_3 = pow(1.0 + pow(lam_HI/2.25,0.376),-3.72);
    double cooling_rate = term_1 * term_2 * term_3;
    return fmax(cooling_rate, 1e-100);
}

KOKKOS_INLINE_FUNCTION
double recombination_cooling_case_B_HeII(double T){
    // From Hui & Gnedin 1997
    double lam_HeI = 570670.0 / T;
    double term_1 = KB * T * 1.26E-14;
    double term_2 = pow(lam_HeI,0.75);
    double cooling_rate = term_1 * term_2;
    return fmax(cooling_rate, 1e-100);
}

KOKKOS_INLINE_FUNCTION
double recombination_cooling_case_B_HeIII(double T){
    // From Hui & Gnedin 1997
    double lam_HeII = 1263030.0 / T;
    double term_1 = 8.0 * 3.435E-30 * T;
    double term_2 = pow(lam_HeII,1.97);
    double term_3 = pow(1.0 + pow(lam_HeII/2.25,0.376),-3.72);
    double cooling_rate = term_1 * term_2 * term_3;
    return fmax(cooling_rate, 1e-100);
}


KOKKOS_INLINE_FUNCTION
double dielectronic_recombination_cooling_HeII(double T){
    // From Black 1981
    double term_1 = 1.24E-13 * pow(T,-1.5);
    double term_2 = exp(-470000.0/T);
    double term_3 = 1.0 + (0.3 * exp(-94000.0/T));
    double cooling_rate = term_1 * term_2 * term_3;
    return fmax(cooling_rate, 1e-100);
}


KOKKOS_INLINE_FUNCTION
double collisional_excitation_cooling_HI(double T){
    // From Seon 2020
    double cooling_rate = 4.13e-19 * exp(-117744.0/T);
    return fmax(cooling_rate, 1e-100);
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
double collisional_excitation_cooling_HeII(double T){
    // From Cen 1992
    double term_1 = 5.54E-17 * pow(T,-0.397);
    double term_2 = 1.0 /(1.0 + sqrt(T/1E5));
    double term_3 = exp(-473638.0/T);
    double cooling_rate = term_1 * term_2 * term_3;
    return fmax(cooling_rate, 1e-100);
}

KOKKOS_INLINE_FUNCTION
double bremmstrahlung(double T){
    // Osterbrock and Ferland 2006
    double cooling_rate = 1.42E-27 * sqrt(T);
    return fmax(cooling_rate, 1e-100);
}

KOKKOS_INLINE_FUNCTION
double compton_cooling(double T, double a){

    double Tcmb_over_a = 2.727/a;

    // Haiman+ 1996
    double term_1 = 1.017E-37;
    double term_2 = pow(Tcmb_over_a, 4);
    double term_3 = T - Tcmb_over_a;
    double cooling_rate = term_1 * term_2 * term_3;
    return cooling_rate;
}

KOKKOS_INLINE_FUNCTION
double H2_cooling(double nH, double nH2, double T){
    // From Moseley 2021
    double T3 = 1E-3 * T;
    double n1 = 50.0;
    double n2 = 450.0;
    double n3 = 25.0;
    double n4 = 900.0;

    double x1 = nH + (5.0 * nH2);
    double x2 = nH + (4.5 * nH2);
    double x3 = nH + (0.75 * nH2);
    double x4 = nH + (0.05 * nH2);

    double f1 = 1.1E-25 * sqrt(T3) * exp(-0.51 / T3);
    f1 = f1 * (((0.7 * x1) / (1.0 + (x1/n1))) + ((0.3 * x1)/(1.0 + (x1/(10.0*n1)))));

    double f2 = 2.0E-25 * T3 * exp(-1.0 / T3);
    f2 = f2 * (((0.35 * x2) / (1.0 + (x2/n2))) + ((0.65 * x2)/(1.0 + (x2/(10.0*n2)))));

    double f3 = 2.4E-24 * (pow(T3,1.50)) * exp(-2.0 / T3);
    f3 = f3 * (x3 / (1.0 + (x3/n3)));

    double f4 = 1.7E-23 * (pow(T3,1.50)) * exp(-4.0 / T3);
    f4 = f4 * (((0.45 * x4) / (1.0 + (x4/n4))) + ((0.55 * x4)/(1.0 + (x4/(10.0*n4)))));

    return fmax(nH2 * (f1 + f2 + f3 + f4), 1e-100);
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

KOKKOS_FUNCTION
double get_high_t_cooling_rates(double T, double ne,
                                const CompactIonData& n_and_ion_fracs,
                                const Element* elements,
                                double temp_smooth,
                                const TabulatedData& tabData){
    // Computes the high-temperature cooling rates
    // derived from Harley's custom cloudy models

    double loc_T = fmax(T,1000.0);
    double log_T = log10(loc_T);
    double t_min = 3.0;
    double t_max = 9.0;
    double dt = 0.05;
    double total_metal_cooling_rate = 0.0;
    double two_e4_overT = 2000.0/loc_T;
    double t_scale_fac = exp(-1.0 * pow(two_e4_overT, 5));

    // bounds for temperature --> no cooling
    if (log_T < t_min){
        return total_metal_cooling_rate;
    }

    // Still cool above 10^9 K but set a bound
    if (log_T > t_max){
        log_T = t_max;
    }

    // Prepare for 1D interpolation
    int idx_low = (int)floor((log_T - t_min)/dt);

	// Uniform grid: high_t_cooling_temp[k] = t_min + k * dt
    double frac_high = (log_T - (t_min + idx_low * dt)) / dt;
    double frac_low = 1.0 - frac_high;

    // Loop over all ions -- not including H and He
    for (int i = 3; i < 27; i++) {
        if (n_and_ion_fracs.n_element_new[i] > MIN_METAL_COOL_DENS){
            // Get the element number density
            double n_element = n_and_ion_fracs.n_element_new[i];
            const int ei = elem_row(i);
            for (int j = 0; j < (elements[i].n_ions + elements[i].n_mol); j++) {
                // Get the element ion fraction
                double n_ion = n_and_ion_fracs[i].ion_fracs_new[j];

                // Interpolate the cooling rate for the ion
                double loc_cooling_rate = (frac_low * tabData.high_t_cooling_rates(idx_low, ei, j));
                loc_cooling_rate += (frac_high * tabData.high_t_cooling_rates(idx_low+1, ei, j));
                loc_cooling_rate = exp10(loc_cooling_rate);

                // Smooth with temperature if necessary
                double loc_temp_smooth = 1.0;
                if (tabData.high_t_cooling_rates_tflag(ei,j)) {
                    loc_temp_smooth = temp_smooth;
                }

                // Multiply cooling rate by ion and electron number densities
                total_metal_cooling_rate += (n_element * n_ion * ne * loc_cooling_rate * loc_temp_smooth);
            }
        }
    }

    return fmax(total_metal_cooling_rate * t_scale_fac,1e-100);
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

KOKKOS_FUNCTION
double three_level(double g_0, double g_1, double g_2,
                   double lam_10, double lam_20, double lam_21, 
                   double A_10, double A_20, double A_21, 
                   double z, double T, double n_ion, 
                   double ne, double nH, double nHp, 
                   double nHe, double nHep, double nHepp, 
                   double nH2, 
                   int idx_0, int idx_1, int idx_2,
                   const TabulatedData& tabData
                ){
    // Harley's three-level ion solver

    // ! Initialize the result
    double three_level_cooling = 1E-50;

    double T_cmb = 2.725 * (1.0 + z); //! CMB temperature

    // ! Get log T and enforce bounds
    double logT = log10(T);

    // Enforce bounds on temperature
    double tmin = 1.0;
    double tmax = 4.975;
    double delta_temp = 0.025;
    logT = fmax(logT,1.0);
    logT = fmin(logT,tmax);
    double lam_10_minus4 = lam_10 * 1E-4;
    double lam_20_minus4 = lam_20 * 1E-4;
    double lam_21_minus4 = lam_21 * 1E-4;

    double nu_10 = C_CGS / lam_10_minus4; //! Hz
    double nu_20 = C_CGS / lam_20_minus4; //! Hz
    double nu_21 = C_CGS / lam_21_minus4; //! Hz

    double E_10 = (H_PLANCK * C_CGS / lam_10_minus4) / KB; //! E/K (K)
    double E_20 = (H_PLANCK * C_CGS / lam_20_minus4) / KB; //! E/K (K)
    double E_21 = (H_PLANCK * C_CGS / lam_21_minus4) / KB; //! E/K (K)

    double B_01 = A_10 * (pow(lam_10_minus4, 3)) * (g_1 / g_0) / (2.0 * H_PLANCK * C_CGS);
    double B_02 = A_20 * (pow(lam_20_minus4, 3)) * (g_2 / g_0) / (2.0 * H_PLANCK * C_CGS);
    double B_12 = A_21 * (pow(lam_21_minus4, 3)) * (g_2 / g_1) / (2.0 * H_PLANCK * C_CGS);

    double B_10 = (g_0 / g_1) * B_01;
    double B_20 = (g_0 / g_2) * B_02;
    double B_21 = (g_1 / g_2) * B_12;

    //! CMB black body spectrum
    double B_nu_10 = (2.0 * H_PLANCK * pow(nu_10, 3) / (C_CGS*C_CGS)) / (exp(H_PLANCK * nu_10 / (KB * T_cmb)) - 1.0);
    double B_nu_20 = (2.0 * H_PLANCK * pow(nu_20, 3) / (C_CGS*C_CGS)) / (exp(H_PLANCK * nu_20 / (KB * T_cmb)) - 1.0);
    double B_nu_21 = (2.0 * H_PLANCK * pow(nu_21, 3) / (C_CGS*C_CGS)) / (exp(H_PLANCK * nu_21 / (KB * T_cmb)) - 1.0);

    // ! Find temperature index.
    int itemp_low = (int)floor((logT - tmin)/delta_temp);
    int itemp_high = itemp_low + 1; 
    double frac_high = (logT - (tmin + ((double)itemp_low * delta_temp))) / delta_temp;
    double frac_low = 1.0 - frac_high;

    // ! All collison strengths are in cm^3 s^-1
    // ! For collisions with o-H2
    double q10_oH2 = (tabData.fs_cool_tab(idx_0, itemp_low, 6) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 6) * frac_high);
    double q20_oH2 = (tabData.fs_cool_tab(idx_1, itemp_low, 6) * frac_low) + (tabData.fs_cool_tab(idx_1, itemp_high, 6) * frac_high);
    double q21_oH2 = (tabData.fs_cool_tab(idx_2, itemp_low, 6) * frac_low) + (tabData.fs_cool_tab(idx_2, itemp_high, 6) * frac_high);

    // ! For collisions with p-H2
    double q10_pH2 = (tabData.fs_cool_tab(idx_0, itemp_low, 7) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 7) * frac_high);
    double q20_pH2 = (tabData.fs_cool_tab(idx_1, itemp_low, 7) * frac_low) + (tabData.fs_cool_tab(idx_1, itemp_high, 7) * frac_high);
    double q21_pH2 = (tabData.fs_cool_tab(idx_2, itemp_low, 7) * frac_low) + (tabData.fs_cool_tab(idx_2, itemp_high, 7) * frac_high);

    // ! For collisions with H
    double q10_H = (tabData.fs_cool_tab(idx_0, itemp_low, 4) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 4) * frac_high);
    double q20_H = (tabData.fs_cool_tab(idx_1, itemp_low, 4) * frac_low) + (tabData.fs_cool_tab(idx_1, itemp_high, 4) * frac_high);
    double q21_H = (tabData.fs_cool_tab(idx_2, itemp_low, 4) * frac_low) + (tabData.fs_cool_tab(idx_2, itemp_high, 4) * frac_high);

    // ! For collisions with H+
    double q10_Hp = (tabData.fs_cool_tab(idx_0, itemp_low, 1) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 1) * frac_high);
    double q20_Hp = (tabData.fs_cool_tab(idx_1, itemp_low, 1) * frac_low) + (tabData.fs_cool_tab(idx_1, itemp_high, 1) * frac_high);
    double q21_Hp = (tabData.fs_cool_tab(idx_2, itemp_low, 1) * frac_low) + (tabData.fs_cool_tab(idx_2, itemp_high, 1) * frac_high);

    // ! For collisions with e
    double q10_e = (tabData.fs_cool_tab(idx_0, itemp_low, 0) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 0) * frac_high);
    double q20_e = (tabData.fs_cool_tab(idx_1, itemp_low, 0) * frac_low) + (tabData.fs_cool_tab(idx_1, itemp_high, 0) * frac_high);
    double q21_e = (tabData.fs_cool_tab(idx_2, itemp_low, 0) * frac_low) + (tabData.fs_cool_tab(idx_2, itemp_high, 0) * frac_high);

    // ! For collisions with He
    double q10_He = (tabData.fs_cool_tab(idx_0, itemp_low, 5) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 5) * frac_high);
    double q20_He = (tabData.fs_cool_tab(idx_1, itemp_low, 5) * frac_low) + (tabData.fs_cool_tab(idx_1, itemp_high, 5) * frac_high);
    double q21_He = (tabData.fs_cool_tab(idx_2, itemp_low, 5) * frac_low) + (tabData.fs_cool_tab(idx_2, itemp_high, 5) * frac_high);

    // ! For collisions with He+
    double q10_Hep = (tabData.fs_cool_tab(idx_0, itemp_low, 2) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 2) * frac_high);
    double q20_Hep = (tabData.fs_cool_tab(idx_1, itemp_low, 2) * frac_low) + (tabData.fs_cool_tab(idx_1, itemp_high, 2) * frac_high);
    double q21_Hep = (tabData.fs_cool_tab(idx_2, itemp_low, 2) * frac_low) + (tabData.fs_cool_tab(idx_2, itemp_high, 2) * frac_high);

    // ! For collisions with He++
    double q10_Hepp = (tabData.fs_cool_tab(idx_0, itemp_low, 3) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 3) * frac_high);
    double q20_Hepp = (tabData.fs_cool_tab(idx_1, itemp_low, 3) * frac_low) + (tabData.fs_cool_tab(idx_1, itemp_high, 3) * frac_high);
    double q21_Hepp = (tabData.fs_cool_tab(idx_2, itemp_low, 3) * frac_low) + (tabData.fs_cool_tab(idx_2, itemp_high, 3) * frac_high);

    // ! Net collision strengths note the 75-25 ortho-para H2
    double C_10 = (q10_e * ne) + (q10_H * nH) + (q10_Hp * nHp) + (q10_oH2 * 0.75 * nH2) + (q10_pH2 * 0.25 * nH2) + (q10_He * nHe) + (q10_Hep * nHep) + (q10_Hepp * nHepp);
    double C_20 = (q20_e * ne) + (q20_H * nH) + (q20_Hp * nHp) + (q20_oH2 * 0.75 * nH2) + (q20_pH2 * 0.25 * nH2) + (q20_He * nHe) + (q20_Hep * nHep) + (q20_Hepp * nHepp);
    double C_21 = (q21_e * ne) + (q21_H * nH) + (q21_Hp * nHp) + (q21_oH2 * 0.75 * nH2) + (q21_pH2 * 0.25 * nH2) + (q21_He * nHe) + (q21_Hep * nHep) + (q21_Hepp * nHepp);

    double C_01 = C_10 * (g_1/g_0) * exp(-1.0 * E_10 / T);
    double C_02 = C_20 * (g_2/g_0) * exp(-1.0 * E_20 / T);
    double C_12 = C_21 * (g_2/g_1) * exp(-1.0 * E_21 / T);

    C_01 += (B_01*B_nu_10);
    C_02 += (B_02*B_nu_20);
    C_12 += (B_12*B_nu_21);

    C_10 += (B_10*B_nu_10);
    C_20 += (B_20*B_nu_20);
    C_21 += (B_21*B_nu_21);

    // ! Analytic solution to the 3 level system (see paul goldsmith papers)
    // ! Done this way to avoid numerical errors
    double n2_n1_top = (C_12 * (C_01 + C_02)) + (C_02 * (A_10 + C_10));
    double n2_n1_bot = ((A_21 + C_21 + C_20) * (C_01 + C_02)) - (C_20 * C_02);
    double n2_n1 = n2_n1_top / n2_n1_bot;

    double n1_n0_top = ((A_21 + C_21 + C_20) * (C_01 + C_02)) - (C_20*C_02);
    double n1_n0_bot = ((A_21 + C_21 + C_20) * (A_10 + C_10)) + (C_20*C_12);
    double n1_n0 = n1_n0_top / n1_n0_bot;

    double n_0 = 1.0 / (1.0 + n1_n0 + (n2_n1 * n1_n0));
    double n_1 = 1.0 / (1.0 + n2_n1 + (1.0/n1_n0));
    double n_2 = 1.0 / (1.0 + (1.0/n2_n1) + ((1.0/n2_n1)*(1.0/n1_n0)));

    // ! Cooling and heating rates
    double cool_0 = (A_10 + (B_10 * B_nu_10)) * E_10 * KB * n_1 * n_ion;
    double cool_1 = (A_20 + (B_20 * B_nu_20)) * E_20 * KB * n_2 * n_ion;
    double cool_2 = (A_21 + (B_21 * B_nu_21)) * E_21 * KB * n_2 * n_ion;

    double heat_0 = B_01 * B_nu_10 * E_10 * KB * n_0 * n_ion;
    double heat_1 = B_02 * B_nu_20 * E_20 * KB * n_0 * n_ion;
    double heat_2 = B_12 * B_nu_21 * E_21 * KB * n_1 * n_ion;

    // ! Total cooling rate
    three_level_cooling = (cool_0 + cool_1 + cool_2) - (heat_0 + heat_1 + heat_2);

    return three_level_cooling;
}

KOKKOS_FUNCTION
double two_level(double g_0, double g_1, double lam_10, double A_10,
                double z, double T, double n_ion, double ne, 
                double nH, double nHp, double nHe, double nHep, 
                double nHepp, double nH2, int idx_0, const TabulatedData& tabData){
    // Harley's two-level ion solver

    // ! Initialize the result
    double two_level_cooling = 1E-50;

    double T_cmb = 2.725 * (1.0 + z); //! CMB temperature

    // ! Get log T and enforce bounds
    double logT = log10(T);

    // Enforce bounds on temperature
    double tmin = 1.0;
    double tmax = 4.975;
    double delta_temp = 0.025;
    logT = fmax(logT,tmin);
    logT = fmin(logT,tmax);
    double lam_10_minus4 = lam_10 * 1E-4; 

    double nu_10 = C_CGS / lam_10_minus4; //! Hz

    double E_10 = (H_PLANCK * C_CGS / lam_10_minus4) / KB; //! E/K (K)

    double B_01 = A_10 * pow(lam_10_minus4, 3) * (g_1 / g_0) / (2.0 * H_PLANCK * C_CGS);

    double B_10 = (g_0 / g_1) * B_01;

    //! CMB black body spectrum
    double B_nu_10 = (2.0 * H_PLANCK * pow(nu_10, 3) / (C_CGS*C_CGS)) / (exp(H_PLANCK * nu_10 / (KB * T_cmb)) - 1.0);

    // ! Find temperature index.
    int itemp_low = (int)floor((logT - tmin)/delta_temp);
    int itemp_high = itemp_low + 1;
    double frac_high = (logT - (tmin + ((double)itemp_low * delta_temp))) / delta_temp;
    double frac_low = 1.0 - frac_high;

    // ! All collison strengths are in cm^3 s^-1
    // ! For collisions with o-H2
    double q10_oH2 = (tabData.fs_cool_tab(idx_0, itemp_low, 6) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 6) * frac_high);

    // ! For collisions with p-H2
    double q10_pH2 = (tabData.fs_cool_tab(idx_0, itemp_low, 7) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 7) * frac_high);

    // ! For collisions with H
    double q10_H = (tabData.fs_cool_tab(idx_0, itemp_low, 4) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 4) * frac_high);

    // ! For collisions with H+
    double q10_Hp = (tabData.fs_cool_tab(idx_0, itemp_low, 1) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 1) * frac_high);

    // ! For collisions with e
    double q10_e = (tabData.fs_cool_tab(idx_0, itemp_low, 0) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 0) * frac_high);

    // ! For collisions with He
    double q10_He = (tabData.fs_cool_tab(idx_0, itemp_low, 5) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 5) * frac_high);

    // ! For collisions with He+
    double q10_Hep = (tabData.fs_cool_tab(idx_0, itemp_low, 2) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 2) * frac_high);

    // ! For collisions with He++
    double q10_Hepp = (tabData.fs_cool_tab(idx_0, itemp_low, 3) * frac_low) + (tabData.fs_cool_tab(idx_0, itemp_high, 3) * frac_high);

    // ! Net collision strengths note the 75-25 ortho-para H2
    double C_10 = (q10_e * ne) + (q10_H * nH) + (q10_Hp * nHp) + (q10_oH2 * 0.75 * nH2) + (q10_pH2 * 0.25 * nH2) + (q10_He * nHe) + (q10_Hep * nHep) + (q10_Hepp * nHepp);

    double C_01 = C_10 * (g_1/g_0) * exp(-1.0 * E_10 / T);

    // ! Analytic solution to the 2 level system (modified version of paul goldsmith papers)
    // ! Done this way to avoid numerical errors
    double t1 = ( (B_01*B_nu_10) + C_01 );
    double t2 = ( A_10 + (B_10*B_nu_10) + C_10 );

    double nu_over_nl = t1 / t2;
    double n_0 = 1.0 / (nu_over_nl + 1.0);
    double n_1 = 1.0 / ((1.0/nu_over_nl) + 1.0);

    // ! Cooling and heating rates
    double cool_0 = (A_10 + (B_10 * B_nu_10)) * E_10 * KB * n_1 * n_ion;

    double heat_0 = B_01 * B_nu_10 * E_10 * KB * n_0 * n_ion;

    // ! Total cooling rate
    two_level_cooling = (cool_0 - heat_0);

    return two_level_cooling;
}

/* OI fine structure calculation function */
KOKKOS_INLINE_FUNCTION
double OI_fine_structure(double T, double n_ion, double nH, double nHp, 
                         double ne, double nH2, double nHe, double nHep, 
                         double nHepp, double z, const TabulatedData& tabData){
    /* Degeneracy variables */
    double g_0 = 5.0;
    double g_1 = 3.0;
    double g_2 = 1.0;

    /* Transition rates (s^-1) */
    double A_10 = 8.910E-5;
    double A_20 = 1.340E-10;
    double A_21 = 1.750E-5;

    /* Wavelengths (microns) */
    double lam_10 = 63.1679;
    double lam_20 = 44.0453;
    double lam_21 = 145.495;

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      7, 8, 9, tabData);
}

/* OIII fine structure calculation function */
KOKKOS_INLINE_FUNCTION
double OIII_fine_structure(double T, double n_ion, double nH, double nHp, 
                           double ne, double nH2, double nHe, double nHep, 
                           double nHepp, double z, const TabulatedData& tabData){
    /* Degeneracy variables */
    double g_0 = 1.0;
    double g_1 = 3.0;
    double g_2 = 5.0;

    /* Transition rates (s^-1) */
    double A_10 = 2.597e-5;
    double A_20 = 3.170e-11;
    double A_21 = 9.760e-5;

    /* Wavelengths (microns) */
    double lam_10 = 88.3323;
    double lam_20 = 32.6523;
    double lam_21 = 51.8004;

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      12, 13, 14, tabData);
}

KOKKOS_INLINE_FUNCTION
double CI_fine_structure(double T, double n_ion, double nH, double nHp, 
                           double ne, double nH2, double nHe, double nHep, 
                           double nHepp, double z, const TabulatedData& tabData){
    /* Degeneracy variables */
    double g_0 = 1.0;
    double g_1 = 3.0;
    double g_2 = 5.0;

    /* Transition rates (s^-1) */
    double A_10 = 7.930E-8;
    double A_20 = 1.000E-30;
    double A_21 = 2.650E-7;

    /* Wavelengths (microns) */
    double lam_10 = 609.590;
    double lam_20 = 230.352;
    double lam_21 = 370.269;

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      1, 2, 3, tabData);
}

KOKKOS_INLINE_FUNCTION
double CII_fine_structure(double T, double n_ion, double nH, double nHp, 
                          double ne, double nH2, double nHe, double nHep, 
                          double nHepp, double z, const TabulatedData& tabData){
    /* Degeneracy variables */
    double g_0 = 2.0;
    double g_1 = 4.0;

    /* Transition rates (s^-1) */
    double A_10 = 2.290E-6;

    /* Wavelengths (microns) */
    double lam_10 = 157.636;

    return two_level(g_0, g_1,
                    lam_10,
                    A_10,
                    z, T,
                    n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                    0,
                    tabData);
}

KOKKOS_INLINE_FUNCTION
double NII_fine_structure(double T, double n_ion, double nH, double nHp, 
                          double ne, double nH2, double nHe, double nHep, 
                          double nHepp, double z, const TabulatedData& tabData){
    /* Degeneracy variables */
    double g_0 = 1.0;
    double g_1 = 3.0;
    double g_2 = 5.0;

    /* Transition rates (s^-1) */
    double A_10 = 2.080E-06;
    double A_20 = 1.000E-30;
    double A_21 = 7.460E-06;

    /* Wavelengths (microns) */
    double lam_10 = 205.244;
    double lam_20 = 76.4318;
    double lam_21 = 121.767;

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      4, 5, 6, tabData);
}

KOKKOS_INLINE_FUNCTION
double SiI_fine_structure(double T, double n_ion, double nH, double nHp, 
                          double ne, double nH2, double nHe, double nHep, 
                          double nHepp, double z, const TabulatedData& tabData){
    /* Degeneracy variables */
    double g_0 = 1.0;
    double g_1 = 3.0;
    double g_2 = 5.0;

    /* Transition rates (s^-1) */
    double A_10 = 8.250E-6;
    double A_20 = 3.490E-10;
    double A_21 = 4.210E-5;

    /* Wavelengths (microns) */
    double lam_10 = 129.641;
    double lam_20 = 44.7993;
    double lam_21 = 68.4548;

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      21, 22, 23, tabData);
}

KOKKOS_INLINE_FUNCTION
double SiII_fine_structure(double T, double n_ion, double nH, double nHp, 
                           double ne, double nH2, double nHe, double nHep, 
                           double nHepp, double z, const TabulatedData& tabData){
    /* Degeneracy variables */
    double g_0 = 2.0;
    double g_1 = 4.0;

    /* Transition rates (s^-1) */
    double A_10 = 2.131E-4;

    /* Wavelengths (microns) */
    double lam_10 = 34.8046;

    return two_level(g_0, g_1,
                    lam_10,
                    A_10,
                    z, T,
                    n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                    11, tabData);
}

KOKKOS_INLINE_FUNCTION
double NeII_fine_structure(double T, double n_ion, double nH, double nHp, 
                           double ne, double nH2, double nHe, double nHep, 
                           double nHepp, double z, const TabulatedData& tabData){
    /* Degeneracy variables */
    double g_0 = 4.0;
    double g_1 = 2.0;

    /* Transition rates (s^-1) */
    double A_10 = 8.590E-3;

    /* Wavelengths (microns) */
    double lam_10 = 12.8101;

    return two_level(g_0, g_1,
                    lam_10,
                    A_10,
                    z, T,
                    n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                    10, tabData);
}

KOKKOS_INLINE_FUNCTION
double FeI_fine_structure(double T, double n_ion, double nH, double nHp, 
                          double ne, double nH2, double nHe, double nHep, 
                          double nHepp, double z, const TabulatedData& tabData){
    /* Degeneracy variables */
    double g_0 = 9.0;
    double g_1 = 7.0;
    double g_2 = 5.0;
    
    /* Transition rates (s^-1) */
    double A_10 = 2.510E-3;
    double A_20 = 1.000E-30;
    double A_21 = 1.560E-3;

    /* Wavelengths (microns) */
    double lam_10 = 24.0358;
    double lam_20 = 14.2005;
    double lam_21 = 34.7038;

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      15, 16, 17, tabData);
}

KOKKOS_INLINE_FUNCTION
double FeII_fine_structure(double T, double n_ion, double nH, double nHp, 
                           double ne, double nH2, double nHe, double nHep, 
                           double nHepp, double z, const TabulatedData& tabData){
    /* Degeneracy variables */
    double g_0 = 10.0;
    double g_1 = 8.0;
    double g_2 = 6.0;

    /* Transition rates (s^-1) */
    double A_10 = 2.050E-3;
    double A_20 = 1.000E-30;
    double A_21 = 1.560E-3;

    /* Wavelengths (microns) */
    double lam_10 = 25.9811;
    double lam_20 = 14.9731;
    double lam_21 = 35.3394;

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      18, 19, 20, tabData);
}

KOKKOS_INLINE_FUNCTION
double SI_fine_structure(double T, double n_ion, double nH, double nHp, 
                         double ne, double nH2, double nHe, double nHep, 
                         double nHepp, double z, const TabulatedData& tabData){
    /* Degeneracy variables */
    double g_0 = 5.0;
    double g_1 = 3.0;
    double g_2 = 1.0;

    /* Transition rates (s^-1) */
    double A_10 = 1.400E-3;
    double A_20 = 7.050E-8;
    double A_21 = 3.020E-4;

    /* Wavelengths (microns) */
    double lam_10 = 25.2421;
    double lam_20 = 17.4278;
    double lam_21 = 56.2957;

    /* Call three_level function */
    return three_level(g_0, g_1, g_2,
                      lam_10, lam_20, lam_21,
                      A_10, A_20, A_21,
                      z, T,
                      n_ion, ne, nH, nHp, nHe, nHep, nHepp, nH2,
                      24, 25, 26, tabData);
}

KOKKOS_INLINE_FUNCTION
double dust_recombination_cooling(double T, double G0, double ne, double f_dg, double nH){
    // Dust recombination cooling
    double beta_drc = 0.74 / pow(T,0.068);
    double dust_recomb_cool = (1.5 * 4.65E-30) * pow(T,0.94) * pow(G0 * sqrt(T) / (0.5*ne),beta_drc) * ne * 0.5 * f_dg * nH;

    return dust_recomb_cool;
}

KOKKOS_INLINE_FUNCTION
double dust_gas_collisional_cooling(double T, double G0, double xH2, double aexp, double nH, double f_dg){
    double dust_hc_const = 1.0E-33; //! For atomic dominated regions
    dust_hc_const += (3.8E-33 - 1.0E-33) * fmax(fmin(xH2,1.0),0.0);

    double T_dust = 16.4 * pow(1.7 * G0,1.0/6.0);
    T_dust = fmax( T_dust, 2.725/aexp ); // ! Limit dust temp minimum to CMB temp

    double dust_coll_cool = dust_hc_const * sqrt(T) * (T - T_dust) * ( 1.0 - ( 0.80 * exp(-75.0/T) ) );
    return dust_coll_cool * nH * nH * f_dg;
}

KOKKOS_INLINE_FUNCTION
double PE_efficiency(double G0, double T, double ne){
    double phi_pah = 0.5; //! wolfire+(03)
    double fact = G0 * sqrt(T) / (ne * phi_pah);
    double PE_efficiency = (4.9E-2/(1.0 + 4E-3 * pow(fact,0.73))) + (3.7E-2 * pow(T/1E4,0.7) / (1.0 + 2E-4 * fact));
    return fmax(PE_efficiency, 1e-100);
}

KOKKOS_INLINE_FUNCTION
double photoelectric_heating(double T, double G0, double ne, double f_dg, double nH){
    double eps_PE = PE_efficiency(G0, T, ne);
    double HratePE = 1.50 * 1.3E-24 * eps_PE * G0 * f_dg * nH; //! [erg/cm3/s]
    return fmax(HratePE, 1e-100);
}

KOKKOS_INLINE_FUNCTION
double cosmic_ray_heating(double xe, double n_HI, double n_HeI, double n_H2,
                          double ne, double xi_h_cr,
                          const CompactIonData& n_and_ion_fracs,
                          const Element* elements,
                          const TabulatedData& tabData
                        ){
    double q_cr = 6.43 * ( 1.0 + 4.06 * sqrt( xe / (0.07 + xe) ) ) * EV_2_ERG; //! Bialy 2019

    double phi_s = secondary_cr_rates(xe);
    double total_cosmic_ray_ionization_rate = xi_h_cr * (1.0 + phi_s);

    double HI_cosmic_ray_ionization_rate_primary = 1.0 * xi_h_cr;
    double He_cosmic_ray_ionization_rate_primary = 1.1 * xi_h_cr;
    double H2_cosmic_ray_ionization_rate_primary = 2.0 * xi_h_cr;

    double cr_heating_tot = 0.0;

    // HI
    cr_heating_tot +=  ( n_HI * HI_cosmic_ray_ionization_rate_primary * q_cr ); // erg/s/cm^3

    // HeI
    cr_heating_tot +=  ( n_HeI * He_cosmic_ray_ionization_rate_primary * q_cr ); // erg/s/cm^3

    // H2
    cr_heating_tot += ( n_H2 * H2_cosmic_ray_ionization_rate_primary * q_cr ); // erg/s/cm^3

    // Direct heating of the electrons from cosmic rays
    cr_heating_tot += (ne * xi_h_cr * 287.0 * EV_2_ERG); // erg/s/cm^3

    // Cosmic ray heating on metals
    // --> Here we skip H and He as it's done separately above
    double induced_UV_heat_scale_fac = H2_cosmic_ray_ionization_rate_primary / 1E-16;
    for (int i = 3; i < 27; i++) {
        double n_element = n_and_ion_fracs.n_element_new[i];
        if (n_element < MIN_COOL_ION) continue;

        int n_ions = (elements[i].n_ions + elements[i].n_mol);
        const int ei = elem_row(i);

        for (int j = 0; j < n_ions - 1; j++) { // This comes from ionization --> no heating for final ion
            double x_ion = n_and_ion_fracs[i].ion_fracs_new[j];

            // This is the primary heating term from ionization
            cr_heating_tot += (n_element * x_ion * q_cr * tabData.cosmic_ray_ionization_rates(ei, j) * total_cosmic_ray_ionization_rate); // erg/s/cm^3

            // This is the secondary heating term from induced UV emission
            // Only happend for the ground state
            if (j == 0){
                cr_heating_tot += ( n_element * x_ion
                                    * tabData.cosmic_ray_ionization_rates_induced_UV_heat(ei)
                                    * tabData.cosmic_ray_ionization_rates_induced_UV(ei) * EV_2_ERG
                                    * induced_UV_heat_scale_fac ); // erg/s/cm^3
            }
        }
    }

    return cr_heating_tot;
}

KOKKOS_INLINE_FUNCTION
double photoheating_UVB(const CompactIonData& n_and_ion_fracs,
                        const Element* elements,
                        const TabulatedData& tabData){
    // Photoheating contribution from a UVB

    double heating_rate = 0.0;

    // Loop over all elements
    for (int i = 1; i < 27; i++) {
        double n_element = n_and_ion_fracs.n_element_new[i];
        if (n_element < MIN_COOL_ION) continue;

        int n_ions = (elements[i].n_ions + elements[i].n_mol);
        const int ei = elem_row(i);
        for (int j = 0; j < n_ions - 1; j++) { // This comes from ionization --> no heating for final ion
            heating_rate += (n_element * n_and_ion_fracs[i].ion_fracs_new[j] * tabData.HM12_UVB_z(ei, j, 1)); // erg/s/cm^3
        }
    }

    return heating_rate;
}

KOKKOS_INLINE_FUNCTION
double photoheating_UVB_G0(double G0,
                           const CompactIonData& n_and_ion_fracs,
                           const TabulatedData& tabData){
    // Photoheating from the G0 background
    // Note that this only impacts the ground state

    double heating_rate = 0.0;

    // Loop over all elements (no need for H and He here)
    for (int i = 3; i < 27; i++) {
        double n_element = n_and_ion_fracs.n_element_new[i];
        if (n_element < MIN_COOL_ION) continue;

        heating_rate += n_element * n_and_ion_fracs[i].ion_fracs_new[0] * tabData.G0_heating_rates(elem_row(i)) * G0; // erg/s/cm^3
    }

    return heating_rate;
}

KOKKOS_INLINE_FUNCTION
double Epump(double nH, double T, double xH2, double xHI){
    //! Energy converted to heat from UV pumping
    //! see Appendix A http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1990ApJ...365..620B&amp;data_type=PDF_HIGH&amp;whole_paper=YES&amp;type=PRINTER&amp;filetype=.pdf
    double Crad = 2.0E-7; //     !radiation de-excitation rate Burton 1990
    double Cdex = (1.0E-12)*((1.4*xH2*exp(-18100.0/(T + 1200.0))) + (1.0*xHI*exp(-1000.0/T)))*sqrt(T)*nH; // !collisional de-excitation rate Burton 1990
    double Cfrac = Cdex / (Cdex + Crad);
    double Epump_loc = 2.0 * Cfrac * EV_2_ERG;  //!ergs
    return Epump_loc;
}

KOKKOS_INLINE_FUNCTION
double H2_heating_bialy(double G0, double nH2, double nH, double T, double xH2, double xHI,
                        double xHII, double xe, double f_dg, double xi_h2_cr){
    // Heating from H2 formation and destruction following Bialy 2018
    double D0 = 5.68E-11;
    double I_UV = 1.7 * G0;

    double E_pump = 1.12 * EV_2_ERG;
    double n_crit = 1.1E5 / sqrt(T/1000.0);
    double ncrit_factor = 1.0 / (1.0 + (n_crit/nH));

    // heating from H2 pumping
    double Hrate_H2_pump = 9.0 * D0 * I_UV * E_pump * nH2 * ncrit_factor; // erg/s/cm^3

    // heating from H2 photodissociation
    double E_pd = 0.4 * EV_2_ERG;
    double Hrate_H2_pd = D0 * I_UV * E_pd * nH2; // erg/s/cm^3

    // heating from H2 formation
    double Hrate_H2_form = 0.0;

    // heating from H2 formation --> dust channel
    double E_form1_dust = 0.2 * EV_2_ERG;
    double E_form2_dust = 4.48 * EV_2_ERG * ncrit_factor;
    double H2_formation_rate_dust = alpha_H2_dust(T, f_dg);

    Hrate_H2_form += H2_formation_rate_dust * (E_form1_dust + E_form2_dust) * nH * nH * xHI; // erg/s/cm^3

    // heating from H2 formation --> primordial channel
    double E_form1_prim = 0.6 * EV_2_ERG;
    double E_form2_prim = 3.13 * EV_2_ERG * ncrit_factor;
    double H2_formation_rate_prim = alpha_H2_prim(T, xe, xi_h2_cr, G0, xHI, xHII, nH);

    Hrate_H2_form += H2_formation_rate_prim * (E_form1_prim + E_form2_prim) * nH * nH * xHI; // erg/s/cm^3
    
    return fmax(Hrate_H2_pd + Hrate_H2_form + Hrate_H2_pump, 1e-100);
}

struct CTReaction { bool is_ionization; int stage; int Z; double dE; };

KOKKOS_INLINE_FUNCTION
double CT_heat_cool(double T, const CompactIonData& n_and_ion_fracs,
                    const TabulatedData& tabData){
    // Heating and cooling from charge exchange reactions

    double ct_hc_rate = 0.0;

    if (T > 1E5){
        return ct_hc_rate;
    }


    double loc_nHI = n_and_ion_fracs.n_element_new[1] * n_and_ion_fracs[1].ion_fracs_new[0];
    double loc_nHII = n_and_ion_fracs.n_element_new[1] * n_and_ion_fracs[1].ion_fracs_new[1];

    // Charge transfer reactions, one table entry per reaction:
    //   is_ionization : true  --> H+ + X(n)  -> X(n+1) + H,  rate scales with nHII
    //                   false --> H  + X(n)  -> X(n-1) + H+, rate scales with nHI
    //   stage         : ionization stage of X, which is also its ion_fracs_new index
    //   Z             : atomic number of X
    //   dE            : energy released by the reaction [eV], negative if absorbed
    static constexpr CTReaction ct_reactions[] = {
        // Helium
        {false, 1,  2,  10.99}, // H + He+ -> He + H+

        // Carbon
        {true,  0,  6,   2.34}, // H+ + C -> C+ + H
        {false, 1,  6,  -2.34}, // H + C+ -> C + H+
        {false, 2,  6,   4.01}, // H + C++ -> C+ + H+
        {false, 3,  6,   5.73}, // H + C+++ -> C++ + H+
        {false, 4,  6,  11.3 }, // H + C++++ -> C+++ + H+

        // Nitrogen
        {true,  0,  7,  -0.94}, // H+ + N -> N+ + H
        {false, 1,  7,   0.94}, // H + N+ -> N + H+
        {false, 2,  7,   4.56}, // H + N++ -> N+ + H+
        {false, 3,  7,   6.40}, // H + N+++ -> N++ + H+
        {false, 4,  7,  11.0 }, // H + N++++ -> N+++ + H+

        // Oxygen
        {true,  0,  8,  -0.02}, // H+ + O -> O+ + H
        {false, 1,  8,   0.02}, // H + O+ -> O + H+
        {false, 2,  8,   6.65}, // H + O++ -> O+ + H+
        {false, 3,  8,   5.00}, // H + O+++ -> O++ + H+
        {false, 4,  8,   8.47}, // H + O++++ -> O+++ + H+

        // Neon
        {false, 3, 10,   5.82}, // H + Ne+++ -> Ne++ + H+
        {false, 4, 10,   8.60}, // H + Ne++++ -> Ne+++ + H+

        // Magnesium
        {true,  0, 12,   1.52}, // H+ + Mg -> Mg+ + H
        {true,  1, 12,  -1.44}, // H+ + Mg+ -> Mg++ + H+
        {false, 2, 12,   1.44}, // H + Mg++ -> Mg+ + H+
        {false, 3, 12,   5.73}, // H + Mg+++ -> Mg++ + H+
        {false, 4, 12,   8.60}, // H + Mg++++ -> Mg+++ + H+

        // Silicon
        {true,  0, 14,   0.12}, // H+ + Si -> Si+ + H
        {true,  1, 14,  -2.72}, // H+ + Si+ -> Si++ + H+
        {false, 2, 14,   2.72}, // H + Si++ -> Si+ + H+
        {false, 3, 14,   4.23}, // H + Si+++ -> Si++ + H+
        {false, 4, 14,   7.49}, // H + Si++++ -> Si+++ + H+

        // Sulfur
        {false, 1, 16,  -3.24}, // H + S+ -> S + H+
        {false, 3, 16,   5.73}, // H + S+++ -> S++ + H+
        {false, 4, 16,   8.60}, // H + S++++ -> S+++ + H+

        // Iron
        {true,  1, 26,  -2.56}, // H+ + Fe+ -> Fe++ + H+
        {false, 2, 26,   2.56}, // H + Fe++ -> Fe+ + H+
        {false, 3, 26,   6.30}, // H + Fe+++ -> Fe++ + H+
        {false, 4, 26,  10.0 }, // H + Fe++++ -> Fe+++ + H+
    };

    // Don't unroll to save on instructions (saves ~30% of SASS instructions for cooling kernel)
#pragma unroll 1
    for (const CTReaction& r : ct_reactions) {
        double ct_rate = r.is_ionization ? charge_transfer_ionization(r.stage, r.Z, T, tabData)
                                         : charge_transfer_recombination(r.stage, r.Z, T, tabData);
        double n_H_partner = r.is_ionization ? loc_nHII : loc_nHI;

        ct_hc_rate += ct_rate * n_H_partner * n_and_ion_fracs.n_element_new[r.Z] * n_and_ion_fracs[r.Z].ion_fracs_new[r.stage] * r.dE * EV_2_ERG;
    }

    return ct_hc_rate;
}

    
KOKKOS_FUNCTION
double local_photoheating(std::array<double, N_GROUPS>& dNp,
                          const CompactIonData& n_and_ion_fracs,
                          const Element* elements,
                          const TabulatedData& tabData) {
    // Heading due to the local radiation field

    double heating_rate = 0.0; 

    for (int i = 1; i < 27; i++) { // Loop over elements
        double n_element = n_and_ion_fracs.n_element_new[i];
        if (n_element < MIN_COOL_ION) continue;

        int n_ions = (elements[i].n_ions + elements[i].n_mol);
        const int ei = elem_row(i);

        for (int j = 0; j < n_ions - 1; j++) { // Loop over ions --> no heating for final ion

            for (int k = 0; k < N_GROUPS; k++) { // Loop over photon groups
                heating_rate += (dNp[k] * n_and_ion_fracs.n_element_new[i] * n_and_ion_fracs[i].ion_fracs_new[j] * tabData.cs_ph(ei, j, k, 2));
            } // End loop over photon groups
        } // End loop over ions
    } // End loop over elements
    return fmax(heating_rate,1e-100);
}

KOKKOS_INLINE_FUNCTION
double CO_cooling_koyama_00(double nH, double nH2, double nHI, double nCO, double T) {
    // CO cooling
    double T_pivot = 3080.0;
    double term_a = (3.3e6 * pow(T / 1.0e3, 0.75)) / nH;

    // CO Rotational cooling
    double rot_numerator = nCO * 4.0 * pow(KB * T, 2) * 9.7e-8;
    double rot_denominator = 2.76 * KB * (1.0 + term_a + 1.5 * sqrt(term_a));
    double rot = rot_numerator / rot_denominator;

    // CO Vibrational cooling (H2 collisions)
    double vib_H2 = nH2 * nCO * T_pivot * KB * 4.3e-14 * T * exp(-pow(3.14e5 / T, 0.333)) * exp(-T_pivot / T);

    // CO Vibrational cooling (HI collisions)
    double vib_H = nHI * nCO * T_pivot * KB * 3.0e-12 * sqrt(T) * exp(-pow(2000.0 / T, 3.43)) * exp(-T_pivot / T);

    // Sum the rate
    double rate = rot + vib_H2 + vib_H;

    return fmax(rate,1e-100);
}

KOKKOS_FUNCTION
double all_cooling(double T, double ne, double aexp, const CompactIonData& n_and_ion_fracs, 
                   const Element* elements,
                   std::array<double, N_GROUPS>& dNp,
                   double nCO,
                   double G0_local, double UV_background_G0, double f_shd_H2,
                   double f_dg, double xe, double xi_h_cr, double xi_h2_cr,
                   double ss_factor,
                   bool rt_advect,
                   const PhysicsFlags& flags,
                   const TabulatedData& tabData){
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

    double nH    = n_and_ion_fracs.n_element_new[1];
    double xHI   = n_and_ion_fracs[1].ion_fracs_new[0];
    double xHII  = n_and_ion_fracs[1].ion_fracs_new[1];
    double xH2   = (elements[1].n_mol > 0) ? n_and_ion_fracs[1].ion_fracs_new[2] : 0.0;
    double nH_I  = nH * xHI;
    double nH_II = nH * xHII;
    double nH2   = nH * xH2 * 0.5;

    double nHe_I   = n_and_ion_fracs.n_element_new[2] * n_and_ion_fracs[2].ion_fracs_new[0];
    double nHe_II  = n_and_ion_fracs.n_element_new[2] * n_and_ion_fracs[2].ion_fracs_new[1];
    double nHe_III = n_and_ion_fracs.n_element_new[2] * n_and_ion_fracs[2].ion_fracs_new[2];

    // metal cool smoothing parameters
    double metal_cool_smooth_f1 = 0.5 * (tanh( (5E-3) * ( T - 1.E4 ) ) + 1.0 );
    double metal_cool_smooth_f2 = 0.5 * (tanh( (5E-3) * ( (-1.0 * T) + 1.E4 ) ) + 1.0 );

    // Cooling from primordial species
    double cooling_HI = (collisional_ionization_cooling_HI(T) + collisional_excitation_cooling_HI(T)) * ne * nH_I;
    double cooling_HII = recombination_cooling_case_B_HII(T) * ne * nH_II; 
    double cooling_HeI = collisional_ionization_cooling_HeI(T) * ne * nHe_I;
    double cooling_HeII = (collisional_ionization_cooling_HeII(T) + collisional_excitation_cooling_HeII(T) + recombination_cooling_case_B_HeII(T) + dielectronic_recombination_cooling_HeII(T)) * ne * nHe_II;
    double cooling_HeIII = recombination_cooling_case_B_HeIII(T) * ne * nHe_III;
    double cooling_bremmstrahlung = bremmstrahlung(T) * ne * (nH_II + nHe_II + (4.0 * nHe_III));
    double cooling_compton = compton_cooling(T,aexp) * ne;
    double cooling_H2 = H2_cooling(nH_I, nH2, T);

    double total_primordial_cooling = cooling_HI \
                                      + cooling_HII \
                                      + cooling_HeI \
                                      + cooling_HeII \
                                      + cooling_HeIII \
                                      + cooling_bremmstrahlung \
                                      + cooling_compton \
                                      + cooling_H2;

    // Metal line cooling -- high temperature
    double high_T_metal_cooling = 0.0;
    high_T_metal_cooling = get_high_t_cooling_rates(T, ne,
                                                    n_and_ion_fracs, elements,
                                                    metal_cool_smooth_f1,
                                                    tabData);

    // Metal line cooling -- low temperature (fine structure)
    double n_ion_fs;
    double z = (1.0 / aexp) - 1.0;
    double total_fine_structure = 0.0;
    if (T < 1.1E4){
        // CI cooling
        n_ion_fs = n_and_ion_fracs.n_element_new[6] * n_and_ion_fracs[6].ion_fracs_new[0];
        if (n_ion_fs > MIN_COOL_ION) {
            double cooling_fine_structure_CI = CI_fine_structure(T, n_ion_fs, nH_I, nH_II, 
                                                               ne, nH2, nHe_I, nHe_II, 
                                                               nHe_III, z,
                                                               tabData);
            total_fine_structure += cooling_fine_structure_CI;
        }
        
        // CII cooling
        n_ion_fs = n_and_ion_fracs.n_element_new[6] * n_and_ion_fracs[6].ion_fracs_new[1];
        if (n_ion_fs > MIN_COOL_ION) {
            double cooling_fine_structure_CII = CII_fine_structure(T, n_ion_fs, nH_I, nH_II, 
                                                                 ne, nH2, nHe_I, nHe_II, 
                                                                 nHe_III, z,
                                                                 tabData);
            //printf("CII cool: %e\n",cooling_fine_structure_CII);
            total_fine_structure += cooling_fine_structure_CII;
        }
        
        // NII cooling
        n_ion_fs = n_and_ion_fracs.n_element_new[7] * n_and_ion_fracs[7].ion_fracs_new[1];
        if (n_ion_fs > MIN_COOL_ION) {
            double cooling_fine_structure_NII = NII_fine_structure(T, n_ion_fs, nH_I, nH_II, 
                                                                 ne, nH2, nHe_I, nHe_II, 
                                                                 nHe_III, z,
                                                                 tabData);
            total_fine_structure += cooling_fine_structure_NII;
        }
        
        // OI cooling
        n_ion_fs = n_and_ion_fracs.n_element_new[8] * n_and_ion_fracs[8].ion_fracs_new[0];
        if (n_ion_fs > MIN_COOL_ION) {
            double cooling_fine_structure_OI = OI_fine_structure(T, n_ion_fs, nH_I, nH_II, 
                                                               ne, nH2, nHe_I, nHe_II, 
                                                               nHe_III, z,
                                                               tabData);
            //printf("OI cool: %e\n",cooling_fine_structure_OI);
            total_fine_structure += cooling_fine_structure_OI;
        }

        // OIII cooling
        n_ion_fs = n_and_ion_fracs.n_element_new[8] * n_and_ion_fracs[8].ion_fracs_new[2];
        if (n_ion_fs > MIN_COOL_ION) {
            double cooling_fine_structure_OIII = OIII_fine_structure(T, n_ion_fs, nH_I, nH_II, 
                                                                   ne, nH2, nHe_I, nHe_II, 
                                                                   nHe_III, z,
                                                                   tabData);
            total_fine_structure += cooling_fine_structure_OIII;
        }

        // NeII cooling
        n_ion_fs = n_and_ion_fracs.n_element_new[10] * n_and_ion_fracs[10].ion_fracs_new[1];
        if (n_ion_fs > MIN_COOL_ION) {
            double cooling_fine_structure_NeII = NeII_fine_structure(T, n_ion_fs, nH_I, nH_II, 
                                                                     ne, nH2, nHe_I, nHe_II, 
                                                                     nHe_III, z,
                                                                     tabData);
            total_fine_structure += cooling_fine_structure_NeII;
        }
        
        // SiI cooling
        n_ion_fs = n_and_ion_fracs.n_element_new[14] * n_and_ion_fracs[14].ion_fracs_new[0];
        if (n_ion_fs > MIN_COOL_ION) {
            double cooling_fine_structure_SiI = SiI_fine_structure(T, n_ion_fs, nH_I, nH_II, 
                                                                 ne, nH2, nHe_I, nHe_II, 
                                                                 nHe_III, z,
                                                                 tabData);
            total_fine_structure += cooling_fine_structure_SiI;
        }
        
        // SiII cooling
        n_ion_fs = n_and_ion_fracs.n_element_new[14] * n_and_ion_fracs[14].ion_fracs_new[1];
        if (n_ion_fs > MIN_COOL_ION) {
            double cooling_fine_structure_SiII = SiII_fine_structure(T, n_ion_fs, nH_I, nH_II, 
                                                                   ne, nH2, nHe_I, nHe_II, 
                                                                   nHe_III, z,
                                                                   tabData);
            total_fine_structure += cooling_fine_structure_SiII;
        }
        
        // SI cooling
        n_ion_fs = n_and_ion_fracs.n_element_new[16] * n_and_ion_fracs[16].ion_fracs_new[0];
        if (n_ion_fs > MIN_COOL_ION) {
            double cooling_fine_structure_SI = SI_fine_structure(T, n_ion_fs, nH_I, nH_II, 
                                                               ne, nH2, nHe_I, nHe_II, 
                                                               nHe_III, z,
                                                               tabData);
            total_fine_structure += cooling_fine_structure_SI;
        }
        
        // FeI cooling
        n_ion_fs = n_and_ion_fracs.n_element_new[26] * n_and_ion_fracs[26].ion_fracs_new[0];
        if (n_ion_fs > MIN_COOL_ION) {
            double cooling_fine_structure_FeI = FeI_fine_structure(T, n_ion_fs, nH_I, nH_II, 
                                                                 ne, nH2, nHe_I, nHe_II, 
                                                                   nHe_III, z,
                                                                   tabData);
            total_fine_structure += cooling_fine_structure_FeI;
        }
        
        // FeII cooling
        n_ion_fs = n_and_ion_fracs.n_element_new[26] * n_and_ion_fracs[26].ion_fracs_new[1];
        if (n_ion_fs > MIN_COOL_ION) {
            double cooling_fine_structure_FeII = FeII_fine_structure(T, n_ion_fs, nH_I, nH_II, 
                                                                   ne, nH2, nHe_I, nHe_II, 
                                                                   nHe_III, z,
                                                                   tabData);
            total_fine_structure += cooling_fine_structure_FeII;
        }
           
    }
    // Smooth the fine structure cooling with temeprature if needed
    total_fine_structure *= metal_cool_smooth_f2;

    // Dust cooling
    double dust_cooling = 0.0;
    dust_cooling = dust_recombination_cooling(T, G0_local + UV_background_G0, ne, f_dg, nH);
    dust_cooling += dust_gas_collisional_cooling(T, G0_local + UV_background_G0, xH2*2.0, aexp, nH, f_dg);

    // CO cooling
    double CO_cooling = CO_cooling_koyama_00(nH, nH2, nH_I, nCO, T);

    // Photoelectric heating --> note factor of 1.7 is because IUV
    double photoelectric_heat = photoelectric_heating(T, G0_local + UV_background_G0, ne, f_dg, nH);

    // Cosmic ray heating
    double cosmic_ray_heat = 0.0;
    if (flags.include_cosmic_ray_ionization) {
		cosmic_ray_heat = cosmic_ray_heating(xe, nH_I, nHe_I, nH2, ne, xi_h_cr,
                                             n_and_ion_fracs, elements, tabData);
	}

    // Photoheating from the UV background
	double uvb_photoheat = 0.0;
    if (flags.include_HM12_UVB) {
        uvb_photoheat = photoheating_UVB(n_and_ion_fracs, elements, tabData);
	    uvb_photoheat *= ss_factor; // Account for self-shielding
	}

    // Photoheating from the G0 FUV background
    double uvb_photoheat_G0 = photoheating_UVB_G0(UV_background_G0, n_and_ion_fracs,  tabData);

    // Heating from H2 formation and destruction
    double h2_heat = 0.0;
    if (elements[1].n_mol > 0) {
        h2_heat = H2_heating_bialy(G0_local*f_shd_H2 + UV_background_G0, nH2, nH, T, xH2, xHI, xHII, xe, f_dg, xi_h2_cr);
    }

    // Heating and cooling from charge transfer reactions
    double charge_transfer_heat_cool = 0.0;
    if (flags.include_charge_exchange) {
		charge_transfer_heat_cool = CT_heat_cool(T, n_and_ion_fracs, tabData);
	}

    double local_phot_heat = 0.0;
    if (rt_advect) {
        local_phot_heat += local_photoheating(dNp, n_and_ion_fracs, elements, tabData);
    }

    // Sum all of the cooling rates
    double total_cooling = total_primordial_cooling + high_T_metal_cooling + total_fine_structure + dust_cooling + CO_cooling;

    // Sum all of the heating rates
    double total_heating = photoelectric_heat + cosmic_ray_heat + uvb_photoheat + uvb_photoheat_G0 + h2_heat + charge_transfer_heat_cool + local_phot_heat;

    return total_heating - total_cooling;
}
