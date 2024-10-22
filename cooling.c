// cooling.c
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "constants.h"

// Constants
const double MIN_METAL_COOL_DENS = 1E-10; //minimum number density to calc cooling

// Initialize the relevant arrays
const int N_HIGH_T_COOLING_TEMP = 121;
double high_t_cooling_temp[N_HIGH_T_COOLING_TEMP];
double high_t_cooling_rates[N_HIGH_T_COOLING_TEMP][27][27];
double fs_cool_tab[27][160][8];  // Array for fine structure cooling rates

double collisional_ionization_cooling_HI(double T){
    // From Cen 1992
    double term_1 = 1.27E-21 * sqrt(T);
    double term_2 = 1.0 /(1.0 + sqrt(T/1E5));
    double term_3 = exp(-157809.1/T);
    double cooling_rate = term_1 * term_2 * term_3;
    return cooling_rate;
}

double collisional_ionization_cooling_HeI(double T){
    // From Cen 1992
    double term_1 = 9.38E-22 * sqrt(T);
    double term_2 = 1.0 /(1.0 + sqrt(T/1E5));
    double term_3 = exp(-285335.4/T);
    double cooling_rate = term_1 * term_2 * term_3;
    return cooling_rate;
}

double collisional_ionization_cooling_HeII(double T){
    // From Cen 1992
    double term_1 = 4.95E-22 * sqrt(T);
    double term_2 = 1.0 /(1.0 + sqrt(T/1E5));
    double term_3 = exp(-631515.0/T);
    double cooling_rate = term_1 * term_2 * term_3;
    return cooling_rate;
}

double recombination_cooling_case_B_HII(double T){
    // From Hui & Gnedin 1997
    double lam_HI = 315614.0 / T;
    double term_1 = 3.435E-30 * T;
    double term_2 = pow(lam_HI,1.97);
    double term_3 = pow(1.0 + pow(lam_HI/2.25,0.376),-3.72);
    double cooling_rate = term_1 * term_2 * term_3;
    return cooling_rate;
}

double recombination_cooling_case_B_HeII(double T){
    // From Hui & Gnedin 1997
    double lam_HeI = 570670.0 / T;
    double term_1 = KB * T * 1.26E-14;
    double term_2 = pow(lam_HeI,0.75);
    double cooling_rate = term_1 * term_2;
    return cooling_rate;
}

double recombination_cooling_case_B_HeIII(double T){
    // From Hui & Gnedin 1997
    double lam_HeII = 1263030.0 / T;
    double term_1 = 8.0 * 3.435E-30 * T;
    double term_2 = pow(lam_HeII,1.97);
    double term_3 = pow(1.0 + pow(lam_HeII/2.25,0.376),-3.72);
    double cooling_rate = term_1 * term_2 * term_3;
    return cooling_rate;
}

double dielectronic_recombination_cooling_HeII(double T){
    // From Black 1981
    double term_1 = 1.24E-13 * pow(T,-1.5);
    double term_2 = exp(-470000.0/T);
    double term_3 = 1.0 + (0.3 * exp(-94000.0/T));
    double cooling_rate = term_1 * term_2 * term_3;
    return cooling_rate;
}

double collisional_excitation_cooling_HI(double T){
    // From Cen 1992
    double term_1 = 7.5E-19;
    double term_2 = 1.0 /(1.0 + sqrt(T/1E5));
    double term_3 = exp(-118348.0/T);
    double cooling_rate = term_1 * term_2 * term_3;
    return cooling_rate;
}

double collisional_excitation_cooling_HeII(double T){
    // From Cen 1992
    double term_1 = 5.54E-17 * pow(T,-0.397);
    double term_2 = 1.0 /(1.0 + sqrt(T/1E5));
    double term_3 = exp(-473638.0/T);
    double cooling_rate = term_1 * term_2 * term_3;
    return cooling_rate;
}

double bremmstrahlung(double T){
    // Osterbrock and Ferland 2006
    double cooling_rate = 1.42E-27 * sqrt(T);
    return cooling_rate;
}

double compton_cooling(double T, double a){
    // Haiman+ 1996
    double term_1 = 1.017E-37;
    double term_2 = pow(2.727/a,4.0);
    double term_3 = T - (2.727/a);
    double cooling_rate = term_1 * term_2 * term_3;
    return cooling_rate;
}

double H2_cooling(double T, double nH){
    // Galli + Palla 1998
    double log_T = log10(T);
    double lam_H2_nH0 = -103.0 + (97.59 * log_T) - \
                         48.05 * (log_T * log_T) + \
                         10.80 * (log_T * log_T * log_T) - \
                         0.9032 * (log_T * log_T * log_T * log_T);

    // Hollenbach + Mckee 1979
    double T3 = T/1000.;
    double LTE_rot = (9.5E-22 * pow(T3,3.76)) / (1.0 + 0.12*pow(T3,2.1));
    LTE_rot *= exp(-pow(0.13/T3,3));
    LTE_rot += 1E-24 * exp(-0.51/T3);

    double LTE_vib = 6.7E-19 * exp(-5.86/T3);
    LTE_vib += (1.6E-18 * exp(-11.7/T3));

    double LTE_limit = LTE_rot + LTE_vib;

    double n_crit_over_nH = LTE_limit / lam_H2_nH0;

    double cooling_rate = LTE_limit / (1.0 + n_crit_over_nH);

    return cooling_rate;
}

void initialize_high_temperature_metal_cooling(){
    // Cloudy tables of metal line cooling
    // which are valid at high temperature
    FILE *file;

    printf("Initializing high temperature cooling tables\n");

    // Temperatures
    file = fopen("./data/high_T_cooling/temperatures.dat", "r");
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
    for (int i = 0; i < N_HIGH_T_COOLING_TEMP; i++) {
        for (int j = 0; j < 27; j++) {
            for (int k = 0; k < 27; k++) {
                high_t_cooling_rates[i][j][k] = 0.0;
            }
        }
    }

    // Carbon
    file = fopen("./data/high_T_cooling/CARBON/all_cool.dat", "r");
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
    file = fopen("./data/high_T_cooling/NITROGEN/all_cool.dat", "r");
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
    file = fopen("./data/high_T_cooling/OXYGEN/all_cool.dat", "r");
    if (file == NULL) {
        printf("Error: Could not open high temperature cooling oxygen rates\n");
    }

    // Reading data from the file into the 2D array
    for (int j = 0; j < 8; j++) { 
        for (int i = 0; i < N_HIGH_T_COOLING_TEMP; i++) {
            fscanf(file, "%lf", &high_t_cooling_rates[i][8][j]);
        }
    }
    // Close the file
    fclose(file);

    // Neon
    file = fopen("./data/high_T_cooling/NEON/all_cool.dat", "r");
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
    file = fopen("./data/high_T_cooling/MAGNESIUM/all_cool.dat", "r");
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
    file = fopen("./data/high_T_cooling/SILICON/all_cool.dat", "r");
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
    file = fopen("./data/high_T_cooling/SULPHUR/all_cool.dat", "r");
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
    file = fopen("./data/high_T_cooling/IRON/all_cool.dat", "r");
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
}

double get_high_t_cooling_rates(double T, double ne, 
                                double *element_number_densities,
                                double *element_number_ions,
                                double **element_ion_fractions){
    // Computes the high-temperature cooling rates
    // derived from Harley's custom cloudy models

    double log_T = log10(T);
    double t_min = 3.0;
    double t_max = 9.0;
    double dt = 0.05;
    double total_metal_cooling_rate = 0.0;

    // bounds for temperature --> no cooling
    if (log_T < t_min){
        return total_metal_cooling_rate;
    }

    // Still cool above 10^9 K but set a bound
    if (log_T > t_max){
        log_T = t_max;
    }

    // Prepare for 1D interpolation
    int idx_low = (int)floor((log_T - t_max)/dt);

    double frac_high = (log_T - (3.0 + ((float)idx_low * dt))) / dt;
    double frac_low = 1.0 - frac_high;

    // Loop over all ions
    for (int i = 0; i < 27; i++) {
        if (element_number_densities[i] > MIN_METAL_COOL_DENS){
            // Get the element number density
            double n_element = element_number_densities[i];
            for (int j = 0; j < element_number_ions[i]; j++) { 
                // Get the element ion fraction
                double n_ion = element_ion_fractions[i][j];

                // Interpolate the cooling rate for the ion
                double loc_cooling_rate = frac_low * high_t_cooling_rates[idx_low][i][j];
                loc_cooling_rate += frac_high * high_t_cooling_rates[idx_low+1][i][j];
                loc_cooling_rate = pow(10.0,loc_cooling_rate);

                // Multiply cooling rate by ion and electron number densities
                total_metal_cooling_rate += n_element * n_ion * ne * total_metal_cooling_rate;
            }
        }
    }

    return total_metal_cooling_rate;
}

void init_fine_structure_tables(){
    // Initialization for fine structure cooling tables
    printf("Initializing fine cooling tables\n");

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

    char fs_cooling_dir[256] = "./data/fine_structure_data/";

    char file_path[512];
    FILE* file;
    bool ok;

    // Check that all files exist and concatenate directory and filename
    for (int i = 0; i < N_LINES; i++) {
        snprintf(file_path, sizeof(file_path), "%s%s", fs_cooling_dir, file_names[i]);
        file = fopen(file_path, "r");
        ok = (file != NULL);
        if (!ok) {
            printf("Cannot access fine structure cooling file %s\n", file_path);
        }
        fclose(file);
    }

    // Read in the fine structure data
    for (int j = 0; j < N_LINES; j++) {
        snprintf(file_path, sizeof(file_path), "%s%s", fs_cooling_dir, file_names[j]);
        file = fopen(file_path, "r");
        if (!file) {
            printf("Failed to open file %s\n", file_path);
        }

        // Loop over temperatures
        for (int i = 0; i < 160; i++) {
            for (int k = 0; k < 8; k++) {
                if (fscanf(file, "%lf", &fs_cool_tab[j][i][k]) != 1) {
                    printf("Error reading data from file %s\n", file_path);
                    fclose(file);
                }
            }
        }

        fclose(file);
    }
}

double three_level(double g_0, double g_1, double g_2, 
                   double lam_10, double lam_20, double lam_21, 
                   double A_10, double A_20, double A_21, 
                   double z, double T, double n_ion, 
                   double ne, double nH, double nHp, 
                   double nHe, double nHep, double nHepp, 
                   double nH2, 
                   int idx_0, int idx_1, int idx_2){
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
    logT = fmin(logT,5.0);

    double nu_10 = C_CGS / (lam_10 * 1E-4); //! Hz
    double nu_20 = C_CGS / (lam_20 * 1E-4); //! Hz
    double nu_21 = C_CGS / (lam_21 * 1E-4); //! Hz

    double E_10 = (H_PLANCK * C_CGS / (lam_10 * 1E-4)) / KB; //! E/K (K)
    double E_20 = (H_PLANCK * C_CGS / (lam_20 * 1E-4)) / KB; //! E/K (K)
    double E_21 = (H_PLANCK * C_CGS / (lam_21 * 1E-4)) / KB; //! E/K (K)

    double B_01 = A_10 * (pow(lam_10 * 1.E-4,3.0)) * (g_1 / g_0) / (2.0 * H_PLANCK * C_CGS);
    double B_02 = A_20 * (pow(lam_20 * 1.E-4,3.0)) * (g_2 / g_0) / (2.0 * H_PLANCK * C_CGS);
    double B_12 = A_21 * (pow(lam_21 * 1.E-4,3.0)) * (g_2 / g_1) / (2.0 * H_PLANCK * C_CGS);

    double B_10 = (g_0 / g_1) * B_01;
    double B_20 = (g_0 / g_2) * B_02;
    double B_21 = (g_1 / g_2) * B_12;

    //! CMB black body spectrum
    double B_nu_10 = (2.0 * H_PLANCK * pow(nu_10,3.0) / (C_CGS*C_CGS)) / (exp(H_PLANCK * nu_10 / (KB * T_cmb)) - 1.0);
    double B_nu_20 = (2.0 * H_PLANCK * pow(nu_20,3.0) / (C_CGS*C_CGS)) / (exp(H_PLANCK * nu_20 / (KB * T_cmb)) - 1.0);
    double B_nu_21 = (2.0 * H_PLANCK * pow(nu_21,3.0) / (C_CGS*C_CGS)) / (exp(H_PLANCK * nu_21 / (KB * T_cmb)) - 1.0);

    // ! Find temperature index.
    int itemp_low = (int)floor( ((logT - tmin) / (tmax - tmin)) * 160.0);
    double frac_high = (logT - (tmin + ((float)itemp_low * delta_temp))) / delta_temp;
    double frac_low = 1.0 - frac_high;

    // ! All collison strengths are in cm^3 s^-1
    // ! For collisions with o-H2
    // !rate_electron,rate_HII,rate_HeII,rate_HeIII,rate_HI,rate_HeI,rate_oH2,rate_pH2
    // double q10_oH2 = interp_coll_1d(fs_cool_tab(idx_0,:,7), logT);
    // double q20_oH2 = interp_coll_1d(fs_cool_tab(idx_1,:,7), logT);
    // double q21_oH2 = interp_coll_1d(fs_cool_tab(idx_2,:,7), logT);
    double q10_oH2 = (fs_cool_tab[idx_0][itemp_low][6] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][6] * frac_high);
    double q20_oH2 = (fs_cool_tab[idx_1][itemp_low][6] * frac_low) + (fs_cool_tab[idx_1][itemp_low+1][6] * frac_high);
    double q21_oH2 = (fs_cool_tab[idx_2][itemp_low][6] * frac_low) + (fs_cool_tab[idx_2][itemp_low+1][6] * frac_high);

    // ! For collisions with p-H2
    // double q10_pH2 = interp_coll_1d(fs_cool_tab(idx_0,:,8), logT);
    // double q20_pH2 = interp_coll_1d(fs_cool_tab(idx_1,:,8), logT);
    // double q21_pH2 = interp_coll_1d(fs_cool_tab(idx_2,:,8), logT);
    double q10_pH2 = (fs_cool_tab[idx_0][itemp_low][7] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][7] * frac_high);
    double q20_pH2 = (fs_cool_tab[idx_1][itemp_low][7] * frac_low) + (fs_cool_tab[idx_1][itemp_low+1][7] * frac_high);
    double q21_pH2 = (fs_cool_tab[idx_2][itemp_low][7] * frac_low) + (fs_cool_tab[idx_2][itemp_low+1][7] * frac_high);

    // ! For collisions with H
    // double q10_H = interp_coll_1d(fs_cool_tab(idx_0,:,5), logT);
    // double q20_H = interp_coll_1d(fs_cool_tab(idx_1,:,5), logT);
    // double q21_H = interp_coll_1d(fs_cool_tab(idx_2,:,5), logT);
    double q10_H = (fs_cool_tab[idx_0][itemp_low][4] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][4] * frac_high);
    double q20_H = (fs_cool_tab[idx_1][itemp_low][4] * frac_low) + (fs_cool_tab[idx_1][itemp_low+1][4] * frac_high);
    double q21_H = (fs_cool_tab[idx_2][itemp_low][4] * frac_low) + (fs_cool_tab[idx_2][itemp_low+1][4] * frac_high);

    // ! For collisions with H+
    // double q10_Hp = interp_coll_1d(fs_cool_tab(idx_0,:,2), logT);
    // double q20_Hp = interp_coll_1d(fs_cool_tab(idx_1,:,2), logT);
    // double q21_Hp = interp_coll_1d(fs_cool_tab(idx_2,:,2), logT);
    double q10_Hp = (fs_cool_tab[idx_0][itemp_low][1] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][1] * frac_high);
    double q20_Hp = (fs_cool_tab[idx_1][itemp_low][1] * frac_low) + (fs_cool_tab[idx_1][itemp_low+1][1] * frac_high);
    double q21_Hp = (fs_cool_tab[idx_2][itemp_low][1] * frac_low) + (fs_cool_tab[idx_2][itemp_low+1][1] * frac_high);

    // ! For collisions with e
    // double q10_e = interp_coll_1d(fs_cool_tab(idx_0,:,1), logT);
    // double q20_e = interp_coll_1d(fs_cool_tab(idx_1,:,1), logT);
    // double q21_e = interp_coll_1d(fs_cool_tab(idx_2,:,1), logT);
    double q10_e = (fs_cool_tab[idx_0][itemp_low][0] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][0] * frac_high);
    double q20_e = (fs_cool_tab[idx_1][itemp_low][0] * frac_low) + (fs_cool_tab[idx_1][itemp_low+1][0] * frac_high);
    double q21_e = (fs_cool_tab[idx_2][itemp_low][0] * frac_low) + (fs_cool_tab[idx_2][itemp_low+1][0] * frac_high);

    // ! For collisions with He
    // double q10_He = interp_coll_1d(fs_cool_tab(idx_0,:,6), logT);
    // double q20_He = interp_coll_1d(fs_cool_tab(idx_1,:,6), logT);
    // double q21_He = interp_coll_1d(fs_cool_tab(idx_2,:,6), logT);
    double q10_He = (fs_cool_tab[idx_0][itemp_low][5] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][5] * frac_high);
    double q20_He = (fs_cool_tab[idx_1][itemp_low][5] * frac_low) + (fs_cool_tab[idx_1][itemp_low+1][5] * frac_high);
    double q21_He = (fs_cool_tab[idx_2][itemp_low][5] * frac_low) + (fs_cool_tab[idx_2][itemp_low+1][5] * frac_high);

    // ! For collisions with He+
    // double q10_Hep = interp_coll_1d(fs_cool_tab(idx_0,:,3), logT);
    // double q20_Hep = interp_coll_1d(fs_cool_tab(idx_1,:,3), logT);
    // double q21_Hep = interp_coll_1d(fs_cool_tab(idx_2,:,3), logT);
    double q10_Hep = (fs_cool_tab[idx_0][itemp_low][2] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][2] * frac_high);
    double q20_Hep = (fs_cool_tab[idx_1][itemp_low][2] * frac_low) + (fs_cool_tab[idx_1][itemp_low+1][2] * frac_high);
    double q21_Hep = (fs_cool_tab[idx_2][itemp_low][2] * frac_low) + (fs_cool_tab[idx_2][itemp_low+1][2] * frac_high);

    // ! For collisions with He++
    // double q10_Hepp = interp_coll_1d(fs_cool_tab(idx_0,:,4), logT);
    // double q20_Hepp = interp_coll_1d(fs_cool_tab(idx_1,:,4), logT);
    // double q21_Hepp = interp_coll_1d(fs_cool_tab(idx_2,:,4), logT);
    double q10_Hepp = (fs_cool_tab[idx_0][itemp_low][3] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][3] * frac_high);
    double q20_Hepp = (fs_cool_tab[idx_1][itemp_low][3] * frac_low) + (fs_cool_tab[idx_1][itemp_low+1][3] * frac_high);
    double q21_Hepp = (fs_cool_tab[idx_2][itemp_low][3] * frac_low) + (fs_cool_tab[idx_2][itemp_low+1][3] * frac_high);

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

double two_level(double g_0, double g_1, double lam_10, double A_10,
                double z, double T, double n_ion, double ne, 
                double nH, double nHp, double nHe, double nHep, 
                double nHepp, double nH2, int idx_0){
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

    double nu_10 = C_CGS / (lam_10 * 1E-4); //! Hz

    double E_10 = (H_PLANCK * C_CGS / (lam_10 * 1E-4)) / KB; //! E/K (K)

    double B_01 = A_10 * (pow(lam_10 * 1.E-4,3.0)) * (g_1 / g_0) / (2.0 * H_PLANCK * C_CGS);

    double B_10 = (g_0 / g_1) * B_01;

    //! CMB black body spectrum
    double B_nu_10 = (2.0 * H_PLANCK * pow(nu_10,3.0) / (C_CGS*C_CGS)) / (exp(H_PLANCK * nu_10 / (KB * T_cmb)) - 1.0);

    // ! Find temperature index.
    int itemp_low = (int)floor( ((logT - tmin) / (tmax - tmin)) * 160.0);
    double frac_high = (logT - (tmin + ((float)itemp_low * delta_temp))) / delta_temp;
    double frac_low = 1.0 - frac_high;

    // ! All collison strengths are in cm^3 s^-1
    // ! For collisions with o-H2
    // !rate_electron,rate_HII,rate_HeII,rate_HeIII,rate_HI,rate_HeI,rate_oH2,rate_pH2
    // double q10_oH2 = interp_coll_1d(fs_cool_tab(idx_0,:,7), logT);
    double q10_oH2 = (fs_cool_tab[idx_0][itemp_low][6] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][6] * frac_high);

    // ! For collisions with p-H2
    // double q10_pH2 = interp_coll_1d(fs_cool_tab(idx_0,:,8), logT);
    double q10_pH2 = (fs_cool_tab[idx_0][itemp_low][7] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][7] * frac_high);

    // ! For collisions with H
    // double q10_H = interp_coll_1d(fs_cool_tab(idx_0,:,5), logT);
    double q10_H = (fs_cool_tab[idx_0][itemp_low][4] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][4] * frac_high);

    // ! For collisions with H+
    // double q10_Hp = interp_coll_1d(fs_cool_tab(idx_0,:,2), logT);
    double q10_Hp = (fs_cool_tab[idx_0][itemp_low][1] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][1] * frac_high);

    // ! For collisions with e
    // double q10_e = interp_coll_1d(fs_cool_tab(idx_0,:,1), logT);
    double q10_e = (fs_cool_tab[idx_0][itemp_low][0] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][0] * frac_high);

    // ! For collisions with He
    // double q10_He = interp_coll_1d(fs_cool_tab(idx_0,:,6), logT);
    double q10_He = (fs_cool_tab[idx_0][itemp_low][5] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][5] * frac_high);

    // ! For collisions with He+
    // double q10_Hep = interp_coll_1d(fs_cool_tab(idx_0,:,3), logT);
    double q10_Hep = (fs_cool_tab[idx_0][itemp_low][2] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][2] * frac_high);

    // ! For collisions with He++
    // double q10_Hepp = interp_coll_1d(fs_cool_tab(idx_0,:,4), logT);
    double q10_Hepp = (fs_cool_tab[idx_0][itemp_low][3] * frac_low) + (fs_cool_tab[idx_0][itemp_low+1][3] * frac_high);

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


double all_cooling(double T, double ne, double aexp, double *element_number_densities, 
                   double *element_number_ions, double **element_ion_fractions){
    // Main cooling driver

    double nH_I = element_number_ions[1] * element_ion_fractions[1][0];
    double nH_II = element_number_ions[1] * element_ion_fractions[1][1];
    double nH2 = element_number_ions[1] * element_ion_fractions[1][2] * 0.5;
    double nHe_I = element_number_ions[2] * element_ion_fractions[2][0];
    double nHe_II = element_number_ions[2] * element_ion_fractions[2][1];
    double nHe_III = element_number_ions[2] * element_ion_fractions[2][2];

    // Cooling from primordial species
    double cooling_HI = (collisional_ionization_cooling_HI(T) + collisional_excitation_cooling_HI(T)) * ne * nH_I;
    double cooling_HII = recombination_cooling_case_B_HII(T) * ne * nH_II; 
    double cooling_HeI = collisional_ionization_cooling_HeI(T) * ne * nHe_I;
    double cooling_HeII = (collisional_ionization_cooling_HeII(T) + collisional_excitation_cooling_HeII(T) + recombination_cooling_case_B_HeII(T) + dielectronic_recombination_cooling_HeII(T)) * ne * nHe_II;
    double cooling_HeIII = recombination_cooling_case_B_HeIII(T) * ne * nHe_III;
    double cooling_bremmstrahlung = bremmstrahlung(T) * ne * (nH_II + nHe_II + (4.0 * nHe_III));
    double cooling_compton = compton_cooling(T,aexp) * ne;
    double cooling_H2 = H2_cooling(T, nH_I) * nH_I * nH2;

    // Metal line cooling
    double high_T_metal_cooling = 0.0;
    if (T > 1e3){
        high_T_metal_cooling = get_high_t_cooling_rates(T, ne, 
                                                        element_number_densities,
                                                        element_number_ions,
                                                        element_ion_fractions);
    }

    // Primordial non-equilibrium cooling rate
    // double total_net_cooling 
    return 0;
}
