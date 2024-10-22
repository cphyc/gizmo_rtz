// cooling.c
#include <stdio.h>
#include <math.h>
#include "constants.h"

// Constants
const double MIN_METAL_COOL_DENS = 1E-10; //minimum number density to calc cooling

// Initialize the relevant arrays
const int N_HIGH_T_COOLING_TEMP = 121;
double high_t_cooling_temp[N_HIGH_T_COOLING_TEMP];
double high_t_cooling_rates[N_HIGH_T_COOLING_TEMP][27][27];

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

    // TODO(code): zero out the rates that we calculate analytically!!!

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
