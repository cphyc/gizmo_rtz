// photoionization_UVB.c
#include <stdio.h>
#include <math.h>
#include "photoionization_UVB.h"  // Include the header file

const int N_UVB_POINTS = 60;

// Initialize the UVB arrays
double HM12_UVB_redshifts[N_UVB_POINTS];
double HM12_UVB_hydrogen[N_UVB_POINTS][1];
double HM12_UVB_helium[N_UVB_POINTS][2];
double HM12_UVB_carbon[N_UVB_POINTS][6];
double HM12_UVB_nitrogen[N_UVB_POINTS][7];
double HM12_UVB_oxygen[N_UVB_POINTS][8];
double HM12_UVB_neon[N_UVB_POINTS][10];
double HM12_UVB_magnesium[N_UVB_POINTS][12];
double HM12_UVB_silicon[N_UVB_POINTS][14];
double HM12_UVB_sulfur[N_UVB_POINTS][16];
double HM12_UVB_iron[N_UVB_POINTS][26];

// Array that holds the UVB for each ion at a given redshift
double HM12_UVB_z[27][27];

void load_UVB_data() {
    // Load in all of the UV background tables
    //TODO(code): can probably abstract this into a function...
    FILE *file;

    printf("Initializing UV background data\n");

    // Redshifts
    file = fopen("./data/HM12/redshifts.dat", "r");
    if (file == NULL) {
        printf("Error: Could not open Redshifts UVB file\n");
    }

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        fscanf(file, "%lf", &HM12_UVB_redshifts[i]);
    }
    // Close the file
    fclose(file);

    // Hydrogen
    file = fopen("./data/HM12/hydrogen_pi.dat", "r");
    if (file == NULL) {
        printf("Error: Could not open Hydrogen UVB file\n");
    }

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 1; j++) {
            fscanf(file, "%lf", &HM12_UVB_hydrogen[i][j]);
        }
    }
    // Close the file
    fclose(file);

    // Helium
    file = fopen("./data/HM12/helium_pi.dat", "r");
    if (file == NULL) {
        printf("Error: Could not open Helium UVB file\n");
    }

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 2; j++) {
            fscanf(file, "%lf", &HM12_UVB_helium[i][j]);
        }
    }
    // Close the file
    fclose(file);

    // Carbon
    file = fopen("./data/HM12/carbon_pi.dat", "r");
    if (file == NULL) {
        printf("Error: Could not open Carbon UVB file\n");
    }

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 6; j++) {
            fscanf(file, "%lf", &HM12_UVB_carbon[i][j]);
        }
    }
    // Close the file
    fclose(file);

    // Nitrogen
    file = fopen("./data/HM12/nitrogen_pi.dat", "r");
    if (file == NULL) {
        printf("Error: Could not open Nitrogen UVB file\n");
    }

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 7; j++) {
            fscanf(file, "%lf", &HM12_UVB_nitrogen[i][j]);
        }
    }
    // Close the file
    fclose(file);

    // Oxygen
    file = fopen("./data/HM12/oxygen_pi.dat", "r");
    if (file == NULL) {
        printf("Error: Could not open Oxygen UVB file\n");
    }

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 8; j++) {
            fscanf(file, "%lf", &HM12_UVB_oxygen[i][j]);
        }
    }
    // Close the file
    fclose(file);

    // Neon
    file = fopen("./data/HM12/neon_pi.dat", "r");
    if (file == NULL) {
        printf("Error: Could not open Neon UVB file\n");
    }

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 10; j++) {
            fscanf(file, "%lf", &HM12_UVB_neon[i][j]);
        }
    }
    // Close the file
    fclose(file);

    // Magnesium
    file = fopen("./data/HM12/magnesium_pi.dat", "r");
    if (file == NULL) {
        printf("Error: Could not open Magnesium UVB file\n");
    }

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 12; j++) {
            fscanf(file, "%lf", &HM12_UVB_magnesium[i][j]);
        }
    }
    // Close the file
    fclose(file);

    // Silicon
    file = fopen("./data/HM12/silicon_pi.dat", "r");
    if (file == NULL) {
        printf("Error: Could not open Silicon UVB file\n");
    }

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 14; j++) {
            fscanf(file, "%lf", &HM12_UVB_silicon[i][j]);
        }
    }
    // Close the file
    fclose(file);

    // Sulfur
    file = fopen("./data/HM12/sulfur_pi.dat", "r");
    if (file == NULL) {
        printf("Error: Could not open Sulfur UVB file\n");
    }

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 16; j++) {
            fscanf(file, "%lf", &HM12_UVB_sulfur[i][j]);
        }
    }
    // Close the file
    fclose(file);

    // Iron
    file = fopen("./data/HM12/iron_pi.dat", "r");
    if (file == NULL) {
        printf("Error: Could not open Iron UVB file\n");
    }

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 26; j++) {
            fscanf(file, "%lf", &HM12_UVB_iron[i][j]);
        }
    }
    // Close the file
    fclose(file);
}

void update_UVB(double redshift){
    /* 
    interpolates the UVB for all ions
    to a particular redshift
    */

    double scale_low;
    double scale_high;

    // No interpolation beyong redshift bounds --> set the UVB to zero
    if (redshift < HM12_UVB_redshifts[0] || redshift >= HM12_UVB_redshifts[N_UVB_POINTS-1]){
        for (int i = 0; i < 27; i++) {
            for (int j = 0; j < 27; j++) {
                HM12_UVB_z[i][j]  = 0.0;
            }
        }
    } else {
        // Loop through the array to find the interval in which x_interp lies
        for (int i = 0; i < N_UVB_POINTS; i++) {
            if (redshift >= HM12_UVB_redshifts[i] && redshift < HM12_UVB_redshifts[i+1]) {
                
                // Perform linear interpolation
                scale_high = (redshift - HM12_UVB_redshifts[i]) / (HM12_UVB_redshifts[i+1] - HM12_UVB_redshifts[i]);
                scale_low  = 1.0 - scale_high;

                for (int j = 0; j < 27; j++) {
                    if (j<1)  HM12_UVB_z[1][j]  = (scale_low * HM12_UVB_hydrogen[i][j])  + (scale_high * HM12_UVB_hydrogen[i+1][j]);
                    if (j<2)  HM12_UVB_z[2][j]  = (scale_low * HM12_UVB_helium[i][j])    + (scale_high * HM12_UVB_helium[i+1][j]);
                    if (j<6)  HM12_UVB_z[6][j]  = (scale_low * HM12_UVB_carbon[i][j])    + (scale_high * HM12_UVB_carbon[i+1][j]);
                    if (j<7)  HM12_UVB_z[7][j]  = (scale_low * HM12_UVB_nitrogen[i][j])  + (scale_high * HM12_UVB_nitrogen[i+1][j]);
                    if (j<8)  HM12_UVB_z[8][j]  = (scale_low * HM12_UVB_oxygen[i][j])    + (scale_high * HM12_UVB_oxygen[i+1][j]);
                    if (j<10) HM12_UVB_z[10][j] = (scale_low * HM12_UVB_neon[i][j])      + (scale_high * HM12_UVB_neon[i+1][j]);
                    if (j<12) HM12_UVB_z[12][j] = (scale_low * HM12_UVB_magnesium[i][j]) + (scale_high * HM12_UVB_magnesium[i+1][j]);
                    if (j<14) HM12_UVB_z[14][j] = (scale_low * HM12_UVB_silicon[i][j])   + (scale_high * HM12_UVB_silicon[i+1][j]);
                    if (j<16) HM12_UVB_z[16][j] = (scale_low * HM12_UVB_sulfur[i][j])    + (scale_high * HM12_UVB_sulfur[i+1][j]);
                    if (j<26) HM12_UVB_z[26][j]  = (scale_low * HM12_UVB_iron[i][j])      + (scale_high * HM12_UVB_iron[i+1][j]);
                }
                
            }
        }
    }
}