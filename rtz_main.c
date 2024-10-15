#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "recombination.h"
#include "collisional_ionization.h"
#include "cosmic_ray_ionization.h"
#include "photoionization_UVB.h"
#include "charge_transfer.h"
#include "dust_recombination.h"

const int MAX_ELEMENTS = 27;   // Define a constant named MAX_SIZE
const double MIN_XION = 1E-20; // Minimum ion fraction
const double MIN_XION_FLOOR = 1E-10; // Minimum ion fraction
const double CONV_ABS = 1E-2; // Value required for convergence
const double X_PCT_RULE = 0.1; // Value required for convergence

// Define the struct
typedef struct {
    int atomic_number;
    int n_ions;
    double atomic_mass;
    double z_solar;
    double G0_photo_rate;
} Element;

// Structs for element properties
void initialize_elements(Element *elements) {
    // Initialize
    for (int i = 0; i < MAX_ELEMENTS; i++) {
        elements[i].atomic_number = -1;
        elements[i].n_ions = -1;
        elements[i].atomic_mass = -1.0;
        elements[i].z_solar = -1.0;
        elements[i].G0_photo_rate = 0.0;
    }

    // Element 1: Hydrogen
    elements[1].atomic_number = 1;
    elements[1].atomic_mass = 1.008;
    elements[1].z_solar = 1.0;
    elements[1].G0_photo_rate = 0.0; // No subionizing PI
    elements[1].n_ions = elements[1].atomic_number + 1;

    // Element 2: Helium
    elements[2].atomic_number = 2;
    elements[2].atomic_mass = 4.0026;
    elements[2].z_solar = 8.51E-02;
    elements[2].G0_photo_rate = 0.0; // No subionizing PI
    elements[2].n_ions = elements[2].atomic_number + 1;

    // Element 6: Carbon
    elements[6].atomic_number = 6;
    elements[6].atomic_mass = 12.0107;
    elements[6].z_solar = 2.69E-04;
    elements[6].G0_photo_rate = 3.39E-10;
    elements[6].n_ions = elements[6].atomic_number + 1;

    // Element 7: Nitrogen
    elements[7].atomic_number = 7;
    elements[7].atomic_mass = 14.0067;
    elements[7].z_solar = 6.76E-05;
    elements[7].G0_photo_rate = 0.0; // No subionizing PI
    elements[7].n_ions = elements[7].atomic_number + 1;

    // Element 8: Oxygen
    elements[8].atomic_number = 8;
    elements[8].atomic_mass = 15.9994;
    elements[8].z_solar = 4.90E-04;
    elements[8].G0_photo_rate = 0.0; // No subionizing PI
    elements[8].n_ions = elements[8].atomic_number + 1;
    elements[8].n_ions = 6; 

    // Element 10: Neon
    elements[10].atomic_number = 10;
    elements[10].atomic_mass = 20.1797;
    elements[10].z_solar = 8.51E-05;
    elements[10].G0_photo_rate = 0.0; // No subionizing PI
    elements[10].n_ions = elements[10].atomic_number + 1;

    // Element 12: Magnesium
    elements[12].atomic_number = 12;
    elements[12].atomic_mass = 24.305;
    elements[12].z_solar = 3.98E-05;
    elements[12].G0_photo_rate = 6.59E-11; 
    elements[12].n_ions = elements[12].atomic_number + 1;

    // Element 14: Silicon
    elements[14].atomic_number = 14;
    elements[14].atomic_mass = 28.0855;
    elements[14].z_solar = 3.24E-05;
    elements[14].G0_photo_rate = 4.47E-09; 
    elements[14].n_ions = elements[14].atomic_number + 1;

    // Element 16: Sulfur
    elements[16].atomic_number = 16;
    elements[16].atomic_mass = 32.065;
    elements[16].z_solar = 1.32E-05;
    elements[16].G0_photo_rate = 1.13E-09; 
    elements[16].n_ions = elements[16].atomic_number + 1;

    // Element 26: Iron
    elements[26].atomic_number = 26;
    elements[26].atomic_mass = 55.854;
    elements[26].z_solar = 3.16E-05;
    elements[26].G0_photo_rate = 4.71E-10;
    elements[26].n_ions = elements[26].atomic_number + 1;
}

typedef struct {
    double n_element;  // n_oxygen cm^-3
    double *ion_fracs; // 9
    double *ion_fracs_new; // 9
} ParticleIonData;

// Structs for element properties
void initialize_ion_fracs(Element *elements, ParticleIonData *n_and_ion_fracs, double hdens) {
    // Loop over all elements
    for (int i = 1; i < MAX_ELEMENTS; i++) { 
        if (elements[i].atomic_number < 1) continue;

        n_and_ion_fracs[i].n_element = hdens * elements[i].z_solar;
        n_and_ion_fracs[i].ion_fracs = (double *)malloc((elements[i].n_ions) * sizeof(double));
        n_and_ion_fracs[i].ion_fracs_new = (double *)malloc((elements[i].n_ions) * sizeof(double));
        for (int j = 0; j < elements[i].n_ions; j++) { 
            n_and_ion_fracs[i].ion_fracs[j] = 0.0;
            n_and_ion_fracs[i].ion_fracs_new[j] = 0.0;
        }
        n_and_ion_fracs[i].ion_fracs[elements[i].n_ions-1] = 1.0;
        n_and_ion_fracs[i].ion_fracs_new[elements[i].n_ions-1] = 1.0;
    }
}

double get_ne(Element *elements, ParticleIonData *n_and_ion_fracs, bool use_new) {
    /*
    Returns the electron number density by looping over all elements 
    and summing their contributions
    */
    double ne = 0E0;

    //Loop over all elements
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;

        //Get the number of ions
        int n_ions = elements[i].n_ions;
        //Loop over all ions
        for (int j = 0; j < n_ions; j++) {
            if (use_new){
                ne += n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs_new[j] * j;
            } else {
                ne += n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs[j] * j;
            }
        } //End loop ion fracs
    } //End loop elements

    return fmax(ne,MIN_XION);
}

void reduce_xion(double ion_fracs[], int size) {
    /*
    Makes sure that the ion fractions sum to one and that 
    the ion fractions don't go below some minimum value
    */
    double total_ion_frac = 0.0;

    //Loop over all ions and get the total
    for (int i = 0; i < size; i++) {
        ion_fracs[i] = fmax(ion_fracs[i],MIN_XION);
        total_ion_frac += ion_fracs[i];
        // total_ion_frac += fmax(ion_fracs[i],MIN_XION);
    }
    //Loop again to normalize
    for (int i = 0; i < size; i++) {
        ion_fracs[i] /= total_ion_frac;
    }
}

void copyArray(double src[], double dest[], int size) {
    for (int i = 0; i < size; i++) {
        dest[i] = src[i];
    }
}

void get_chemical_eqm(Element *elements, ParticleIonData *n_and_ion_fracs, double TK, 
                      double total_dt, double UV_background_G0,
                      double primary_cosmic_ray_ionization_rate,
                      double dust_to_gas_mass_ratio_over_mw) {
    /*
    Computes chemical equilibrium

    if total_dt [s] is <=0 --> run to equilibrium otherwise stop at a fixed time
    */

    //Parameters
    double ddt = 3.15E16; // Initial timestep [s]
    // If we are not running to convergence, try to take one single step
    if (total_dt > 0) {
        ddt = total_dt;
    }

    // Convergence variables 
    int success_counter     = 0;
    int convergence_counter = 0;
    int total_iterations    = 0;
    bool model_converged    = false;
    bool convergence_bool   = true;
    bool x_percent_rule     = true;
    double total_time       = 0.E0;
    double residual;  
    double ne_residual;  
    double max_residual;
    double max_residual_all;  

    //Physical variables
    double ne; //Electron number density
    double ne_initial; //Electron number density for convergence
    double cr; //Creation rate
    double de; //Destruction rate
    double xe; //Electron fraction (n_electron/nH)
    double phi_s; // Secondary electron CR factor
    double total_cosmic_ray_ionization_rate; //Total CR ionization rate (acounting for secondaries)
    double cosmic_ray_scale_factor; //Most rates normalized to 1e-16

    int n_ions; //Number of ions --> reset per element
    int atomic_number;

    // Set the cosmic ray scale factor -> only needed for secondary UVs
    cosmic_ray_scale_factor = primary_cosmic_ray_ionization_rate / 1E-16;

    //# infinite loop
    while (1) {  // Infinite loop
        total_iterations += 1;
        //Loop over elements and set the ion_fracs new to
        //the old ion fracs
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (elements[i].atomic_number < 1) continue;

            int size = elements[i].n_ions;
            for (int j = 0; j < size; j++) {
                n_and_ion_fracs[i].ion_fracs_new[j] = n_and_ion_fracs[i].ion_fracs[j];
            }
        } // End element loop

        // First get the electron fraction
        ne = get_ne(elements,n_and_ion_fracs,false);
        // Update the electron fraction
        xe = ne / n_and_ion_fracs[1].n_element;
        // Measure phi_s for secondary CR ionization
        phi_s = secondary_cr_rates(xe);
        // Now calculate the total CR ionization rate
        total_cosmic_ray_ionization_rate = primary_cosmic_ray_ionization_rate * (1.0 + phi_s);

        // Initialize the X percent rule
        x_percent_rule = true;

        // Convergence
        convergence_bool = true;
        max_residual_all = -999.;

        // Make sure the timestep isn't larger than how long we
        // need to evolve for, otherwise shorten it
        if (total_dt > 0.0) {
            if ((total_time+ddt) > total_dt){
                ddt = total_dt - total_time;
            }
        }

        // Loop over all elements
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (elements[i].atomic_number < 1) continue;

            // Get the atomic number
            atomic_number = elements[i].atomic_number;

            // Get the number of ions
            n_ions = elements[i].n_ions;

            // Initialize the max residual
            max_residual = 0.E0;

            // Loop over the number of ions
            for (int j = 0; j < n_ions; j++) {

                // Update the HI number density in case it has change e.g. due to charge exchange
                double HI_number_density = n_and_ion_fracs[1].ion_fracs_new[0] * n_and_ion_fracs[1].n_element;
                double dust_effective_number_density = HI_number_density * dust_to_gas_mass_ratio_over_mw;

                /////////////////////////
                //       Creation      //
                /////////////////////////
                cr = 0.E0;

                // Recombinations of the more excited ionization state
                if (j < (n_ions - 1)) cr += recombination(TK, j+1, i) * ne * n_and_ion_fracs[i].ion_fracs_new[j+1]; 

                // Collisional ionization of the less excited state
                if (j > 0) cr += collisional_ionization(TK, j-1, i) * ne * n_and_ion_fracs[i].ion_fracs_new[j-1];

                // Photoionization of the less excited state
                if (j > 0) cr += HM12_UVB_z[i][j-1] * n_and_ion_fracs[i].ion_fracs_new[j-1];

                // Photoionization by sub-ionizing ISRF --> only impacts lowest ionization states
                if (j == 1) cr += UV_background_G0 * elements[i].G0_photo_rate * n_and_ion_fracs[i].ion_fracs_new[j-1];

                // Cosmic ray ionization of the less excited state
                if (j > 0) cr += cosmic_ray_ionization_rates[i][j-1] * total_cosmic_ray_ionization_rate * n_and_ion_fracs[i].ion_fracs_new[j-1];

                // Cosmic ray ionization of the less excited state from induced UV
                if (j == 1) cr += cosmic_ray_ionization_rates_induced_UV[i] * cosmic_ray_scale_factor * n_and_ion_fracs[i].ion_fracs_new[j-1];

                // Recombination on dust for the more excited state
                if (j < (n_ions - 1)) cr += dust_recombination_rates(j+1, i, TK, UV_background_G0, ne) * dust_effective_number_density * n_and_ion_fracs[i].ion_fracs_new[j+1];

                /////////////////////////
                //     Destruction     //
                /////////////////////////
                de = 0.E0;

                // Collisional ionization 
                if (j < (n_ions - 1)) de += collisional_ionization(TK, j, i) * ne;

                // Photoionization
                if (j < (n_ions - 1)) de += HM12_UVB_z[i][j];

                // Photoionization by sub-ionizing ISRF --> only impacts lowest ionization states
                if (j == 0) de += UV_background_G0 * elements[i].G0_photo_rate;

                // Recombination
                if (j > 0) de += recombination(TK, j, i) * ne;

                // Cosmic ray ionization
                if (j < (n_ions - 1)) de += cosmic_ray_ionization_rates[i][j] * total_cosmic_ray_ionization_rate;

                // Cosmic ray ionization from induced UV
                if (j == 0) de += cosmic_ray_ionization_rates_induced_UV[i] * cosmic_ray_scale_factor;

                // Recombination on dust
                if (j > 0) de += dust_recombination_rates(j, i, TK, UV_background_G0, ne) * dust_effective_number_density;

                /////////////////////////
                //   Charge Transfer   //
                /////////////////////////
                // Note, this was split off due to cross species
                // coupling. Saves us an extra double loop
                // Charge exchange
                if (i == 1) {  // If element is hydrogen
                    // Loop over all other elements
                    for (int ii = 2; ii < MAX_ELEMENTS; ii++){
                        if (elements[ii].atomic_number < 1) continue;

                        // Loop over all other ionization states
                        for (int jj = 0; jj < elements[ii].n_ions; jj++) {
                            double paired_ion_number_density = n_and_ion_fracs[ii].ion_fracs_new[jj] * n_and_ion_fracs[ii].n_element;
                            if (j == 0) { // H
                                cr += charge_transfer_ionization(jj,ii,TK) * n_and_ion_fracs[i].ion_fracs_new[j+1] * paired_ion_number_density; //! Example:  O + H+ => O+ + H
                                de += charge_transfer_recombination(jj,ii,TK) * paired_ion_number_density; //! Example:  O+ + H => O + H+
                            } else { // H+
                                de += charge_transfer_ionization(jj,ii,TK) * paired_ion_number_density; //! Example:  O + H+ => O+ + H
                                cr += charge_transfer_recombination(jj,ii,TK) * n_and_ion_fracs[i].ion_fracs_new[j-1] * paired_ion_number_density; //! Example:  O+ + H => O + H+
                            }
                        } // end loop over ionization states
                    } // end loop over other elements
                } else { // All other elements
                    double HI_number_density = n_and_ion_fracs[1].ion_fracs_new[0] * n_and_ion_fracs[1].n_element;
                    double HII_number_density = n_and_ion_fracs[1].ion_fracs_new[1] * n_and_ion_fracs[1].n_element;

                    if (j > 0) {
                        // Ionization from less excited state
                        cr += charge_transfer_ionization(j-1,i,TK) * n_and_ion_fracs[i].ion_fracs_new[j-1] * HII_number_density; //! Example:  O + H+ => O+ + H   

                        // Charge exchange recombination 
                        de += charge_transfer_recombination(j,i,TK) * HI_number_density; //! Example:  O+ + H => O + H+
                    }    

                    if (j < (n_ions - 1)) {
                        // Charge exchange ionization
                        de += charge_transfer_ionization(j,i,TK) * HII_number_density; //! Example:  O + H+ => O+ + H   

                        // Charge exchange recombination from the more excited state
                        cr += charge_transfer_recombination(j+1,i,TK) * n_and_ion_fracs[i].ion_fracs_new[j+1] * HI_number_density; //! Example:  O+ + H => O + H+
                    }           
                } 

                /////////////////////////
                //       Update        //
                /////////////////////////
                // The update
                n_and_ion_fracs[i].ion_fracs_new[j] = (cr*ddt + n_and_ion_fracs[i].ion_fracs_new[j])/(1.0 + de*ddt);

                // Make sure ions sum to 1
                int size = elements[i].n_ions;
                reduce_xion(n_and_ion_fracs[i].ion_fracs_new, size);

                // Get the new electron fraction
                ne_initial = ne;
                ne = get_ne(elements,n_and_ion_fracs,true);
                xe = ne / n_and_ion_fracs[1].n_element;
                phi_s = secondary_cr_rates(xe);
                total_cosmic_ray_ionization_rate = primary_cosmic_ray_ionization_rate * (1.0 + phi_s);

                /////////////////////////
                //     Convergence     //
                /////////////////////////
                // X% rule
                residual = fabs(n_and_ion_fracs[i].ion_fracs_new[j] - n_and_ion_fracs[i].ion_fracs[j]) / fmax(n_and_ion_fracs[i].ion_fracs[j],MIN_XION_FLOOR);
                
                // electron density residual
                ne_residual = fabs(ne - ne_initial) / fmax(ne,MIN_XION_FLOOR);
                residual = fmax(residual,ne_residual);

                // Update the max residual
                max_residual = fmax(max_residual,residual);

                // Max residual over all elements
                max_residual_all = fmax(max_residual_all,max_residual);

                //TODO(code): X percent rule a parameter
                if (residual > X_PCT_RULE){
                    // Declare the x-percent rule was violated
                    x_percent_rule = false;
                    // Break the loop over ions
                    break;
                }
            } // End ion loop

            //If the X% rule is violated restart with shorter timestep
            if (!x_percent_rule){
                // Reset the success counter
                success_counter = 0;
                // Break out of the element loop
                break;
            }

            // Check if the model has converged
            if (max_residual > CONV_ABS) convergence_bool = false;

        } // End element loop

        /////////////////////////
        //   More Convergence  //
        /////////////////////////

        // Set new dt --> Strategy from Nick Gnedin
        max_residual_all /= X_PCT_RULE;
        if (max_residual_all > 1.0) {
            ddt = 0.9 * ddt / sqrt(2.+max_residual_all); 
        } else {
            ddt = 0.9 * ddt / pow(0.01+max_residual_all,0.3);
        }
        // ddt = ddt * fmin(0.9*pow(X_PCT_RULE/max_residual_all,1.0/2.0),5.0); // --> ChatGPT suggestion
        ddt = fmin(ddt,1E20);

        // Check if the model has converged
        if (convergence_bool){
            // If yes, increment counter
            convergence_counter += 1;
        } else {
            // Else reset the counter
            convergence_counter = 0;
        }

        // If all elements completed
        // Loop over elements and ions and 
        // set old values to new values
        if (x_percent_rule) {
            for (int i = 1; i < MAX_ELEMENTS; i++) {
                if (elements[i].atomic_number < 1) continue;

                int size = elements[i].n_ions;
                for (int j = 0; j < size; j++) {
                    n_and_ion_fracs[i].ion_fracs[j] = n_and_ion_fracs[i].ion_fracs_new[j];
                }
            } // End element loop
        }

        // Check if we are converged
        //TODO(code): make convergence number a parameter
        if (convergence_counter > 100) model_converged = true;


        // Update the total time
        total_time += ddt;

        // Break the infinite loop if the timestep has been reached
        if (total_dt > 0.0) {
            //TODO(code): Double check this parameter
            if (fabs(total_time-total_dt)/total_dt < 1e-6) break;
        }

        // Finish the calculation if model converged
        if (model_converged) break; //break from infinite loop

    } // End infinite loop

    // printf("Total iterations %d\n",total_iterations);
}

int main() {
    double redshift = 20.5;          // Redshift of the calculation
    double UV_background_G0 = 0.0;  // G0 (sub ionizing radiaiton field)
    double generic_cosmic_ray_ionization_rate = 0.0; // Background CR ionization rate
    double dust_to_gas_mass_ratio_over_mw = 1.0; // dust to gas mass ratio normalized to the MW

    // Declare an array (list) of 3 elements (structs)
    Element elements[MAX_ELEMENTS];
    ParticleIonData n_and_ion_fracs[MAX_ELEMENTS];

    // Initialize the elements
    initialize_elements(elements);

    // Initialize the ion fracs
    initialize_ion_fracs(elements,n_and_ion_fracs,1E-3);

    // Initialize the UV background data
    load_UVB_data(); // First load the UVB data from files
    update_UVB(redshift); // Interpolate to the correct redshift

    // Initialize cosmic ray rates
    initialize_cr_rates();

    // Initialize the charge transfer rates
    load_ct_rates();

    // // Loop over temperature and get the ion fractions
    for (int i = 0; i < 500; i++) {
        //double TK = pow(10.0,3.5+((double)i*0.01));
        double TK = pow(10.0,8.0-((double)i*0.01));
        get_chemical_eqm(elements, n_and_ion_fracs, TK, -1.0, UV_background_G0,
                         generic_cosmic_ray_ionization_rate, dust_to_gas_mass_ratio_over_mw);
        printf("%e ",TK);
        for (int j = 0; j < 6; j++) {
            printf("%e ",n_and_ion_fracs[8].ion_fracs[j]);
        }
        printf("\n");
    }

    return 0;
}
