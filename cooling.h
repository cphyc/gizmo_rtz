// cooling.h
#ifndef COOLING_H
#define COOLING_H

// Metal cooling table initialization
void initialize_high_temperature_metal_cooling();
void init_fine_structure_tables();
double all_cooling(double T, double ne, double aexp, double *element_number_densities, 
                   double *element_number_ions, double **element_ion_fractions,
                   double G0, double f_dg, double xe, double xi_h_cr, double xi_h2_cr,
                   double ss_factor);

#endif // COOLING_H