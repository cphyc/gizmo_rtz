// cosmic_ray_ionization.h
#ifndef COSMIC_RAY_IONIZATION
#define COSMIC_RAY_IONIZATION

// Initialization of cosmic ray rates tables
void initialize_cr_rates();
double secondary_cr_rates(double xe);

// Array that contains cosmic ray ionization values
// This is currently a bit of waste of space but it's small
// enough that it shouldn't matter
// Index is element number | Ion
extern double cosmic_ray_ionization_rates[27][27];
extern double cosmic_ray_ionization_rates_induced_UV[27];

#endif