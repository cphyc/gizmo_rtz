// molecules.h
#ifndef MOLECULES
#define MOLECULES

// H2 formation on dust and via the primordial H- channel
double alpha_H2(double T, double dust_to_gas_mass_ratio_over_mw, double xe, double H2_cosmic_ray_ionization_rate, double G0, double xHI, double xHII);

// Collosional H2 destruction
double beta_H2(double T, double nH, double xHI, double xH2, double xHe, double ne, double nHI, double nH2, double nHeI);

#endif