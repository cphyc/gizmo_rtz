// charge_transfer.h
// See https://iopscience.iop.org/article/10.1086/324035/pdf
// Note that this approximation is only good between 10 and 10^3 K
#pragma once

#include <Kokkos_Core.hpp>

double dust_rec_coefs[27][7] = {
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // NA
    { 12.25E0, 8.074E-6, 1.378E0, 5.087E2, 1.586E-2, 0.4723E0, 1.102E-5 }, // Hydrogen
    { 5.572E0, 3.185E-7, 1.512E0, 5.115E3, 3.903E-7, 0.4956E0, 5.494E-7 }, // Helium
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // NA
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // NA
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // NA
    { 45.58E0, 6.089E-3, 1.128E0, 4.331E2, 4.845E-2, 0.8120E0, 1.333E-4 }, // Carbon
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // Nitrogen
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // Oxygen
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // NA
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // Neon
    { 2.178E0, 1.732E-7, 2.133E0, 1.029E4, 1.859E-6, 1.0341E0, 3.223E-5, }, // Sodium
    { 2.510E0, 8.116E-8, 1.864E0, 6.170E4, 2.169E-6, 0.9605E0, 7.232E-5 }, // Magnesium
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // Aluminum
    { 2.166E0, 5.678E-8, 1.874E0, 4.375E4, 1.635E-6, 0.8964E0, 7.538E-5 }, // Silicon
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // NA
    { 3.064E0, 7.769E-5, 1.319E0, 1.087E2, 3.475E-1, 0.4790E0, 4.689E-2 }, // Sulfur
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // Chlorine
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // Argon
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // NA
    { 1.636E0, 8.208E-9, 2.289E0, 1.254E5, 1.349E-9, 1.1506E0, 7.204E-4 }, // Calcium
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // NA
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // NA
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // NA
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // NA
    { 0.000E0, 0.000E0,  0.000E0, 0.000E0, 0.000E0,  0.0000E0, 0.000E0 }, // NA
    { 1.701E0, 9.554E-8, 1.851E0, 5.763E4, 4.116E-8, 0.9456E0, 2.198E-5 }, // Iron
};



KOKKOS_FUNCTION
double dust_recombination_rates(int ion, int nelem, double T, double G, double ne, const Kokkos::View<float[N_TAB_ELEMENTS][7], Kokkos::LayoutRight>& dust_rec_coefs){
    double dust_rec_rate = 0.0;
    double dr_sf = 1.0;

    double loc_T = T;

    // No dust recombination except for the first ionization state.
    // Maybe this will change later...
    if (ion != 1)
    {
        return dust_rec_rate;
    }

    if (T > 1E4) {
        // No dust recombination at high temperatures
        return dust_rec_rate;
    } else if (T > 1E3) {
        // Scale down if gtr than 1.d3
        dr_sf = exp(-1.0 * T / 1E3) / exp(-1.0);
    }

    loc_T = fmax(T,10.0);

    const int erow = elem_row(nelem);

    // First check to make sure that all elements are not zero
    double row_sum = 0.0;
    for (int i = 0; i < 7; i++) {
        row_sum += fabs(dust_rec_coefs(erow, i));
    }

    // In this case there is nothing to compute
    if (row_sum <= 0.0) {
        return dust_rec_rate;
    }

    // Extra fac on the denominator to avoid divide by zero
    double phi = (G + 1E-8) * sqrt(loc_T) / (ne + 1E-10); // units K^1/2 cm^3

    double a1 = dust_rec_coefs(erow, 1) * pow(phi, dust_rec_coefs(erow, 2));
    double a2 = dust_rec_coefs(erow, 3) * pow(T, dust_rec_coefs(erow, 4));
    double a3 = (-1.0 * dust_rec_coefs(erow, 5)) - (dust_rec_coefs(erow, 6) * log(T));

    dust_rec_rate = 1.E-14 * dust_rec_coefs(erow, 0);
    dust_rec_rate = dust_rec_rate / (1.0 + (a1 * (1.0 + (a2 * pow(phi, a3)))));
    dust_rec_rate = dust_rec_rate * dr_sf;

    //   ! Rescale the dust recombination rates due to a PAH normalization issue
    //   ! Zubko assumes 3.3e-5 C in PAH / H atom
    //   ! Weingartner & Draine assume 6e-5 C in PAH / H atom
    dust_rec_rate = dust_rec_rate * (3.3E-5 / 6.0E-5);

    dust_rec_rate = fmax(dust_rec_rate,1E-100);

    return dust_rec_rate;
}
