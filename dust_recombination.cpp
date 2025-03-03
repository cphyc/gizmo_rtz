// cosmic_transfer.c
// See https://iopscience.iop.org/article/10.1086/324035/pdf
// Note that this approximation is only good between 10 and 10^3 K
#include <stdio.h>
#include <math.h>
#include <Kokkos_Core.hpp>


KOKKOS_FUNCTION
double dust_recombination_rates(int ion, int nelem, double T, double G, double ne, Kokkos::View<double[27][7]> dust_rec_coefs){
    double dust_rec_rate = 0.0;
    double dr_sf = 1.0;

    // No dust recombombination except for the first ionization state.
    // Maybe this will change later...
    if (ion != 1)
    {
        return dust_rec_rate;
    }

    if (T > 1E4)
    {
        // No dust recombination at high temperatures
        return dust_rec_rate;
    } else if (T > 1E3)
    {
        // Scale down if gtr than 1.d3
        dr_sf = exp(-1.0 * T / 1E3) / exp(-1.0);
    }

    if (T < 10.0)
    {
        // No dust recombination at very low temperatures
        return dust_rec_rate;
    }

    // First check to make sure that all elements are not zero
    double row_sum = 0.0;
    for (int i = 0; i < 7; i++) {
        row_sum += fabs(dust_rec_coefs(nelem, i));
    } 

    // In this case there is nothing to compute
    if (row_sum <= 0.0)
    {
        return dust_rec_rate;
    }

    // Extra fac on the denominator to avoid divide by zero
    double phi = (G + 1E-8) * sqrt(T) / (ne + 1E-10); // units K^1/2 cm^3

    double a1 = dust_rec_coefs(nelem, 1) * pow(phi, dust_rec_coefs(nelem, 2));
    double a2 = dust_rec_coefs(nelem, 3) * pow(T, dust_rec_coefs(nelem, 4));
    double a3 = (-1.0 * dust_rec_coefs(nelem, 5)) - (dust_rec_coefs(nelem, 6) * log(T));

    dust_rec_rate = 1.E-14 * dust_rec_coefs(nelem, 0);
    dust_rec_rate = dust_rec_rate / (1.0 + (a1 * (1.0 + (a2 * pow(phi, a3)))));
    dust_rec_rate = dust_rec_rate * dr_sf;

    //   ! Rescale the dust recombination rates due to a PAH normalization issue
    //   ! Zubko assumes 3.3e-5 C in PAH / H atom
    //   ! Weingartner & Draine assume 6e-5 C in PAH / H atom
    dust_rec_rate = dust_rec_rate * (3.3E-5 / 6.0E-5);

    return dust_rec_rate;
}