// charge_transfer.h
#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"


double CTRecomb[6][4][31];
double CTIon[7][3][31];

void load_ct_rates(std::string path_to_data){
    // Load the charge transfer ionization
    // and recombination rates from file

    FILE *file;

    // Ionization
    file = fopen((path_to_data + "/charge_transfer/ct_ionization.dat").c_str(), "r");
    if (file == NULL) {
        printf("Error: Could not open charge transfer ionization file\n");
    }

    // Reading data from the file into the 3D array
    for (int i = 3; i < 31; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 7; k++) {
                fscanf(file, "%lf", &CTIon[k][j][i]);
            }
        }
    }
    // Close the file
    fclose(file);

    // Zero out helium
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 7; k++) {
            CTIon[k][j][2] = 0.0;
        }
    }

    // Recombination
    file = fopen((path_to_data + "/charge_transfer/ct_recombination.dat").c_str(), "r");
    if (file == NULL) {
        printf("Error: Could not open charge transfer recombination file\n");
    }

    // Reading data from the file into the 3D array
    for (int i = 2; i < 31; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 6; k++) {
                fscanf(file, "%lf", &CTRecomb[k][j][i]);
            }
        }
    }
    // Close the file
    fclose(file);

}

KOKKOS_FUNCTION
Real charge_transfer_recombination(int ion, int nelem, Real T, const TabulatedData& tabData){
    // ion is stage of ionization, 2 for the ion going to the atom
    // nelem is atomic number of element, 2 up to 30
    // Example:  O+ + H => O + H+ is HCTRecom(2,8,1e4)
    // Note that temperature is in linear scale

    Real ct_recomb = Real(0.0);

    if (ion == 0)
    {
        return ct_recomb;
    }

    int ipIon = ion - 1;
    //  use statistical charge transfer for ion > 4
    if( ion > 4 )
    {
        ct_recomb = Real(1.92E-9) * (Real)(ipIon+1);
        return ct_recomb;
    }

    const int erow = elem_row(nelem);

    // Make sure te is between temp. boundaries; set constant outside of range
    Real tused = Real(0.0);
    tused = rtz_math::fmin(rtz_math::fmax(T,tabData.CTRecomb(4, ipIon, erow)),tabData.CTRecomb(5, ipIon, erow));
    tused *= Real(1E-4);
    tused = rtz_math::fmax(tused,Real(1E-10)); //! harley added to prevent zero temperature

    // The interpolation equation
    ct_recomb = tabData.CTRecomb(0, ipIon, erow) * Real(1E-9) * rtz_math::pow(tused,tabData.CTRecomb(1, ipIon, erow)) * (Real(1.) + tabData.CTRecomb(2, ipIon, erow) * rtz_math::exp(tabData.CTRecomb(3, ipIon, erow)*tused) );

    // Clamp
    ct_recomb = rtz_math::fmax(ct_recomb,Real(1E-100));

    return ct_recomb;
}

KOKKOS_FUNCTION
Real charge_transfer_ionization(int ion, int nelem, Real T, const TabulatedData& tabData){
    // ion is stage of ionization, 1 for atom
    // nelem is atomic number of element, 2 up to 30
    // Example:  O + H+ => O+ + H is HCTIon(1,8,1e4)
    // Note that temperature is in linear scale

    Real ct_ion = Real(0.0);

    int ipIon = ion;
    if( ipIon > 1 )
    {
        ct_ion = Real(0.0);
        return ct_ion;
    }

    const int erow = elem_row(nelem);

    // ! Make sure te is between temp. boundaries; set constant outside of range
    Real tused = Real(0.0);
    tused = rtz_math::fmin(rtz_math::fmax(T, tabData.CTIon(4, ipIon, erow)), tabData.CTIon(5, ipIon, erow));
    tused *= Real(1E-4);
    tused = rtz_math::fmax(tused,Real(1E-10)); //! harley added to prevent zero temperature

    // ! the interpolation equation
    ct_ion = tabData.CTIon(0, ipIon, erow) * Real(1E-9) * rtz_math::pow(tused,tabData.CTIon(1, ipIon, erow)) * (Real(1.) + tabData.CTIon(2, ipIon, erow) * rtz_math::exp(tabData.CTIon(3, ipIon, erow)*tused) ) * rtz_math::exp(-Real(1.0) * tabData.CTIon(6, ipIon, erow)/tused);

    // Clamp
    ct_ion = rtz_math::fmax(ct_ion,Real(1E-100));

    return ct_ion;
}
