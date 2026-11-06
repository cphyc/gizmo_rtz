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

#ifdef RTZ_FLOAT32_CHARGE_TRANSFER
using ChargeTransferReal = float;
namespace charge_transfer_math = rtz_float_math;
#else
using ChargeTransferReal = Real;
namespace charge_transfer_math = rtz_math;
#endif

template<typename Table>
KOKKOS_FUNCTION
ChargeTransferReal charge_transfer_recombination(int ion, int nelem, ChargeTransferReal T, const Table& tabData){
    // ion is stage of ionization, 2 for the ion going to the atom
    // nelem is atomic number of element, 2 up to 30
    // Example:  O+ + H => O + H+ is HCTRecom(2,8,1e4)
    // Note that temperature is in linear scale

    ChargeTransferReal ct_recomb = ChargeTransferReal(0.0);

    if (ion == 0)
    {
        return ct_recomb;
    }

    int ipIon = ion - 1;
    //  use statistical charge transfer for ion > 4
    if( ion > 4 )
    {
        ct_recomb = ChargeTransferReal(1.92E-9) * (ChargeTransferReal)(ipIon+1);
        return ct_recomb;
    }

    const int erow = elem_row(nelem);

    // A zero-amplitude fit yields only the rate floor, even if an unused
    // exponential overflows (fmax maps the resulting NaN to the same floor).
    if (ChargeTransferReal(tabData.CTRecomb(0, ipIon, erow)) == ChargeTransferReal(0.0))
        return ChargeTransferReal(1E-100);

    // Make sure te is between temp. boundaries; set constant outside of range
    ChargeTransferReal tused = ChargeTransferReal(0.0);
    tused = charge_transfer_math::fmin(charge_transfer_math::fmax(T,ChargeTransferReal(tabData.CTRecomb(4, ipIon, erow))),ChargeTransferReal(tabData.CTRecomb(5, ipIon, erow)));
    tused *= ChargeTransferReal(1E-4);
    tused = charge_transfer_math::fmax(tused,ChargeTransferReal(1E-10)); //! harley added to prevent zero temperature

    // The interpolation equation
    ct_recomb = ChargeTransferReal(tabData.CTRecomb(0, ipIon, erow)) * ChargeTransferReal(1E-9) * charge_transfer_math::pow(tused,ChargeTransferReal(tabData.CTRecomb(1, ipIon, erow))) * (ChargeTransferReal(1.) + ChargeTransferReal(tabData.CTRecomb(2, ipIon, erow)) * charge_transfer_math::exp(ChargeTransferReal(tabData.CTRecomb(3, ipIon, erow))*tused) );

    // Clamp
    ct_recomb = charge_transfer_math::fmax(ct_recomb,ChargeTransferReal(1E-100));

    return ct_recomb;
}

template<typename Table>
KOKKOS_FUNCTION
ChargeTransferReal charge_transfer_ionization(int ion, int nelem, ChargeTransferReal T, const Table& tabData){
    // ion is stage of ionization, 1 for atom
    // nelem is atomic number of element, 2 up to 30
    // Example:  O + H+ => O+ + H is HCTIon(1,8,1e4)
    // Note that temperature is in linear scale

    ChargeTransferReal ct_ion = ChargeTransferReal(0.0);

    int ipIon = ion;
    if( ipIon > 1 )
    {
        ct_ion = ChargeTransferReal(0.0);
        return ct_ion;
    }

    const int erow = elem_row(nelem);

    if (ChargeTransferReal(tabData.CTIon(0, ipIon, erow)) == ChargeTransferReal(0.0))
        return ChargeTransferReal(1E-100);

    // ! Make sure te is between temp. boundaries; set constant outside of range
    ChargeTransferReal tused = ChargeTransferReal(0.0);
    tused = charge_transfer_math::fmin(charge_transfer_math::fmax(T, ChargeTransferReal(tabData.CTIon(4, ipIon, erow))), ChargeTransferReal(tabData.CTIon(5, ipIon, erow)));
    tused *= ChargeTransferReal(1E-4);
    tused = charge_transfer_math::fmax(tused,ChargeTransferReal(1E-10)); //! harley added to prevent zero temperature

    // ! the interpolation equation
    ct_ion = ChargeTransferReal(tabData.CTIon(0, ipIon, erow)) * ChargeTransferReal(1E-9) * charge_transfer_math::pow(tused,ChargeTransferReal(tabData.CTIon(1, ipIon, erow))) * (ChargeTransferReal(1.) + ChargeTransferReal(tabData.CTIon(2, ipIon, erow)) * charge_transfer_math::exp(ChargeTransferReal(tabData.CTIon(3, ipIon, erow))*tused) ) * charge_transfer_math::exp(-ChargeTransferReal(1.0) * ChargeTransferReal(tabData.CTIon(6, ipIon, erow))/tused);

    // Clamp
    ct_ion = charge_transfer_math::fmax(ct_ion,ChargeTransferReal(1E-100));

    return ct_ion;
}
