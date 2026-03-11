// collisional_ionization.h
#pragma once

#include "types.hpp"
#include <Kokkos_Core.hpp>

template<int N>
void copy1D (double src[N], Kokkos::View<double[N]> &data_d) {
    data_d = Kokkos::View<double[N]>("data_d", N);
    auto data_h = Kokkos::create_mirror_view(data_d);
    for (int i = 0; i < N; i++)
        data_h(i) = src[i];

    Kokkos::deep_copy(data_d, data_h);
}
void init_collisional_ionization(std::string path_to_data, TabulatedData& tabData) {
    FILE *file;
    double data[27][27][5] = {};

    printf("Initializing collisional ionization rates\n");

    // Check that file exists
    file = fopen((path_to_data + "/collisional_ionization.dat").c_str(), "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open %s/collisional_ionization.dat\n", path_to_data.c_str());
        return;
    }

    // Load in the data
    for (int i = 1; i < 27; i++) { // Loop over elements
        for (int j = i; j > 0; j--) { // Loop over number of electrons
            // Read 5 doubles per line
            int items = fscanf(file, "%lf %lf %lf %lf %lf",
                        &data[i][j][0], &data[i][j][1],
                        &data[i][j][2], &data[i][j][3],
                        &data[i][j][4]);

            // Basic error checking
            if (items != 5) {
                fprintf(stderr, "Error parsing collisional ionization data at i=%d j=%d\n", i, j);
                fclose(file);
                return;
            }
        }
    }
    fclose(file);

    auto h = Kokkos::create_mirror_view(tabData.collisional_ionization_data);
    for (int i = 0; i < 27; i++)
        for (int j = 0; j < 27; j++)
            for (int k = 0; k < 5; k++)
                h(i, j, k) = data[i][j][k];
    Kokkos::deep_copy(tabData.collisional_ionization_data, h);
};

KOKKOS_INLINE_FUNCTION
double coll_ion(double T, double dE, double A, double X, double K, double P){
    // Eqn 1 of Voronov 1997
    double U = dE / (T * 8.61732814974056E-05); // #! K --> eV

    if (U > 80.0) {
        return 0.0;
    }

    return A * (1. + P * sqrt(U)) * pow(U,K) * exp(-U) / (X + U);
}

KOKKOS_FUNCTION
double collisional_ionization(double T, int ion, int element_idx, const TabulatedData& tabData) {
    /*
    collisional ionization rate
    */

    double col = 0.0;

    int n_electron = element_idx - ion;

    if (n_electron > 0) {
        double dE = tabData.collisional_ionization_data(element_idx, n_electron, 0);
        double P  = tabData.collisional_ionization_data(element_idx, n_electron, 1);
        double A  = tabData.collisional_ionization_data(element_idx, n_electron, 2);
        double X  = tabData.collisional_ionization_data(element_idx, n_electron, 3);
        double K  = tabData.collisional_ionization_data(element_idx, n_electron, 4);

        col = coll_ion(T, dE, A, X, K, P);
    }

    col = fmax(col,1e-100);

    return col;
}
