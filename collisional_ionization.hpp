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
void init_collisional_ionization(std::string path_to_data, RawTabulatedData& tabData) {
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
    for (int i = 0; i < 27; i++) {
        int row = elem_row(i);
        if (row < 0) continue;
        for (int j = 0; j < 27; j++)
            for (int k = 0; k < 5; k++)
                h(row, j, k) = data[i][j][k];
    }
    Kokkos::deep_copy(tabData.collisional_ionization_data, h);
};

// Per-file arithmetic precision; table loading retains its original storage.
#ifdef RTZ_FLOAT32_COLLISIONAL
using CollisionalReal = float;
namespace collisional_math = rtz_float_math;
#else
using CollisionalReal = Real;
namespace collisional_math = rtz_math;
#endif

KOKKOS_INLINE_FUNCTION
CollisionalReal coll_ion(CollisionalReal T, CollisionalReal dE, CollisionalReal A, CollisionalReal X, CollisionalReal K, CollisionalReal P){
    // Eqn 1 of Voronov 1997
    CollisionalReal U = dE / (T * CollisionalReal(8.61732814974056E-05)); // #! K --> eV

    if (U > CollisionalReal(80.0)) {
        return CollisionalReal(0.0);
    }

    return A * (CollisionalReal(1.) + P * collisional_math::sqrt(U)) * collisional_math::pow(U,K) * collisional_math::exp(-U) / (X + U);
}

template<typename Table>
KOKKOS_FUNCTION
CollisionalReal collisional_ionization(CollisionalReal T, int ion, int element_idx, const Table& tabData) {
    /*
    collisional ionization rate
    */

    CollisionalReal col = CollisionalReal(0.0);

    int n_electron = element_idx - ion;

    if (n_electron > 0) {
        const int erow = elem_row(element_idx);
        CollisionalReal dE = tabData.collisional_ionization_data(erow, n_electron, 0);
        CollisionalReal P  = tabData.collisional_ionization_data(erow, n_electron, 1);
        CollisionalReal A  = tabData.collisional_ionization_data(erow, n_electron, 2);
        CollisionalReal X  = tabData.collisional_ionization_data(erow, n_electron, 3);
        CollisionalReal K  = tabData.collisional_ionization_data(erow, n_electron, 4);

        col = coll_ion(T, dE, A, X, K, P);
    }

    col = collisional_math::fmax(col,CollisionalReal(1e-100));

    return col;
}
