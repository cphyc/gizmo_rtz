#pragma once
#include <Kokkos_Core.hpp>


typedef struct
{
    int atomic_number;
    int n_ions;
    int n_mol;
    double atomic_mass;
    double z_solar;
    double G0_photo_rate;
    double depletion;
} Element;

typedef struct
{
    Kokkos::View<double***> HM12_UVB_z;
    Kokkos::View<double*> G0_heating_rates;
    Kokkos::View<double**> cosmic_ray_ionization_rates;
    Kokkos::View<double*> cosmic_ray_ionization_rates_induced_UV;
    Kokkos::View<double*> cosmic_ray_ionization_rates_induced_UV_heat;
    Kokkos::View<double**> dust_rec_coefs;
    Kokkos::View<double***> CTIon;
    Kokkos::View<double***> CTRecomb;
    Kokkos::View<double***> fs_cool_tab;

    Kokkos::View<double***> high_t_cooling_rates;
    Kokkos::View<bool**> high_t_cooling_rates_tflag;

} TabulatedData;
