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

    Kokkos::View<double[7][6]> RR_rates_carbon;
    Kokkos::View<double[7][9]> DR_rates_c_carbon;
    Kokkos::View<double[7][9]> DR_rates_e_carbon;
    Kokkos::View<double[8][6]> RR_rates_nitrogen;
    Kokkos::View<double[8][9]> DR_rates_c_nitrogen;
    Kokkos::View<double[8][9]> DR_rates_e_nitrogen;
    Kokkos::View<double[9][6]> RR_rates_oxygen;
    Kokkos::View<double[9][9]> DR_rates_c_oxygen;
    Kokkos::View<double[9][9]> DR_rates_e_oxygen;
    Kokkos::View<double[11][6]> RR_rates_neon;
    Kokkos::View<double[11][9]> DR_rates_c_neon;
    Kokkos::View<double[11][9]> DR_rates_e_neon;
    Kokkos::View<double[13][6]> RR_rates_magnesium;
    Kokkos::View<double[13][9]> DR_rates_c_magnesium;
    Kokkos::View<double[13][9]> DR_rates_e_magnesium;
    Kokkos::View<double[15][6]> RR_rates_silicon;
    Kokkos::View<double[15][9]> DR_rates_c_silicon;
    Kokkos::View<double[15][9]> DR_rates_e_silicon;
    Kokkos::View<double[17][6]> RR_rates_sulfur;
    Kokkos::View<double[17][9]> DR_rates_c_sulfur;
    Kokkos::View<double[17][9]> DR_rates_e_sulfur;
    Kokkos::View<double[17][2]> RR_rates_alt_sulfur;
    Kokkos::View<double[17][4]> DR_rates_alt_sulfur;
    Kokkos::View<double[27][6]> RR_rates_iron;
    Kokkos::View<double[27][9]> DR_rates_c_iron;
    Kokkos::View<double[27][9]> DR_rates_e_iron;
    Kokkos::View<double[27][2]> RR_rates_alt_iron;
    Kokkos::View<double[27][8]> DR_rates_alt_iron;

    Kokkos::View<double[6]> dE_carbon;
    Kokkos::View<double[6]> A_carbon;
    Kokkos::View<double[6]> X_carbon;
    Kokkos::View<double[6]> K_carbon;
    Kokkos::View<double[6]> P_carbon;

    Kokkos::View<double[8]> dE_oxygen;
    Kokkos::View<double[8]> A_oxygen;
    Kokkos::View<double[8]> X_oxygen;
    Kokkos::View<double[8]> K_oxygen;
    Kokkos::View<double[8]> P_oxygen;

    Kokkos::View<double[7]> dE_nitrogen;
    Kokkos::View<double[7]> A_nitrogen;
    Kokkos::View<double[7]> X_nitrogen;
    Kokkos::View<double[7]> K_nitrogen;
    Kokkos::View<double[7]> P_nitrogen;

    Kokkos::View<double[10]> dE_neon;
    Kokkos::View<double[10]> A_neon;
    Kokkos::View<double[10]> X_neon;
    Kokkos::View<double[10]> K_neon;
    Kokkos::View<double[10]> P_neon;

    Kokkos::View<double[12]> dE_magnesium;
    Kokkos::View<double[12]> A_magnesium;
    Kokkos::View<double[12]> X_magnesium;
    Kokkos::View<double[12]> K_magnesium;
    Kokkos::View<double[12]> P_magnesium;

    Kokkos::View<double[14]> dE_silicon;
    Kokkos::View<double[14]> A_silicon;
    Kokkos::View<double[14]> X_silicon;
    Kokkos::View<double[14]> K_silicon;
    Kokkos::View<double[14]> P_silicon;

    Kokkos::View<double[16]> dE_sulfur;
    Kokkos::View<double[16]> A_sulfur;
    Kokkos::View<double[16]> X_sulfur;
    Kokkos::View<double[16]> K_sulfur;
    Kokkos::View<double[16]> P_sulfur;

    Kokkos::View<double[26]> dE_iron;
    Kokkos::View<double[26]> A_iron;
    Kokkos::View<double[26]> X_iron;
    Kokkos::View<double[26]> K_iron;
    Kokkos::View<double[26]> P_iron;

} TabulatedData;
