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
    Kokkos::View<double[27][27][2]> HM12_UVB_z {"HM12_UVB_z", 27, 27, 2};
    Kokkos::View<double[27]> G0_heating_rates {"G0_heating_rates", 27};
    Kokkos::View<double[27][27]> cosmic_ray_ionization_rates {"cosmic_ray_ionization_rates", 27, 27};
    Kokkos::View<double[27]> cosmic_ray_ionization_rates_induced_UV {"cosmic_ray_ionization_rates_induced_UV", 27};
    Kokkos::View<double[27]> cosmic_ray_ionization_rates_induced_UV_heat {"cosmic_ray_ionization_rates_induced_UV_heat", 27};
    Kokkos::View<double[27][7]> dust_rec_coefs {"dust_rec_coefs", 27, 7};
    Kokkos::View<double[7][3][31]> CTIon {"CTIon", 7, 3, 31};
    Kokkos::View<double[6][4][31]> CTRecomb {"CTRecomb", 6, 4, 31};
    Kokkos::View<double[27][160][8]> fs_cool_tab {"fs_cool_tab", 27, 160, 8};

    Kokkos::View<double[121][27][27]> high_t_cooling_rates {"high_t_cooling_rates", 121, 27, 27};
    Kokkos::View<bool[27][27]> high_t_cooling_rates_tflag {"high_t_cooling_rates_tflag", 27, 27};

    Kokkos::View<double[7][6]> RR_rates_carbon {"RR_rates_carbon", 7, 6};
    Kokkos::View<double[7][9]> DR_rates_c_carbon {"DR_rates_c_carbon", 7, 9};
    Kokkos::View<double[7][9]> DR_rates_e_carbon {"DR_rates_e_carbon", 7, 9};
    Kokkos::View<double[8][6]> RR_rates_nitrogen {"RR_rates_nitrogen", 8, 6};
    Kokkos::View<double[8][9]> DR_rates_c_nitrogen {"DR_rates_c_nitrogen", 8, 9};
    Kokkos::View<double[8][9]> DR_rates_e_nitrogen {"DR_rates_e_nitrogen", 8, 9};
    Kokkos::View<double[9][6]> RR_rates_oxygen {"RR_rates_oxygen", 9, 6};
    Kokkos::View<double[9][9]> DR_rates_c_oxygen {"DR_rates_c_oxygen", 9, 9};
    Kokkos::View<double[9][9]> DR_rates_e_oxygen {"DR_rates_e_oxygen", 9, 9};
    Kokkos::View<double[11][6]> RR_rates_neon {"RR_rates_neon", 11, 6};
    Kokkos::View<double[11][9]> DR_rates_c_neon {"DR_rates_c_neon", 11, 9};
    Kokkos::View<double[11][9]> DR_rates_e_neon {"DR_rates_e_neon", 11, 9};
    Kokkos::View<double[13][6]> RR_rates_magnesium {"RR_rates_magnesium", 13, 6};
    Kokkos::View<double[13][9]> DR_rates_c_magnesium {"DR_rates_c_magnesium", 13, 9};
    Kokkos::View<double[13][9]> DR_rates_e_magnesium {"DR_rates_e_magnesium", 13, 9};
    Kokkos::View<double[15][6]> RR_rates_silicon {"RR_rates_silicon", 15, 6};
    Kokkos::View<double[15][9]> DR_rates_c_silicon {"DR_rates_c_silicon", 15, 9};
    Kokkos::View<double[15][9]> DR_rates_e_silicon {"DR_rates_e_silicon", 15, 9};
    Kokkos::View<double[17][6]> RR_rates_sulfur {"RR_rates_sulfur", 17, 6};
    Kokkos::View<double[17][9]> DR_rates_c_sulfur {"DR_rates_c_sulfur", 17, 9};
    Kokkos::View<double[17][9]> DR_rates_e_sulfur {"DR_rates_e_sulfur", 17, 9};
    Kokkos::View<double[17][2]> RR_rates_alt_sulfur {"RR_rates_alt_sulfur", 17, 2};
    Kokkos::View<double[17][4]> DR_rates_alt_sulfur {"DR_rates_alt_sulfur", 17, 4};
    Kokkos::View<double[27][6]> RR_rates_iron {"RR_rates_iron", 27, 6};
    Kokkos::View<double[27][9]> DR_rates_c_iron {"DR_rates_c_iron", 27, 9};
    Kokkos::View<double[27][9]> DR_rates_e_iron {"DR_rates_e_iron", 27, 9};
    Kokkos::View<double[27][2]> RR_rates_alt_iron {"RR_rates_alt_iron", 27, 2};
    Kokkos::View<double[27][8]> DR_rates_alt_iron {"DR_rates_alt_iron", 27, 8};

    Kokkos::View<double[6]> dE_carbon {"dE_carbon", 6};
    Kokkos::View<double[6]> A_carbon {"A_carbon", 6};
    Kokkos::View<double[6]> X_carbon {"X_carbon", 6};
    Kokkos::View<double[6]> K_carbon {"K_carbon", 6};
    Kokkos::View<double[6]> P_carbon {"P_carbon", 6};

    Kokkos::View<double[7]> dE_nitrogen {"dE_nitrogen", 7};
    Kokkos::View<double[7]> A_nitrogen {"A_nitrogen", 7};
    Kokkos::View<double[7]> X_nitrogen {"X_nitrogen", 7};
    Kokkos::View<double[7]> K_nitrogen {"K_nitrogen", 7};
    Kokkos::View<double[7]> P_nitrogen {"P_nitrogen", 7};

    Kokkos::View<double[8]> dE_oxygen {"dE_oxygen", 8};
    Kokkos::View<double[8]> A_oxygen {"A_oxygen", 8};
    Kokkos::View<double[8]> X_oxygen {"X_oxygen", 8};
    Kokkos::View<double[8]> K_oxygen {"K_oxygen", 8};
    Kokkos::View<double[8]> P_oxygen {"P_oxygen", 8};

    Kokkos::View<double[10]> dE_neon {"dE_neon", 10};
    Kokkos::View<double[10]> A_neon {"A_neon", 10};
    Kokkos::View<double[10]> X_neon {"X_neon", 10};
    Kokkos::View<double[10]> K_neon {"K_neon", 10};
    Kokkos::View<double[10]> P_neon {"P_neon", 10};

    Kokkos::View<double[12]> dE_magnesium {"dE_magnesium", 12};
    Kokkos::View<double[12]> A_magnesium {"A_magnesium", 12};
    Kokkos::View<double[12]> X_magnesium {"X_magnesium", 12};
    Kokkos::View<double[12]> K_magnesium {"K_magnesium", 12};
    Kokkos::View<double[12]> P_magnesium {"P_magnesium", 12};

    Kokkos::View<double[14]> dE_silicon {"dE_silicon", 14};
    Kokkos::View<double[14]> A_silicon {"A_silicon", 14};
    Kokkos::View<double[14]> X_silicon {"X_silicon", 14};
    Kokkos::View<double[14]> K_silicon {"K_silicon", 14};
    Kokkos::View<double[14]> P_silicon {"P_silicon", 14};

    Kokkos::View<double[16]> dE_sulfur {"dE_sulfur", 16};
    Kokkos::View<double[16]> A_sulfur {"A_sulfur", 16};
    Kokkos::View<double[16]> X_sulfur {"X_sulfur", 16};
    Kokkos::View<double[16]> K_sulfur {"K_sulfur", 16};
    Kokkos::View<double[16]> P_sulfur {"P_sulfur", 16};

    Kokkos::View<double[26]> dE_iron {"dE_iron", 26};
    Kokkos::View<double[26]> A_iron {"A_iron", 26};
    Kokkos::View<double[26]> X_iron {"X_iron", 26};
    Kokkos::View<double[26]> K_iron {"K_iron", 26};
    Kokkos::View<double[26]> P_iron {"P_iron", 26};

} TabulatedData;
