#pragma once
#include <Kokkos_Core.hpp>
// Create shorthand for double kokkos views
#define MAX_ELEMENTS 27
// Maximum total ion+molecule states across all elements:
// H(3) + He(3) + C(7) + N(8) + O(9) + Ne(11) + Mg(13) + Si(15) + S(17) + Fe(27) = 113
#define MAX_TOTAL_IONS 113
#define N_GROUPS 1
#define N_DIM 3
using Array2D = std::array<std::array<double, MAX_ELEMENTS>, MAX_ELEMENTS>;
using Array2D_RT = std::array<std::array<double, N_DIM>, N_GROUPS>;


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

// Runtime physics flags (controls which processes are active in the solver)
struct PhysicsFlags {
    bool include_collisional_ionization = true;
    bool include_photoionization = true;
    bool include_cosmic_ray_ionization = true;
    bool include_HM12_UVB = true;
    bool include_dust_recombination = true;
    bool include_charge_exchange = true;
};

typedef struct
{
    double n_element;
    double n_element_new;
    std::array<double, MAX_ELEMENTS> ion_fracs;
    std::array<double, MAX_ELEMENTS> ion_fracs_new;
} ParticleIonData;

// Compact 2D array for element-indexed ion data. Uses flat MAX_TOTAL_IONS
// storage with per-element offsets, preserving [element][ion] access syntax.
struct CompactArray2D {
    double data[MAX_TOTAL_IONS];
    int offsets[MAX_ELEMENTS];

    struct RowProxy {
        double* row;
        KOKKOS_INLINE_FUNCTION double& operator[](int j) { return row[j]; }
        KOKKOS_INLINE_FUNCTION const double& operator[](int j) const { return row[j]; }
    };
    struct ConstRowProxy {
        const double* row;
        KOKKOS_INLINE_FUNCTION const double& operator[](int j) const { return row[j]; }
    };

    KOKKOS_INLINE_FUNCTION RowProxy operator[](int i) { return {data + offsets[i]}; }
    KOKKOS_INLINE_FUNCTION ConstRowProxy operator[](int i) const { return {data + offsets[i]}; }

    KOKKOS_INLINE_FUNCTION
    void init_offsets(const Element* elements) {
        for (int i = 0; i < MAX_ELEMENTS; i++) offsets[i] = 0;
        for (int i = 0; i < MAX_TOTAL_IONS; i++) data[i] = 0.0;
        int offset = 0;
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (elements[i].atomic_number < 1) continue;
            offsets[i] = offset;
            offset += elements[i].n_ions + elements[i].n_mol;
        }
    }
};

// Compact version that stores ion fractions in flat arrays with element offsets.
// Provides the same n_and_ion_fracs[element].ion_fracs[ion] access pattern as
// std::array<ParticleIonData, MAX_ELEMENTS> but uses MAX_TOTAL_IONS storage
// instead of MAX_ELEMENTS * MAX_ELEMENTS.
struct CompactIonData {
    double n_element[MAX_ELEMENTS];
    double n_element_new[MAX_ELEMENTS];
    double ion_fracs_flat[MAX_TOTAL_IONS];
    double ion_fracs_new_flat[MAX_TOTAL_IONS];
    int offsets[MAX_ELEMENTS]; // offset into flat arrays, -1 if inactive

    // Proxy to support n_and_ion_fracs[i].ion_fracs[j] syntax
    struct ElementProxy {
        double& n_element;
        double& n_element_new;
        double* ion_fracs;
        double* ion_fracs_new;
    };
    struct ConstElementProxy {
        const double& n_element;
        const double& n_element_new;
        const double* ion_fracs;
        const double* ion_fracs_new;
    };

    KOKKOS_INLINE_FUNCTION
    ElementProxy operator[](int i) {
        return {n_element[i], n_element_new[i],
                ion_fracs_flat + offsets[i],
                ion_fracs_new_flat + offsets[i]};
    }
    KOKKOS_INLINE_FUNCTION
    ConstElementProxy operator[](int i) const {
        return {n_element[i], n_element_new[i],
                ion_fracs_flat + offsets[i],
                ion_fracs_new_flat + offsets[i]};
    }

    KOKKOS_INLINE_FUNCTION
    void init_offsets(const Element* elements) {
        for (int i = 0; i < MAX_ELEMENTS; i++) {
            n_element[i] = 0.0;
            n_element_new[i] = 0.0;
            offsets[i] = 0;  // safe default (points to start of flat array)
        }
        for (int i = 0; i < MAX_TOTAL_IONS; i++) {
            ion_fracs_flat[i] = 0.0;
            ion_fracs_new_flat[i] = 0.0;
        }
        int offset = 0;
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (elements[i].atomic_number < 1) continue;
            offsets[i] = offset;
            offset += elements[i].n_ions + elements[i].n_mol;
        }
    }
};

// Define the structure equivalent to the Fortran type
typedef struct {
    double E_th;
    double E_max;
    double E_0;
    double sig_0;
    double y_a;
    double P;
    double y_w;
    double y_0;
    double y_1;
} CrossSections;


typedef struct {
    double E_th; // @HarleyKatz IS THIS CORRECT?
    double E_0;
    double sig_0;
    double y_a;
    double P;
    double y_w;
    int l;
} CrossSectionsNL;

typedef struct
{
    Kokkos::View<double[27][27][2]> HM12_UVB_z {"HM12_UVB_z"};
    Kokkos::View<double[27]> G0_heating_rates {"G0_heating_rates"};
    Kokkos::View<double[27][27]> cosmic_ray_ionization_rates {"cosmic_ray_ionization_rates"};
    Kokkos::View<double[27]> cosmic_ray_ionization_rates_induced_UV {"cosmic_ray_ionization_rates_induced_UV"};
    Kokkos::View<double[27]> cosmic_ray_ionization_rates_induced_UV_heat {"cosmic_ray_ionization_rates_induced_UV_heat"};
    Kokkos::View<double[27][7]> dust_rec_coefs {"dust_rec_coefs"};
    Kokkos::View<double[7][3][31]> CTIon {"CTIon"};
    Kokkos::View<double[6][4][31]> CTRecomb {"CTRecomb"};
    Kokkos::View<double[27][160][8]> fs_cool_tab {"fs_cool_tab"};

    Kokkos::View<double[121][27][27]> high_t_cooling_rates {"high_t_cooling_rates"};
    Kokkos::View<bool[27][27]> high_t_cooling_rates_tflag {"high_t_cooling_rates_tflag"};

    Kokkos::View<double[3][9]> DR_rates_c_helium {"DR_rates_c_helium"};
    Kokkos::View<double[3][9]> DR_rates_e_helium {"DR_rates_e_helium"};
    Kokkos::View<double[7][6]> RR_rates_carbon {"RR_rates_carbon"};
    Kokkos::View<double[7][9]> DR_rates_c_carbon {"DR_rates_c_carbon"};
    Kokkos::View<double[7][9]> DR_rates_e_carbon {"DR_rates_e_carbon"};
    Kokkos::View<double[8][6]> RR_rates_nitrogen {"RR_rates_nitrogen"};
    Kokkos::View<double[8][9]> DR_rates_c_nitrogen {"DR_rates_c_nitrogen"};
    Kokkos::View<double[8][9]> DR_rates_e_nitrogen {"DR_rates_e_nitrogen"};
    Kokkos::View<double[9][6]> RR_rates_oxygen {"RR_rates_oxygen"};
    Kokkos::View<double[9][9]> DR_rates_c_oxygen {"DR_rates_c_oxygen"};
    Kokkos::View<double[9][9]> DR_rates_e_oxygen {"DR_rates_e_oxygen"};
    Kokkos::View<double[11][6]> RR_rates_neon {"RR_rates_neon"};
    Kokkos::View<double[11][9]> DR_rates_c_neon {"DR_rates_c_neon"};
    Kokkos::View<double[11][9]> DR_rates_e_neon {"DR_rates_e_neon"};
    Kokkos::View<double[13][6]> RR_rates_magnesium {"RR_rates_magnesium"};
    Kokkos::View<double[13][9]> DR_rates_c_magnesium {"DR_rates_c_magnesium"};
    Kokkos::View<double[13][9]> DR_rates_e_magnesium {"DR_rates_e_magnesium"};
    Kokkos::View<double[15][6]> RR_rates_silicon {"RR_rates_silicon"};
    Kokkos::View<double[15][9]> DR_rates_c_silicon {"DR_rates_c_silicon"};
    Kokkos::View<double[15][9]> DR_rates_e_silicon {"DR_rates_e_silicon"};
    Kokkos::View<double[17][6]> RR_rates_sulfur {"RR_rates_sulfur"};
    Kokkos::View<double[17][9]> DR_rates_c_sulfur {"DR_rates_c_sulfur"};
    Kokkos::View<double[17][9]> DR_rates_e_sulfur {"DR_rates_e_sulfur"};
    Kokkos::View<double[17][2]> RR_rates_alt_sulfur {"RR_rates_alt_sulfur"};
    Kokkos::View<double[17][4]> DR_rates_alt_sulfur {"DR_rates_alt_sulfur"};
    Kokkos::View<double[27][6]> RR_rates_iron {"RR_rates_iron"};
    Kokkos::View<double[27][9]> DR_rates_c_iron {"DR_rates_c_iron"};
    Kokkos::View<double[27][9]> DR_rates_e_iron {"DR_rates_e_iron"};
    Kokkos::View<double[27][2]> RR_rates_alt_iron {"RR_rates_alt_iron"};
    Kokkos::View<double[27][8]> DR_rates_alt_iron {"DR_rates_alt_iron"};

    Kokkos::View<double[6]> dE_carbon {"dE_carbon"};
    Kokkos::View<double[6]> A_carbon {"A_carbon"};
    Kokkos::View<double[6]> X_carbon {"X_carbon"};
    Kokkos::View<double[6]> K_carbon {"K_carbon"};
    Kokkos::View<double[6]> P_carbon {"P_carbon"};

    Kokkos::View<double[7]> dE_nitrogen {"dE_nitrogen"};
    Kokkos::View<double[7]> A_nitrogen {"A_nitrogen"};
    Kokkos::View<double[7]> X_nitrogen {"X_nitrogen"};
    Kokkos::View<double[7]> K_nitrogen {"K_nitrogen"};
    Kokkos::View<double[7]> P_nitrogen {"P_nitrogen"};

    Kokkos::View<double[8]> dE_oxygen {"dE_oxygen"};
    Kokkos::View<double[8]> A_oxygen {"A_oxygen"};
    Kokkos::View<double[8]> X_oxygen {"X_oxygen"};
    Kokkos::View<double[8]> K_oxygen {"K_oxygen"};
    Kokkos::View<double[8]> P_oxygen {"P_oxygen"};

    Kokkos::View<double[10]> dE_neon {"dE_neon"};
    Kokkos::View<double[10]> A_neon {"A_neon"};
    Kokkos::View<double[10]> X_neon {"X_neon"};
    Kokkos::View<double[10]> K_neon {"K_neon"};
    Kokkos::View<double[10]> P_neon {"P_neon"};

    Kokkos::View<double[12]> dE_magnesium {"dE_magnesium"};
    Kokkos::View<double[12]> A_magnesium {"A_magnesium"};
    Kokkos::View<double[12]> X_magnesium {"X_magnesium"};
    Kokkos::View<double[12]> K_magnesium {"K_magnesium"};
    Kokkos::View<double[12]> P_magnesium {"P_magnesium"};

    Kokkos::View<double[14]> dE_silicon {"dE_silicon"};
    Kokkos::View<double[14]> A_silicon {"A_silicon"};
    Kokkos::View<double[14]> X_silicon {"X_silicon"};
    Kokkos::View<double[14]> K_silicon {"K_silicon"};
    Kokkos::View<double[14]> P_silicon {"P_silicon"};

    Kokkos::View<double[16]> dE_sulfur {"dE_sulfur"};
    Kokkos::View<double[16]> A_sulfur {"A_sulfur"};
    Kokkos::View<double[16]> X_sulfur {"X_sulfur"};
    Kokkos::View<double[16]> K_sulfur {"K_sulfur"};
    Kokkos::View<double[16]> P_sulfur {"P_sulfur"};

    Kokkos::View<double[26]> dE_iron {"dE_iron"};
    Kokkos::View<double[26]> A_iron {"A_iron"};
    Kokkos::View<double[26]> X_iron {"X_iron"};
    Kokkos::View<double[26]> K_iron {"K_iron"};
    Kokkos::View<double[26]> P_iron {"P_iron"};

    // Cloudy radiative recombination data (from rad_rec.dat)
    Kokkos::View<double[30][30][2]> rrec {"rrec"};
    Kokkos::View<double[30][30][4]> rnew {"rnew"};
    Kokkos::View<double[13][3]> fe {"fe"};

    // Collisional ionization data (from collisional_ionization.dat)
    Kokkos::View<double[27][27][5]> collisional_ionization_data {"collisional_ionization_data"};

    // Verner cross sections
    Kokkos::View<CrossSections[27][27]> photoionization_cross_sections_tot {"photoionization_cross_sections_tot"};
    Kokkos::View<CrossSectionsNL[27][27][8]> photoionization_cross_sections_nl {"photoionization_cross_sections_nl"};

    // Photoionization cross sections and photoheating rates for each group, element, and ion
    Kokkos::View<double[27][27][N_GROUPS][3]> cs_ph {"cs_ph"};

    // Photon bins
    Kokkos::View<double[N_GROUPS]> group_E_min {"group_E_min"};
    Kokkos::View<double[N_GROUPS]> group_E_max {"group_E_max"};

    // Mean photon energy [eV] of each group, weighted over the blackbody spectrum
    Kokkos::View<double[N_GROUPS]> group_energy {"group_energy"};

    // Dust absorption cross section per H [cm^2/H] from the BARE-GR-S model
    Kokkos::View<double[N_DUST_WAV]> dust_wav {"dust_wav"};
    Kokkos::View<double[N_DUST_WAV]> dust_cabs {"dust_cabs"}; 
    Kokkos::View<double[N_DUST_WAV]> dust_log_wav {"dust_log_wav"};
    Kokkos::View<double[N_DUST_WAV]> dust_log_cabs {"dust_log_cabs"};

    // Blackbody-averaged dust absorption rate
    Kokkos::View<double[N_GROUPS]> cs_dust {"cs_dust"};

    // Reduced speed of light factor
    Kokkos::View<double> c_frac {"c_frac"};

} TabulatedData;
