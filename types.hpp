#pragma once
#include <Kokkos_Core.hpp>
#include <cstdint>
// Create shorthand for double kokkos views
#define MAX_ELEMENTS 27
// Maximum total ion+molecule states across all elements:
// H(3) + He(3) + C(7) + N(8) + O(9) + Ne(11) + Mg(13) + Si(15) + S(17) + Fe(27) = 113
#define MAX_TOTAL_IONS 113
// Number of elements with tabulated data (see initialize_elements)
#define N_TAB_ELEMENTS 10

// Maps an atomic number to its row in the element-compacted tables
KOKKOS_INLINE_FUNCTION
constexpr int elem_row(int Z) {
    switch (Z) {
        case  1: return 0; // H
        case  2: return 1; // He
        case  6: return 2; // C
        case  7: return 3; // N
        case  8: return 4; // O
        case 10: return 5; // Ne
        case 12: return 6; // Mg
        case 14: return 7; // Si
        case 16: return 8; // S
        case 26: return 9; // Fe
        default: return -1;
    }
}
#define N_GROUPS 6
#define N_DIM 3
#define N_DUST_WAV 1000
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

// Reduced-precision storage (enabled with -DRTZ_REDUCED_PRECISION):

// Compact version that stores ion fractions in flat arrays with element offsets.
// Provides the same n_and_ion_fracs[element].ion_fracs[ion] access pattern as
// std::array<ParticleIonData, MAX_ELEMENTS> but uses MAX_TOTAL_IONS storage
// instead of MAX_ELEMENTS * MAX_ELEMENTS.
struct CompactIonData {
#ifdef RTZ_REDUCED_PRECISION
    using ndens_t = float;
    using xion_t = float;
#else
    using ndens_t = double;
    using xion_t = double;
#endif

    ndens_t n_element[MAX_ELEMENTS];
    ndens_t n_element_new[MAX_ELEMENTS];
    xion_t ion_fracs_flat[MAX_TOTAL_IONS];
    xion_t ion_fracs_new_flat[MAX_TOTAL_IONS];
    int offsets[MAX_ELEMENTS]; // offset into flat arrays, 0 for inactive elements

    // Proxy to support n_and_ion_fracs[i].ion_fracs[j] syntax
    template <typename ND, typename XI>
    struct ElementProxyT {
        ND& n_element;
        ND& n_element_new;
        XI* ion_fracs;
        XI* ion_fracs_new;
    };
    using ElementProxy = ElementProxyT<ndens_t, xion_t>;
    using ConstElementProxy = ElementProxyT<const ndens_t, const xion_t>;

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
            n_element[i] = ndens_t(0);
            n_element_new[i] = ndens_t(0);
            offsets[i] = 0;
        }
        for (int i = 0; i < MAX_TOTAL_IONS; i++) {
            ion_fracs_flat[i] = xion_t(0);
            ion_fracs_new_flat[i] = xion_t(0);
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

// One term of the dielectronic recombination sum alpha_DR = T^(-3/2) * sum_i
// c_i * exp(-E_i/T). Energy and coefficient are interleaved and 16-byte
// aligned so a thread fetches the pair with a single-sector access instead of
// two gathers into separate tables. e == 0 marks an unused term; the tables
// only have trailing zeros, so a zero e terminates the row.
struct alignas(16) DRTerm {
    double e;
    double c;
};

struct TabulatedData
{
    Kokkos::View<float[N_TAB_ELEMENTS][27][2], Kokkos::LayoutRight> HM12_UVB_z {"HM12_UVB_z"};
    Kokkos::View<float[N_TAB_ELEMENTS]> G0_heating_rates {"G0_heating_rates"};
    Kokkos::View<float[N_TAB_ELEMENTS][27], Kokkos::LayoutRight> cosmic_ray_ionization_rates {"cosmic_ray_ionization_rates"};
    Kokkos::View<float[N_TAB_ELEMENTS]> cosmic_ray_ionization_rates_induced_UV {"cosmic_ray_ionization_rates_induced_UV"};
    Kokkos::View<float[N_TAB_ELEMENTS]> cosmic_ray_ionization_rates_induced_UV_heat {"cosmic_ray_ionization_rates_induced_UV_heat"};
    Kokkos::View<float[N_TAB_ELEMENTS][7], Kokkos::LayoutRight> dust_rec_coefs {"dust_rec_coefs"};
    Kokkos::View<float[7][3][N_TAB_ELEMENTS], Kokkos::LayoutRight> CTIon {"CTIon"};
    Kokkos::View<float[6][4][N_TAB_ELEMENTS], Kokkos::LayoutRight> CTRecomb {"CTRecomb"};
    Kokkos::View<float[27][160][8], Kokkos::LayoutRight> fs_cool_tab {"fs_cool_tab"};

    Kokkos::View<float[121][N_TAB_ELEMENTS][27], Kokkos::LayoutRight> high_t_cooling_rates {"high_t_cooling_rates"};
    Kokkos::View<bool[N_TAB_ELEMENTS][27], Kokkos::LayoutRight> high_t_cooling_rates_tflag {"high_t_cooling_rates_tflag"};

    Kokkos::View<DRTerm[3][9], Kokkos::LayoutRight> DR_rates_helium {"DR_rates_helium"};
    Kokkos::View<double[7][6], Kokkos::LayoutRight> RR_rates_carbon {"RR_rates_carbon"};
    Kokkos::View<DRTerm[7][9], Kokkos::LayoutRight> DR_rates_carbon {"DR_rates_carbon"};
    Kokkos::View<double[8][6], Kokkos::LayoutRight> RR_rates_nitrogen {"RR_rates_nitrogen"};
    Kokkos::View<DRTerm[8][9], Kokkos::LayoutRight> DR_rates_nitrogen {"DR_rates_nitrogen"};
    Kokkos::View<double[9][6], Kokkos::LayoutRight> RR_rates_oxygen {"RR_rates_oxygen"};
    Kokkos::View<DRTerm[9][9], Kokkos::LayoutRight> DR_rates_oxygen {"DR_rates_oxygen"};
    Kokkos::View<double[11][6], Kokkos::LayoutRight> RR_rates_neon {"RR_rates_neon"};
    Kokkos::View<DRTerm[11][9], Kokkos::LayoutRight> DR_rates_neon {"DR_rates_neon"};
    Kokkos::View<double[13][6], Kokkos::LayoutRight> RR_rates_magnesium {"RR_rates_magnesium"};
    Kokkos::View<DRTerm[13][9], Kokkos::LayoutRight> DR_rates_magnesium {"DR_rates_magnesium"};
    Kokkos::View<double[15][6], Kokkos::LayoutRight> RR_rates_silicon {"RR_rates_silicon"};
    Kokkos::View<DRTerm[15][9], Kokkos::LayoutRight> DR_rates_silicon {"DR_rates_silicon"};
    Kokkos::View<double[17][6], Kokkos::LayoutRight> RR_rates_sulfur {"RR_rates_sulfur"};
    Kokkos::View<DRTerm[17][9], Kokkos::LayoutRight> DR_rates_sulfur {"DR_rates_sulfur"};
    Kokkos::View<double[17][2], Kokkos::LayoutRight> RR_rates_alt_sulfur {"RR_rates_alt_sulfur"};
    Kokkos::View<double[17][4], Kokkos::LayoutRight> DR_rates_alt_sulfur {"DR_rates_alt_sulfur"};
    Kokkos::View<double[27][6], Kokkos::LayoutRight> RR_rates_iron {"RR_rates_iron"};
    Kokkos::View<DRTerm[27][9], Kokkos::LayoutRight> DR_rates_iron {"DR_rates_iron"};
    Kokkos::View<double[27][2], Kokkos::LayoutRight> RR_rates_alt_iron {"RR_rates_alt_iron"};
    Kokkos::View<double[27][8], Kokkos::LayoutRight> DR_rates_alt_iron {"DR_rates_alt_iron"};

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
    Kokkos::View<double[30][N_TAB_ELEMENTS][2], Kokkos::LayoutRight> rrec {"rrec"};
    Kokkos::View<double[30][N_TAB_ELEMENTS][4], Kokkos::LayoutRight> rnew {"rnew"};
    Kokkos::View<double[13][3], Kokkos::LayoutRight> fe {"fe"};

    // Collisional ionization data (from collisional_ionization.dat)
    Kokkos::View<double[N_TAB_ELEMENTS][27][5], Kokkos::LayoutRight> collisional_ionization_data {"collisional_ionization_data"};

    // Verner cross sections
    Kokkos::View<CrossSections[27][27], Kokkos::LayoutRight> photoionization_cross_sections_tot {"photoionization_cross_sections_tot"};
    Kokkos::View<CrossSectionsNL[27][27][8], Kokkos::LayoutRight> photoionization_cross_sections_nl {"photoionization_cross_sections_nl"};

    // Photoionization cross sections and photoheating rates for each group, element, and ion
    Kokkos::View<double[N_TAB_ELEMENTS][27][N_GROUPS][3], Kokkos::LayoutRight> cs_ph {"cs_ph"};

    Kokkos::View<double[52]> sco_col1 {"sigma_CO_1"};
    Kokkos::View<double[52]> sco_col2 {"sigma_CO_2"};
    Kokkos::View<double[43]> sh2_col1 {"sigma_H2_1"};
    Kokkos::View<double[43]> sh2_col2 {"sigma_H2_2"};

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

    // Explicitly host-only destructor to avoid device context warnings
    ~TabulatedData() = default;

};
