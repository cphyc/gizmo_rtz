#pragma once
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <vector>
#include <stdexcept>

#include "constants.hpp"

// Create shorthand for double kokkos views
#define MAX_ELEMENTS 27
// Maximum total ion+molecule states across all elements:
// H(3) + He(3) + C(7) + N(8) + O(9) + Ne(11) + Mg(13) + Si(15) + S(17) + Fe(27) = 113
//
// This is the bound on CompactIonData's two flat ion-fraction arrays, and it is
// storage that every cell carries whether or not the configured network uses it.
// A run with only H, He, C and O needs 22 of the 113 slots, so the default wastes
// ~0.7 kB per cell of bandwidth in every kernel that touches the solver state.
// Override it at configure time to match the network actually being run
// (-DDYABLO_PRISM_MAX_TOTAL_IONS=24); CoolingUpdate_PRISM_GPU checks the
// configured network against it at startup and aborts if it would not fit.
#ifndef MAX_TOTAL_IONS
#define MAX_TOTAL_IONS 113
#endif
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
 // Maps a row in the element-compacted tables to its atomic number
KOKKOS_INLINE_FUNCTION
constexpr int row_elem(int row) {
    switch (row) {
        case 0: return  1; // H
        case 1: return  2; // He
        case 2: return  6; // C
        case 3: return  7; // N
        case 4: return  8; // O
        case 5: return 10; // Ne
        case 6: return 12; // Mg
        case 7: return 14; // Si
        case 8: return 16; // S
        case 9: return 26; // Fe
        default: return -1;
    }
}


#define N_GROUPS 7
#define N_DIM 3
#define N_DUST_WAV 1000
using Array2D = std::array<std::array<Real, MAX_ELEMENTS>, MAX_ELEMENTS>;
using Array2D_RT = std::array<std::array<Real, N_DIM>, N_GROUPS>;


typedef struct
{
    int atomic_number;
    int n_ions;
    int n_mol;
    Real atomic_mass;
    Real z_solar;
    Real G0_photo_rate;
    Real depletion;
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
    Real n_element;
    Real n_element_new;
    std::array<Real, MAX_ELEMENTS> ion_fracs;
    std::array<Real, MAX_ELEMENTS> ion_fracs_new;
} ParticleIonData;

// Reduced-precision storage (enabled with -DRTZ_REDUCED_PRECISION):

// Compact version that stores ion fractions in flat arrays with element offsets.
// Provides the same n_and_ion_fracs[element].ion_fracs[ion] access pattern as
// std::array<ParticleIonData, MAX_ELEMENTS> but uses MAX_TOTAL_IONS storage
// instead of MAX_ELEMENTS * MAX_ELEMENTS.
// Immutable network maps live once in host/device accessible memory.
struct SharedIonOffsets {
    std::array<int, MAX_ELEMENTS> key;
    Kokkos::View<int*, Kokkos::SharedSpace> values;
};
inline std::vector<SharedIonOffsets>& ion_offset_cache() {
    static std::vector<SharedIonOffsets> cache;
    return cache;
}
inline void clear_ion_offset_cache() { ion_offset_cache().clear(); }
inline const int* shared_ion_offsets(const Element* elements, const int* network) {
    std::array<int, MAX_ELEMENTS> key{};
    int end = 0;
    for (int i = 1; i < MAX_ELEMENTS; ++i) {
        key[i] = end;
        if (elements[i].atomic_number > 0 && (!network || network[i] != -1))
            end += elements[i].n_ions + elements[i].n_mol;
    }
    if (end > MAX_TOTAL_IONS) throw std::runtime_error("ion network exceeds MAX_TOTAL_IONS");
    for (const auto& entry : ion_offset_cache())
        if (entry.key == key) return entry.values.data();
    SharedIonOffsets entry{key, Kokkos::View<int*, Kokkos::SharedSpace>("ion_offsets", MAX_ELEMENTS)};
    for (int i = 0; i < MAX_ELEMENTS; ++i) entry.values(i) = key[i];
    ion_offset_cache().push_back(entry);
    return entry.values.data();
}

struct CompactIonData {
#if defined(RTZ_REDUCED_PRECISION) || defined(RTZ_FLOAT32_ARITHMETIC)
    using ndens_t = float;
    using xion_t = float;
#else
    using ndens_t = Real;
    using xion_t = Real;
#endif

    ndens_t n_element[MAX_ELEMENTS];
    ndens_t n_element_new[MAX_ELEMENTS];
    xion_t ion_fracs_flat[MAX_TOTAL_IONS];
    xion_t ion_fracs_new_flat[MAX_TOTAL_IONS];
    const int* offsets = nullptr; // shared immutable network map

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

    // `in_network`, when given, is a per-element map with -1 for elements the
    // run's ion network does not include (dyablo's ions2passive). Without it
    // every element initialize_elements knows about reserves its full
    // ionization ladder, which is 113 slots whatever the .ini actually lists.
    void init_offsets(const Element* elements, const int* in_network = nullptr) {
        for (int i = 0; i < MAX_ELEMENTS; i++) {
            n_element[i] = ndens_t(0);
            n_element_new[i] = ndens_t(0);

        }
        for (int i = 0; i < MAX_TOTAL_IONS; i++) {
            ion_fracs_flat[i] = xion_t(0);
            ion_fracs_new_flat[i] = xion_t(0);
        }
        offsets = shared_ion_offsets(elements, in_network);
    }
};

// Define the structure equivalent to the Fortran type
typedef struct {
    Real E_th;
    Real E_max;
    Real E_0;
    Real sig_0;
    Real y_a;
    Real P;
    Real y_w;
    Real y_0;
    Real y_1;
} CrossSections;


typedef struct {
    Real E_th; // @HarleyKatz IS THIS CORRECT?
    Real E_0;
    Real sig_0;
    Real y_a;
    Real P;
    Real y_w;
    int l;
} CrossSectionsNL;

// One term of the dielectronic recombination sum alpha_DR = T^(-3/2) * sum_i
// c_i * exp(-E_i/T). Energy and coefficient are interleaved and 16-byte
// aligned so a thread fetches the pair with a single-sector access instead of
// two gathers into separate tables. e == 0 marks an unused term; the tables
// only have trailing zeros, so a zero e terminates the row.
struct alignas(16) DRTerm {
    Real e;
    Real c;
};

template<class MemorySpace>
struct RawTabulatedDataStorage
{
    using memory_space = MemorySpace;
    using execution_space = typename MemorySpace::execution_space;
    Kokkos::View<float[N_TAB_ELEMENTS][27][2], Kokkos::LayoutRight, MemorySpace> HM12_UVB_z {"HM12_UVB_z"};
    Kokkos::View<float[N_TAB_ELEMENTS], MemorySpace> G0_heating_rates {"G0_heating_rates"};
    Kokkos::View<float[N_TAB_ELEMENTS][27], Kokkos::LayoutRight, MemorySpace> cosmic_ray_ionization_rates {"cosmic_ray_ionization_rates"};
    Kokkos::View<float[N_TAB_ELEMENTS], MemorySpace> cosmic_ray_ionization_rates_induced_UV {"cosmic_ray_ionization_rates_induced_UV"};
    Kokkos::View<float[N_TAB_ELEMENTS], MemorySpace> cosmic_ray_ionization_rates_induced_UV_heat {"cosmic_ray_ionization_rates_induced_UV_heat"};
    Kokkos::View<float[N_TAB_ELEMENTS][7], Kokkos::LayoutRight, MemorySpace> dust_rec_coefs {"dust_rec_coefs"};
    Kokkos::View<float[7][3][N_TAB_ELEMENTS], Kokkos::LayoutRight, MemorySpace> CTIon {"CTIon"};
    Kokkos::View<float[6][4][N_TAB_ELEMENTS], Kokkos::LayoutRight, MemorySpace> CTRecomb {"CTRecomb"};
    Kokkos::View<float[27][160][8], Kokkos::LayoutRight, MemorySpace> fs_cool_tab {"fs_cool_tab"};

    Kokkos::View<float[121][N_TAB_ELEMENTS][27], Kokkos::LayoutRight, MemorySpace> high_t_cooling_rates {"high_t_cooling_rates"};
    Kokkos::View<bool[N_TAB_ELEMENTS][27], Kokkos::LayoutRight, MemorySpace> high_t_cooling_rates_tflag {"high_t_cooling_rates_tflag"};

    Kokkos::View<DRTerm[3][9], Kokkos::LayoutRight, MemorySpace> DR_rates_helium {"DR_rates_helium"};
    Kokkos::View<Real[7][6], Kokkos::LayoutRight, MemorySpace> RR_rates_carbon {"RR_rates_carbon"};
    Kokkos::View<DRTerm[7][9], Kokkos::LayoutRight, MemorySpace> DR_rates_carbon {"DR_rates_carbon"};
    Kokkos::View<Real[8][6], Kokkos::LayoutRight, MemorySpace> RR_rates_nitrogen {"RR_rates_nitrogen"};
    Kokkos::View<DRTerm[8][9], Kokkos::LayoutRight, MemorySpace> DR_rates_nitrogen {"DR_rates_nitrogen"};
    Kokkos::View<Real[9][6], Kokkos::LayoutRight, MemorySpace> RR_rates_oxygen {"RR_rates_oxygen"};
    Kokkos::View<DRTerm[9][9], Kokkos::LayoutRight, MemorySpace> DR_rates_oxygen {"DR_rates_oxygen"};
    Kokkos::View<Real[11][6], Kokkos::LayoutRight, MemorySpace> RR_rates_neon {"RR_rates_neon"};
    Kokkos::View<DRTerm[11][9], Kokkos::LayoutRight, MemorySpace> DR_rates_neon {"DR_rates_neon"};
    Kokkos::View<Real[13][6], Kokkos::LayoutRight, MemorySpace> RR_rates_magnesium {"RR_rates_magnesium"};
    Kokkos::View<DRTerm[13][9], Kokkos::LayoutRight, MemorySpace> DR_rates_magnesium {"DR_rates_magnesium"};
    Kokkos::View<Real[15][6], Kokkos::LayoutRight, MemorySpace> RR_rates_silicon {"RR_rates_silicon"};
    Kokkos::View<DRTerm[15][9], Kokkos::LayoutRight, MemorySpace> DR_rates_silicon {"DR_rates_silicon"};
    Kokkos::View<Real[17][6], Kokkos::LayoutRight, MemorySpace> RR_rates_sulfur {"RR_rates_sulfur"};
    Kokkos::View<DRTerm[17][9], Kokkos::LayoutRight, MemorySpace> DR_rates_sulfur {"DR_rates_sulfur"};
    Kokkos::View<Real[17][2], Kokkos::LayoutRight, MemorySpace> RR_rates_alt_sulfur {"RR_rates_alt_sulfur"};
    Kokkos::View<Real[17][4], Kokkos::LayoutRight, MemorySpace> DR_rates_alt_sulfur {"DR_rates_alt_sulfur"};
    Kokkos::View<Real[27][6], Kokkos::LayoutRight, MemorySpace> RR_rates_iron {"RR_rates_iron"};
    Kokkos::View<DRTerm[27][9], Kokkos::LayoutRight, MemorySpace> DR_rates_iron {"DR_rates_iron"};
    Kokkos::View<Real[27][2], Kokkos::LayoutRight, MemorySpace> RR_rates_alt_iron {"RR_rates_alt_iron"};
    Kokkos::View<Real[27][8], Kokkos::LayoutRight, MemorySpace> DR_rates_alt_iron {"DR_rates_alt_iron"};

    Kokkos::View<Real[6], MemorySpace> dE_carbon {"dE_carbon"};
    Kokkos::View<Real[6], MemorySpace> A_carbon {"A_carbon"};
    Kokkos::View<Real[6], MemorySpace> X_carbon {"X_carbon"};
    Kokkos::View<Real[6], MemorySpace> K_carbon {"K_carbon"};
    Kokkos::View<Real[6], MemorySpace> P_carbon {"P_carbon"};

    Kokkos::View<Real[7], MemorySpace> dE_nitrogen {"dE_nitrogen"};
    Kokkos::View<Real[7], MemorySpace> A_nitrogen {"A_nitrogen"};
    Kokkos::View<Real[7], MemorySpace> X_nitrogen {"X_nitrogen"};
    Kokkos::View<Real[7], MemorySpace> K_nitrogen {"K_nitrogen"};
    Kokkos::View<Real[7], MemorySpace> P_nitrogen {"P_nitrogen"};

    Kokkos::View<Real[8], MemorySpace> dE_oxygen {"dE_oxygen"};
    Kokkos::View<Real[8], MemorySpace> A_oxygen {"A_oxygen"};
    Kokkos::View<Real[8], MemorySpace> X_oxygen {"X_oxygen"};
    Kokkos::View<Real[8], MemorySpace> K_oxygen {"K_oxygen"};
    Kokkos::View<Real[8], MemorySpace> P_oxygen {"P_oxygen"};

    Kokkos::View<Real[10], MemorySpace> dE_neon {"dE_neon"};
    Kokkos::View<Real[10], MemorySpace> A_neon {"A_neon"};
    Kokkos::View<Real[10], MemorySpace> X_neon {"X_neon"};
    Kokkos::View<Real[10], MemorySpace> K_neon {"K_neon"};
    Kokkos::View<Real[10], MemorySpace> P_neon {"P_neon"};

    Kokkos::View<Real[12], MemorySpace> dE_magnesium {"dE_magnesium"};
    Kokkos::View<Real[12], MemorySpace> A_magnesium {"A_magnesium"};
    Kokkos::View<Real[12], MemorySpace> X_magnesium {"X_magnesium"};
    Kokkos::View<Real[12], MemorySpace> K_magnesium {"K_magnesium"};
    Kokkos::View<Real[12], MemorySpace> P_magnesium {"P_magnesium"};

    Kokkos::View<Real[14], MemorySpace> dE_silicon {"dE_silicon"};
    Kokkos::View<Real[14], MemorySpace> A_silicon {"A_silicon"};
    Kokkos::View<Real[14], MemorySpace> X_silicon {"X_silicon"};
    Kokkos::View<Real[14], MemorySpace> K_silicon {"K_silicon"};
    Kokkos::View<Real[14], MemorySpace> P_silicon {"P_silicon"};

    Kokkos::View<Real[16], MemorySpace> dE_sulfur {"dE_sulfur"};
    Kokkos::View<Real[16], MemorySpace> A_sulfur {"A_sulfur"};
    Kokkos::View<Real[16], MemorySpace> X_sulfur {"X_sulfur"};
    Kokkos::View<Real[16], MemorySpace> K_sulfur {"K_sulfur"};
    Kokkos::View<Real[16], MemorySpace> P_sulfur {"P_sulfur"};

    Kokkos::View<Real[26], MemorySpace> dE_iron {"dE_iron"};
    Kokkos::View<Real[26], MemorySpace> A_iron {"A_iron"};
    Kokkos::View<Real[26], MemorySpace> X_iron {"X_iron"};
    Kokkos::View<Real[26], MemorySpace> K_iron {"K_iron"};
    Kokkos::View<Real[26], MemorySpace> P_iron {"P_iron"};

    // Cloudy radiative recombination data (from rad_rec.dat)
    Kokkos::View<Real[30][N_TAB_ELEMENTS][2], Kokkos::LayoutRight, MemorySpace> rrec {"rrec"};
    Kokkos::View<Real[30][N_TAB_ELEMENTS][4], Kokkos::LayoutRight, MemorySpace> rnew {"rnew"};
    Kokkos::View<Real[13][3], Kokkos::LayoutRight, MemorySpace> fe {"fe"};

    // Collisional ionization data (from collisional_ionization.dat)
    Kokkos::View<Real[N_TAB_ELEMENTS][27][5], Kokkos::LayoutRight, MemorySpace> collisional_ionization_data {"collisional_ionization_data"};

    // Verner cross sections
    Kokkos::View<CrossSections[27][27], Kokkos::LayoutRight, MemorySpace> photoionization_cross_sections_tot {"photoionization_cross_sections_tot"};
    Kokkos::View<CrossSectionsNL[27][27][8], Kokkos::LayoutRight, MemorySpace> photoionization_cross_sections_nl {"photoionization_cross_sections_nl"};

    // Photoionization cross sections and photoheating rates for each group, element, and ion
    Kokkos::View<Real[N_TAB_ELEMENTS][27][N_GROUPS][3], Kokkos::LayoutRight, MemorySpace> cs_ph {"cs_ph"};

    Kokkos::View<Real[52], MemorySpace> sco_col1 {"sigma_CO_1"};
    Kokkos::View<Real[52], MemorySpace> sco_col2 {"sigma_CO_2"};
    Kokkos::View<Real[43], MemorySpace> sh2_col1 {"sigma_H2_1"};
    Kokkos::View<Real[43], MemorySpace> sh2_col2 {"sigma_H2_2"};

    // Photon bins
    Kokkos::View<Real[N_GROUPS], MemorySpace> group_E_min {"group_E_min"};
    Kokkos::View<Real[N_GROUPS], MemorySpace> group_E_max {"group_E_max"};

    // Mean photon energy [eV] of each group, weighted over the blackbody spectrum
    Kokkos::View<Real[N_GROUPS], MemorySpace> group_energy {"group_energy"};

    // Dust absorption cross section per H [cm^2/H] from the BARE-GR-S model
    Kokkos::View<Real[N_DUST_WAV], MemorySpace> dust_wav {"dust_wav"};
    Kokkos::View<Real[N_DUST_WAV], MemorySpace> dust_cabs {"dust_cabs"};
    Kokkos::View<Real[N_DUST_WAV], MemorySpace> dust_log_wav {"dust_log_wav"};
    Kokkos::View<Real[N_DUST_WAV], MemorySpace> dust_log_cabs {"dust_log_cabs"};

    // Blackbody-averaged dust absorption rate
    Kokkos::View<Real[N_GROUPS], MemorySpace> cs_dust {"cs_dust"};

    // Reduced speed of light factor
    Kokkos::View<Real, MemorySpace> c_frac {"c_frac"};

    // Explicitly host-only destructor to avoid device context warnings
    ~RawTabulatedDataStorage() = default;

};

using RawTabulatedData = RawTabulatedDataStorage<Kokkos::HostSpace>;
using DeviceRawTabulatedData = RawTabulatedDataStorage<typename Kokkos::DefaultExecutionSpace::memory_space>;
template<class Destination, class Source>
void copy_raw_tables(Destination& destination, const Source& source) {
    Kokkos::deep_copy(destination.HM12_UVB_z, source.HM12_UVB_z);
    Kokkos::deep_copy(destination.G0_heating_rates, source.G0_heating_rates);
    Kokkos::deep_copy(destination.cosmic_ray_ionization_rates, source.cosmic_ray_ionization_rates);
    Kokkos::deep_copy(destination.cosmic_ray_ionization_rates_induced_UV, source.cosmic_ray_ionization_rates_induced_UV);
    Kokkos::deep_copy(destination.cosmic_ray_ionization_rates_induced_UV_heat, source.cosmic_ray_ionization_rates_induced_UV_heat);
    Kokkos::deep_copy(destination.dust_rec_coefs, source.dust_rec_coefs);
    Kokkos::deep_copy(destination.CTIon, source.CTIon);
    Kokkos::deep_copy(destination.CTRecomb, source.CTRecomb);
    Kokkos::deep_copy(destination.fs_cool_tab, source.fs_cool_tab);
    Kokkos::deep_copy(destination.high_t_cooling_rates, source.high_t_cooling_rates);
    Kokkos::deep_copy(destination.high_t_cooling_rates_tflag, source.high_t_cooling_rates_tflag);
    Kokkos::deep_copy(destination.DR_rates_helium, source.DR_rates_helium);
    Kokkos::deep_copy(destination.RR_rates_carbon, source.RR_rates_carbon);
    Kokkos::deep_copy(destination.DR_rates_carbon, source.DR_rates_carbon);
    Kokkos::deep_copy(destination.RR_rates_nitrogen, source.RR_rates_nitrogen);
    Kokkos::deep_copy(destination.DR_rates_nitrogen, source.DR_rates_nitrogen);
    Kokkos::deep_copy(destination.RR_rates_oxygen, source.RR_rates_oxygen);
    Kokkos::deep_copy(destination.DR_rates_oxygen, source.DR_rates_oxygen);
    Kokkos::deep_copy(destination.RR_rates_neon, source.RR_rates_neon);
    Kokkos::deep_copy(destination.DR_rates_neon, source.DR_rates_neon);
    Kokkos::deep_copy(destination.RR_rates_magnesium, source.RR_rates_magnesium);
    Kokkos::deep_copy(destination.DR_rates_magnesium, source.DR_rates_magnesium);
    Kokkos::deep_copy(destination.RR_rates_silicon, source.RR_rates_silicon);
    Kokkos::deep_copy(destination.DR_rates_silicon, source.DR_rates_silicon);
    Kokkos::deep_copy(destination.RR_rates_sulfur, source.RR_rates_sulfur);
    Kokkos::deep_copy(destination.DR_rates_sulfur, source.DR_rates_sulfur);
    Kokkos::deep_copy(destination.RR_rates_alt_sulfur, source.RR_rates_alt_sulfur);
    Kokkos::deep_copy(destination.DR_rates_alt_sulfur, source.DR_rates_alt_sulfur);
    Kokkos::deep_copy(destination.RR_rates_iron, source.RR_rates_iron);
    Kokkos::deep_copy(destination.DR_rates_iron, source.DR_rates_iron);
    Kokkos::deep_copy(destination.RR_rates_alt_iron, source.RR_rates_alt_iron);
    Kokkos::deep_copy(destination.DR_rates_alt_iron, source.DR_rates_alt_iron);
    Kokkos::deep_copy(destination.dE_carbon, source.dE_carbon);
    Kokkos::deep_copy(destination.A_carbon, source.A_carbon);
    Kokkos::deep_copy(destination.X_carbon, source.X_carbon);
    Kokkos::deep_copy(destination.K_carbon, source.K_carbon);
    Kokkos::deep_copy(destination.P_carbon, source.P_carbon);
    Kokkos::deep_copy(destination.dE_nitrogen, source.dE_nitrogen);
    Kokkos::deep_copy(destination.A_nitrogen, source.A_nitrogen);
    Kokkos::deep_copy(destination.X_nitrogen, source.X_nitrogen);
    Kokkos::deep_copy(destination.K_nitrogen, source.K_nitrogen);
    Kokkos::deep_copy(destination.P_nitrogen, source.P_nitrogen);
    Kokkos::deep_copy(destination.dE_oxygen, source.dE_oxygen);
    Kokkos::deep_copy(destination.A_oxygen, source.A_oxygen);
    Kokkos::deep_copy(destination.X_oxygen, source.X_oxygen);
    Kokkos::deep_copy(destination.K_oxygen, source.K_oxygen);
    Kokkos::deep_copy(destination.P_oxygen, source.P_oxygen);
    Kokkos::deep_copy(destination.dE_neon, source.dE_neon);
    Kokkos::deep_copy(destination.A_neon, source.A_neon);
    Kokkos::deep_copy(destination.X_neon, source.X_neon);
    Kokkos::deep_copy(destination.K_neon, source.K_neon);
    Kokkos::deep_copy(destination.P_neon, source.P_neon);
    Kokkos::deep_copy(destination.dE_magnesium, source.dE_magnesium);
    Kokkos::deep_copy(destination.A_magnesium, source.A_magnesium);
    Kokkos::deep_copy(destination.X_magnesium, source.X_magnesium);
    Kokkos::deep_copy(destination.K_magnesium, source.K_magnesium);
    Kokkos::deep_copy(destination.P_magnesium, source.P_magnesium);
    Kokkos::deep_copy(destination.dE_silicon, source.dE_silicon);
    Kokkos::deep_copy(destination.A_silicon, source.A_silicon);
    Kokkos::deep_copy(destination.X_silicon, source.X_silicon);
    Kokkos::deep_copy(destination.K_silicon, source.K_silicon);
    Kokkos::deep_copy(destination.P_silicon, source.P_silicon);
    Kokkos::deep_copy(destination.dE_sulfur, source.dE_sulfur);
    Kokkos::deep_copy(destination.A_sulfur, source.A_sulfur);
    Kokkos::deep_copy(destination.X_sulfur, source.X_sulfur);
    Kokkos::deep_copy(destination.K_sulfur, source.K_sulfur);
    Kokkos::deep_copy(destination.P_sulfur, source.P_sulfur);
    Kokkos::deep_copy(destination.dE_iron, source.dE_iron);
    Kokkos::deep_copy(destination.A_iron, source.A_iron);
    Kokkos::deep_copy(destination.X_iron, source.X_iron);
    Kokkos::deep_copy(destination.K_iron, source.K_iron);
    Kokkos::deep_copy(destination.P_iron, source.P_iron);
    Kokkos::deep_copy(destination.rrec, source.rrec);
    Kokkos::deep_copy(destination.rnew, source.rnew);
    Kokkos::deep_copy(destination.fe, source.fe);
    Kokkos::deep_copy(destination.collisional_ionization_data, source.collisional_ionization_data);
    Kokkos::deep_copy(destination.photoionization_cross_sections_tot, source.photoionization_cross_sections_tot);
    Kokkos::deep_copy(destination.photoionization_cross_sections_nl, source.photoionization_cross_sections_nl);
    Kokkos::deep_copy(destination.cs_ph, source.cs_ph);
    Kokkos::deep_copy(destination.sco_col1, source.sco_col1);
    Kokkos::deep_copy(destination.sco_col2, source.sco_col2);
    Kokkos::deep_copy(destination.sh2_col1, source.sh2_col1);
    Kokkos::deep_copy(destination.sh2_col2, source.sh2_col2);
    Kokkos::deep_copy(destination.group_E_min, source.group_E_min);
    Kokkos::deep_copy(destination.group_E_max, source.group_E_max);
    Kokkos::deep_copy(destination.group_energy, source.group_energy);
    Kokkos::deep_copy(destination.dust_wav, source.dust_wav);
    Kokkos::deep_copy(destination.dust_cabs, source.dust_cabs);
    Kokkos::deep_copy(destination.dust_log_wav, source.dust_log_wav);
    Kokkos::deep_copy(destination.dust_log_cabs, source.dust_log_cabs);
    Kokkos::deep_copy(destination.cs_dust, source.cs_dust);
    Kokkos::deep_copy(destination.c_frac, source.c_frac);
}

#include "compact_tables.hpp"
