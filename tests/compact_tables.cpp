#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstring>
#include <tuple>
#include <cstdio>
#include <stdexcept>
#include "types.hpp"

struct ReverseMapping {
    std::size_t columns, entries;
    ReverseMapping(const std::array<std::size_t, 2>& shape, const TableNetwork&)
        : columns(shape[1]), entries(shape[0] * shape[1]) {}
    KOKKOS_INLINE_FUNCTION bool contains(const std::array<int, 2>&) const { return true; }
    KOKKOS_INLINE_FUNCTION std::size_t flat(const std::array<int, 2>& i) const {
        return entries - 1 - (i[0] * columns + i[1]);
    }
    KOKKOS_INLINE_FUNCTION std::size_t size() const { return entries; }
};
static_assert(TableIndexMapping<ReverseMapping, 2>);
static_assert(!TableIndexMapping<ReverseMapping, 3>);
static_assert(!TableIndexMapping<int, 2>);

template<class T> T sample(std::size_t i) {
    T x{};
    if constexpr (std::is_arithmetic_v<T>) x = static_cast<T>(i % 10007 + 1);
    else if constexpr (requires { x.e; }) { x.e = i + 1; x.c = i + 2; }
    else { x.E_th = i + 1; x.E_0 = i + 2; }
    return x;
}
template<class T> bool equal(T a, T b) {
    if constexpr (std::is_arithmetic_v<T>) return a == b;
    else if constexpr (requires { a.e; }) return a.e == b.e && a.c == b.c;
    else return a.E_th == b.E_th && a.E_0 == b.E_0;
}
template<class Raw> void fill(Raw& raw) {
    auto host = Kokkos::create_mirror_view(raw);
    for (std::size_t i = 0; i < raw.size(); ++i) host.data()[i] = sample<typename Raw::non_const_value_type>(i);
    Kokkos::deep_copy(raw, host);
}
template<class Raw, class Compact> void check(const Raw& raw, const Compact& compact) {
    if constexpr (requires { compact.values; }) {
        auto source = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), raw);
        auto packed = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), compact.values);
        constexpr int rank = Raw::rank;
        std::vector<bool> visited(packed.size(), false);
        for (std::size_t linear = 0; linear < raw.size(); ++linear) {
            std::array<int, rank> index{}; auto rest = linear;
            for (int axis = rank - 1; axis >= 0; --axis) { index[axis] = rest % raw.extent(axis); rest /= raw.extent(axis); }
            // Test every representable original index, excluding inactive rows.
            if (!compact.contains(index)) continue;
            const auto flat = compact.flat(index);
            if (flat >= packed.size() || visited[flat])
                throw std::runtime_error("mapping is out of bounds or aliases included indices");
            visited[flat] = true;
            if (!equal(std::apply([&](auto... axis) { return source(axis...); }, index), packed(flat)))
                throw std::runtime_error("compact lookup differs from raw");
        }
    }
}
template<class View> std::size_t data_bytes(const View& view) {
    if constexpr (requires { view.values; })
        return view.values.size() * sizeof(typename View::value_type);
    else return view.size() * sizeof(typename View::non_const_value_type);
}
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        std::array<Element, MAX_ELEMENTS> elements{};
        for (int z : {1, 2, 8, 26}) {
            elements[z].atomic_number = z;
            elements[z].n_ions = z == 26 ? 5 : z + 1;
        }
        CompactIonData a{}, b{};
        a.init_offsets(elements.data()); b.init_offsets(elements.data());
        if (a.offsets != b.offsets) throw std::runtime_error("offsets are not shared");
        RawTabulatedData raw;
        fill(raw.HM12_UVB_z);
        fill(raw.G0_heating_rates);
        fill(raw.cosmic_ray_ionization_rates);
        fill(raw.cosmic_ray_ionization_rates_induced_UV);
        fill(raw.cosmic_ray_ionization_rates_induced_UV_heat);
        fill(raw.dust_rec_coefs);
        fill(raw.CTIon);
        fill(raw.CTRecomb);
        fill(raw.fs_cool_tab);
        fill(raw.high_t_cooling_rates);
        fill(raw.high_t_cooling_rates_tflag);
        fill(raw.DR_rates_helium);
        fill(raw.RR_rates_carbon);
        fill(raw.DR_rates_carbon);
        fill(raw.RR_rates_nitrogen);
        fill(raw.DR_rates_nitrogen);
        fill(raw.RR_rates_oxygen);
        fill(raw.DR_rates_oxygen);
        fill(raw.RR_rates_neon);
        fill(raw.DR_rates_neon);
        fill(raw.RR_rates_magnesium);
        fill(raw.DR_rates_magnesium);
        fill(raw.RR_rates_silicon);
        fill(raw.DR_rates_silicon);
        fill(raw.RR_rates_sulfur);
        fill(raw.DR_rates_sulfur);
        fill(raw.RR_rates_alt_sulfur);
        fill(raw.DR_rates_alt_sulfur);
        fill(raw.RR_rates_iron);
        fill(raw.DR_rates_iron);
        fill(raw.RR_rates_alt_iron);
        fill(raw.DR_rates_alt_iron);
        fill(raw.dE_carbon);
        fill(raw.A_carbon);
        fill(raw.X_carbon);
        fill(raw.K_carbon);
        fill(raw.P_carbon);
        fill(raw.dE_nitrogen);
        fill(raw.A_nitrogen);
        fill(raw.X_nitrogen);
        fill(raw.K_nitrogen);
        fill(raw.P_nitrogen);
        fill(raw.dE_oxygen);
        fill(raw.A_oxygen);
        fill(raw.X_oxygen);
        fill(raw.K_oxygen);
        fill(raw.P_oxygen);
        fill(raw.dE_neon);
        fill(raw.A_neon);
        fill(raw.X_neon);
        fill(raw.K_neon);
        fill(raw.P_neon);
        fill(raw.dE_magnesium);
        fill(raw.A_magnesium);
        fill(raw.X_magnesium);
        fill(raw.K_magnesium);
        fill(raw.P_magnesium);
        fill(raw.dE_silicon);
        fill(raw.A_silicon);
        fill(raw.X_silicon);
        fill(raw.K_silicon);
        fill(raw.P_silicon);
        fill(raw.dE_sulfur);
        fill(raw.A_sulfur);
        fill(raw.X_sulfur);
        fill(raw.K_sulfur);
        fill(raw.P_sulfur);
        fill(raw.dE_iron);
        fill(raw.A_iron);
        fill(raw.X_iron);
        fill(raw.K_iron);
        fill(raw.P_iron);
        fill(raw.rrec);
        fill(raw.rnew);
        fill(raw.fe);
        fill(raw.collisional_ionization_data);
        fill(raw.photoionization_cross_sections_tot);
        fill(raw.photoionization_cross_sections_nl);
        fill(raw.cs_ph);
        fill(raw.sco_col1);
        fill(raw.sco_col2);
        fill(raw.sh2_col1);
        fill(raw.sh2_col2);
        fill(raw.group_E_min);
        fill(raw.group_E_max);
        fill(raw.group_energy);
        fill(raw.dust_wav);
        fill(raw.dust_cabs);
        fill(raw.dust_log_wav);
        fill(raw.dust_log_cabs);
        fill(raw.cs_dust);
        fill(raw.c_frac);
        CompactTabulatedData compact(raw, elements.data());
        if (compact.network.offsets.data() != a.offsets)
            throw std::runtime_error("table and cell offsets are not shared");
        check(raw.HM12_UVB_z, compact.HM12_UVB_z);
        check(raw.G0_heating_rates, compact.G0_heating_rates);
        check(raw.cosmic_ray_ionization_rates, compact.cosmic_ray_ionization_rates);
        check(raw.cosmic_ray_ionization_rates_induced_UV, compact.cosmic_ray_ionization_rates_induced_UV);
        check(raw.cosmic_ray_ionization_rates_induced_UV_heat, compact.cosmic_ray_ionization_rates_induced_UV_heat);
        check(raw.dust_rec_coefs, compact.dust_rec_coefs);
        check(raw.CTIon, compact.CTIon);
        check(raw.CTRecomb, compact.CTRecomb);
        check(raw.fs_cool_tab, compact.fs_cool_tab);
        check(raw.high_t_cooling_rates, compact.high_t_cooling_rates);
        check(raw.high_t_cooling_rates_tflag, compact.high_t_cooling_rates_tflag);
        check(raw.DR_rates_helium, compact.DR_rates_helium);
        check(raw.RR_rates_carbon, compact.RR_rates_carbon);
        check(raw.DR_rates_carbon, compact.DR_rates_carbon);
        check(raw.RR_rates_nitrogen, compact.RR_rates_nitrogen);
        check(raw.DR_rates_nitrogen, compact.DR_rates_nitrogen);
        check(raw.RR_rates_oxygen, compact.RR_rates_oxygen);
        check(raw.DR_rates_oxygen, compact.DR_rates_oxygen);
        check(raw.RR_rates_neon, compact.RR_rates_neon);
        check(raw.DR_rates_neon, compact.DR_rates_neon);
        check(raw.RR_rates_magnesium, compact.RR_rates_magnesium);
        check(raw.DR_rates_magnesium, compact.DR_rates_magnesium);
        check(raw.RR_rates_silicon, compact.RR_rates_silicon);
        check(raw.DR_rates_silicon, compact.DR_rates_silicon);
        check(raw.RR_rates_sulfur, compact.RR_rates_sulfur);
        check(raw.DR_rates_sulfur, compact.DR_rates_sulfur);
        check(raw.RR_rates_alt_sulfur, compact.RR_rates_alt_sulfur);
        check(raw.DR_rates_alt_sulfur, compact.DR_rates_alt_sulfur);
        check(raw.RR_rates_iron, compact.RR_rates_iron);
        check(raw.DR_rates_iron, compact.DR_rates_iron);
        check(raw.RR_rates_alt_iron, compact.RR_rates_alt_iron);
        check(raw.DR_rates_alt_iron, compact.DR_rates_alt_iron);
        check(raw.dE_carbon, compact.dE_carbon);
        check(raw.A_carbon, compact.A_carbon);
        check(raw.X_carbon, compact.X_carbon);
        check(raw.K_carbon, compact.K_carbon);
        check(raw.P_carbon, compact.P_carbon);
        check(raw.dE_nitrogen, compact.dE_nitrogen);
        check(raw.A_nitrogen, compact.A_nitrogen);
        check(raw.X_nitrogen, compact.X_nitrogen);
        check(raw.K_nitrogen, compact.K_nitrogen);
        check(raw.P_nitrogen, compact.P_nitrogen);
        check(raw.dE_oxygen, compact.dE_oxygen);
        check(raw.A_oxygen, compact.A_oxygen);
        check(raw.X_oxygen, compact.X_oxygen);
        check(raw.K_oxygen, compact.K_oxygen);
        check(raw.P_oxygen, compact.P_oxygen);
        check(raw.dE_neon, compact.dE_neon);
        check(raw.A_neon, compact.A_neon);
        check(raw.X_neon, compact.X_neon);
        check(raw.K_neon, compact.K_neon);
        check(raw.P_neon, compact.P_neon);
        check(raw.dE_magnesium, compact.dE_magnesium);
        check(raw.A_magnesium, compact.A_magnesium);
        check(raw.X_magnesium, compact.X_magnesium);
        check(raw.K_magnesium, compact.K_magnesium);
        check(raw.P_magnesium, compact.P_magnesium);
        check(raw.dE_silicon, compact.dE_silicon);
        check(raw.A_silicon, compact.A_silicon);
        check(raw.X_silicon, compact.X_silicon);
        check(raw.K_silicon, compact.K_silicon);
        check(raw.P_silicon, compact.P_silicon);
        check(raw.dE_sulfur, compact.dE_sulfur);
        check(raw.A_sulfur, compact.A_sulfur);
        check(raw.X_sulfur, compact.X_sulfur);
        check(raw.K_sulfur, compact.K_sulfur);
        check(raw.P_sulfur, compact.P_sulfur);
        check(raw.dE_iron, compact.dE_iron);
        check(raw.A_iron, compact.A_iron);
        check(raw.X_iron, compact.X_iron);
        check(raw.K_iron, compact.K_iron);
        check(raw.P_iron, compact.P_iron);
        check(raw.rrec, compact.rrec);
        check(raw.rnew, compact.rnew);
        check(raw.fe, compact.fe);
        check(raw.collisional_ionization_data, compact.collisional_ionization_data);
        check(raw.photoionization_cross_sections_tot, compact.photoionization_cross_sections_tot);
        check(raw.photoionization_cross_sections_nl, compact.photoionization_cross_sections_nl);
        check(raw.cs_ph, compact.cs_ph);
        check(raw.sco_col1, compact.sco_col1);
        check(raw.sco_col2, compact.sco_col2);
        check(raw.sh2_col1, compact.sh2_col1);
        check(raw.sh2_col2, compact.sh2_col2);
        check(raw.group_E_min, compact.group_E_min);
        check(raw.group_E_max, compact.group_E_max);
        check(raw.group_energy, compact.group_energy);
        check(raw.dust_wav, compact.dust_wav);
        check(raw.dust_cabs, compact.dust_cabs);
        check(raw.dust_log_wav, compact.dust_log_wav);
        check(raw.dust_log_cabs, compact.dust_log_cabs);
        check(raw.cs_dust, compact.cs_dust);
        check(raw.c_frac, compact.c_frac);
        if (!compact.collisional_ionization_data.contains({elem_row(26), 26, 0}))
            throw std::runtime_error("neutral iron electron-count entry omitted");
        if (compact.collisional_ionization_data.contains({elem_row(26), 21, 0}))
            throw std::runtime_error("inactive iron stage retained");
        if (!compact.rrec.contains({26, elem_row(26), 0}))
            throw std::runtime_error("Cloudy electron-count entry omitted");
        if (compact.high_t_cooling_rates.flat({0, 0, 1}) - compact.high_t_cooling_rates.flat({0, 0, 0}) != 1 ||
            compact.high_t_cooling_rates.flat({1, 0, 0}) - compact.high_t_cooling_rates.flat({0, 0, 0}) != static_cast<std::size_t>(compact.network.ions))
            throw std::runtime_error("temperature/ion axis order changed");
        Kokkos::View<int[2][3], Kokkos::HostSpace> custom_raw("custom_mapping_test");
        fill(custom_raw);
        CompactTableView<decltype(custom_raw), ReverseMapping> custom(custom_raw, compact.network);
        check(custom_raw, custom);
        auto reversed = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), custom.values);
        if (reversed(0) != custom_raw(1, 2)) throw std::runtime_error("custom policy ignored");
        Kokkos::View<int[2][3], Kokkos::LayoutLeft, Kokkos::HostSpace> left_raw("left_layout");
        fill(left_raw);
        CompactTableView<decltype(left_raw), ReverseMapping> left(left_raw, compact.network);
        check(left_raw, left);
        int errors = 0;
        Kokkos::parallel_reduce("custom_mapping_device_check", 6, KOKKOS_LAMBDA(int i, int& count) {
            if (custom(i / 3, i % 3) != i + 1) ++count;
        }, errors);
        if (errors) throw std::runtime_error("custom mapping failed on execution space");
        std::size_t raw_bytes = 0, compact_bytes = 0;
        raw_bytes += data_bytes(raw.HM12_UVB_z); compact_bytes += data_bytes(compact.HM12_UVB_z);
        raw_bytes += data_bytes(raw.G0_heating_rates); compact_bytes += data_bytes(compact.G0_heating_rates);
        raw_bytes += data_bytes(raw.cosmic_ray_ionization_rates); compact_bytes += data_bytes(compact.cosmic_ray_ionization_rates);
        raw_bytes += data_bytes(raw.cosmic_ray_ionization_rates_induced_UV); compact_bytes += data_bytes(compact.cosmic_ray_ionization_rates_induced_UV);
        raw_bytes += data_bytes(raw.cosmic_ray_ionization_rates_induced_UV_heat); compact_bytes += data_bytes(compact.cosmic_ray_ionization_rates_induced_UV_heat);
        raw_bytes += data_bytes(raw.dust_rec_coefs); compact_bytes += data_bytes(compact.dust_rec_coefs);
        raw_bytes += data_bytes(raw.CTIon); compact_bytes += data_bytes(compact.CTIon);
        raw_bytes += data_bytes(raw.CTRecomb); compact_bytes += data_bytes(compact.CTRecomb);
        raw_bytes += data_bytes(raw.fs_cool_tab); compact_bytes += data_bytes(compact.fs_cool_tab);
        raw_bytes += data_bytes(raw.high_t_cooling_rates); compact_bytes += data_bytes(compact.high_t_cooling_rates);
        raw_bytes += data_bytes(raw.high_t_cooling_rates_tflag); compact_bytes += data_bytes(compact.high_t_cooling_rates_tflag);
        raw_bytes += data_bytes(raw.DR_rates_helium); compact_bytes += data_bytes(compact.DR_rates_helium);
        raw_bytes += data_bytes(raw.RR_rates_carbon); compact_bytes += data_bytes(compact.RR_rates_carbon);
        raw_bytes += data_bytes(raw.DR_rates_carbon); compact_bytes += data_bytes(compact.DR_rates_carbon);
        raw_bytes += data_bytes(raw.RR_rates_nitrogen); compact_bytes += data_bytes(compact.RR_rates_nitrogen);
        raw_bytes += data_bytes(raw.DR_rates_nitrogen); compact_bytes += data_bytes(compact.DR_rates_nitrogen);
        raw_bytes += data_bytes(raw.RR_rates_oxygen); compact_bytes += data_bytes(compact.RR_rates_oxygen);
        raw_bytes += data_bytes(raw.DR_rates_oxygen); compact_bytes += data_bytes(compact.DR_rates_oxygen);
        raw_bytes += data_bytes(raw.RR_rates_neon); compact_bytes += data_bytes(compact.RR_rates_neon);
        raw_bytes += data_bytes(raw.DR_rates_neon); compact_bytes += data_bytes(compact.DR_rates_neon);
        raw_bytes += data_bytes(raw.RR_rates_magnesium); compact_bytes += data_bytes(compact.RR_rates_magnesium);
        raw_bytes += data_bytes(raw.DR_rates_magnesium); compact_bytes += data_bytes(compact.DR_rates_magnesium);
        raw_bytes += data_bytes(raw.RR_rates_silicon); compact_bytes += data_bytes(compact.RR_rates_silicon);
        raw_bytes += data_bytes(raw.DR_rates_silicon); compact_bytes += data_bytes(compact.DR_rates_silicon);
        raw_bytes += data_bytes(raw.RR_rates_sulfur); compact_bytes += data_bytes(compact.RR_rates_sulfur);
        raw_bytes += data_bytes(raw.DR_rates_sulfur); compact_bytes += data_bytes(compact.DR_rates_sulfur);
        raw_bytes += data_bytes(raw.RR_rates_alt_sulfur); compact_bytes += data_bytes(compact.RR_rates_alt_sulfur);
        raw_bytes += data_bytes(raw.DR_rates_alt_sulfur); compact_bytes += data_bytes(compact.DR_rates_alt_sulfur);
        raw_bytes += data_bytes(raw.RR_rates_iron); compact_bytes += data_bytes(compact.RR_rates_iron);
        raw_bytes += data_bytes(raw.DR_rates_iron); compact_bytes += data_bytes(compact.DR_rates_iron);
        raw_bytes += data_bytes(raw.RR_rates_alt_iron); compact_bytes += data_bytes(compact.RR_rates_alt_iron);
        raw_bytes += data_bytes(raw.DR_rates_alt_iron); compact_bytes += data_bytes(compact.DR_rates_alt_iron);
        raw_bytes += data_bytes(raw.dE_carbon); compact_bytes += data_bytes(compact.dE_carbon);
        raw_bytes += data_bytes(raw.A_carbon); compact_bytes += data_bytes(compact.A_carbon);
        raw_bytes += data_bytes(raw.X_carbon); compact_bytes += data_bytes(compact.X_carbon);
        raw_bytes += data_bytes(raw.K_carbon); compact_bytes += data_bytes(compact.K_carbon);
        raw_bytes += data_bytes(raw.P_carbon); compact_bytes += data_bytes(compact.P_carbon);
        raw_bytes += data_bytes(raw.dE_nitrogen); compact_bytes += data_bytes(compact.dE_nitrogen);
        raw_bytes += data_bytes(raw.A_nitrogen); compact_bytes += data_bytes(compact.A_nitrogen);
        raw_bytes += data_bytes(raw.X_nitrogen); compact_bytes += data_bytes(compact.X_nitrogen);
        raw_bytes += data_bytes(raw.K_nitrogen); compact_bytes += data_bytes(compact.K_nitrogen);
        raw_bytes += data_bytes(raw.P_nitrogen); compact_bytes += data_bytes(compact.P_nitrogen);
        raw_bytes += data_bytes(raw.dE_oxygen); compact_bytes += data_bytes(compact.dE_oxygen);
        raw_bytes += data_bytes(raw.A_oxygen); compact_bytes += data_bytes(compact.A_oxygen);
        raw_bytes += data_bytes(raw.X_oxygen); compact_bytes += data_bytes(compact.X_oxygen);
        raw_bytes += data_bytes(raw.K_oxygen); compact_bytes += data_bytes(compact.K_oxygen);
        raw_bytes += data_bytes(raw.P_oxygen); compact_bytes += data_bytes(compact.P_oxygen);
        raw_bytes += data_bytes(raw.dE_neon); compact_bytes += data_bytes(compact.dE_neon);
        raw_bytes += data_bytes(raw.A_neon); compact_bytes += data_bytes(compact.A_neon);
        raw_bytes += data_bytes(raw.X_neon); compact_bytes += data_bytes(compact.X_neon);
        raw_bytes += data_bytes(raw.K_neon); compact_bytes += data_bytes(compact.K_neon);
        raw_bytes += data_bytes(raw.P_neon); compact_bytes += data_bytes(compact.P_neon);
        raw_bytes += data_bytes(raw.dE_magnesium); compact_bytes += data_bytes(compact.dE_magnesium);
        raw_bytes += data_bytes(raw.A_magnesium); compact_bytes += data_bytes(compact.A_magnesium);
        raw_bytes += data_bytes(raw.X_magnesium); compact_bytes += data_bytes(compact.X_magnesium);
        raw_bytes += data_bytes(raw.K_magnesium); compact_bytes += data_bytes(compact.K_magnesium);
        raw_bytes += data_bytes(raw.P_magnesium); compact_bytes += data_bytes(compact.P_magnesium);
        raw_bytes += data_bytes(raw.dE_silicon); compact_bytes += data_bytes(compact.dE_silicon);
        raw_bytes += data_bytes(raw.A_silicon); compact_bytes += data_bytes(compact.A_silicon);
        raw_bytes += data_bytes(raw.X_silicon); compact_bytes += data_bytes(compact.X_silicon);
        raw_bytes += data_bytes(raw.K_silicon); compact_bytes += data_bytes(compact.K_silicon);
        raw_bytes += data_bytes(raw.P_silicon); compact_bytes += data_bytes(compact.P_silicon);
        raw_bytes += data_bytes(raw.dE_sulfur); compact_bytes += data_bytes(compact.dE_sulfur);
        raw_bytes += data_bytes(raw.A_sulfur); compact_bytes += data_bytes(compact.A_sulfur);
        raw_bytes += data_bytes(raw.X_sulfur); compact_bytes += data_bytes(compact.X_sulfur);
        raw_bytes += data_bytes(raw.K_sulfur); compact_bytes += data_bytes(compact.K_sulfur);
        raw_bytes += data_bytes(raw.P_sulfur); compact_bytes += data_bytes(compact.P_sulfur);
        raw_bytes += data_bytes(raw.dE_iron); compact_bytes += data_bytes(compact.dE_iron);
        raw_bytes += data_bytes(raw.A_iron); compact_bytes += data_bytes(compact.A_iron);
        raw_bytes += data_bytes(raw.X_iron); compact_bytes += data_bytes(compact.X_iron);
        raw_bytes += data_bytes(raw.K_iron); compact_bytes += data_bytes(compact.K_iron);
        raw_bytes += data_bytes(raw.P_iron); compact_bytes += data_bytes(compact.P_iron);
        raw_bytes += data_bytes(raw.rrec); compact_bytes += data_bytes(compact.rrec);
        raw_bytes += data_bytes(raw.rnew); compact_bytes += data_bytes(compact.rnew);
        raw_bytes += data_bytes(raw.fe); compact_bytes += data_bytes(compact.fe);
        raw_bytes += data_bytes(raw.collisional_ionization_data); compact_bytes += data_bytes(compact.collisional_ionization_data);
        raw_bytes += data_bytes(raw.photoionization_cross_sections_tot); compact_bytes += data_bytes(compact.photoionization_cross_sections_tot);
        raw_bytes += data_bytes(raw.photoionization_cross_sections_nl); compact_bytes += data_bytes(compact.photoionization_cross_sections_nl);
        raw_bytes += data_bytes(raw.cs_ph); compact_bytes += data_bytes(compact.cs_ph);
        raw_bytes += data_bytes(raw.sco_col1); compact_bytes += data_bytes(compact.sco_col1);
        raw_bytes += data_bytes(raw.sco_col2); compact_bytes += data_bytes(compact.sco_col2);
        raw_bytes += data_bytes(raw.sh2_col1); compact_bytes += data_bytes(compact.sh2_col1);
        raw_bytes += data_bytes(raw.sh2_col2); compact_bytes += data_bytes(compact.sh2_col2);
        raw_bytes += data_bytes(raw.group_E_min); compact_bytes += data_bytes(compact.group_E_min);
        raw_bytes += data_bytes(raw.group_E_max); compact_bytes += data_bytes(compact.group_E_max);
        raw_bytes += data_bytes(raw.group_energy); compact_bytes += data_bytes(compact.group_energy);
        raw_bytes += data_bytes(raw.dust_wav); compact_bytes += data_bytes(compact.dust_wav);
        raw_bytes += data_bytes(raw.dust_cabs); compact_bytes += data_bytes(compact.dust_cabs);
        raw_bytes += data_bytes(raw.dust_log_wav); compact_bytes += data_bytes(compact.dust_log_wav);
        raw_bytes += data_bytes(raw.dust_log_cabs); compact_bytes += data_bytes(compact.dust_log_cabs);
        raw_bytes += data_bytes(raw.cs_dust); compact_bytes += data_bytes(compact.cs_dust);
        raw_bytes += data_bytes(raw.c_frac); compact_bytes += data_bytes(compact.c_frac);
        if (compact_bytes >= raw_bytes) throw std::runtime_error("table storage did not shrink");
        std::printf("table data bytes: raw=%zu compact=%zu; cell offset member=%zu bytes\n",
                    raw_bytes, compact_bytes, sizeof(a.offsets));
        clear_ion_offset_cache();
    }
    Kokkos::finalize();
}
