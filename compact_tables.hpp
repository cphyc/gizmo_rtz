#pragma once
#include <utility>
#include "table_index_mapping.hpp"

// The container knows only the mapping contract. Index semantics, selection,
// and mapping ownership are entirely delegated to the policy type.
template<class Raw, class Mapping>
    requires TableIndexMapping<Mapping, Raw::rank>
struct CompactTableView {
    using value_type = typename Raw::non_const_value_type;
    static constexpr int rank = Raw::rank;
    Kokkos::View<value_type*> values;
    std::array<std::size_t, rank> dimensions;
    Mapping mapping;
    static auto shape(const Raw& raw) {
        std::array<std::size_t, rank> result{};
        for (int axis = 0; axis < rank; ++axis) result[axis] = raw.extent(axis);
        return result;
    }
    KOKKOS_INLINE_FUNCTION std::size_t extent(int axis) const { return dimensions[axis]; }
    KOKKOS_INLINE_FUNCTION bool contains(const std::array<int, rank>& index) const { return mapping.contains(index); }
    KOKKOS_INLINE_FUNCTION std::size_t flat(const std::array<int, rank>& index) const { return mapping.flat(index); }
    template<class... Index>
    KOKKOS_INLINE_FUNCTION value_type& operator()(Index... index) const {
        static_assert(sizeof...(Index) == rank);
        return values(mapping.flat({static_cast<int>(index)...}));
    }
    template<class Host, std::size_t... Axis>
    static auto read(const Host& host, const std::array<int, rank>& index, std::index_sequence<Axis...>) {
        return host(index[Axis]...);
    }
    CompactTableView(const Raw& raw, const TableNetwork& network)
        : dimensions(shape(raw)), mapping(dimensions, network) {
        values = decltype(values)("compact_table", mapping.size());
        auto destination = Kokkos::create_mirror_view(values);
        const auto source = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), raw);
        for (std::size_t linear = 0; linear < raw.size(); ++linear) {
            auto rest = linear;
            std::array<int, rank> index{};
            for (int axis = rank - 1; axis >= 0; --axis) {
                index[axis] = rest % dimensions[axis]; rest /= dimensions[axis];
            }
            if (!mapping.contains(index)) continue;
            destination(mapping.flat(index)) = read(source, index, std::make_index_sequence<rank>{});
        }
        Kokkos::deep_copy(values, destination);
    }
};

template<class Raw>
auto device_table_copy(const Raw& raw) {
    Kokkos::View<typename Raw::data_type, typename Raw::array_layout> copy("constant_table");
    Kokkos::deep_copy(copy, raw);
    return copy;
}

struct CompactTabulatedData {
    TableNetwork network;
    CompactTableView<decltype(RawTabulatedData::HM12_UVB_z), ElementIonMapping<3, 0, 1>> HM12_UVB_z;
    CompactTableView<decltype(RawTabulatedData::G0_heating_rates), ElementMapping<1, 0>> G0_heating_rates;
    CompactTableView<decltype(RawTabulatedData::cosmic_ray_ionization_rates), ElementIonMapping<2, 0, 1>> cosmic_ray_ionization_rates;
    CompactTableView<decltype(RawTabulatedData::cosmic_ray_ionization_rates_induced_UV), ElementMapping<1, 0>> cosmic_ray_ionization_rates_induced_UV;
    CompactTableView<decltype(RawTabulatedData::cosmic_ray_ionization_rates_induced_UV_heat), ElementMapping<1, 0>> cosmic_ray_ionization_rates_induced_UV_heat;
    CompactTableView<decltype(RawTabulatedData::dust_rec_coefs), ElementMapping<2, 0>> dust_rec_coefs;
    CompactTableView<decltype(RawTabulatedData::CTIon), ElementMapping<3, 2>> CTIon;
    CompactTableView<decltype(RawTabulatedData::CTRecomb), ElementMapping<3, 2>> CTRecomb;
    CompactTableView<decltype(RawTabulatedData::fs_cool_tab), FineStructureMapping> fs_cool_tab;
    CompactTableView<decltype(RawTabulatedData::high_t_cooling_rates), ElementIonMapping<3, 1, 2>> high_t_cooling_rates;
    CompactTableView<decltype(RawTabulatedData::high_t_cooling_rates_tflag), ElementIonMapping<2, 0, 1>> high_t_cooling_rates_tflag;
    CompactTableView<decltype(RawTabulatedData::DR_rates_helium), ElementFitMapping<2, 2, 0>> DR_rates_helium;
    CompactTableView<decltype(RawTabulatedData::RR_rates_carbon), ElementFitMapping<2, 6, 0>> RR_rates_carbon;
    CompactTableView<decltype(RawTabulatedData::DR_rates_carbon), ElementFitMapping<2, 6, 0>> DR_rates_carbon;
    CompactTableView<decltype(RawTabulatedData::RR_rates_nitrogen), ElementFitMapping<2, 7, 0>> RR_rates_nitrogen;
    CompactTableView<decltype(RawTabulatedData::DR_rates_nitrogen), ElementFitMapping<2, 7, 0>> DR_rates_nitrogen;
    CompactTableView<decltype(RawTabulatedData::RR_rates_oxygen), ElementFitMapping<2, 8, 0>> RR_rates_oxygen;
    CompactTableView<decltype(RawTabulatedData::DR_rates_oxygen), ElementFitMapping<2, 8, 0>> DR_rates_oxygen;
    CompactTableView<decltype(RawTabulatedData::RR_rates_neon), ElementFitMapping<2, 10, 0>> RR_rates_neon;
    CompactTableView<decltype(RawTabulatedData::DR_rates_neon), ElementFitMapping<2, 10, 0>> DR_rates_neon;
    CompactTableView<decltype(RawTabulatedData::RR_rates_magnesium), ElementFitMapping<2, 12, 0>> RR_rates_magnesium;
    CompactTableView<decltype(RawTabulatedData::DR_rates_magnesium), ElementFitMapping<2, 12, 0>> DR_rates_magnesium;
    CompactTableView<decltype(RawTabulatedData::RR_rates_silicon), ElementFitMapping<2, 14, 0>> RR_rates_silicon;
    CompactTableView<decltype(RawTabulatedData::DR_rates_silicon), ElementFitMapping<2, 14, 0>> DR_rates_silicon;
    CompactTableView<decltype(RawTabulatedData::RR_rates_sulfur), ElementFitMapping<2, 16, 0>> RR_rates_sulfur;
    CompactTableView<decltype(RawTabulatedData::DR_rates_sulfur), ElementFitMapping<2, 16, 0>> DR_rates_sulfur;
    CompactTableView<decltype(RawTabulatedData::RR_rates_alt_sulfur), ElementFitMapping<2, 16, 0>> RR_rates_alt_sulfur;
    CompactTableView<decltype(RawTabulatedData::DR_rates_alt_sulfur), ElementFitMapping<2, 16, 0>> DR_rates_alt_sulfur;
    CompactTableView<decltype(RawTabulatedData::RR_rates_iron), ElementFitMapping<2, 26, 0>> RR_rates_iron;
    CompactTableView<decltype(RawTabulatedData::DR_rates_iron), ElementFitMapping<2, 26, 0>> DR_rates_iron;
    CompactTableView<decltype(RawTabulatedData::RR_rates_alt_iron), ElementFitMapping<2, 26, 0>> RR_rates_alt_iron;
    CompactTableView<decltype(RawTabulatedData::DR_rates_alt_iron), ElementFitMapping<2, 26, 0>> DR_rates_alt_iron;
    CompactTableView<decltype(RawTabulatedData::dE_carbon), ElementFitMapping<1, 6, 0>> dE_carbon;
    CompactTableView<decltype(RawTabulatedData::A_carbon), ElementFitMapping<1, 6, 0>> A_carbon;
    CompactTableView<decltype(RawTabulatedData::X_carbon), ElementFitMapping<1, 6, 0>> X_carbon;
    CompactTableView<decltype(RawTabulatedData::K_carbon), ElementFitMapping<1, 6, 0>> K_carbon;
    CompactTableView<decltype(RawTabulatedData::P_carbon), ElementFitMapping<1, 6, 0>> P_carbon;
    CompactTableView<decltype(RawTabulatedData::dE_nitrogen), ElementFitMapping<1, 7, 0>> dE_nitrogen;
    CompactTableView<decltype(RawTabulatedData::A_nitrogen), ElementFitMapping<1, 7, 0>> A_nitrogen;
    CompactTableView<decltype(RawTabulatedData::X_nitrogen), ElementFitMapping<1, 7, 0>> X_nitrogen;
    CompactTableView<decltype(RawTabulatedData::K_nitrogen), ElementFitMapping<1, 7, 0>> K_nitrogen;
    CompactTableView<decltype(RawTabulatedData::P_nitrogen), ElementFitMapping<1, 7, 0>> P_nitrogen;
    CompactTableView<decltype(RawTabulatedData::dE_oxygen), ElementFitMapping<1, 8, 0>> dE_oxygen;
    CompactTableView<decltype(RawTabulatedData::A_oxygen), ElementFitMapping<1, 8, 0>> A_oxygen;
    CompactTableView<decltype(RawTabulatedData::X_oxygen), ElementFitMapping<1, 8, 0>> X_oxygen;
    CompactTableView<decltype(RawTabulatedData::K_oxygen), ElementFitMapping<1, 8, 0>> K_oxygen;
    CompactTableView<decltype(RawTabulatedData::P_oxygen), ElementFitMapping<1, 8, 0>> P_oxygen;
    CompactTableView<decltype(RawTabulatedData::dE_neon), ElementFitMapping<1, 10, 0>> dE_neon;
    CompactTableView<decltype(RawTabulatedData::A_neon), ElementFitMapping<1, 10, 0>> A_neon;
    CompactTableView<decltype(RawTabulatedData::X_neon), ElementFitMapping<1, 10, 0>> X_neon;
    CompactTableView<decltype(RawTabulatedData::K_neon), ElementFitMapping<1, 10, 0>> K_neon;
    CompactTableView<decltype(RawTabulatedData::P_neon), ElementFitMapping<1, 10, 0>> P_neon;
    CompactTableView<decltype(RawTabulatedData::dE_magnesium), ElementFitMapping<1, 12, 0>> dE_magnesium;
    CompactTableView<decltype(RawTabulatedData::A_magnesium), ElementFitMapping<1, 12, 0>> A_magnesium;
    CompactTableView<decltype(RawTabulatedData::X_magnesium), ElementFitMapping<1, 12, 0>> X_magnesium;
    CompactTableView<decltype(RawTabulatedData::K_magnesium), ElementFitMapping<1, 12, 0>> K_magnesium;
    CompactTableView<decltype(RawTabulatedData::P_magnesium), ElementFitMapping<1, 12, 0>> P_magnesium;
    CompactTableView<decltype(RawTabulatedData::dE_silicon), ElementFitMapping<1, 14, 0>> dE_silicon;
    CompactTableView<decltype(RawTabulatedData::A_silicon), ElementFitMapping<1, 14, 0>> A_silicon;
    CompactTableView<decltype(RawTabulatedData::X_silicon), ElementFitMapping<1, 14, 0>> X_silicon;
    CompactTableView<decltype(RawTabulatedData::K_silicon), ElementFitMapping<1, 14, 0>> K_silicon;
    CompactTableView<decltype(RawTabulatedData::P_silicon), ElementFitMapping<1, 14, 0>> P_silicon;
    CompactTableView<decltype(RawTabulatedData::dE_sulfur), ElementFitMapping<1, 16, 0>> dE_sulfur;
    CompactTableView<decltype(RawTabulatedData::A_sulfur), ElementFitMapping<1, 16, 0>> A_sulfur;
    CompactTableView<decltype(RawTabulatedData::X_sulfur), ElementFitMapping<1, 16, 0>> X_sulfur;
    CompactTableView<decltype(RawTabulatedData::K_sulfur), ElementFitMapping<1, 16, 0>> K_sulfur;
    CompactTableView<decltype(RawTabulatedData::P_sulfur), ElementFitMapping<1, 16, 0>> P_sulfur;
    CompactTableView<decltype(RawTabulatedData::dE_iron), ElementFitMapping<1, 26, 0>> dE_iron;
    CompactTableView<decltype(RawTabulatedData::A_iron), ElementFitMapping<1, 26, 0>> A_iron;
    CompactTableView<decltype(RawTabulatedData::X_iron), ElementFitMapping<1, 26, 0>> X_iron;
    CompactTableView<decltype(RawTabulatedData::K_iron), ElementFitMapping<1, 26, 0>> K_iron;
    CompactTableView<decltype(RawTabulatedData::P_iron), ElementFitMapping<1, 26, 0>> P_iron;
    CompactTableView<decltype(RawTabulatedData::rrec), ElementIonMapping<3, 1, 0, TabulatedElementRow, ElectronCountIndex>> rrec;
    CompactTableView<decltype(RawTabulatedData::rnew), ElementIonMapping<3, 1, 0, TabulatedElementRow, ElectronCountIndex>> rnew;
    decltype(device_table_copy(std::declval<decltype(RawTabulatedData::fe)>())) fe;
    CompactTableView<decltype(RawTabulatedData::collisional_ionization_data), ElementIonMapping<3, 0, 1, TabulatedElementRow, ElectronCountIndex>> collisional_ionization_data;
    CompactTableView<decltype(RawTabulatedData::photoionization_cross_sections_tot), ElementIonMapping<2, 0, 1, AtomicNumberIndex>> photoionization_cross_sections_tot;
    CompactTableView<decltype(RawTabulatedData::photoionization_cross_sections_nl), ElementIonMapping<3, 0, 1, AtomicNumberIndex>> photoionization_cross_sections_nl;
    CompactTableView<decltype(RawTabulatedData::cs_ph), ElementIonMapping<4, 0, 1>> cs_ph;
    decltype(device_table_copy(std::declval<decltype(RawTabulatedData::sco_col1)>())) sco_col1;
    decltype(device_table_copy(std::declval<decltype(RawTabulatedData::sco_col2)>())) sco_col2;
    decltype(device_table_copy(std::declval<decltype(RawTabulatedData::sh2_col1)>())) sh2_col1;
    decltype(device_table_copy(std::declval<decltype(RawTabulatedData::sh2_col2)>())) sh2_col2;
    decltype(device_table_copy(std::declval<decltype(RawTabulatedData::group_E_min)>())) group_E_min;
    decltype(device_table_copy(std::declval<decltype(RawTabulatedData::group_E_max)>())) group_E_max;
    decltype(device_table_copy(std::declval<decltype(RawTabulatedData::group_energy)>())) group_energy;
    decltype(device_table_copy(std::declval<decltype(RawTabulatedData::dust_wav)>())) dust_wav;
    decltype(device_table_copy(std::declval<decltype(RawTabulatedData::dust_cabs)>())) dust_cabs;
    decltype(device_table_copy(std::declval<decltype(RawTabulatedData::dust_log_wav)>())) dust_log_wav;
    decltype(device_table_copy(std::declval<decltype(RawTabulatedData::dust_log_cabs)>())) dust_log_cabs;
    decltype(device_table_copy(std::declval<decltype(RawTabulatedData::cs_dust)>())) cs_dust;
    decltype(device_table_copy(std::declval<decltype(RawTabulatedData::c_frac)>())) c_frac;
    CompactTabulatedData(const RawTabulatedData& raw, const Element* elements, const int* active = nullptr)
        : network(elements, active),
          HM12_UVB_z(raw.HM12_UVB_z, network),
          G0_heating_rates(raw.G0_heating_rates, network),
          cosmic_ray_ionization_rates(raw.cosmic_ray_ionization_rates, network),
          cosmic_ray_ionization_rates_induced_UV(raw.cosmic_ray_ionization_rates_induced_UV, network),
          cosmic_ray_ionization_rates_induced_UV_heat(raw.cosmic_ray_ionization_rates_induced_UV_heat, network),
          dust_rec_coefs(raw.dust_rec_coefs, network),
          CTIon(raw.CTIon, network),
          CTRecomb(raw.CTRecomb, network),
          fs_cool_tab(raw.fs_cool_tab, network),
          high_t_cooling_rates(raw.high_t_cooling_rates, network),
          high_t_cooling_rates_tflag(raw.high_t_cooling_rates_tflag, network),
          DR_rates_helium(raw.DR_rates_helium, network),
          RR_rates_carbon(raw.RR_rates_carbon, network),
          DR_rates_carbon(raw.DR_rates_carbon, network),
          RR_rates_nitrogen(raw.RR_rates_nitrogen, network),
          DR_rates_nitrogen(raw.DR_rates_nitrogen, network),
          RR_rates_oxygen(raw.RR_rates_oxygen, network),
          DR_rates_oxygen(raw.DR_rates_oxygen, network),
          RR_rates_neon(raw.RR_rates_neon, network),
          DR_rates_neon(raw.DR_rates_neon, network),
          RR_rates_magnesium(raw.RR_rates_magnesium, network),
          DR_rates_magnesium(raw.DR_rates_magnesium, network),
          RR_rates_silicon(raw.RR_rates_silicon, network),
          DR_rates_silicon(raw.DR_rates_silicon, network),
          RR_rates_sulfur(raw.RR_rates_sulfur, network),
          DR_rates_sulfur(raw.DR_rates_sulfur, network),
          RR_rates_alt_sulfur(raw.RR_rates_alt_sulfur, network),
          DR_rates_alt_sulfur(raw.DR_rates_alt_sulfur, network),
          RR_rates_iron(raw.RR_rates_iron, network),
          DR_rates_iron(raw.DR_rates_iron, network),
          RR_rates_alt_iron(raw.RR_rates_alt_iron, network),
          DR_rates_alt_iron(raw.DR_rates_alt_iron, network),
          dE_carbon(raw.dE_carbon, network),
          A_carbon(raw.A_carbon, network),
          X_carbon(raw.X_carbon, network),
          K_carbon(raw.K_carbon, network),
          P_carbon(raw.P_carbon, network),
          dE_nitrogen(raw.dE_nitrogen, network),
          A_nitrogen(raw.A_nitrogen, network),
          X_nitrogen(raw.X_nitrogen, network),
          K_nitrogen(raw.K_nitrogen, network),
          P_nitrogen(raw.P_nitrogen, network),
          dE_oxygen(raw.dE_oxygen, network),
          A_oxygen(raw.A_oxygen, network),
          X_oxygen(raw.X_oxygen, network),
          K_oxygen(raw.K_oxygen, network),
          P_oxygen(raw.P_oxygen, network),
          dE_neon(raw.dE_neon, network),
          A_neon(raw.A_neon, network),
          X_neon(raw.X_neon, network),
          K_neon(raw.K_neon, network),
          P_neon(raw.P_neon, network),
          dE_magnesium(raw.dE_magnesium, network),
          A_magnesium(raw.A_magnesium, network),
          X_magnesium(raw.X_magnesium, network),
          K_magnesium(raw.K_magnesium, network),
          P_magnesium(raw.P_magnesium, network),
          dE_silicon(raw.dE_silicon, network),
          A_silicon(raw.A_silicon, network),
          X_silicon(raw.X_silicon, network),
          K_silicon(raw.K_silicon, network),
          P_silicon(raw.P_silicon, network),
          dE_sulfur(raw.dE_sulfur, network),
          A_sulfur(raw.A_sulfur, network),
          X_sulfur(raw.X_sulfur, network),
          K_sulfur(raw.K_sulfur, network),
          P_sulfur(raw.P_sulfur, network),
          dE_iron(raw.dE_iron, network),
          A_iron(raw.A_iron, network),
          X_iron(raw.X_iron, network),
          K_iron(raw.K_iron, network),
          P_iron(raw.P_iron, network),
          rrec(raw.rrec, network),
          rnew(raw.rnew, network),
          fe(device_table_copy(raw.fe)),
          collisional_ionization_data(raw.collisional_ionization_data, network),
          photoionization_cross_sections_tot(raw.photoionization_cross_sections_tot, network),
          photoionization_cross_sections_nl(raw.photoionization_cross_sections_nl, network),
          cs_ph(raw.cs_ph, network),
          sco_col1(device_table_copy(raw.sco_col1)),
          sco_col2(device_table_copy(raw.sco_col2)),
          sh2_col1(device_table_copy(raw.sh2_col1)),
          sh2_col2(device_table_copy(raw.sh2_col2)),
          group_E_min(device_table_copy(raw.group_E_min)),
          group_E_max(device_table_copy(raw.group_E_max)),
          group_energy(device_table_copy(raw.group_energy)),
          dust_wav(device_table_copy(raw.dust_wav)),
          dust_cabs(device_table_copy(raw.dust_cabs)),
          dust_log_wav(device_table_copy(raw.dust_log_wav)),
          dust_log_cabs(device_table_copy(raw.dust_log_cabs)),
          cs_dust(device_table_copy(raw.cs_dust)),
          c_frac(device_table_copy(raw.c_frac)) {}
};
