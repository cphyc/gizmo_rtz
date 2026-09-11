// table_verify.hpp
#pragma once

// Startup verification of the element-compacted TabulatedData arrays.
//
// The host-side rate tables are indexed by atomic number; TabulatedData stores
// them with one row per tabulated element (row = elem_row(Z), N_TAB_ELEMENTS
// rows). Every host->device copy of such a table has to apply that mapping. A
// plain dense copy compiles just as happily and silently gives row r the data of
// Z = r, which swaps and zeroes rates instead of failing -- so this reads each
// table back off the device and checks it against its host source. Any table
// copied with the wrong helper aborts the run here rather than quietly changing
// the physics.
//
// Covers the tables filled by the copy_data_* helpers. Those filled directly
// into tabData (collisional_ionization_data, rrec, rnew, cs_ph) keep no host copy
// to compare against; they apply elem_row() inline at the point of the write.

template <typename V>
auto compacted_table_mirror(const V& view) {
    auto host = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(host, view);
    return host;
}

void verify_compacted_tables(const TabulatedData& tab) {
    int bad = 0;

    // The views are float; compare against the host double narrowed the same way.
    auto check = [&](const char* name, int erow, int Z, double got, double want) {
        if (got == want) return;
        if (bad < 20)
            fprintf(stderr, "  %-44s row %d (Z=%2d): has %.8g, expected %.8g\n",
                    name, erow, Z, got, want);
        bad++;
    };

    auto h_uvb    = compacted_table_mirror(tab.HM12_UVB_z);
    auto h_g0     = compacted_table_mirror(tab.G0_heating_rates);
    auto h_cr     = compacted_table_mirror(tab.cosmic_ray_ionization_rates);
    auto h_cruv   = compacted_table_mirror(tab.cosmic_ray_ionization_rates_induced_UV);
    auto h_cruvh  = compacted_table_mirror(tab.cosmic_ray_ionization_rates_induced_UV_heat);
    auto h_dust   = compacted_table_mirror(tab.dust_rec_coefs);
    auto h_ction  = compacted_table_mirror(tab.CTIon);
    auto h_ctrec  = compacted_table_mirror(tab.CTRecomb);
    auto h_hit    = compacted_table_mirror(tab.high_t_cooling_rates);
    auto h_hitf   = compacted_table_mirror(tab.high_t_cooling_rates_tflag);

    for (int erow = 0; erow < N_TAB_ELEMENTS; erow++) {
        const int Z = row_elem(erow);

        check("G0_heating_rates", erow, Z,
              h_g0(erow), (float)G0_heating_rates[Z]);
        check("cosmic_ray_ionization_rates_induced_UV", erow, Z,
              h_cruv(erow), (float)cosmic_ray_ionization_rates_induced_UV[Z]);
        check("cosmic_ray_ionization_rates_induced_UV_heat", erow, Z,
              h_cruvh(erow), (float)cosmic_ray_ionization_rates_induced_UV_heat[Z]);

        for (int j = 0; j < 7; j++)
            check("dust_rec_coefs", erow, Z, h_dust(erow, j), (float)dust_rec_coefs[Z][j]);

        for (int j = 0; j < 27; j++) {
            check("cosmic_ray_ionization_rates", erow, Z,
                  h_cr(erow, j), (float)cosmic_ray_ionization_rates[Z][j]);
            check("high_t_cooling_rates_tflag", erow, Z,
                  h_hitf(erow, j), high_t_cooling_rates_tflag[Z][j]);
            for (int k = 0; k < 2; k++)
                check("HM12_UVB_z", erow, Z, h_uvb(erow, j, k), (float)HM12_UVB_z[Z][j][k]);
        }

        // Element index is the trailing dimension for the charge-transfer tables.
        for (int k = 0; k < 7; k++)
            for (int j = 0; j < 3; j++)
                check("CTIon", erow, Z, h_ction(k, j, erow), (float)CTIon[k][j][Z]);
        for (int k = 0; k < 6; k++)
            for (int j = 0; j < 4; j++)
                check("CTRecomb", erow, Z, h_ctrec(k, j, erow), (float)CTRecomb[k][j][Z]);

        // ...and the middle dimension for the high-T metal cooling rates.
        for (int it = 0; it < N_HIGH_T_COOLING_TEMP; it++)
            for (int j = 0; j < 27; j++)
                check("high_t_cooling_rates", erow, Z,
                      h_hit(it, erow, j), (float)high_t_cooling_rates[it][Z][j]);
    }

    if (bad > 0) {
        fprintf(stderr,
                "\nFATAL: %d element-compacted table entries do not match their host source.\n"
                "       A table is being copied with a plain copy_data_* instead of the\n"
                "       element-remapping copy_data_*_elem (see table_verify.hpp).\n", bad);
        abort();
    }
}
