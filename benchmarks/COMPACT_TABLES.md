# Network-specific table storage

`RawTabulatedData` contains the original dense shapes in host memory. Loaders,
UV updates and raw-table validation use that format. Spectrum setup runs in
the original execution space using temporary dense device staging tables;
only its results are copied back to the host cache.
`CompactTabulatedData(raw, elements, active)` creates device tables for a fixed
network; `active[z] == -1` excludes an element. Omitting `active` includes all
initialized elements. Ion counts come from `Element::n_ions` / `n_mol`, not
from a cell's current nonzero populations: stages must remain available as
ionization changes.

`CompactIonData::offsets` is now a pointer to one immutable integer array in
Kokkos shared memory, instead of 27 integers in every cell. Identical network
layouts reuse the array. Compact tables use that same array for their ion
bases, with generic shared element-row and ion-count maps. Table-specific policies
own any additional mappings. Cell
initialization is host-side; it binds the map and zeros cell state. The cache
must be cleared before Kokkos finalization (`clear_ion_offset_cache`); both
Python and standalone teardown do this.

The compact accessors preserve raw logical indices while changing physical
storage. In particular:

- `(element row, stage, ...)` tables use the shared ion base plus stage.
- Collisional and Cloudy tables translate electron-count indices to stage
  before applying that base. This matters for truncated ladders.
- Per-element fit tables allocate only active stage rows, or zero entries
  for excluded elements.
- Charge-transfer element axes are packed without changing their fit axes.
- Fine-structure transition IDs have a separate shared row map, selected by
  the owning element and ion stage.
- Fixed global wavelength/group data are copied to device memory as needed.

Raw dense tables stay on the host; compact data and the shared maps are the
only table data required by device evolution. Python constructs the compact
set from its explicit element list. `RTZ::get_tabData()` supplies a compact
set for standalone drivers; `get_rawTabData()` exposes the host setup data.
Physics functions accept either table layout through inferred template types.

Validation: `compact_tables_test` fills every raw field with distinct values
and checks all included indices against the compact accessors. It includes
noncontiguous elements, a five-stage iron ladder, reversed electron counts,
and pointer equality between cell and table offset maps. The CPU solver also
matches the saved raw-table float32 output bit for bit on the 256-cell subset.
The CUDA build and mapping tests pass. The full 10,000-cell GPU output is
bit-for-bit identical to the pre-compaction float32 result in every returned
array (`compact-policy-exact-gpu.json`). The exact gate compares output bit
patterns, without relative or absolute tolerances.

## Mapping contract

`TableNetwork` stores only generic element rows, ion counts, and shared ion
bases. It has no transition catalog or table-specific axes. `CompactTableView`
is parameterized by a mapping type constrained by the C++20
`TableIndexMapping` concept:

- Construct from the raw shape and network metadata.
- `contains(indices) -> bool` selects raw logical indices.
- `flat(indices) -> size_t` maps included indices into packed storage.
- `size() -> size_t` gives the required allocation.

Query methods must be device-callable. Mapping policies own their lookup
storage and semantics; the container only iterates raw indices and copies
included values. Implementations cover element axes, element/ion axes with
stage or electron-count conventions, per-element fits, and selected rows.
Fine-structure row membership is supplied to the selected-row policy through
`IonRows` template arguments. The network itself knows nothing about it.

Tests also instantiate an unrelated reverse-order mapping, reject malformed
mapping types at compile time, test both raw memory layouts, and execute the
custom accessor on the configured Kokkos execution space.

The default axis policies merge packed axes at their first original position;
remaining axes keep their order. For example, `[temperature][element][ion]`
becomes `[temperature][packed ion]`, retaining contiguous adjacent ions.
An initial ion-leading layout measured 6.0102 s versus 4.3012 s for raw tables;
that measurement is preserved in `compact-policy-leading-gpu.json`. It also
failed the harness's bitwise-like strict gate, although added relative error
was only 0.0177% for reference ions above 1e-4 and 0.000189% for temperature.

The initial host-only spectrum setup introduced tiny GPU differences, which
were not acceptable for this layout-only change. Restoring the original
setup execution space eliminated them completely. Dense device staging is
released after setup; there is no persistent dense device table cache.

In the H/He/O/five-stage-Fe unit fixture (double coefficients), packed table
data occupies 124,611 bytes versus 786,246 raw bytes (84.15% less). These are
payload sizes for that fixture, excluding allocation bookkeeping and small
shared index maps. The cell's offset member is one 8-byte pointer rather than
27 integers. The previously measured float32 approximation error against the
double solver is unchanged; this change adds no output differences.

Final timing: 6.611024 s median for the five warmed GPU solves, versus
4.301220 s for the raw-table float32 reference. This implementation reduces
table/cell metadata storage but is **not a speedup** on this workload. The
exact gate passes every output bit in all four arrays across 10,000 cells.
See `results/compact-policy-final-gpu.json`. Default-double CPU regressions:
18 pass, three long sweep tests excluded. Float32 CUDA API/physics checks:
13 pass, eight float64-specific/tight-golden tests excluded. Mapping tests
pass on CPU and CUDA, including custom policies, both source layouts,
noncontiguous elements, truncated ladders, and shared offset identity.
