#pragma once
#include <concepts>

// Network metadata only. No table shapes, index conventions, or transition IDs.
struct TableNetwork {
    Kokkos::View<int*, Kokkos::SharedSpace> rows, counts;
    Kokkos::View<const int*, Kokkos::SharedSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> offsets;
    int elements = 0, ions = 0;
    TableNetwork(const Element* element, const int* active = nullptr)
        : rows("table_rows", MAX_ELEMENTS), counts("table_counts", MAX_ELEMENTS),
          offsets(shared_ion_offsets(element, active), MAX_ELEMENTS) {
        for (int z = 0; z < MAX_ELEMENTS; ++z) {
            rows(z) = -1; counts(z) = 0;
            if (element[z].atomic_number < 1 || (active && active[z] == -1)) continue;
            rows(z) = elements++;
            counts(z) = element[z].n_ions + element[z].n_mol;
            ions += counts(z);
        }
    }
};

// A mapping owns its indexing semantics and any lookup storage it needs.
// It is constructed on the host and its query operations must be device callable.
template<class Mapping, int Rank>
concept TableIndexMapping = requires(const Mapping& mapping,
                                    const std::array<int, Rank>& index,
                                    const std::array<std::size_t, Rank>& shape,
                                    const TableNetwork& network) {
    Mapping(shape, network);
    { mapping.contains(index) } -> std::same_as<bool>;
    { mapping.flat(index) } -> std::same_as<std::size_t>;
    { mapping.size() } -> std::same_as<std::size_t>;
};

template<int Rank, int... Packed>
struct PackedAxes {
    static_assert(((Packed >= 0 && Packed < Rank) && ...));
    std::array<std::size_t, Rank> shape{};
    explicit PackedAxes(const std::array<std::size_t, Rank>& dimensions) : shape(dimensions) {}
    KOKKOS_INLINE_FUNCTION static bool packed(int axis) { return ((axis == Packed) || ...); }
    KOKKOS_INLINE_FUNCTION std::size_t tail_size() const {
        std::size_t size = 1;
        for (int axis = 0; axis < Rank; ++axis) if (!packed(axis)) size *= shape[axis];
        return size;
    }
    KOKKOS_INLINE_FUNCTION std::size_t append(std::size_t row, const std::array<int, Rank>& index, std::size_t entries) const {
        std::size_t result = 0;
        bool inserted = false;
        for (int axis = 0; axis < Rank; ++axis) {
            if (packed(axis)) {
                if (!inserted) { result = result * entries + row; inserted = true; }
            } else result = result * shape[axis] + index[axis];
        }
        return result;
    }
};
struct TabulatedElementRow {
    KOKKOS_INLINE_FUNCTION static int atomic_number(int row) { return row_elem(row); }
};
struct AtomicNumberIndex {
    KOKKOS_INLINE_FUNCTION static int atomic_number(int z) { return z; }
};
struct IonStageIndex {
    KOKKOS_INLINE_FUNCTION static int stage(int, int index) { return index; }
};
struct ElectronCountIndex {
    KOKKOS_INLINE_FUNCTION static int stage(int z, int index) { return z - index; }
};

template<int Rank, int ElementAxis, class ElementIndex = TabulatedElementRow>
struct ElementMapping : PackedAxes<Rank, ElementAxis> {
    const int* rows;
    int element_count;
    ElementMapping(const std::array<std::size_t, Rank>& shape, const TableNetwork& network)
        : PackedAxes<Rank, ElementAxis>(shape), rows(network.rows.data()), element_count(network.elements) {}
    KOKKOS_INLINE_FUNCTION bool contains(const std::array<int, Rank>& index) const {
        int z = ElementIndex::atomic_number(index[ElementAxis]);
        return z >= 0 && z < MAX_ELEMENTS && rows[z] >= 0;
    }
    KOKKOS_INLINE_FUNCTION std::size_t flat(const std::array<int, Rank>& index) const {
        return this->append(rows[ElementIndex::atomic_number(index[ElementAxis])], index, element_count);
    }
    KOKKOS_INLINE_FUNCTION std::size_t size() const { return element_count * this->tail_size(); }
};

template<int Rank, int ElementAxis, int IonAxis,
         class ElementIndex = TabulatedElementRow, class IonIndex = IonStageIndex>
struct ElementIonMapping : PackedAxes<Rank, ElementAxis, IonAxis> {
    static_assert(ElementAxis != IonAxis);
    const int *rows, *counts, *offsets;
    int ion_count;
    ElementIonMapping(const std::array<std::size_t, Rank>& shape, const TableNetwork& network)
        : PackedAxes<Rank, ElementAxis, IonAxis>(shape), rows(network.rows.data()),
          counts(network.counts.data()), offsets(network.offsets.data()), ion_count(network.ions) {}
    KOKKOS_INLINE_FUNCTION bool contains(const std::array<int, Rank>& index) const {
        int z = ElementIndex::atomic_number(index[ElementAxis]);
        if (z < 0 || z >= MAX_ELEMENTS || rows[z] < 0) return false;
        int stage = IonIndex::stage(z, index[IonAxis]);
        return stage >= 0 && stage < counts[z];
    }
    KOKKOS_INLINE_FUNCTION std::size_t flat(const std::array<int, Rank>& index) const {
        int z = ElementIndex::atomic_number(index[ElementAxis]);
        return this->append(offsets[z] + IonIndex::stage(z, index[IonAxis]), index, ion_count);
    }
    KOKKOS_INLINE_FUNCTION std::size_t size() const { return ion_count * this->tail_size(); }
};

template<int Rank, int Z, int IonAxis = 0>
struct ElementFitMapping : PackedAxes<Rank, IonAxis> {
    int ion_count;
    ElementFitMapping(const std::array<std::size_t, Rank>& shape, const TableNetwork& network)
        : PackedAxes<Rank, IonAxis>(shape),
          ion_count(std::min<std::size_t>(shape[IonAxis], network.counts(Z))) {}
    KOKKOS_INLINE_FUNCTION bool contains(const std::array<int, Rank>& index) const {
        return index[IonAxis] >= 0 && index[IonAxis] < ion_count;
    }
    KOKKOS_INLINE_FUNCTION std::size_t flat(const std::array<int, Rank>& index) const {
        return this->append(index[IonAxis], index, ion_count);
    }
    KOKKOS_INLINE_FUNCTION std::size_t size() const { return ion_count * this->tail_size(); }
};

// Domain-specific row membership is supplied as types, outside TableNetwork.
template<int Z, int Stage, int... Rows>
struct IonRows {
    template<class RowMap>
    static void select(const TableNetwork& network, RowMap& mapping, int& count) {
        if (network.rows(Z) >= 0 && Stage < network.counts(Z))
            ((mapping(Rows) = count++), ...);
    }
};

template<int Rank, int Axis, class... Groups>
struct SelectedRowsMapping : PackedAxes<Rank, Axis> {
    Kokkos::View<int*, Kokkos::SharedSpace> row_map;
    int row_count = 0;
    SelectedRowsMapping(const std::array<std::size_t, Rank>& shape, const TableNetwork& network)
        : PackedAxes<Rank, Axis>(shape), row_map("selected_rows", shape[Axis]) {
        for (std::size_t row = 0; row < shape[Axis]; ++row) row_map(row) = -1;
        (Groups::select(network, row_map, row_count), ...);
    }
    KOKKOS_INLINE_FUNCTION bool contains(const std::array<int, Rank>& index) const {
        return row_map(index[Axis]) >= 0;
    }
    KOKKOS_INLINE_FUNCTION std::size_t flat(const std::array<int, Rank>& index) const {
        return this->append(row_map(index[Axis]), index, row_count);
    }
    KOKKOS_INLINE_FUNCTION std::size_t size() const { return row_count * this->tail_size(); }
};

using FineStructureMapping = SelectedRowsMapping<3, 0,
    IonRows<6, 1, 0>, IonRows<6, 0, 1, 2, 3>, IonRows<7, 1, 4, 5, 6>,
    IonRows<8, 0, 7, 8, 9>, IonRows<10, 1, 10>, IonRows<14, 1, 11>,
    IonRows<8, 2, 12, 13, 14>, IonRows<26, 0, 15, 16, 17>,
    IonRows<26, 1, 18, 19, 20>, IonRows<14, 0, 21, 22, 23>, IonRows<16, 1, 24, 25, 26>>;
