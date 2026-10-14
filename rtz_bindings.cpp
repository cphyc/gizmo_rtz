#include "dyablo_api.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

using Solver = RTZ<false, false, false, true, false>;

constexpr double kCellSizeCm = 4.4 * 3.086e18;
constexpr double kBlackbodyTemperature = 1.0e4;
constexpr int kConvergenceIterations = 20000;

const std::unordered_map<std::string, int>& element_numbers() {
    static const std::unordered_map<std::string, int> values{
        {"h", 1}, {"hydrogen", 1},
        {"he", 2}, {"helium", 2},
        {"c", 6}, {"carbon", 6},
        {"n", 7}, {"nitrogen", 7},
        {"o", 8}, {"oxygen", 8},
        {"ne", 10}, {"neon", 10},
        {"mg", 12}, {"magnesium", 12},
        {"si", 14}, {"silicon", 14},
        {"s", 16}, {"sulfur", 16},
        {"fe", 26}, {"iron", 26},
    };
    return values;
}

std::string lowercase(std::string value) {
    for (char& character : value)
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    return value;
}

int parse_element(const std::string& name) {
    const auto found = element_numbers().find(lowercase(name));
    if (found == element_numbers().end())
        throw std::invalid_argument("unsupported element name: " + name);
    return found->second;
}

struct IonColumn {
    int atomic_number;
    int stage;
};

IonColumn parse_ion(nb::handle value) {
    if (nb::isinstance<nb::str>(value)) {
        const std::string name = nb::cast<std::string>(value);
        const auto separator = name.rfind('_');
        if (separator == std::string::npos || separator == 0 || separator + 1 == name.size())
            throw std::invalid_argument(
                "ion strings must have the form '<element>_<stage>', e.g. 'hydrogen_01'");
        const std::string stage_text = name.substr(separator + 1);
        std::size_t parsed = 0;
        int stage = 0;
        try {
            stage = std::stoi(stage_text, &parsed);
        } catch (const std::exception&) {
            throw std::invalid_argument("ion stage must be a positive integer: " + name);
        }
        if (parsed != stage_text.size())
            throw std::invalid_argument("ion stage must be a positive integer: " + name);
        return {parse_element(name.substr(0, separator)), stage};
    }

    if (!nb::isinstance<nb::tuple>(value))
        throw std::invalid_argument(
            "each ion must be a '<element>_<stage>' string or an (element, stage) pair");
    const nb::tuple pair = nb::cast<nb::tuple>(value);
    if (pair.size() != 2)
        throw std::invalid_argument("ion pairs must contain exactly an element and a stage");
    return {parse_element(nb::cast<std::string>(pair[0])), nb::cast<int>(pair[1])};
}

struct Inputs {
    std::vector<int> element_numbers;
    std::vector<IonColumn> ions;
    std::vector<std::pair<double, double>> photon_bands;
};

Inputs parse_inputs(nb::sequence element_list,
                    nb::sequence ion_list,
                    nb::sequence photon_bands) {
    Inputs inputs;
    std::array<bool, MAX_ELEMENTS> active_elements{};

    for (nb::handle value : element_list) {
        const std::string name = nb::cast<std::string>(value);
        const int atomic_number = parse_element(name);
        if (active_elements[atomic_number])
            throw std::invalid_argument("element_list contains a duplicate element: " + name);
        active_elements[atomic_number] = true;
        inputs.element_numbers.push_back(atomic_number);
    }

    if (!active_elements[1])
        throw std::invalid_argument("element_list must include hydrogen");

    std::array<std::array<bool, MAX_ELEMENTS + 1>, MAX_ELEMENTS> seen_ions{};
    for (nb::handle value : ion_list) {
        const IonColumn ion = parse_ion(value);
        if (ion.stage < 1 || ion.stage > ion.atomic_number + 1)
            throw std::invalid_argument("ionization stage is outside the supported range");
        if (!active_elements[ion.atomic_number])
            throw std::invalid_argument("each ion's element must also appear in element_list");
        if (seen_ions[ion.atomic_number][ion.stage])
            throw std::invalid_argument("ion_list contains a duplicate ion stage");
        seen_ions[ion.atomic_number][ion.stage] = true;
        inputs.ions.push_back(ion);
    }

    if (inputs.ions.empty())
        throw std::invalid_argument("ion_list must contain at least one ion stage");

    for (int atomic_number : inputs.element_numbers) {
        bool has_stage = false;
        for (int stage = 1; stage <= atomic_number + 1; ++stage)
            has_stage = has_stage || seen_ions[atomic_number][stage];
        if (!has_stage)
            throw std::invalid_argument(
                "every element in element_list must have at least one entry in ion_list");
    }

    const std::size_t band_count = static_cast<std::size_t>(nb::len(photon_bands));
    if (band_count == 0 || band_count > N_GROUPS)
        throw std::invalid_argument("photon_bands must contain between 1 and 6 energy intervals");
    for (nb::handle value : photon_bands) {
        if (!nb::isinstance<nb::sequence>(value))
            throw std::invalid_argument("each photon band must be a (lower_eV, upper_eV) pair");
        const nb::sequence band = nb::cast<nb::sequence>(value);
        if (nb::len(band) != 2)
            throw std::invalid_argument("each photon band must have exactly two energy bounds");
        const double lower = nb::cast<double>(band[0]);
        const double upper = nb::cast<double>(band[1]);
        if (!std::isfinite(lower) || !std::isfinite(upper) || lower <= 0.0 || lower >= upper)
            throw std::invalid_argument("photon bands must have finite positive lower < upper bounds");
        inputs.photon_bands.emplace_back(lower, upper);
    }
    return inputs;
}

std::vector<double> parse_times(nb::handle times, bool& scalar_time) {
    std::vector<double> result;
    if (nb::ndarray_check(times)) {
        const auto array =
            nb::cast<nb::ndarray<nb::numpy, const double, nb::ndim<1>>>(times);
        scalar_time = false;
        result.reserve(array.shape(0));
        for (std::size_t index = 0; index < array.shape(0); ++index)
            result.push_back(array.data()[index * array.stride(0)]);
    } else if (nb::isinstance<nb::sequence>(times)) {
        scalar_time = false;
        const nb::sequence sequence = nb::cast<nb::sequence>(times);
        result.reserve(static_cast<std::size_t>(nb::len(sequence)));
        for (nb::handle value : sequence)
            result.push_back(nb::cast<double>(value));
    } else {
        scalar_time = true;
        result.push_back(nb::cast<double>(times));
    }

    double previous = -1.0;
    for (double time : result) {
        if (!std::isfinite(time) || time < 0.0 || time <= previous)
            throw std::invalid_argument("times must be finite, nonnegative, and strictly increasing");
        previous = time;
    }
    return result;
}

std::filesystem::path resolve_data_path(const std::string& requested_path) {
    std::filesystem::path path = requested_path;
    if (path.empty()) {
        if (const char* environment_path = std::getenv("RTZ_DATA_DIR"))
            path = environment_path;
        else
            path = GIZMO_RTZ_DATA_DIR;
    }
    if (!std::filesystem::is_directory(path))
        throw std::invalid_argument("RTZ data directory does not exist: " + path.string());
    return std::filesystem::weakly_canonical(path);
}

class Runtime {
public:
    explicit Runtime(const std::filesystem::path& data_path)
        : data_path_(data_path), owns_kokkos_(!Kokkos::is_initialized()) {
        if (owns_kokkos_)
            Kokkos::initialize();
        try {
            solver_ = std::make_unique<Solver>(data_path_.string());
            solver_->set_reduced_speed_of_light_factor(1.0);
            solver_->set_UV_background_G0(0.0070977);
            solver_->need_to_update_UVB(0.0);
        } catch (...) {
            solver_.reset();
            if (owns_kokkos_)
                Kokkos::finalize();
            throw;
        }
    }

    ~Runtime() { solver_.reset(); }

    Solver& solver() { return *solver_; }
    const std::filesystem::path& data_path() const { return data_path_; }
    bool owns_kokkos() const { return owns_kokkos_; }

    void set_photon_bands(const std::vector<std::pair<double, double>>& bands) {
        if (bands == configured_bands_)
            return;
        std::array<double, N_GROUPS> lower{};
        std::array<double, N_GROUPS> upper{};
        for (int group = 0; group < N_GROUPS; ++group) {
            const auto& band = bands[std::min<std::size_t>(group, bands.size() - 1)];
            lower[group] = band.first;
            upper[group] = band.second;
        }
        solver_->set_photon_groups(lower, upper);
        solver_->need_to_update_cross_sections(kBlackbodyTemperature);
        configured_bands_ = bands;
    }

private:
    std::filesystem::path data_path_;
    bool owns_kokkos_;
    std::unique_ptr<Solver> solver_;
    std::vector<std::pair<double, double>> configured_bands_;
};

Runtime*& runtime_instance() {
    static Runtime* runtime = nullptr;
    return runtime;
}

Runtime& get_runtime(const std::filesystem::path& data_path) {
    // Keep CUDA/Kokkos resources alive until process teardown. Destructing this
    // singleton from a DSO static destructor can occur after CUDA unloads.
    Runtime*& runtime = runtime_instance();
    if (!runtime)
        runtime = new Runtime(data_path);
    else if (runtime->data_path() != data_path)
        throw std::invalid_argument(
            "RTZ was already initialized with a different data directory in this process");
    return *runtime;
}

void finalize_runtime() {
    Runtime*& runtime = runtime_instance();
    if (!runtime)
        return;
    const bool owns_kokkos = runtime->owns_kokkos();
    delete runtime;
    runtime = nullptr;
    if (owns_kokkos && Kokkos::is_initialized())
        Kokkos::finalize();
}

struct CellState {
    CompactIonData chemistry{};
    double temperature = 0.0;
    double nCO = 0.0;
    double out_Tmu = 0.0;
    double out_mu = 0.0;
    int iterations = 0;
    std::array<double, N_GROUPS> photons{};
    Array2D_RT flux{};
};

void copy_state_to_matrix(const CellState& cell,
                          double* row,
                          const Inputs& inputs,
                          std::size_t density_offset,
                          std::size_t ion_offset,
                          std::size_t photon_offset) {
    row[0] = cell.temperature;
    for (std::size_t column = 0; column < inputs.element_numbers.size(); ++column)
        row[density_offset + column] = cell.chemistry.n_element[inputs.element_numbers[column]];
    for (std::size_t column = 0; column < inputs.ions.size(); ++column) {
        const IonColumn ion = inputs.ions[column];
        row[ion_offset + column] = cell.chemistry[ion.atomic_number].ion_fracs[ion.stage - 1];
    }
    for (std::size_t group = 0; group < inputs.photon_bands.size(); ++group)
        row[photon_offset + group] = cell.photons[group];
}

nb::object solve(nb::ndarray<nb::numpy, const double, nb::ndim<2>> input,
                 nb::handle times_arg,
                 nb::sequence element_list,
                 nb::sequence ion_list,
                 nb::sequence photon_bands,
                 const std::string& requested_data_path,
                 int max_iterations) {
    if (input.shape(0) == 0)
        throw std::invalid_argument("state must contain at least one cell");
    if (max_iterations <= 0)
        throw std::invalid_argument("max_iterations must be positive");

    const Inputs inputs = parse_inputs(element_list, ion_list, photon_bands);
    bool scalar_time = false;
    const std::vector<double> times = parse_times(times_arg, scalar_time);

    const std::size_t n_cells = input.shape(0);
    if (n_cells > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("the number of cells exceeds Kokkos's supported range");
    const std::size_t n_elements = inputs.element_numbers.size();
    const std::size_t n_ions = inputs.ions.size();
    const std::size_t n_bands = inputs.photon_bands.size();
    const std::size_t density_offset = 1;
    const std::size_t ion_offset = density_offset + n_elements;
    const std::size_t photon_offset = ion_offset + n_ions;
    const std::size_t n_columns = photon_offset + n_bands;
    if (input.shape(1) != n_columns)
        throw std::invalid_argument(
            "state must have columns [temperature, element densities, ion fractions, photon densities]");

    std::array<Element, MAX_ELEMENTS> elements{};
    initialize_elements(elements, false);
    std::array<int, MAX_ELEMENTS> network{};
    network.fill(-1);
    for (int atomic_number : inputs.element_numbers)
        network[atomic_number] = 0;
    std::vector<CellState> cells(n_cells);
    for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index) {
        const auto input_value = [&](std::size_t column) {
            return input.data()[cell_index * input.stride(0) + column * input.stride(1)];
        };
        CellState& cell = cells[cell_index];
        cell.chemistry.init_offsets(elements.data(), network.data());
        cell.temperature = input_value(0);
        if (!std::isfinite(cell.temperature) || cell.temperature <= 0.0)
            throw std::invalid_argument("all temperatures must be finite and positive");

        for (std::size_t column = 0; column < n_elements; ++column) {
            const double density = input_value(density_offset + column);
            if (!std::isfinite(density) || density < 0.0)
                throw std::invalid_argument("element densities must be finite and nonnegative");
            const int atomic_number = inputs.element_numbers[column];
            cell.chemistry.n_element[atomic_number] = density;
            cell.chemistry.n_element_new[atomic_number] = density;
        }
        if (cell.chemistry.n_element[1] <= 0.0)
            throw std::invalid_argument("hydrogen density must be positive");

        std::array<double, MAX_ELEMENTS> ion_sums{};
        for (std::size_t column = 0; column < n_ions; ++column) {
            const double fraction = input_value(ion_offset + column);
            if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0)
                throw std::invalid_argument("ion fractions must be finite and between 0 and 1");
            const IonColumn ion = inputs.ions[column];
            cell.chemistry[ion.atomic_number].ion_fracs[ion.stage - 1] = fraction;
            cell.chemistry[ion.atomic_number].ion_fracs_new[ion.stage - 1] = fraction;
            ion_sums[ion.atomic_number] += fraction;
        }
        for (int atomic_number : inputs.element_numbers) {
            if (std::abs(ion_sums[atomic_number] - 1.0) > 1e-8)
                throw std::invalid_argument(
                    "listed ion fractions for each element must sum to one");
        }
        for (std::size_t group = 0; group < n_bands; ++group) {
            const double density = input_value(photon_offset + group);
            if (!std::isfinite(density) || density < 0.0)
                throw std::invalid_argument("photon number densities must be finite and nonnegative");
            cell.photons[group] = density;
        }
    }

    const std::size_t output_size = times.size() * n_cells * n_columns;
    auto output_data = std::make_unique<double[]>(output_size);
    if (!times.empty() && times.back() > 0.0) {
        const std::filesystem::path data_path = resolve_data_path(requested_data_path);
        Runtime& runtime = get_runtime(data_path);
        runtime.set_photon_bands(inputs.photon_bands);
        Solver solver = runtime.solver();
        Kokkos::View<CellState*> device_cells("rtz_python_cells", n_cells);
        auto host_cells = Kokkos::create_mirror_view(device_cells);
        for (std::size_t index = 0; index < n_cells; ++index)
            host_cells(index) = cells[index];
        Kokkos::deep_copy(device_cells, host_cells);

        double previous_time = 0.0;
        const PhysicsFlags flags{};
        const std::array<double, N_GROUPS> dNpdt{};
        const Array2D_RT dFpdt{};
        const auto elements_d = solver.elements_d;
        for (std::size_t time_index = 0; time_index < times.size(); ++time_index) {
            const double time = times[time_index];
            const double delta_time = time - previous_time;
            if (delta_time > 0.0) {
                Kokkos::parallel_for(
                    "rtz_python_solve",
                    Kokkos::RangePolicy<>(0, n_cells),
                    KOKKOS_LAMBDA(const std::size_t cell_index) {
                        CellState& cell = device_cells(cell_index);
                        const Element* device_elements = elements_d.data();
                        const double ne = get_ne(device_elements, cell.chemistry, false);
                        const double mu = get_mu(ne, device_elements, cell.chemistry,
                                                 false, false, cell.nCO);
                        const double Tmu = cell.temperature / mu;
                        solver.solve_chemistry_and_cooling(
                            Tmu, kCellSizeCm, 1.0, delta_time,
                            cell.chemistry, cell.nCO, cell.photons, cell.flux,
                            cell.out_Tmu, cell.out_mu,
                            kConvergenceIterations, max_iterations, cell.iterations,
                            flags, false, dNpdt, dFpdt);
                        cell.temperature = cell.out_Tmu * cell.out_mu;
                    });
                Kokkos::deep_copy(host_cells, device_cells);
                for (std::size_t index = 0; index < n_cells; ++index) {
                    if (host_cells(index).iterations > max_iterations)
                        throw std::runtime_error(
                            "RTZ exceeded max_iterations before reaching a requested time");
                }
            }
            for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index) {
                cells[cell_index] = host_cells(cell_index);
                double* output_row =
                    output_data.get() + (time_index * n_cells + cell_index) * n_columns;
                copy_state_to_matrix(cells[cell_index], output_row, inputs,
                                     density_offset, ion_offset, photon_offset);
            }
            previous_time = time;
        }
    } else {
        for (std::size_t time_index = 0; time_index < times.size(); ++time_index) {
            for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index) {
                double* output_row =
                    output_data.get() + (time_index * n_cells + cell_index) * n_columns;
                copy_state_to_matrix(cells[cell_index], output_row, inputs,
                                     density_offset, ion_offset, photon_offset);
            }
        }
    }

    nb::capsule owner(output_data.get(), [](void* pointer) noexcept {
        delete[] static_cast<double*>(pointer);
    });
    double* result_data = output_data.release();
    if (scalar_time) {
        nb::ndarray<nb::numpy, double, nb::ndim<2>> result(
            result_data, {n_cells, n_columns}, owner);
        return nb::cast(result);
    }
    nb::ndarray<nb::numpy, double, nb::ndim<3>> result(
        result_data, {times.size(), n_cells, n_columns}, owner);
    return nb::cast(result);
}

} // namespace

NB_MODULE(_rtz, module) {
    module.doc() = "Kokkos-backed Python bindings for the RTZ chemistry and cooling solver";
    module.def("_finalize", &finalize_runtime);
    module.def(
        "solve",
        &solve,
        nb::arg("state"),
        nb::arg("times"),
        nb::arg("element_list"),
        nb::arg("ion_list"),
        nb::arg("photon_bands"),
        nb::arg("data_path") = "",
        nb::arg("max_iterations") = 100000,
        "Evolve each state row from t=0 to the requested time(s). A scalar time "
        "returns an (Ncell, Ncolumns) array; a sequence returns "
        "(Ntimes, Ncell, Ncolumns), including the initial state when time=0.");
}
