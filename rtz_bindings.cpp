#include <cassert>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dyablo_api_GPU.cpp"

namespace nb = nanobind;

namespace {

using namespace prism_gpu;
using Solver = RTZ<false, true, true, true, true>;

constexpr double kCellSizeCm = 4.4 * 3.086e18;

const std::unordered_map<std::string, int> &element_numbers() {
    static const std::unordered_map<std::string, int> values{
        {"hydrogen", 1},
        {"helium", 2},
        {"carbon", 6},
        {"nitrogen", 7},
        {"oxygen", 8},
        {"neon", 10},
        {"magnesium", 12},
        {"silicon", 14},
        {"sulfur", 16},
        {"iron", 26},
    };
    return values;
}

int parse_element(const std::string &name) {
    const auto found = element_numbers().find(name);
    if (found == element_numbers().end())
        throw std::invalid_argument("unsupported element name: " + name);
    return found->second;
}

struct IonColumn {
    int atomic_number;
    int stage;
};

IonColumn parse_ion(nb::handle value) {
    if (!nb::isinstance<nb::tuple>(value))
        throw std::invalid_argument("each ion must be an (element, stage) tuple");
    const nb::tuple pair = nb::cast<nb::tuple>(value);
    if (pair.size() != 2)
        throw std::invalid_argument("ion pairs must contain exactly an element and a stage");
    if (!nb::isinstance<nb::int_>(pair[1]) || nb::isinstance<nb::bool_>(pair[1]))
        throw std::invalid_argument("ionization stage must be an integer");
    return {parse_element(nb::cast<std::string>(pair[0])), nb::cast<int>(pair[1])};
}

struct Inputs {
    std::vector<int> element_numbers;
    std::vector<IonColumn> ions;
    std::vector<std::pair<double, double>> photon_bands;
};

Inputs parse_inputs(nb::sequence element_list, nb::sequence ion_list, nb::sequence photon_bands) {
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

    if (!active_elements[1] || !active_elements[2])
        throw std::invalid_argument("element_list must include hydrogen and helium");

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
        if (!nb::isinstance<nb::tuple>(value))
            throw std::invalid_argument("each photon band must be a (lower_eV, upper_eV) pair");
        const nb::tuple band = nb::cast<nb::tuple>(value);
        if (band.size() != 2)
            throw std::invalid_argument("each photon band must have exactly two energy bounds");
        const double lower = nb::cast<double>(band[0]);
        const double upper = nb::cast<double>(band[1]);
        if (!std::isfinite(lower) || !std::isfinite(upper) || lower <= 0.0 || lower >= upper)
            throw std::invalid_argument(
                "photon bands must have finite positive lower < upper bounds");
        inputs.photon_bands.emplace_back(lower, upper);
    }
    return inputs;
}

std::vector<double> parse_times(nb::ndarray<nb::numpy, const double, nb::ndim<1>> times) {
    std::vector<double> result;
    result.reserve(times.shape(0));

    double previous = -1.0;
    for (std::size_t index = 0; index < times.shape(0); ++index) {
        const double time = times.data()[index * times.stride(0)];
        if (!std::isfinite(time) || time < 0.0 || time <= previous)
            throw std::invalid_argument(
                "times must be finite, nonnegative, and strictly increasing");
        result.push_back(time);
        previous = time;
    }
    return result;
}

std::vector<double> copy_1d_array(nb::ndarray<nb::numpy, const double, nb::ndim<1>> array) {
    std::vector<double> result;
    result.reserve(array.shape(0));
    for (std::size_t index = 0; index < array.shape(0); ++index)
        result.push_back(array.data()[index * array.stride(0)]);
    return result;
}

std::filesystem::path resolve_data_path(const std::string &requested_path) {
    std::filesystem::path path = requested_path;
    if (path.empty()) {
        if (const char *environment_path = std::getenv("RTZ_DATA_DIR"))
            path = environment_path;
        else
            path = GIZMO_RTZ_DATA_DIR;
    }
    if (!std::filesystem::is_directory(path))
        throw std::invalid_argument("RTZ data directory does not exist: " + path.string());
    return std::filesystem::weakly_canonical(path);
}

bool early_kokkos_owner = false;
void ensure_cell_runtime() {
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
        early_kokkos_owner = true;
    }
}

class Runtime {
  public:
    explicit Runtime(const std::filesystem::path &data_path)
        : data_path_(data_path), owns_kokkos_(!Kokkos::is_initialized() || early_kokkos_owner) {
        if (!Kokkos::is_initialized())
            Kokkos::initialize();
        try {
            solver_ = std::make_unique<Solver>(data_path_.string());
            solver_->set_reduced_speed_of_light_factor(1.0);
        } catch (...) {
            solver_.reset();
            if (owns_kokkos_)
                Kokkos::finalize();
            throw;
        }
    }

    ~Runtime() {
        solver_.reset();
    }

    Solver &solver() {
        return *solver_;
    }

    const std::filesystem::path &data_path() const {
        return data_path_;
    }

    bool owns_kokkos() const {
        return owns_kokkos_;
    }

    void
    configure(double redshift,
              double uv_g0,
              double T_blackbody,
              const std::vector<std::pair<double, double>> &bands,
              const std::vector<double> &sed_frequency,
              const std::vector<double> &sed_fnu) {
        if (bands != configured_bands_ || T_blackbody_ != T_blackbody ||
            sed_frequency != configured_sed_frequency_ || sed_fnu != configured_sed_fnu_) {
            std::array<Real, N_GROUPS> lower{};
            std::array<Real, N_GROUPS> upper{};
            for (int group = 0; group < N_GROUPS; ++group) {
                const auto &band = bands[std::min<std::size_t>(group, bands.size() - 1)];
                lower[group] = band.first;
                upper[group] = band.second;
            }
            solver_->set_photon_groups(lower, upper);
            solver_->need_to_update_cross_sections(T_blackbody, sed_frequency, sed_fnu);
            configured_bands_ = bands;
            T_blackbody_ = T_blackbody;
            configured_sed_frequency_ = sed_frequency;
            configured_sed_fnu_ = sed_fnu;
        }
        if (!configured_redshift_ || *configured_redshift_ != redshift) {
            solver_->need_to_update_UVB(redshift);
            configured_redshift_ = redshift;
        }
        if (!configured_uv_g0_ || *configured_uv_g0_ != uv_g0) {
            solver_->set_UV_background_G0(uv_g0);
            configured_uv_g0_ = uv_g0;
        }
    }

  private:
    std::filesystem::path data_path_;
    bool owns_kokkos_;
    std::unique_ptr<Solver> solver_;
    std::vector<std::pair<double, double>> configured_bands_;
    std::vector<double> configured_sed_frequency_;
    std::vector<double> configured_sed_fnu_;
    std::optional<double> configured_redshift_;
    std::optional<double> configured_uv_g0_;
    double T_blackbody_ = 0.0;
};

Runtime *&runtime_instance() {
    static Runtime *runtime = nullptr;
    return runtime;
}

Runtime &get_runtime(const std::filesystem::path &data_path) {
    // Keep CUDA/Kokkos resources alive until process teardown. Destructing this
    // singleton from a DSO static destructor can occur after CUDA unloads.
    Runtime *&runtime = runtime_instance();
    if (!runtime)
        runtime = new Runtime(data_path);
    else if (runtime->data_path() != data_path)
        throw std::invalid_argument(
            "RTZ was already initialized with a different data directory in this process");
    return *runtime;
}

void finalize_runtime() {
    Runtime *&runtime = runtime_instance();
    if (!runtime) {
        clear_ion_offset_cache();
        if (early_kokkos_owner && Kokkos::is_initialized()) Kokkos::finalize();
        early_kokkos_owner = false;
        return;
    }
    const bool owns_kokkos = runtime->owns_kokkos();
    clear_ion_offset_cache();
    delete runtime;
    runtime = nullptr;
    if (owns_kokkos && Kokkos::is_initialized())
        Kokkos::finalize();
}

struct CellState {
    CompactIonData chemistry{};
    SubcycleState subcycle{};
    Real temperature = 0.0;
    Real nCO = 0.0;
    Real out_Tmu = 0.0;
    Real out_mu = 0.0;
    Real dust_ratio = 0.0;
    Real loc_rho = 0.0;
    Real ss_factor = 1.0;
    Real ddt = 0.0;
    Real total_time = 0.0;
    int iterations = 0;
    bool done = false;
    bool initial_interval = true;
    std::array<Real, N_GROUPS> photons{};
    std::array<Real, N_GROUPS> photons_new{};
    std::array<Real, N_GROUPS> photon_attenuation{};
    std::array<Real, 2> cooling_rates{};
    Array2D_RT flux{};
};

// Keep the descriptor alive once per solve rather than copying all of its
// views and mappings into every subcycle kernel's argument block.
template<class Tables>
class DeviceTableDescriptor {
    Kokkos::View<unsigned char*> storage_;
    Tables* pointer_;
  public:
    static void construct(Tables* pointer, const Tables& tables) {
        Kokkos::parallel_for("rtz_table_descriptor_init", 1, KOKKOS_LAMBDA(int) {
            ::new (static_cast<void*>(pointer)) Tables(tables);
        });
    }
    static void destroy(Tables* pointer) {
        Kokkos::parallel_for("rtz_table_descriptor_destroy", 1, KOKKOS_LAMBDA(int) {
            pointer->~Tables();
        });
        Kokkos::fence();
    }
    explicit DeviceTableDescriptor(const Tables& tables)
        : storage_(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rtz_table_descriptor"), sizeof(Tables)),
          pointer_(reinterpret_cast<Tables*>(storage_.data())) {
        construct(pointer_, tables);
    }
    DeviceTableDescriptor(const DeviceTableDescriptor&) = delete;
    ~DeviceTableDescriptor() {
        destroy(pointer_);
    }
    const Tables* data() const { return pointer_; }
};

void copy_state_to_matrix(const CellState &cell,
                          double *row,
                          const Inputs &inputs,
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

struct SolveParameters {
    Inputs inputs;
    std::vector<double> times;
    std::size_t n_cells;
    std::size_t n_columns;
    std::size_t density_offset;
    std::size_t ion_offset;
    std::size_t photon_offset;
};

struct SolveOptions {
    std::string data_path;
    int max_iterations;
    int conv_its;
    double redshift;
    double uv_g0;
    double T_blackbody;
    std::vector<double> sed_frequency;
    std::vector<double> sed_fnu;
    bool fixed_radiation;
    bool equilibrium;
    bool temperature_is_Tmu;
    bool use_default_depletion;
    PhysicsFlags physics_flags;
};

template <bool include_CO>
SolveParameters validate_parameters(nb::ndarray<nb::numpy, const double, nb::ndim<2>> input,
                                    nb::ndarray<nb::numpy, const double, nb::ndim<1>> times_arg,
                                    nb::sequence element_list,
                                    nb::sequence ion_list,
                                    nb::sequence photon_bands,
                                    const SolveOptions &options) {
    if (input.shape(0) == 0)
        throw std::invalid_argument("state must contain at least one cell");
    if (options.max_iterations <= 0)
        throw std::invalid_argument("max_iterations must be positive");
    if (options.conv_its <= 0)
        throw std::invalid_argument("conv_its must be positive");
    if (!std::isfinite(options.redshift) || options.redshift < 0.0)
        throw std::invalid_argument("redshift must be finite and nonnegative");
    if (!std::isfinite(options.uv_g0) || options.uv_g0 < 0.0)
        throw std::invalid_argument("uv_g0 must be finite and nonnegative");
    if (!std::isfinite(options.T_blackbody) || options.T_blackbody <= 0.0)
        throw std::invalid_argument("T_blackbody must be finite and positive");

    SolveParameters parameters;
    parameters.inputs = parse_inputs(element_list, ion_list, photon_bands);
    if (options.sed_frequency.empty() != options.sed_fnu.empty())
        throw std::invalid_argument("sed_frequency and sed_fnu must both be provided");
    if (!options.sed_frequency.empty()) {
        if (options.sed_frequency.size() < 2 ||
            options.sed_frequency.size() != options.sed_fnu.size())
            throw std::invalid_argument(
                "SED frequency and F_nu arrays must have equal lengths of at least two");
        for (std::size_t i = 0; i < options.sed_frequency.size(); ++i) {
            if (!std::isfinite(options.sed_frequency[i]) || options.sed_frequency[i] <= 0.0 ||
                !std::isfinite(options.sed_fnu[i]) || options.sed_fnu[i] < 0.0 ||
                (i > 0 && options.sed_frequency[i] <= options.sed_frequency[i - 1]))
                throw std::invalid_argument(
                    "SED frequencies must be positive and strictly increasing; F_nu must be "
                    "finite and nonnegative");
        }
        for (const auto &[lower_eV, upper_eV] : parameters.inputs.photon_bands) {
            const double lower_frequency = lower_eV * EV_2_ERG / H_PLANCK;
            const double upper_frequency = upper_eV * EV_2_ERG / H_PLANCK;
            if (options.sed_frequency.front() > lower_frequency ||
                options.sed_frequency.back() < upper_frequency)
                throw std::invalid_argument(
                    "SED frequency samples must cover every photon band");
            bool has_flux = false;
            for (std::size_t i = 0; i + 1 < options.sed_frequency.size(); ++i) {
                const double overlap_lower =
                    std::max(lower_frequency, options.sed_frequency[i]);
                const double overlap_upper =
                    std::min(upper_frequency, options.sed_frequency[i + 1]);
                if (overlap_lower < overlap_upper &&
                    (options.sed_fnu[i] > 0.0 || options.sed_fnu[i + 1] > 0.0)) {
                    has_flux = true;
                    break;
                }
            }
            if (!has_flux)
                throw std::invalid_argument("SED F_nu must be positive within every photon band");
        }
    }
    if constexpr (include_CO) {
        const auto &elements_by_name = element_numbers();
        const bool has_carbon =
            std::find(parameters.inputs.element_numbers.begin(),
                      parameters.inputs.element_numbers.end(),
                      elements_by_name.at("carbon")) != parameters.inputs.element_numbers.end();
        const bool has_oxygen =
            std::find(parameters.inputs.element_numbers.begin(),
                      parameters.inputs.element_numbers.end(),
                      elements_by_name.at("oxygen")) != parameters.inputs.element_numbers.end();
        if (!has_carbon || !has_oxygen)
            throw std::invalid_argument(
                "element_list must include carbon and oxygen when include_CO=True");
    }
    parameters.times = parse_times(times_arg);

    parameters.n_cells = input.shape(0);
    if (parameters.n_cells > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("the number of cells exceeds Kokkos's supported range");
    parameters.density_offset = 1;
    parameters.ion_offset = parameters.density_offset + parameters.inputs.element_numbers.size();
    parameters.photon_offset = parameters.ion_offset + parameters.inputs.ions.size();
    parameters.n_columns = parameters.photon_offset + parameters.inputs.photon_bands.size();
    if (input.shape(1) != parameters.n_columns)
        throw std::invalid_argument("state must have columns [temperature, element densities, ion "
                                    "fractions, photon densities]");

    return parameters;
}

std::vector<CellState>
validate_state_values(nb::ndarray<nb::numpy, const double, nb::ndim<2>> input,
                      const SolveParameters &parameters,
                      bool include_H2) {
    const Inputs &inputs = parameters.inputs;
    const std::size_t n_elements = inputs.element_numbers.size();
    const std::size_t n_ions = inputs.ions.size();
    const std::size_t n_bands = inputs.photon_bands.size();
    std::array<Element, MAX_ELEMENTS> elements{};
    initialize_elements(elements, include_H2);
    std::array<int, MAX_ELEMENTS> network{};
    network.fill(-1);
    for (int atomic_number : inputs.element_numbers)
        network[atomic_number] = 0;
    std::vector<CellState> cells(parameters.n_cells);
    for (std::size_t cell_index = 0; cell_index < parameters.n_cells; ++cell_index) {
        const auto input_value = [&](std::size_t column) {
            return input.data()[cell_index * input.stride(0) + column * input.stride(1)];
        };
        CellState &cell = cells[cell_index];
        cell.chemistry.init_offsets(elements.data(), network.data());
        cell.temperature = input_value(0);
        if (!std::isfinite(cell.temperature) || cell.temperature <= 0.0)
            throw std::invalid_argument("all temperatures must be finite and positive");

        for (std::size_t column = 0; column < n_elements; ++column) {
            const double density = input_value(parameters.density_offset + column);
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
            const double fraction = input_value(parameters.ion_offset + column);
            if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0)
                throw std::invalid_argument("ion fractions must be finite and between 0 and 1");
            const IonColumn ion = inputs.ions[column];
            cell.chemistry[ion.atomic_number].ion_fracs[ion.stage - 1] = fraction;
            cell.chemistry[ion.atomic_number].ion_fracs_new[ion.stage - 1] = fraction;
            ion_sums[ion.atomic_number] += fraction;
        }
        for (int atomic_number : inputs.element_numbers) {
            if (include_H2 && atomic_number == 1) {
                if (ion_sums[atomic_number] > 1.0 + 1e-8)
                    throw std::invalid_argument(
                        "hydrogen ion fractions must sum to at most one when include_H2=True");
                const double h2_fraction = std::max(0.0, 1.0 - ion_sums[atomic_number]);
                cell.chemistry[1].ion_fracs[2] = h2_fraction;
                cell.chemistry[1].ion_fracs_new[2] = h2_fraction;
            } else if (std::abs(ion_sums[atomic_number] - 1.0) > 1e-8) {
                throw std::invalid_argument(
                    "listed ion fractions for each element must sum to one");
            }
        }
        for (std::size_t group = 0; group < n_bands; ++group) {
            const double density = input_value(parameters.photon_offset + group);
            if (!std::isfinite(density) || density < 0.0)
                throw std::invalid_argument(
                    "photon number densities must be finite and nonnegative");
            cell.photons[group] = density;
        }
    }

    return cells;
}

template <bool constant_temperature,
          bool include_H2,
          bool include_CO,
          bool rt_advect,
          bool include_self_shielding>
nb::object run_solver(const SolveParameters &parameters,
                      std::vector<CellState> cells,
                      const SolveOptions &options) {
    const Inputs &inputs = parameters.inputs;
    const std::vector<double> &times = parameters.times;
    const std::size_t n_cells = parameters.n_cells;
    const std::size_t n_columns = parameters.n_columns;
    const std::size_t density_offset = parameters.density_offset;
    const std::size_t ion_offset = parameters.ion_offset;
    const std::size_t photon_offset = parameters.photon_offset;

    const std::size_t output_size = times.size() * n_cells * n_columns;
    auto output_data = std::make_unique<double[]>(output_size);
    if (times.empty() || times.back() <= 0) {
        for (std::size_t time_index = 0; time_index < times.size(); ++time_index) {
            for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index) {
                double *output_row =
                    output_data.get() + (time_index * n_cells + cell_index) * n_columns;
                copy_state_to_matrix(cells[cell_index],
                                     output_row,
                                     inputs,
                                     density_offset,
                                     ion_offset,
                                     photon_offset);
            }
        }
        nb::capsule owner(output_data.get(),
                          [](void *pointer) noexcept { delete[] static_cast<double *>(pointer); });
        double *result_data = output_data.release();
        nb::ndarray<nb::numpy, double, nb::ndim<3>> result(
            result_data, {times.size(), n_cells, n_columns}, owner);
        return nb::cast(result);
    }


    const std::filesystem::path data_path = resolve_data_path(options.data_path);
    Runtime &runtime = get_runtime(data_path);
    runtime.configure(
        options.redshift, options.uv_g0, options.T_blackbody, inputs.photon_bands,
        options.sed_frequency, options.sed_fnu);
    Solver solver = runtime.solver();
    Kokkos::View<CellState *> device_cells("rtz_python_cells", n_cells);
    // Keep chemistry in place; compact only cell indices as subcycles finish.
    Kokkos::View<std::size_t *> active_cells("rtz_python_active_cells", n_cells);
    Kokkos::View<std::size_t *> next_active_cells("rtz_python_next_active_cells", n_cells);
    auto host_cells = Kokkos::create_mirror_view(device_cells);
    std::array<Element, MAX_ELEMENTS> configured_elements{};
    initialize_elements(configured_elements, include_H2);
    if (!options.use_default_depletion) {
        for (Element &element : configured_elements) {
            if (element.atomic_number > 0)
                element.depletion = 1.0;
        }
    }

    std::vector<std::uint16_t> ion_map;
    for (int atomic_number : inputs.element_numbers) {
        if (atomic_number < 3)
            continue;

        const int ion_count = configured_elements[atomic_number].n_ions;
        for (int stage = 0; stage < ion_count; ++stage)
            ion_map.push_back(static_cast<std::uint16_t>((atomic_number << 8) | stage));
    }

    const int n_ion_work = static_cast<int>(ion_map.size());
    Kokkos::View<std::uint16_t *> ion_map_d("rtz_python_ion_map", n_ion_work);
    auto host_ion_map = Kokkos::create_mirror_view(ion_map_d);
    for (int index = 0; index < n_ion_work; ++index)
        host_ion_map(index) = ion_map[index];
    Kokkos::deep_copy(ion_map_d, host_ion_map);

    Kokkos::View<Element *> elements_d("rtz_python_elements", MAX_ELEMENTS);
    auto host_elements = Kokkos::create_mirror_view(elements_d);
    for (int index = 0; index < MAX_ELEMENTS; ++index)
        host_elements(index) = configured_elements[index];
    Kokkos::deep_copy(elements_d, host_elements);
    for (std::size_t index = 0; index < n_cells; ++index)
        host_cells(index) = cells[index];
    Kokkos::deep_copy(device_cells, host_cells);

    double previous_time = 0.0;
    const PhysicsFlags flags = options.physics_flags;
    const std::array<Real, N_GROUPS> dNpdt{};
    const Array2D_RT dFpdt{};
    const Element *device_elements = elements_d.data();
    const std::uint16_t *device_ion_map = ion_map_d.data();
    std::array<int, MAX_ELEMENTS> active_tables{};
    active_tables.fill(-1);
    for (int z : inputs.element_numbers) active_tables[z] = 0;
    const CompactTabulatedData host_tables(solver.get_rawTabData(), configured_elements.data(), active_tables.data());
    const DeviceTableDescriptor<CompactTabulatedData> descriptor(host_tables);
    const auto* device_tables = descriptor.data();
    const Real uv_background = solver.get_UV_background_G0();
    const Real primary_cr_rate = solver.get_generic_cosmic_ray_ionization_rate();
    const bool equilibrium = options.equilibrium;
    const bool temperature_is_Tmu = options.temperature_is_Tmu;
    const bool fixed_radiation = options.fixed_radiation;
    const int conv_its = options.conv_its;
    const int max_iterations = options.max_iterations;
    using TeamPolicy = Kokkos::TeamPolicy<>;
    using TeamMember = TeamPolicy::member_type;
    const std::size_t ion_snapshot_bytes = MAX_TOTAL_IONS * sizeof(CompactIonData::xion_t);
    const std::size_t rate_cache_bytes = 4 * n_ion_work * sizeof(Real);
    const std::size_t ion_scratch_bytes = ion_snapshot_bytes + alignof(Real) - 1 + rate_cache_bytes;

    for (std::size_t time_index = 0; time_index < times.size(); ++time_index) {
        const double time = times[time_index];
        const Real requested_delta_time = time - previous_time;
        if (requested_delta_time == 0.0) {
            for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index) {
                double *output_row =
                    output_data.get() + (time_index * n_cells + cell_index) * n_columns;
                copy_state_to_matrix(cells[cell_index],
                                     output_row,
                                     inputs,
                                     density_offset,
                                     ion_offset,
                                     photon_offset);
            }
            previous_time = time;
            continue;
        }
        assert(requested_delta_time > 0.0);
        const Real total_delta_time = equilibrium ? -1.0 : requested_delta_time;

        Kokkos::parallel_for(
            "rtz_python_interval_init",
            Kokkos::RangePolicy<>(0, n_cells),
            KOKKOS_LAMBDA(const std::size_t cell_index) {
                active_cells(cell_index) = cell_index;
                CellState &cell = device_cells(cell_index);
                const Real initial_ne = get_ne(device_elements, cell.chemistry, false);
                const Real initial_mu = get_mu(
                    initial_ne, device_elements, cell.chemistry, false, include_H2, cell.nCO);
                const Real initial_Tmu = constant_temperature
                                               ? cell.temperature
                                               : (temperature_is_Tmu &&
                                                          cell.initial_interval
                                                      ? cell.temperature
                                                      : cell.temperature / initial_mu);
                cell.dust_ratio = get_dust_mass_and_depletion(device_elements,
                                                              cell.chemistry.n_element[8],
                                                              cell.chemistry.n_element[1],
                                                              cell.nCO);
                cell.loc_rho = get_rho(device_elements, cell.chemistry);
                apply_dust_depletion(device_elements, cell.dust_ratio, cell.chemistry, cell.nCO);
                if constexpr (include_self_shielding)
                    cell.ss_factor = rtz_math::exp(-cell.chemistry[1].n_element / Real(1E-2));
                else
                    cell.ss_factor = Real(1.0);

                cell.subcycle = SubcycleState{};
                cell.subcycle.success_counter = 0;
                cell.subcycle.ne_initial = get_ne(device_elements, cell.chemistry, false);
                cell.subcycle.Tmu_old = initial_Tmu;
                cell.subcycle.Tmu_new = cell.subcycle.Tmu_old;
                cell.ddt = equilibrium
                               ? Real(10000.0) * Real(365.25) * Real(24.0) * Real(60.0) * Real(60.0)
                               : requested_delta_time;
                cell.total_time = Real(0.0);
                cell.iterations = 0;
                cell.done = false;
            });

        std::size_t pending = n_cells;
        std::size_t active_count = n_cells;
        while (pending > 0) {
            Kokkos::parallel_for(
                "rtz_python_prep_radiation",
                Kokkos::RangePolicy<>(0, active_count),
                KOKKOS_LAMBDA(const std::size_t active_index) {
                    const auto& tab_data = *device_tables;
                    const std::size_t cell_index = active_cells(active_index);
                    CellState &cell = device_cells(cell_index);
                    if (cell.done)
                        return;
                    ++cell.iterations;
                    subcycle_prep_iteration<constant_temperature,
                                            include_H2,
                                            include_CO,
                                            rt_advect,
                                            include_self_shielding>(device_elements,
                                                                    cell.chemistry,
                                                                    cell.nCO,
                                                                    total_delta_time,
                                                                    cell.total_time,
                                                                    primary_cr_rate,
                                                                    cell.ddt,
                                                                    cell.subcycle);
                    radiation_step<constant_temperature,
                                   include_H2,
                                   include_CO,
                                   rt_advect,
                                   include_self_shielding>(cell.ddt,
                                                           device_elements,
                                                           cell.chemistry,
                                                           tab_data,
                                                           cell.nCO,
                                                           kCellSizeCm,
                                                           cell.dust_ratio,
                                                           cell.photons,
                                                           cell.photons_new,
                                                           cell.photon_attenuation,
                                                           cell.subcycle,
                                                           Real(0.0),
                                                           nullptr,
                                                           false,
                                                           dNpdt,
                                                           fixed_radiation);
                });

            if constexpr (!constant_temperature) {
                Kokkos::parallel_for(
                    "rtz_python_cooling_rates",
                    Kokkos::RangePolicy<>(0, 2 * active_count),
                    KOKKOS_LAMBDA(const std::size_t lane_index) {
                        const auto& tab_data = *device_tables;
                        const std::size_t cell_index = active_cells(lane_index / 2);
                        CellState &cell = device_cells(cell_index);
                        if (cell.done)
                            return;
                        cooling_rate_lane<constant_temperature,
                                          include_H2,
                                          include_CO,
                                          rt_advect,
                                          include_self_shielding>(static_cast<int>(lane_index % 2),
                                                                  cell.cooling_rates.data(),
                                                                  Real(1.0),
                                                                  device_elements,
                                                                  cell.chemistry,
                                                                  tab_data,
                                                                  flags,
                                                                  cell.nCO,
                                                                  cell.dust_ratio,
                                                                  primary_cr_rate,
                                                                  cell.ss_factor,
                                                                  uv_background,
                                                                  cell.photons_new,
                                                                  cell.subcycle);
                    });
            }

            // H/He is sequential within a cell. Give neighboring cells to
            // neighboring lanes instead of leaving a team idle during it.
            // The following kernel observes these updates on the same stream.
            Kokkos::parallel_for(
                "rtz_python_molecules_hhe",
                Kokkos::RangePolicy<>(0, active_count),
                KOKKOS_LAMBDA(const std::size_t active_index) {
                    const auto& tab_data = *device_tables;
                    CellState &cell = device_cells(active_cells(active_index));
                    if (cell.done)
                        return;

                    if constexpr (!constant_temperature) {
                        cooling_update<constant_temperature,
                                       include_H2,
                                       include_CO,
                                       rt_advect,
                                       include_self_shielding>(
                            cell.cooling_rates.data(), cell.ddt, cell.loc_rho, cell.subcycle);
                    }

                    molecular_step<constant_temperature,
                                   include_H2,
                                   include_CO,
                                   rt_advect,
                                   include_self_shielding>(cell.ddt,
                                                           device_elements,
                                                           cell.chemistry,
                                                           tab_data,
                                                           cell.nCO,
                                                           cell.dust_ratio,
                                                           uv_background,
                                                           cell.photons_new,
                                                           cell.subcycle);

                    HandHe_ions_step<constant_temperature,
                                     include_H2,
                                     include_CO,
                                     rt_advect,
                                     include_self_shielding>(cell.ddt,
                                                             device_elements,
                                                             cell.chemistry,
                                                             tab_data,
                                                             flags,
                                                             cell.dust_ratio,
                                                             uv_background,
                                                             primary_cr_rate,
                                                             cell.ss_factor,
                                                             cell.photons_new,
                                                             cell.subcycle);
                });

            // Metal stages retain one team per cell and their frozen snapshot.
#ifdef KOKKOS_ENABLE_CUDA
            TeamPolicy team_policy(static_cast<int>(active_count), 32);
#else
            TeamPolicy team_policy(static_cast<int>(active_count), Kokkos::AUTO());
#endif
            team_policy.set_scratch_size(0, Kokkos::PerTeam(ion_scratch_bytes));

            Kokkos::parallel_for(
                "rtz_python_metals_commit",
                team_policy,
                KOKKOS_LAMBDA(const TeamMember &team) {
                    const auto& tab_data = *device_tables;
                    const std::size_t cell_index = active_cells(team.league_rank());
                    CellState &cell = device_cells(cell_index);
                    if (cell.done)
                        return;

                    auto *ion_snapshot = static_cast<CompactIonData::xion_t *>(
                        team.team_shmem().get_shmem(ion_snapshot_bytes));
                    auto* rate_cache = static_cast<Real*>(
                        team.team_shmem().get_shmem_aligned(rate_cache_bytes, alignof(Real)));

                    metal_ions_step<constant_temperature,
                                    include_H2,
                                    include_CO,
                                    rt_advect,
                                    include_self_shielding, true, true>(team,
                                                            cell.ddt,
                                                            device_elements,
                                                            cell.chemistry,
                                                            tab_data,
                                                            flags,
                                                            cell.dust_ratio,
                                                            uv_background,
                                                            primary_cr_rate,
                                                            cell.ss_factor,
                                                            cell.photons_new,
                                                            device_ion_map,
                                                            n_ion_work,
                                                            ion_snapshot,
                                                            cell.subcycle,
                                                            rate_cache,
                                                            rate_cache + n_ion_work,
                                                            rate_cache + 2 * n_ion_work,
                                                            rate_cache + 3 * n_ion_work);
                    team.team_barrier();

                    Kokkos::single(Kokkos::PerTeam(team), [&]() {
                        Real ddt = cell.ddt;
                        Real total_time = cell.total_time;
                        subcycle_commit_and_control<constant_temperature,
                                                    include_H2,
                                                    include_CO,
                                                    rt_advect,
                                                    include_self_shielding>(device_elements,
                                                                            tab_data,
                                                                            cell.chemistry,
                                                                            cell.nCO,
                                                                            cell.photons,
                                                                            cell.flux,
                                                                            cell.photons_new,
                                                                            cell.photon_attenuation,
                                                                            ddt,
                                                                            total_time,
                                                                            cell.subcycle,
                                                                            false,
                                                                            dFpdt);
                        cell.ddt = ddt;
                        cell.total_time = total_time;

                        if (equilibrium) {
                            if (cell.subcycle.success_counter > conv_its)
                                cell.done = true;
                        } else if (rtz_math::fabs(cell.total_time - requested_delta_time) /
                                       requested_delta_time <
                                   Real(1e-6)) {
                            cell.done = true;
                        }
                    });
                });

            Kokkos::parallel_reduce(
                "rtz_python_pending",
                Kokkos::RangePolicy<>(0, active_count),
                KOKKOS_LAMBDA(const std::size_t active_index, std::size_t &count) {
                    const std::size_t cell_index = active_cells(active_index);
                    if (!device_cells(cell_index).done)
                        ++count;
                },
                pending);
            if (pending > 0) {
                int over_limit = 0;
                Kokkos::parallel_reduce(
                    "rtz_python_iteration_limit",
                    Kokkos::RangePolicy<>(0, active_count),
                    KOKKOS_LAMBDA(const std::size_t active_index, int &count) {
                        const std::size_t cell_index = active_cells(active_index);
                        const CellState &cell = device_cells(cell_index);
                        if (!cell.done && cell.iterations > max_iterations)
                            ++count;
                    },
                    over_limit);
                if (over_limit > 0) {
                    Kokkos::deep_copy(host_cells, device_cells);
                    std::ostringstream message;
                    message << "RTZ exceeded max_iterations before reaching a requested time";
                    for (std::size_t index = 0; index < n_cells; ++index) {
                        const auto &cell = host_cells(index);
                        if (cell.done || cell.iterations <= max_iterations)
                            continue;
                        message << std::scientific << "; cell=" << index
                                << " elapsed=" << cell.total_time << " dt=" << cell.ddt
                                << " T=" << cell.subcycle.TK
                                << " ne=" << cell.subcycle.ne
                                << " residual=" << cell.subcycle.max_residual_all
                                << " accepted=" << cell.subcycle.x_percent_rule;
                        break;
                    }
                    throw std::runtime_error(message.str());
                }
            }
            if (pending > 0 && pending < active_count) {
                // Stable scan preserves cell order and never moves solver state.
                // No host result: the next kernel consumes the indices in stream order.
                Kokkos::parallel_scan(
                    "rtz_python_compact_active",
                    Kokkos::RangePolicy<>(0, active_count),
                    KOKKOS_LAMBDA(const std::size_t active_index,
                                  std::size_t &offset, const bool final) {
                        const std::size_t cell_index = active_cells(active_index);
                        if (!device_cells(cell_index).done) {
                            if (final)
                                next_active_cells(offset) = cell_index;
                            ++offset;
                        }
                    });
                std::swap(active_cells, next_active_cells);
            }
            active_count = pending;
        }

        Kokkos::parallel_for(
            "rtz_python_interval_finish",
            Kokkos::RangePolicy<>(0, n_cells),
            KOKKOS_LAMBDA(const std::size_t cell_index) {
                CellState &cell = device_cells(cell_index);
                cell.initial_interval = false;
                const Real ne = get_ne(device_elements, cell.chemistry, false);
                cell.out_mu =
                    get_mu(ne, device_elements, cell.chemistry, false, include_H2, cell.nCO);
                cell.out_Tmu = cell.subcycle.Tmu_old;
                remove_dust_depletion(device_elements, cell.dust_ratio, cell.chemistry, cell.nCO);
                if constexpr (!constant_temperature)
                    cell.temperature = cell.out_Tmu * cell.out_mu;
            });
        Kokkos::deep_copy(host_cells, device_cells);

        for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index) {
            cells[cell_index] = host_cells(cell_index);
            double *output_row =
                output_data.get() + (time_index * n_cells + cell_index) * n_columns;
            copy_state_to_matrix(
                cells[cell_index], output_row, inputs, density_offset, ion_offset, photon_offset);
        }
        previous_time = time;
    }

    nb::capsule owner(output_data.get(),
                      [](void *pointer) noexcept { delete[] static_cast<double *>(pointer); });
    double *result_data = output_data.release();
    nb::ndarray<nb::numpy, double, nb::ndim<3>> result(
        result_data, {times.size(), n_cells, n_columns}, owner);
    return nb::cast(result);
}

template <bool constant_temperature,
          bool include_H2,
          bool include_CO,
          bool rt_advect,
          bool include_self_shielding>
nb::object solve_impl(nb::ndarray<nb::numpy, const double, nb::ndim<2>> input,
                      nb::ndarray<nb::numpy, const double, nb::ndim<1>> times_arg,
                      nb::sequence element_list,
                      nb::sequence ion_list,
                      nb::sequence photon_bands,
                      const SolveOptions &options) {
    const SolveParameters parameters =
        validate_parameters<include_CO>(input, times_arg, element_list, ion_list, photon_bands, options);
    ensure_cell_runtime();
    std::vector<CellState> cells = validate_state_values(input, parameters, include_H2);
    return run_solver<constant_temperature,
                      include_H2,
                      include_CO,
                      rt_advect,
                      include_self_shielding>(parameters, std::move(cells), options);
}

template <bool constant_temperature, bool include_H2, bool include_CO, bool rt_advect>
nb::object dispatch_self_shielding(nb::ndarray<nb::numpy, const double, nb::ndim<2>> state,
                                   nb::ndarray<nb::numpy, const double, nb::ndim<1>> times,
                                   nb::sequence elements,
                                   nb::sequence ions,
                                   nb::sequence photon_bands,
                                   const SolveOptions &options,
                                   bool include_self_shielding) {
    if (include_self_shielding)
        return solve_impl<constant_temperature, include_H2, include_CO, rt_advect, true>(
            state, times, elements, ions, photon_bands, options);
    return solve_impl<constant_temperature, include_H2, include_CO, rt_advect, false>(
        state, times, elements, ions, photon_bands, options);
}

template <bool constant_temperature, bool include_H2, bool rt_advect>
nb::object dispatch_CO(nb::ndarray<nb::numpy, const double, nb::ndim<2>> state,
                       nb::ndarray<nb::numpy, const double, nb::ndim<1>> times,
                       nb::sequence elements,
                       nb::sequence ions,
                       nb::sequence photon_bands,
                       const SolveOptions &options,
                       bool include_CO,
                       bool include_self_shielding) {
    if (include_CO)
        return dispatch_self_shielding<constant_temperature, include_H2, true, rt_advect>(
            state, times, elements, ions, photon_bands, options, include_self_shielding);
    return dispatch_self_shielding<constant_temperature, include_H2, false, rt_advect>(
        state, times, elements, ions, photon_bands, options, include_self_shielding);
}

template <bool constant_temperature, bool rt_advect>
nb::object dispatch_H2(nb::ndarray<nb::numpy, const double, nb::ndim<2>> state,
                       nb::ndarray<nb::numpy, const double, nb::ndim<1>> times,
                       nb::sequence elements,
                       nb::sequence ions,
                       nb::sequence photon_bands,
                       const SolveOptions &options,
                       bool include_H2,
                       bool include_CO,
                       bool include_self_shielding) {
    if (include_H2)
        return dispatch_CO<constant_temperature, true, rt_advect>(state,
                                                                  times,
                                                                  elements,
                                                                  ions,
                                                                  photon_bands,
                                                                  options,
                                                                  include_CO,
                                                                  include_self_shielding);
    return dispatch_CO<constant_temperature, false, rt_advect>(
        state, times, elements, ions, photon_bands, options, include_CO, include_self_shielding);
}

template <bool constant_temperature>
nb::object dispatch_rt_advection(nb::ndarray<nb::numpy, const double, nb::ndim<2>> state,
                                 nb::ndarray<nb::numpy, const double, nb::ndim<1>> times,
                                 nb::sequence elements,
                                 nb::sequence ions,
                                 nb::sequence photon_bands,
                                 const SolveOptions &options,
                                 bool include_H2,
                                 bool include_CO,
                                 bool include_self_shielding,
                                 bool rt_advect) {
    if (rt_advect)
        return dispatch_H2<constant_temperature, true>(state,
                                                       times,
                                                       elements,
                                                       ions,
                                                       photon_bands,
                                                       options,
                                                       include_H2,
                                                       include_CO,
                                                       include_self_shielding);
    return dispatch_H2<constant_temperature, false>(state,
                                                    times,
                                                    elements,
                                                    ions,
                                                    photon_bands,
                                                    options,
                                                    include_H2,
                                                    include_CO,
                                                    include_self_shielding);
}

nb::object solve(nb::ndarray<nb::numpy, const double, nb::ndim<2>> state,
                 nb::ndarray<nb::numpy, const double, nb::ndim<1>> times,
                 nb::sequence elements,
                 nb::sequence ions,
                 nb::sequence photon_bands,
                 nb::ndarray<nb::numpy, const double, nb::ndim<1>> sed_frequency_arg,
                 nb::ndarray<nb::numpy, const double, nb::ndim<1>> sed_fnu_arg,
                 const std::string &data_path,
                 int max_iterations,
                 double redshift,
                 double uv_g0,
                 bool constant_temperature,
                 bool include_H2,
                 bool include_CO,
                 bool include_self_shielding,
                 bool fixed_radiation,
                 bool rt_advect,
                 bool equilibrium,
                 int conv_its,
                 double T_blackbody,
                 bool temperature_is_Tmu,
                 bool use_default_depletion,
                 bool include_collisional_ionization,
                 bool include_photoionization,
                 bool include_cosmic_ray_ionization,
                 bool include_HM12_UVB,
                 bool include_dust_recombination,
                 bool include_charge_exchange) {
    PhysicsFlags physics_flags;
    physics_flags.include_collisional_ionization = include_collisional_ionization;
    physics_flags.include_photoionization = include_photoionization;
    physics_flags.include_cosmic_ray_ionization = include_cosmic_ray_ionization;
    physics_flags.include_HM12_UVB = include_HM12_UVB;
    physics_flags.include_dust_recombination = include_dust_recombination;
    physics_flags.include_charge_exchange = include_charge_exchange;
    const SolveOptions options{data_path,
                               max_iterations,
                               conv_its,
                               redshift,
                               uv_g0,
                               T_blackbody,
                               copy_1d_array(sed_frequency_arg),
                               copy_1d_array(sed_fnu_arg),
                               fixed_radiation,
                               equilibrium,
                               temperature_is_Tmu,
                               use_default_depletion,
                               physics_flags};
    if (constant_temperature)
        return dispatch_rt_advection<true>(state,
                                           times,
                                           elements,
                                           ions,
                                           photon_bands,
                                           options,
                                           include_H2,
                                           include_CO,
                                           include_self_shielding,
                                           rt_advect);
    return dispatch_rt_advection<false>(state,
                                        times,
                                        elements,
                                        ions,
                                        photon_bands,
                                        options,
                                        include_H2,
                                        include_CO,
                                        include_self_shielding,
                                        rt_advect);
}

} // namespace

NB_MODULE(_rtz, module) {
    module.def("build_info", []() {
        nb::dict info;
        info["execution_space"] = Kokkos::DefaultExecutionSpace::name();
        info["cooling_bytes"] = sizeof(prism_gpu::CoolingReal);
        info["subcycle_state_bytes"] = sizeof(prism_gpu::SubcycleReal);
        info["constants_bytes"] = sizeof(prism_gpu::ConstantsReal);
        info["cross_section_bytes"] = sizeof(prism_gpu::CrossSectionReal);
        info["uvb_bytes"] = sizeof(prism_gpu::UVBReal);
        info["molecular_bytes"] = sizeof(prism_gpu::MolecularReal);
        info["dust_recombination_bytes"] = sizeof(prism_gpu::DustRecombinationReal);
        info["cosmic_ray_bytes"] = sizeof(prism_gpu::CosmicRayReal);
        info["recombination_bytes"] = sizeof(prism_gpu::RecombinationReal);
        info["charge_transfer_bytes"] = sizeof(prism_gpu::ChargeTransferReal);
        info["collisional_ionization_bytes"] = sizeof(prism_gpu::CollisionalReal);
        info["real_bytes"] = sizeof(prism_gpu::Real);
        info["density_bytes"] = sizeof(prism_gpu::CompactIonData::ndens_t);
        info["ion_fraction_bytes"] = sizeof(prism_gpu::CompactIonData::xion_t);
        info["cell_bytes"] = sizeof(CellState);
        info["table_descriptor_bytes"] = sizeof(prism_gpu::CompactTabulatedData);
        return info;
    });
    module.doc() = "Kokkos-backed Python bindings for the RTZ chemistry and cooling solver";
    module.def("_finalize", &finalize_runtime);
    module.def("solve",
               &solve,
               nb::arg("state"),
               nb::arg("times"),
               nb::arg("element_list"),
               nb::arg("ion_list"),
               nb::arg("photon_bands"),
               nb::arg("sed_frequency"),
               nb::arg("sed_fnu"),
               nb::arg("data_path") = "",
               nb::arg("max_iterations") = 100000,
               nb::arg("redshift") = 0.0,
               nb::arg("uv_g0") = 0.0070977,
               nb::arg("constant_temperature") = false,
               nb::arg("include_H2") = false,
               nb::arg("include_CO") = false,
               nb::arg("include_self_shielding") = false,
               nb::arg("fixed_radiation") = false,
               nb::arg("rt_advect") = true,
               nb::arg("equilibrium") = false,
               nb::arg("conv_its") = 20000,
               nb::arg("T_blackbody") = 1.0e4,
               nb::arg("temperature_is_Tmu") = false,
               nb::arg("use_default_depletion") = true,
               nb::arg("include_collisional_ionization") = true,
               nb::arg("include_photoionization") = true,
               nb::arg("include_cosmic_ray_ionization") = true,
               nb::arg("include_HM12_UVB") = true,
               nb::arg("include_dust_recombination") = true,
               nb::arg("include_charge_exchange") = true,
               "Evolve each state row from t=0 through the requested 1D NumPy time array.");
}
