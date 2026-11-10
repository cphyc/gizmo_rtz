"""Time identical independent galaxy solves with an explicitly selected extension."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import unyt as u

ROOT = Path(__file__).resolve().parents[1]

FULL_SYSTEM_ELEMENTS = (
    ("hydrogen", 1),
    ("helium", 2),
    ("carbon", 6),
    ("nitrogen", 7),
    ("oxygen", 8),
    ("magnesium", 12),
    ("sulfur", 16),
    ("silicon", 14),
    ("iron", 26),
    ("neon", 10),
)
SOLAR_ABUNDANCES = {
    "helium": 8.51e-2,
    "carbon": 2.69e-4,
    "nitrogen": 6.76e-5,
    "oxygen": 4.90e-4,
    "neon": 8.51e-5,
    "magnesium": 3.98e-5,
    "silicon": 3.24e-5,
    "sulfur": 1.32e-5,
    "iron": 3.16e-5,
}


def expand_to_full_system(state, metadata):
    """Add missing atomic species and enable H2/CO chemistry for the benchmark.

    Missing elements are neutral equilibrium seeds; H2 uses the unassigned
    hydrogen fraction and the binding initializes CO to zero.
    """
    source_elements = list(metadata["elements"])
    source_element_index = {name: index for index, name in enumerate(source_elements)}
    full_element_names = [name for name, _ in FULL_SYSTEM_ELEMENTS]
    unknown_elements = set(source_elements) - set(full_element_names)
    if unknown_elements:
        raise ValueError(
            "full-system mode cannot preserve input elements outside its network: "
            + ", ".join(sorted(unknown_elements))
        )
    if len(source_element_index) != len(source_elements):
        raise ValueError("dataset element metadata contains duplicates")
    if "hydrogen" not in source_element_index:
        raise ValueError("full-system mode requires hydrogen densities")

    source_densities = state["densities"]
    if source_densities.shape[1] != len(source_elements):
        raise ValueError("dataset density columns do not match its element metadata")
    hydrogen_density = source_densities[:, source_element_index["hydrogen"]]
    missing_metals = [
        element
        for element, _ in FULL_SYSTEM_ELEMENTS
        if element not in source_element_index and element not in ("hydrogen", "helium")
    ]
    metallicity = (metadata.get("equilibrium_table") or {}).get("metallicity")
    if missing_metals and (
        metallicity is None or not np.isfinite(metallicity) or metallicity <= 0
    ):
        raise ValueError(
            "full-system mode requires a positive metallicity in equilibrium_table metadata"
        )

    densities = np.empty(
        (len(hydrogen_density), len(FULL_SYSTEM_ELEMENTS)), dtype=source_densities.dtype
    )
    for column, (element, _) in enumerate(FULL_SYSTEM_ELEMENTS):
        if element in source_element_index:
            densities[:, column] = source_densities[:, source_element_index[element]]
        elif element == "helium":
            densities[:, column] = hydrogen_density * SOLAR_ABUNDANCES[element]
        else:
            # Match the solver's abundance table for elements absent from the source dataset.
            densities[:, column] = (
                hydrogen_density * metallicity * SOLAR_ABUNDANCES[element]
            )

    ions = [
        (element, stage)
        for element, atomic_number in FULL_SYSTEM_ELEMENTS
        for stage in range(1, atomic_number + 2)
    ]
    ion_index = {ion: index for index, ion in enumerate(ions)}
    source_ions = [(str(element), int(stage)) for element, stage in metadata["ions"]]
    source_fractions = state["ion_fractions"]
    if source_fractions.shape[1] != len(source_ions):
        raise ValueError("dataset ion-fraction columns do not match its ion metadata")
    if len(set(source_ions)) != len(source_ions):
        raise ValueError("dataset ion metadata contains duplicate stages")
    fractions = np.zeros((len(hydrogen_density), len(ions)), dtype=source_fractions.dtype)
    for source_column, ion in enumerate(source_ions):
        if ion not in ion_index:
            raise ValueError(f"dataset ion is outside the full-system network: {ion}")
        fractions[:, ion_index[ion]] = source_fractions[:, source_column]

    source_elements_set = set(source_elements)
    for element, _ in FULL_SYSTEM_ELEMENTS:
        if element not in source_elements_set:
            fractions[:, ion_index[(element, 1)]] = 1.0

    expanded_state = {**state, "densities": densities, "ion_fractions": fractions}
    expanded_metadata = {
        **metadata,
        "elements": full_element_names,
        "ions": ions,
        "physics": {**metadata["physics"], "include_H2": True, "include_CO": True},
    }
    return expanded_state, expanded_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "recombination_lag/galaxy_cells.npz")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path prefix; .npz and .json are appended",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--installed-package",
        action="store_true",
        help="Use the installed Python wrapper instead of the repository wrapper",
    )
    parser.add_argument("--ncell", type=int)
    parser.add_argument(
        "--full-system",
        action="store_true",
        help="Benchmark the full H/He/C/N/O/Mg/S/Si/Fe/Ne network from PIE equilibrium",
    )
    parser.add_argument(
        "--equilibrium-conv-its",
        type=int,
        default=20000,
        help="Convergence counter for full-system PIE initialization",
    )
    parser.add_argument("--dt-yr", type=float, default=100.0)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--accuracy", choices=("exact", "strict", "physical"), default="strict")
    parser.add_argument("--ion-rtol", type=float, default=0.05)
    parser.add_argument("--temperature-rtol", type=float, default=0.01)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--implicit-ions", action="store_true", help="Test the sparse backward-Euler ion solver")
    parser.add_argument("--x-pct-rule", type=float, default=0.2)
    parser.add_argument(
        "--require-float64", action="store_true", help="Require float64 state and all reported arithmetic types"
    )
    parser.add_argument(
        "--require-float64-state",
        action="store_true",
        help="Require float64 global Real and state; permit per-file float32 arithmetic",
    )
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("repetitions must be positive and warmups nonnegative")
    if args.equilibrium_conv_its < 1:
        parser.error("--equilibrium-conv-its must be positive")
    extension = args.extension.resolve()
    spec = importlib.util.spec_from_file_location("rtz._rtz", extension)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    build_info = module.build_info() if hasattr(module, "build_info") else None
    if (args.require_float64 or args.require_float64_state) and (
        build_info is None
        or any(build_info.get(key) != 8 for key in ("real_bytes", "density_bytes", "ion_fraction_bytes"))
        or build_info.get("subcycle_state_bytes", 8) != 8
    ):
        raise RuntimeError(f"float64 global Real/state required; loaded build reports {build_info}")
    if args.require_float64 and any(
        value != 8
        for key, value in build_info.items()
        if key.endswith("_bytes") and key not in ("cell_bytes", "table_descriptor_bytes")
    ):
        raise RuntimeError(f"fully float64 arithmetic required; loaded build reports {build_info}")
    print(f"Build: {build_info}", flush=True)
    if not args.installed_package:
        sys.path.insert(0, str(ROOT / "python"))
    os.environ["RTZ_DATA_DIR"] = str(ROOT / "data")
    import rtz

    solver_parameters = inspect.signature(rtz.solve).parameters
    supports_x_pct_rule = "x_pct_rule" in solver_parameters
    supports_implicit_ions = "implicit_ions" in solver_parameters
    supports_molecular_state = "molecular_state" in solver_parameters
    x_pct_rule_source = (
        "runtime_parameter"
        if supports_x_pct_rule
        else (
            "compile_time_patch"
            if os.environ.get("RTZ_COMMIT_SWEEP_X_PCT_PATCHED") == "1"
            else "revision_default"
        )
    )

    with np.load(args.dataset, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        state = {
            k: data[k][: args.ncell].copy() for k in ("temperature", "densities", "ion_fractions", "photon_densities")
        }
        sed = (data["sed_frequency"] * u.Hz, data["sed_fnu"] * u.erg / (u.s * u.cm**2 * u.Hz))
    if args.full_system:
        state, meta = expand_to_full_system(state, meta)
    physics = dict(meta["physics"])
    if args.full_system:
        physics["fixed_radiation"] = True
    equilibrium_seconds = None
    molecular_state = None
    molecular_state_roundtripped = False
    if args.full_system:
        start = perf_counter()
        equilibrium_options = dict(
            rt_advect=True,
            constant_temperature=True,
            equilibrium=True,
            conv_its=args.equilibrium_conv_its,
            max_iterations=args.max_iterations,
        )
        if supports_x_pct_rule:
            equilibrium_options["x_pct_rule"] = args.x_pct_rule
        equilibrium_history = rtz.solve(
            state,
            np.array([1.0]),
            meta["elements"],
            [tuple(v) for v in meta["ions"]],
            [tuple(v) for v in meta["bands"]],
            sed=sed,
            **physics,
            **equilibrium_options,
        )
        equilibrium_seconds = perf_counter() - start
        state = {
            key: equilibrium_history[key][0]
            for key in ("temperature", "densities", "ion_fractions", "photon_densities")
        }
        if supports_molecular_state and "molecular_state" in equilibrium_history:
            molecular_state = {
                key: value[0]
                for key, value in equilibrium_history["molecular_state"].items()
            }
            molecular_state_roundtripped = True
        print(f"PIE equilibrium initialization: {equilibrium_seconds:.6f} s", flush=True)
    timings = []
    for i in range(args.repetitions + args.warmups):
        start = perf_counter()
        solve_options = dict(
            rt_advect=True,
            constant_temperature=False,
            max_iterations=args.max_iterations,
        )
        if supports_implicit_ions:
            solve_options["implicit_ions"] = args.implicit_ions
        if supports_x_pct_rule:
            solve_options["x_pct_rule"] = args.x_pct_rule
        if molecular_state is not None and supports_molecular_state:
            solve_options["molecular_state"] = molecular_state
        history = rtz.solve(
            state,
            np.array([args.dt_yr * 365.25 * 86400]),
            meta["elements"],
            [tuple(v) for v in meta["ions"]],
            [tuple(v) for v in meta["bands"]],
            sed=sed,
            **physics,
            **solve_options,
        )
        duration = perf_counter() - start
        print(f"{'warmup' if i < args.warmups else i - args.warmups + 1}: {duration:.6f} s", flush=True)
        if i >= args.warmups:
            timings.append(duration)
    arrays = {
        key: np.asarray(history[key])
        for key in ("temperature", "densities", "ion_fractions", "photon_densities")
    }
    if "molecular_state" in history:
        arrays.update(
            {key: np.asarray(value) for key, value in history["molecular_state"].items()}
        )
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("nonfinite solver output")
    result = dict(
        seconds=timings,
        median_seconds=float(np.median(timings)),
        build_info=build_info,
        ncell=len(state["temperature"]),
        dt_yr=args.dt_yr,
        warmups=args.warmups,
        implicit_ions=args.implicit_ions,
        full_system=args.full_system,
        initialization="PIE equilibrium" if args.full_system else "dataset",
        molecular_state_roundtripped=molecular_state_roundtripped,
        equilibrium_seconds=equilibrium_seconds,
        equilibrium_conv_its=args.equilibrium_conv_its if args.full_system else None,
        atomic_ion_stages=len(meta["ions"]),
        molecular_species=[
            species
            for enabled, species in (
                (meta["physics"].get("include_H2", False), "H2"),
                (meta["physics"].get("include_CO", False), "CO"),
            )
            if enabled
        ],
        x_pct_rule=(
            args.x_pct_rule
            if supports_x_pct_rule or os.environ.get("RTZ_COMMIT_SWEEP_X_PCT_PATCHED") == "1"
            else None
        ),
        x_pct_rule_supported=supports_x_pct_rule,
        x_pct_rule_source=x_pct_rule_source,
        max_iterations=args.max_iterations,
        extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        gpu=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw", "--format=csv"],
            text=True,
        ).strip(),
        command=sys.argv,
    )
    if hasattr(module, "last_solve_statistics"):
        counts = module.last_solve_statistics()
        result["per_cell_statistics"] = counts
        result["iteration_summary"] = {
            key: dict(
                total=int(np.sum(value)),
                maximum=int(np.max(value)),
                p50=float(np.quantile(value, 0.5)),
                p90=float(np.quantile(value, 0.9)),
                p99=float(np.quantile(value, 0.99)),
            )
            for key, value in counts.items()
        }
    accuracy_passed = True
    if args.reference:
        with np.load(args.reference, allow_pickle=False) as reference:
            missing = set(arrays) - set(reference.files)
            if missing:
                raise ValueError(
                    "reference is missing output arrays: " + ", ".join(sorted(missing))
                )
            reference = {key: reference[key][:, : len(state["temperature"])] for key in arrays}
            for key, value in arrays.items():
                if reference[key].shape != value.shape:
                    raise ValueError(
                        f"reference {key} has shape {reference[key].shape}, "
                        f"but this run produced {value.shape}"
                    )
            result["nonfinite_values"] = {
                key: int(np.count_nonzero(~np.isfinite(value))) for key, value in arrays.items()
            }
            reference_finite = all(np.isfinite(reference[key]).all() for key in arrays)
            result["max_absolute_difference"] = {
                key: float(np.max(np.abs(arrays[key] - reference[key]))) for key in arrays
            }
            if args.accuracy == "exact":
                result["differing_values"] = {
                    key: int(np.count_nonzero(arrays[key].view(np.uint64) != reference[key].view(np.uint64)))
                    for key in arrays
                }
                accuracy_passed = not any(result["differing_values"].values())
            elif args.accuracy == "physical":
                ion_ref = reference["ion_fractions"]
                mask = ion_ref > 1e-4
                relative = np.abs(arrays["ion_fractions"][mask] / ion_ref[mask] - 1)
                temperature_relative = np.abs(arrays["temperature"] / reference["temperature"] - 1)
                result["accuracy"] = dict(
                    mode="physical",
                    ion_fraction_threshold=1e-4,
                    ion_relative_tolerance=args.ion_rtol,
                    temperature_relative_tolerance=args.temperature_rtol,
                    checked_ion_values=int(mask.sum()),
                    max_ion_relative_error=float(relative.max(initial=0)),
                    ion_violations=int(np.count_nonzero(relative >= args.ion_rtol)),
                    max_temperature_relative_error=float(temperature_relative.max()),
                    temperature_violations=int(np.count_nonzero(temperature_relative >= args.temperature_rtol)),
                    max_small_ion_absolute_error=float(
                        np.abs(arrays["ion_fractions"][~mask] - ion_ref[~mask]).max(initial=0)
                    ),
                )
                accuracy_passed = not (
                    result["accuracy"]["ion_violations"] or result["accuracy"]["temperature_violations"]
                )
                for key in ("densities", "photon_densities"):
                    accuracy_passed &= bool(np.allclose(arrays[key], reference[key], rtol=1e-6, atol=0))
                for key in ("h2_fraction", "co_density"):
                    if key in arrays:
                        accuracy_passed &= bool(
                            np.allclose(
                                arrays[key],
                                reference[key],
                                rtol=args.ion_rtol,
                                atol=1e-12 if key == "h2_fraction" else 1e-30,
                            )
                        )
            else:
                accuracy_passed = all(
                    np.allclose(arrays[key], reference[key], rtol=1e-10, atol=1e-12) for key in arrays
                )
            accuracy_passed &= reference_finite and not any(result["nonfinite_values"].values())
            result["accuracy_passed"] = bool(accuracy_passed)
    output = args.output
    if output.suffix in (".npz", ".json"):
        output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_name(output.name + ".npz"), **arrays)
    output.with_name(output.name + ".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not accuracy_passed:
        raise AssertionError("reference accuracy check failed; results saved for diagnosis")


if __name__ == "__main__":
    main()
