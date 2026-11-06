"""Time identical independent galaxy solves with an explicitly selected extension."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import unyt as u

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "recombination_lag/galaxy_cells.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--installed-package", action="store_true", help="Use the installed Python wrapper instead of the repository wrapper")
    parser.add_argument("--ncell", type=int)
    parser.add_argument("--dt-yr", type=float, default=100.)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--accuracy", choices=("exact", "strict", "physical"), default="strict")
    parser.add_argument("--ion-rtol", type=float, default=.05)
    parser.add_argument("--temperature-rtol", type=float, default=.01)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--require-float64", action="store_true", help="Require float64 state and all reported arithmetic types")
    parser.add_argument("--require-float64-state", action="store_true", help="Require float64 global Real and state; permit per-file float32 arithmetic")
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("repetitions must be positive and warmups nonnegative")
    extension = args.extension.resolve()
    spec = importlib.util.spec_from_file_location("rtz._rtz", extension)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    build_info = module.build_info() if hasattr(module, "build_info") else None
    if (args.require_float64 or args.require_float64_state) and (build_info is None or any(
            build_info.get(key) != 8 for key in ("real_bytes", "density_bytes", "ion_fraction_bytes"))
            or build_info.get("subcycle_state_bytes", 8) != 8):
        raise RuntimeError(f"float64 global Real/state required; loaded build reports {build_info}")
    if args.require_float64 and any(
            value != 8 for key, value in build_info.items()
            if key.endswith("_bytes") and key not in ("cell_bytes", "table_descriptor_bytes")):
        raise RuntimeError(f"fully float64 arithmetic required; loaded build reports {build_info}")
    print(f"Build: {build_info}", flush=True)
    if not args.installed_package:
        sys.path.insert(0, str(ROOT / "python"))
    os.environ["RTZ_DATA_DIR"] = str(ROOT / "data")
    import rtz

    with np.load(args.dataset, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        state = {k: data[k][:args.ncell].copy() for k in
                 ("temperature", "densities", "ion_fractions", "photon_densities")}
        sed = (data["sed_frequency"] * u.Hz,
               data["sed_fnu"] * u.erg / (u.s * u.cm**2 * u.Hz))
    timings = []
    for i in range(args.repetitions + args.warmups):
        start = perf_counter()
        history = rtz.solve(state, np.array([args.dt_yr * 365.25 * 86400]),
                            meta["elements"], [tuple(v) for v in meta["ions"]],
                            [tuple(v) for v in meta["bands"]], sed=sed,
                            **meta["physics"], rt_advect=True, constant_temperature=False,
                            max_iterations=args.max_iterations)
        duration = perf_counter() - start
        print(f"{'warmup' if i < args.warmups else i - args.warmups + 1}: {duration:.6f} s", flush=True)
        if i >= args.warmups:
            timings.append(duration)
    arrays = {key: np.asarray(value) for key, value in history.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("nonfinite solver output")
    result = dict(seconds=timings, median_seconds=float(np.median(timings)),
                  build_info=build_info,
                  ncell=len(state["temperature"]), dt_yr=args.dt_yr,
                  warmups=args.warmups,
                  max_iterations=args.max_iterations,
                  extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
                  dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                  git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw", "--format=csv"], text=True).strip(),
                  command=sys.argv)
    accuracy_passed = True
    if args.reference:
        with np.load(args.reference, allow_pickle=False) as reference:
            result["nonfinite_values"] = {key: int(np.count_nonzero(~np.isfinite(value)))
                                          for key, value in arrays.items()}
            reference_finite = all(np.isfinite(reference[key]).all() for key in arrays)
            result["max_absolute_difference"] = {
                key: float(np.max(np.abs(arrays[key] - reference[key]))) for key in arrays}
            if args.accuracy == "exact":
                result["differing_values"] = {
                    key: int(np.count_nonzero(arrays[key].view(np.uint64) != reference[key].view(np.uint64)))
                    for key in arrays}
                accuracy_passed = not any(result["differing_values"].values())
            elif args.accuracy == "physical":
                ion_ref = reference["ion_fractions"]
                mask = ion_ref > 1e-4
                relative = np.abs(arrays["ion_fractions"][mask] / ion_ref[mask] - 1)
                temperature_relative = np.abs(arrays["temperature"] / reference["temperature"] - 1)
                result["accuracy"] = dict(
                    mode="physical", ion_fraction_threshold=1e-4,
                    ion_relative_tolerance=args.ion_rtol, temperature_relative_tolerance=args.temperature_rtol,
                    checked_ion_values=int(mask.sum()),
                    max_ion_relative_error=float(relative.max(initial=0)),
                    ion_violations=int(np.count_nonzero(relative >= args.ion_rtol)),
                    max_temperature_relative_error=float(temperature_relative.max()),
                    temperature_violations=int(np.count_nonzero(temperature_relative >= args.temperature_rtol)),
                    max_small_ion_absolute_error=float(np.abs(arrays["ion_fractions"][~mask] - ion_ref[~mask]).max(initial=0)))
                accuracy_passed = not (result["accuracy"]["ion_violations"] or
                                       result["accuracy"]["temperature_violations"])
                for key in ("densities", "photon_densities"):
                    accuracy_passed &= bool(np.allclose(arrays[key], reference[key], rtol=1e-6, atol=0))
            else:
                accuracy_passed = all(np.allclose(arrays[key], reference[key], rtol=1e-10, atol=1e-12)
                                      for key in arrays)
            accuracy_passed &= reference_finite and not any(result["nonfinite_values"].values())
            result["accuracy_passed"] = bool(accuracy_passed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not accuracy_passed:
        raise AssertionError("reference accuracy check failed; results saved for diagnosis")


if __name__ == "__main__":
    main()
