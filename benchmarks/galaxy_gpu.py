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
    parser.add_argument("--ncell", type=int)
    parser.add_argument("--dt-yr", type=float, default=100.)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    extension = args.extension.resolve()
    spec = importlib.util.spec_from_file_location("rtz._rtz", extension)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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
    for i in range(args.repetitions + 1):
        start = perf_counter()
        history = rtz.solve(state, np.array([args.dt_yr * 365.25 * 86400]),
                            meta["elements"], [tuple(v) for v in meta["ions"]],
                            [tuple(v) for v in meta["bands"]], sed=sed,
                            **meta["physics"], rt_advect=True, constant_temperature=False)
        duration = perf_counter() - start
        print(f"{'warmup' if i == 0 else i}: {duration:.6f} s", flush=True)
        if i:
            timings.append(duration)
    arrays = {key: np.asarray(value) for key, value in history.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("nonfinite solver output")
    result = dict(seconds=timings, median_seconds=float(np.median(timings)),
                  ncell=len(state["temperature"]), dt_yr=args.dt_yr,
                  extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
                  dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                  git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw", "--format=csv"], text=True).strip(),
                  command=sys.argv)
    if args.reference:
        with np.load(args.reference, allow_pickle=False) as reference:
            result["max_absolute_difference"] = {
                key: float(np.max(np.abs(arrays[key] - reference[key]))) for key in arrays}
            for key in arrays:
                np.testing.assert_allclose(arrays[key], reference[key], rtol=1e-10, atol=1e-12,
                                           err_msg=key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
