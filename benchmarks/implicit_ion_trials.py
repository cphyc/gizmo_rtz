"""Sweep implicit ion limits against the legacy output and record iteration tails."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import unyt as u

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--extension", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reference", type=Path, default=ROOT / "benchmarks/results/baseline.npz")
parser.add_argument("--ncell", type=int, default=10000)
parser.add_argument("--limits", default="0.5,0.25,0.15,0.125,0.1,0.075,0.05")
parser.add_argument("--repetitions", type=int, default=3)
parser.add_argument("--warmups", type=int, default=1)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("rtz._rtz", args.extension.resolve())
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
sys.path.insert(0, str(ROOT / "python"))
os.environ["RTZ_DATA_DIR"] = str(ROOT / "data")
import rtz

dataset = ROOT / "recombination_lag/galaxy_cells.npz"
with np.load(dataset) as data:
    metadata = json.loads(str(data["metadata"]))
    fields = ("temperature", "densities", "ion_fractions", "photon_densities")
    state = {key: data[key][:args.ncell].copy() for key in fields}
    sed = (data["sed_frequency"] * u.Hz,
           data["sed_fnu"] * u.erg / (u.s * u.cm**2 * u.Hz))
with np.load(args.reference) as data:
    reference = {key: data[key][:, :args.ncell] for key in fields}
report = dict(command=sys.argv, build_info=module.build_info(), ncell=args.ncell,
              extension_sha256=hashlib.sha256(args.extension.read_bytes()).hexdigest(),
              dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(), trials=[])
assert all(value == 8 for key, value in report["build_info"].items()
           if key.endswith("_bytes") and key not in ("cell_bytes", "table_descriptor_bytes"))
args.output.parent.mkdir(parents=True, exist_ok=True)

def summary(values):
    return dict(total=int(np.sum(values)), maximum=int(np.max(values)),
                p50=float(np.quantile(values, .5)), p90=float(np.quantile(values, .9)),
                p99=float(np.quantile(values, .99)))

for limit in [None, *[float(x) for x in args.limits.split(",")]]:
    row = dict(implicit_ions=limit is not None, ion_change_limit=limit or .1)
    timings = []
    try:
        for iteration in range(args.warmups + args.repetitions):
            start = perf_counter()
            history = rtz.solve(state, np.array([100 * 365.25 * 86400]),
                                metadata["elements"], [tuple(x) for x in metadata["ions"]],
                                [tuple(x) for x in metadata["bands"]], sed=sed,
                                **metadata["physics"], rt_advect=True,
                                implicit_ions=limit is not None, ion_change_limit=limit or .5)
            elapsed = perf_counter() - start
            if iteration >= args.warmups:
                timings.append(elapsed)
        arrays = {key: np.asarray(value) for key, value in history.items()}
        assert all(np.isfinite(value).all() for value in arrays.values())
        mask = reference["ion_fractions"] > 1e-4
        errors = np.abs(arrays["ion_fractions"][mask] / reference["ion_fractions"][mask] - 1)
        temperature_errors = np.abs(arrays["temperature"] / reference["temperature"] - 1)
        stats = module.last_solve_statistics()
        row.update(seconds=timings, median_seconds=float(np.median(timings)),
                   max_ion_relative_error=float(errors.max(initial=0)),
                   p99_ion_relative_error=float(np.quantile(errors, .99)),
                   ion_violations=int(np.count_nonzero(errors >= 1e-5)),
                   max_temperature_relative_error=float(temperature_errors.max()),
                   temperature_violations=int(np.count_nonzero(temperature_errors >= 1e-5)),
                   iteration_summary={key: summary(value) for key, value in stats.items()},
                   per_cell_statistics=stats, completed=True)
        row["accuracy_passed"] = row["ion_violations"] == row["temperature_violations"] == 0
        tag = "legacy" if limit is None else str(limit).replace(".", "p")
        np.savez_compressed(args.output.with_name(args.output.name + "-" + tag + ".npz"), **arrays)
    except RuntimeError as error:
        row.update(completed=False, error=str(error))
    report["trials"].append(row)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in row.items() if key != "per_cell_statistics"}), flush=True)
