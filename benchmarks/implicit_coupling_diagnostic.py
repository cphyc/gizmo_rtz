"""Compare legacy and implicit integration at shared physical output intervals."""
import argparse
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
parser.add_argument("--intervals", default="1,100,1000,10000,100000")
parser.add_argument("--indices", default="117,5915,1819,44")
parser.add_argument("--constant-temperature", action="store_true")
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("rtz._rtz", args.extension.resolve())
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
sys.path.insert(0, str(ROOT / "python"))
os.environ["RTZ_DATA_DIR"] = str(ROOT / "data")
import rtz

indices = [int(i) for i in args.indices.split(",")]
with np.load(ROOT / "recombination_lag/galaxy_cells.npz") as data:
    meta = json.loads(str(data["metadata"]))
    state = {k: data[k][indices].copy() for k in
             ("temperature", "densities", "ion_fractions", "photon_densities")}
    sed = (data["sed_frequency"] * u.Hz, data["sed_fnu"] * u.erg / (u.s * u.cm**2 * u.Hz))
with np.load(ROOT / "benchmarks/results/metal-all-cache-final.npz") as data:
    coarse = {k: data[k][-1, indices] for k in state}

def errors(actual, reference):
    mask = reference["ion_fractions"] > 1e-4
    return dict(max_ion_relative_error=float(np.max(np.abs(
        actual["ion_fractions"][mask] / reference["ion_fractions"][mask] - 1))),
        max_temperature_relative_error=float(np.max(np.abs(
            actual["temperature"] / reference["temperature"] - 1))))

report = dict(command=sys.argv, cell_indices=indices, constant_temperature=args.constant_temperature,
              build_info=module.build_info(), trials=[])
for count in [int(n) for n in args.intervals.split(",")]:
    reference = None
    for implicit in [False, True]:
        start = perf_counter()
        history = rtz.solve(state, np.linspace(100*365.25*86400/count, 100*365.25*86400, count),
                            meta["elements"], [tuple(x) for x in meta["ions"]],
                            [tuple(x) for x in meta["bands"]], sed=sed, **meta["physics"],
                            rt_advect=True, implicit_ions=implicit, ion_change_limit=.2,
                            constant_temperature=args.constant_temperature)
        final = {k: np.array(v[-1], copy=True) for k, v in history.items()}
        del history
        if reference is None:
            reference = final
        row = dict(intervals=count, implicit=implicit, seconds=perf_counter()-start,
                   **errors(final, reference))
        if not args.constant_temperature:
            row["error_vs_original_coarse_legacy"] = errors(final, coarse)
        report["trials"].append(row)
        tag = "implicit" if implicit else "legacy"
        np.savez_compressed(args.output.with_name(args.output.name + f"-{count}-{tag}.npz"), **final)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(row, flush=True)
