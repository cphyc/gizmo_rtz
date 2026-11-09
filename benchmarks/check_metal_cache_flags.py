"""Compare cached metal chemistry with a parent binary when physics is disabled."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import unyt as u

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--extension", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reference", type=Path)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("rtz._rtz", args.extension.resolve())
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert all(value == 8 for key, value in module.build_info().items()
           if key.endswith("_bytes") and key not in ("cell_bytes", "table_descriptor_bytes"))
sys.path.insert(0, str(ROOT / "python"))
os.environ["RTZ_DATA_DIR"] = str(ROOT / "data")
import rtz

with np.load(ROOT / "recombination_lag/galaxy_cells.npz") as data:
    metadata = json.loads(str(data["metadata"]))
    indices = np.linspace(0, len(data["temperature"]) - 1, 64, dtype=int)
    state = {key: data[key][indices].copy() for key in
             ("temperature", "densities", "ion_fractions", "photon_densities")}
    sed = (data["sed_frequency"] * u.Hz,
           data["sed_fnu"] * u.erg / (u.s * u.cm**2 * u.Hz))

flags = ["include_collisional_ionization", "include_HM12_UVB",
         "include_photoionization", "include_cosmic_ray_ionization",
         "include_dust_recombination", "include_charge_exchange"]
outputs = {}
checks = {}
reference = np.load(args.reference) if args.reference else None
try:
    for disabled in [[], *[[flag] for flag in flags], flags]:
        name = "all_on" if not disabled else "all_off" if len(disabled) > 1 else disabled[0] + "_off"
        physics = {**metadata["physics"], **{flag: False for flag in disabled}}
        result = rtz.solve(state, np.array([100 * 365.25 * 86400]),
                           metadata["elements"], [tuple(x) for x in metadata["ions"]],
                           [tuple(x) for x in metadata["bands"]], sed=sed,
                           **physics, rt_advect=True, constant_temperature=False)
        checks[name] = {}
        for field, values in result.items():
            key = name + "__" + field
            actual = np.asarray(values)
            assert np.isfinite(actual).all(), key
            outputs[key] = actual
            if reference is not None:
                expected = reference[key]
                mask = expected > 1e-4 if field == "ion_fractions" else np.ones(expected.shape, dtype=bool)
                relative = np.abs(actual[mask] - expected[mask]) / np.maximum(np.abs(expected[mask]), 1e-300)
                error = float(np.max(relative, initial=0))
                checks[name][field] = dict(max_relative_error=error,
                                          bit_identical=bool(np.array_equal(actual, expected)))
                assert error <= 1e-5, (key, error)
        print(name, "passed", flush=True)
finally:
    if reference is not None:
        reference.close()
np.savez_compressed(args.output.with_suffix(".npz"), **outputs)
args.output.with_suffix(".json").write_text(json.dumps(checks, indent=2) + "\n")
