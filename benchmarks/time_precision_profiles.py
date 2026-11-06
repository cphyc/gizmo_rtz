"""Re-time the saved, accepted per-file GPU profiles under the strict gate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/galaxy-gpu"
PROFILES = {
    "float64": "ct-cache-float64",
    "charge": "per-file-charge-float32",
    "cosmic": "per-file-cosmic-float32",
    "dust": "per-file-dust-float32",
    "molecules": "per-file-molecules-float32",
    "uvb": "per-file-uvb-float32",
    "selected": "per-file-selected",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", choices=PROFILES)
args = parser.parse_args()
profiles = {args.profile: PROFILES[args.profile]} if args.profile else PROFILES
paths = {name: BUILD / f"{stem}.cpython-314-x86_64-linux-gnu.so"
         for name, stem in profiles.items()}
for path in paths.values():
    if not path.exists():
        raise FileNotFoundError(f"Saved trial binary required: {path}")
for name, path in paths.items():
    output = ROOT / f"benchmarks/results/precision-timing-{name}"
    command = [
        sys.executable, str(ROOT / "benchmarks/galaxy_gpu.py"),
        "--extension", str(path), "--output", str(output),
        "--reference", str(ROOT / "benchmarks/results/baseline.npz"),
        "--accuracy", "physical", "--ion-rtol", "1e-5",
        "--temperature-rtol", "1e-5", "--warmups", "1", "--repetitions", "5",
        "--require-float64" if name == "float64" else "--require-float64-state",
    ]
    print(f"Timing {name} ...", flush=True)
    with (BUILD / f"precision-timing-{name}.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(output.with_suffix(".json").read_text())
    print(f"{name}: {result['median_seconds']:.6f} s; "
          f"max ion rel {result['accuracy']['max_ion_relative_error']:.3g}; "
          f"max T rel {result['accuracy']['max_temperature_relative_error']:.3g}", flush=True)
