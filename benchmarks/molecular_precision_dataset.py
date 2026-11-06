"""Select a T/nH grid from the galaxy input and enable molecular physics."""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=Path("recombination_lag/galaxy_cells.npz"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
with np.load(args.input, allow_pickle=False) as data:
    meta = json.loads(str(data["metadata"]))
    h = meta["elements"].index("hydrogen")
    log_t = np.log10(data["temperature"])
    log_n = np.log10(data["densities"][:, h])
    selected = np.unique([
        np.argmin((log_t - t) ** 2 + (log_n - n) ** 2)
        for t in range(1, 8) for n in (-4, -2, 0, 1, 2, 3)
    ])
    result = {
        key: (data[key][selected] if data[key].ndim and
              data[key].shape[0] == len(log_t) else data[key].copy())
        for key in data.files if key != "metadata"
    }
meta["physics"].update(include_H2=True, include_CO=True, include_self_shielding=True)
meta["precision_probe"] = {"source": str(args.input), "indices": selected.tolist()}
result["metadata"] = np.array(json.dumps(meta, sort_keys=True))
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.output, **result)
print(f"Selected {len(selected)} cells: {args.output}")
