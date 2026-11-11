from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, List

import numpy as np
import numpy.typing as npt

class TSPInstance:
    def __init__(
        self,
        positions: npt.NDArray[np.float_],
        distance_matrix_opt: npt.NDArray[np.float_] | None = None,
        reference_cost: float | None = None,
    ) -> None:
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError(f"Invalid coordinate shape: {pos.shape}")
        self.positions = pos
        self.n = int(pos.shape[0])
        if distance_matrix_opt is None:
            dist = compute_distance_matrix(pos)
        else:
            dist = np.asarray(distance_matrix_opt, dtype=np.float64)
            if dist.shape != (self.n, self.n):
                dist = compute_distance_matrix(pos)
        # Keep tiny diagonal epsilon for numeric stability.
        self.distmat = dist + np.eye(self.n, dtype=np.float64) * 1e-5
        if reference_cost is None:
            self.reference_cost = None
        else:
            try:
                ref = float(reference_cost)
            except Exception:
                ref = float("nan")
            self.reference_cost = (ref if np.isfinite(ref) and ref > 0.0 else None)

dataset_conf = {
    'train': (200,),
    'val':   (20, 50, 100, 200),
    'test':  (20, 50, 100, 200, 500, 1000),
}


def compute_distance_matrix(positions: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Invalid coordinate shape: {arr.shape}")
    diff = arr[:, None, :] - arr[None, :, :]
    return np.linalg.norm(diff, axis=2).astype(np.float64)


def generate_dataset(filepath, n, batch_size=64):
    positions = np.random.random((batch_size, n, 2))
    np.save(filepath, positions)

def generate_datasets(basepath = None):
    import os
    basepath = basepath or os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(basepath, exist_ok=True)

    for mood, problem_sizes in dataset_conf.items():
        np.random.seed(len(mood))
        for n in problem_sizes:
            filepath = os.path.join(basepath, f"{mood}{n}_dataset.npy")
            generate_dataset(filepath, n, batch_size=10 if mood =='train' else 64)


def _to_row_list(value: Any, *, field_name: str) -> List[Any]:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            raise ValueError(f"Dataset field '{field_name}' has invalid ndarray scalar shape.")
        return [value[i] for i in range(int(value.shape[0]))]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(f"Dataset field '{field_name}' must be list/tuple/ndarray.")


def _normalize_row(row: Any) -> TSPInstance:
    if isinstance(row, dict):
        coords = row.get("coordinate", row.get("coords"))
        if coords is None:
            raise ValueError("Row dict has no 'coordinate' field.")
        dist = row.get("distance_matrix", row.get("dist"))
        ref = row.get("reference_cost", None)
        return TSPInstance(
            np.asarray(coords, dtype=np.float64),
            None if dist is None else np.asarray(dist, dtype=np.float64),
            reference_cost=ref,
        )

    if isinstance(row, (list, tuple)) and len(row) >= 1:
        coords = row[0]
        dist = row[1] if len(row) >= 2 else None
        ref = row[2] if len(row) >= 3 else None
        return TSPInstance(
            np.asarray(coords, dtype=np.float64),
            None if dist is None else np.asarray(dist, dtype=np.float64),
            reference_cost=ref,
        )

    raise ValueError(f"Unsupported row format: {type(row)}")


def _load_from_rows(rows: List[Any]) -> list[TSPInstance]:
    out: list[TSPInstance] = []
    for row in rows:
        out.append(_normalize_row(row))
    if not out:
        raise ValueError("No instances loaded from row list.")
    return out


def _load_dataset_npy(fp: Path) -> list[TSPInstance]:
    data = np.load(fp, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[-1] == 2:
        return [TSPInstance(data[i]) for i in range(int(data.shape[0]))]
    if isinstance(data, np.ndarray) and data.dtype == object:
        return _load_from_rows(list(data))
    raise ValueError(
        f"Unsupported .npy dataset format at {fp}. "
        "Expected shape (N,n,2) coordinates or object-array rows."
    )


def _load_dataset_pkl(fp: Path) -> list[TSPInstance]:
    with fp.open("rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict) and "coordinate" in raw:
        coords_all = _to_row_list(raw.get("coordinate"), field_name="coordinate")
        n_rows = len(coords_all)
        dist_raw = raw.get("distance_matrix", None)
        if dist_raw is None:
            dist_all = [None] * n_rows
        else:
            try:
                dist_all = _to_row_list(dist_raw, field_name="distance_matrix")
            except Exception:
                dist_all = [None] * n_rows
            if len(dist_all) != n_rows:
                dist_all = [None] * n_rows

        reference_costs: list[float | None] = [None] * n_rows

        # 1) Prefer dataset-embedded Concorde references: `cost_concorde`.
        cost_concorde_raw = raw.get("cost_concorde", None)
        if cost_concorde_raw is not None:
            vals = np.asarray(cost_concorde_raw, dtype=np.float64)
            if vals.ndim < 1 or int(vals.shape[0]) != n_rows:
                raise ValueError(
                    f"`cost_concorde` shape mismatch in {fp}: expected {n_rows}, got {vals.shape}"
                )
            reference_costs = [
                (float(x) if np.isfinite(float(x)) and float(x) > 0.0 else None)
                for x in vals.tolist()
            ]

        # 2) Otherwise, load sidecar produced by data_generate/compute_concorde_opt.py.
        if not any(v is not None for v in reference_costs):
            repo_root = Path(__file__).resolve().parents[2]
            sidecar_candidates: List[Path] = [
                fp.with_name(f"{fp.stem}_concorde_full.json"),
                fp.parent / "concorde" / f"{fp.stem}_concorde_full.json",
                repo_root / "outputs" / "concorde_opt" / f"{fp.stem}_concorde_full.json",  # legacy
            ]
            sidecar: Path | None = None
            for cand in sidecar_candidates:
                if cand.exists():
                    sidecar = cand
                    break
            if sidecar is None:
                raise ValueError(
                    "Concorde reference costs are required but missing. "
                    "Expected `cost_concorde` in dataset or sidecar at one of: "
                    f"`{fp.with_name(fp.stem + '_concorde_full.json')}`, "
                    f"`{fp.parent / 'concorde' / (fp.stem + '_concorde_full.json')}`, "
                    f"or legacy `{repo_root / 'outputs' / 'concorde_opt' / (fp.stem + '_concorde_full.json')}`"
                )

            obj = json.loads(sidecar.read_text(encoding="utf-8"))
            recs = obj.get("records", None)
            if not isinstance(recs, list) or len(recs) < n_rows:
                raise ValueError(
                    f"Invalid Concorde sidecar `{sidecar}`: records length < {n_rows}"
                )

            parsed: list[float | None] = []
            for i in range(n_rows):
                r = recs[i]
                if not isinstance(r, dict) or str(r.get("status", "")) != "ok":
                    parsed.append(None)
                    continue
                try:
                    vf = float(r.get("opt_float_from_tour", None))
                    parsed.append(vf if np.isfinite(vf) and vf > 0.0 else None)
                except Exception:
                    parsed.append(None)
            reference_costs = parsed

        missing = [i for i, v in enumerate(reference_costs) if v is None]
        if missing:
            preview = ",".join(str(i) for i in missing[:10])
            raise ValueError(
                "Concorde reference costs are incomplete. "
                f"Missing {len(missing)}/{n_rows} items (first indices: {preview})."
            )

        rows = [
            {
                "coordinate": coords_all[i],
                "distance_matrix": dist_all[i],
                "reference_cost": reference_costs[i] if i < len(reference_costs) else None,
            }
            for i in range(n_rows)
        ]
        return _load_from_rows(rows)

    if isinstance(raw, (list, tuple)):
        return _load_from_rows(list(raw))

    raise ValueError(
        f"Unsupported .pkl dataset format at {fp}. "
        "Expected dict with 'coordinate' or list/tuple rows."
    )


def load_dataset(fp) -> list[TSPInstance]:
    path = Path(fp).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _load_dataset_npy(path)
    if suffix == ".pkl":
        return _load_dataset_pkl(path)
    raise ValueError(f"Unsupported dataset suffix: {path.suffix}. Use .npy or .pkl.")


if __name__ == "__main__":
    generate_datasets()
