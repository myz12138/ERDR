from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


InstanceTuple = Tuple[np.ndarray, np.ndarray]


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float64)
    diff = arr[:, None, :] - arr[None, :, :]
    return np.linalg.norm(diff, axis=2).astype(np.float64)


def normalize_instance(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        coords = np.asarray(row.get("coordinate"), dtype=np.float64)
        dist = row.get("distance_matrix")
        if dist is None:
            dist_arr = compute_distance_matrix(coords)
        else:
            dist_arr = np.asarray(dist, dtype=np.float64)
            if dist_arr.shape != (coords.shape[0], coords.shape[0]):
                dist_arr = compute_distance_matrix(coords)
        out = dict(row)
        out["coordinate"] = coords
        out["distance_matrix"] = dist_arr
        return out

    if isinstance(row, (list, tuple)) and len(row) >= 1:
        coords = np.asarray(row[0], dtype=np.float64)
        if len(row) >= 2 and row[1] is not None:
            dist = np.asarray(row[1], dtype=np.float64)
            if dist.shape != (coords.shape[0], coords.shape[0]):
                dist = compute_distance_matrix(coords)
        else:
            dist = compute_distance_matrix(coords)
        return {"coordinate": coords, "distance_matrix": dist}

    raise ValueError("Unsupported instance row format.")


def _to_row_list(value: Any, *, field_name: str) -> List[Any]:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            raise ValueError(f"Dataset field '{field_name}' has invalid ndarray scalar shape.")
        return [value[i] for i in range(int(value.shape[0]))]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(f"Dataset field '{field_name}' must be list/tuple/ndarray.")


def _to_optional_costs(value: Any, *, expected_len: int) -> List[float] | None:
    if value is None:
        return None
    try:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                if expected_len == 1:
                    return [float(value)]
                return None
            if int(value.shape[0]) != int(expected_len):
                return None
            return [float(x) for x in value.tolist()]
        if isinstance(value, (list, tuple)) and len(value) == expected_len:
            return [float(x) for x in value]
    except Exception:
        return None
    return None


def _load_concorde_sidecar_costs(data_path: Path, expected_len: int) -> List[float] | None:
    """
    Read sidecar produced by data_generate/compute_concorde_opt.py.
    Accepted locations:
    1) <dataset_dir>/<dataset_stem>_concorde_full.json
    2) <dataset_dir>/concorde/<dataset_stem>_concorde_full.json
    """
    candidates: List[Path] = [
        data_path.with_name(f"{data_path.stem}_concorde_full.json"),
        data_path.parent / "concorde" / f"{data_path.stem}_concorde_full.json",
    ]
    seen: set[str] = set()
    for sidecar in candidates:
        k = str(sidecar.resolve()).lower()
        if k in seen:
            continue
        seen.add(k)
        if not sidecar.exists():
            continue
        try:
            obj = json.loads(sidecar.read_text(encoding="utf-8"))
            recs = obj.get("records", None)
            if not isinstance(recs, list) or len(recs) < expected_len:
                continue
            out: List[float] = []
            ok = True
            for i in range(expected_len):
                row = recs[i]
                if not isinstance(row, dict) or str(row.get("status", "")) != "ok":
                    ok = False
                    break
                val = float(row.get("opt_float_from_tour"))
                if (not np.isfinite(val)) or val <= 0.0:
                    ok = False
                    break
                out.append(float(val))
            if ok:
                return out
        except Exception:
            continue
    return None


def load_tsp_dataset(data_path: str | Path) -> Tuple[List[InstanceTuple], List[float] | None]:
    """
    Strict Concorde mode:
    - requires Concorde reference costs (cost_concorde or concorde sidecar json)
    - does NOT fallback to legacy `cost` or nearest-neighbor proxies
    """
    p = Path(data_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    with p.open("rb") as f:
        raw = pickle.load(f)

    instances: List[InstanceTuple] = []
    opt_costs: List[float] | None = None

    if isinstance(raw, dict) and "coordinate" in raw:
        coords_all = _to_row_list(raw.get("coordinate"), field_name="coordinate")
        n_rows = len(coords_all)

        dist_all_raw = raw.get("distance_matrix", None)
        if dist_all_raw is None:
            dist_all = [None] * n_rows
        else:
            try:
                dist_all = _to_row_list(dist_all_raw, field_name="distance_matrix")
            except Exception:
                dist_all = [None] * n_rows
            if len(dist_all) != n_rows:
                dist_all = [None] * n_rows

        # Strict priority: explicit concorde costs in dataset > concorde sidecar.
        opt_costs = _to_optional_costs(raw.get("cost_concorde"), expected_len=n_rows)
        if opt_costs is None:
            opt_costs = _load_concorde_sidecar_costs(p, expected_len=n_rows)
        if opt_costs is None:
            raise ValueError(
                "Concorde reference costs are required but missing. "
                "Provide `cost_concorde` in dataset or sidecar at one of: "
                f"`{p.with_name(p.stem + '_concorde_full.json')}`, "
                f"`{p.parent / 'concorde' / (p.stem + '_concorde_full.json')}`."
            )
        bad = [i for i, v in enumerate(opt_costs) if (not np.isfinite(float(v))) or float(v) <= 0.0]
        if bad:
            preview = ",".join(str(i) for i in bad[:10])
            raise ValueError(
                f"Invalid concorde reference costs in dataset: {len(bad)}/{n_rows} bad items "
                f"(first indices: {preview})."
            )
        for coords_raw, dist_raw in zip(coords_all, dist_all):
            row = normalize_instance({"coordinate": coords_raw, "distance_matrix": dist_raw})
            instances.append((row["coordinate"], row["distance_matrix"]))
    elif isinstance(raw, list):
        raise ValueError(
            "List-format dataset is not supported in strict Concorde mode. "
            "Use dict-format dataset with `coordinate` and Concorde reference costs."
        )
    else:
        raise ValueError("Unsupported dataset format; expected list or dict with coordinate list.")

    if not instances:
        raise ValueError(f"No instances loaded from: {p}")
    return instances, opt_costs
