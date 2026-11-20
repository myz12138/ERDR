from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_RE_OPT = re.compile(r"Optimal Solution:\s*([+-]?\d+(?:\.\d+)?)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Concorde reference costs for a TSP dataset pkl, write sidecar json, "
            "and optionally write an output pkl with embedded `cost_concorde`."
        )
    )
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-pkl", type=str, default=None)
    parser.add_argument("--concorde-exe", type=str, default="third_party/concorde_cygwin/concorde.exe")
    parser.add_argument("--scale", type=int, default=10000)
    parser.add_argument("--n-instance", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def _resolve_concorde_exe(raw: str, repo_root: Path) -> Path:
    p = Path(raw).expanduser()
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        # Allow plain command names (e.g., "concorde") from PATH first.
        which_raw = shutil.which(str(raw))
        if which_raw:
            candidates.append(Path(which_raw).resolve())
        candidates.append((repo_root / p).resolve())
        candidates.append((repo_root / "erdr" / "Hercules-tsp" / p).resolve())

    # Fallbacks used across this workspace.
    candidates.append((repo_root / "erdr" / "Hercules-tsp" / "third_party" / "concorde_cygwin" / "concorde.exe").resolve())
    candidates.append((repo_root / ".venv" / "Scripts" / "concorde.exe").resolve())
    # Linux/macOS common binary names from PATH.
    for name in ("concorde", "concorde.exe"):
        w = shutil.which(name)
        if w:
            candidates.append(Path(w).resolve())

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for c in candidates:
        key = str(c).lower()
        if key not in seen:
            unique_candidates.append(c)
            seen.add(key)

    for c in unique_candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Concorde executable not found. Tried:\n- " + "\n- ".join(str(c) for c in unique_candidates)
    )


def _to_coord_rows(raw_coord: Any) -> list[np.ndarray]:
    if isinstance(raw_coord, np.ndarray):
        arr = np.asarray(raw_coord, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[-1] != 2:
            raise ValueError(f"`coordinate` ndarray must be [N, M, 2], got {arr.shape}")
        return [np.asarray(arr[i], dtype=np.float64) for i in range(int(arr.shape[0]))]

    if isinstance(raw_coord, (list, tuple)):
        rows = [np.asarray(x, dtype=np.float64) for x in raw_coord]
        for i, r in enumerate(rows):
            if r.ndim != 2 or r.shape[1] != 2:
                raise ValueError(f"`coordinate[{i}]` must be [M, 2], got {r.shape}")
        return rows

    raise TypeError("`coordinate` must be ndarray/list/tuple")


def _to_dist_rows(raw_dist: Any, n_rows: int) -> list[np.ndarray | None]:
    if raw_dist is None:
        return [None] * n_rows
    try:
        arr = np.asarray(raw_dist, dtype=np.float64)
        if arr.ndim == 3 and int(arr.shape[0]) == n_rows:
            out = [np.asarray(arr[i], dtype=np.float64) for i in range(n_rows)]
            return out
    except Exception:
        pass
    if isinstance(raw_dist, (list, tuple)) and len(raw_dist) == n_rows:
        out2: list[np.ndarray | None] = []
        for x in raw_dist:
            if x is None:
                out2.append(None)
            else:
                dx = np.asarray(x, dtype=np.float64)
                out2.append(dx if dx.ndim == 2 else None)
        return out2
    return [None] * n_rows


def _load_dataset(dataset_path: Path) -> tuple[Any, list[np.ndarray], list[np.ndarray | None]]:
    with dataset_path.open("rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict) and "coordinate" in raw:
        coords = _to_coord_rows(raw["coordinate"])
        dists = _to_dist_rows(raw.get("distance_matrix"), n_rows=len(coords))
        return raw, coords, dists

    if isinstance(raw, (list, tuple)):
        coords_rows: list[np.ndarray] = []
        dist_rows: list[np.ndarray | None] = []
        for i, row in enumerate(raw):
            if isinstance(row, dict):
                c = row.get("coordinate", None)
                if c is None:
                    raise ValueError(f"row[{i}] missing `coordinate`")
                coord = np.asarray(c, dtype=np.float64)
                if coord.ndim != 2 or coord.shape[1] != 2:
                    raise ValueError(f"row[{i}].coordinate must be [M,2], got {coord.shape}")
                coords_rows.append(coord)
                d = row.get("distance_matrix", None)
                if d is None:
                    dist_rows.append(None)
                else:
                    dm = np.asarray(d, dtype=np.float64)
                    dist_rows.append(dm if dm.ndim == 2 else None)
                continue

            if isinstance(row, (list, tuple)) and len(row) >= 1:
                coord = np.asarray(row[0], dtype=np.float64)
                if coord.ndim != 2 or coord.shape[1] != 2:
                    raise ValueError(f"row[{i}][0] must be [M,2], got {coord.shape}")
                coords_rows.append(coord)
                if len(row) >= 2 and row[1] is not None:
                    dm = np.asarray(row[1], dtype=np.float64)
                    dist_rows.append(dm if dm.ndim == 2 else None)
                else:
                    dist_rows.append(None)
                continue

            raise ValueError(f"Unsupported dataset row format at index {i}: {type(row)}")
        return raw, coords_rows, dist_rows

    raise ValueError("Unsupported dataset format. Expected dict with `coordinate` or list/tuple rows.")


def _write_tsplib_euc2d(path: Path, coords: np.ndarray, scale: int) -> None:
    n = int(coords.shape[0])
    lines = [
        f"NAME: {path.stem}",
        "TYPE: TSP",
        f"DIMENSION: {n}",
        "EDGE_WEIGHT_TYPE: EUC_2D",
        "NODE_COORD_SECTION",
    ]
    sc = float(scale)
    for i in range(n):
        x = int(np.rint(float(coords[i, 0]) * sc))
        y = int(np.rint(float(coords[i, 1]) * sc))
        lines.append(f"{i + 1} {x} {y}")
    lines.append("EOF")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dm_scale_to_int(dm: np.ndarray, requested_scale: int) -> tuple[np.ndarray, int]:
    d = np.asarray(dm, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f"distance matrix must be square, got shape={d.shape}")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance matrix has non-finite values")
    if float(np.min(d)) < -1.0e-12:
        raise ValueError("distance matrix has negative values")

    # Concorde expects integer weights; keep conservative bound to avoid overflow.
    limit = int(2_000_000_000)
    max_v = float(np.max(np.abs(d)))
    req = max(1, int(requested_scale))
    if max_v <= 0.0:
        scale = req
    else:
        max_scale = int(limit // max_v)
        scale = min(req, max(1, max_scale))
    scaled = np.rint(d * float(scale)).astype(np.int64)

    # Ensure symmetric integer matrix (TSP).
    scaled = ((scaled + scaled.T) // 2).astype(np.int64, copy=False)
    np.fill_diagonal(scaled, 0)
    if int(np.max(np.abs(scaled))) > limit:
        raise ValueError(
            f"scaled matrix overflows integer bound with requested_scale={requested_scale}, "
            f"applied_scale={scale}"
        )
    return scaled, int(scale)


def _write_tsplib_explicit_full_matrix(path: Path, dm: np.ndarray, scale: int) -> int:
    dm_int, applied_scale = _dm_scale_to_int(dm, requested_scale=scale)
    n = int(dm_int.shape[0])
    lines = [
        f"NAME: {path.stem}",
        "TYPE: TSP",
        f"DIMENSION: {n}",
        "EDGE_WEIGHT_TYPE: EXPLICIT",
        "EDGE_WEIGHT_FORMAT: FULL_MATRIX",
        "EDGE_WEIGHT_SECTION",
    ]
    for i in range(n):
        row = " ".join(str(int(x)) for x in dm_int[i].tolist())
        lines.append(row)
    lines.append("EOF")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return int(applied_scale)


def _parse_sol(sol_path: Path, n_nodes: int) -> list[int]:
    text = sol_path.read_text(encoding="utf-8", errors="ignore")
    toks = [int(x) for x in re.findall(r"-?\d+", text)]
    if not toks:
        raise ValueError(f"Empty .sol file: {sol_path}")

    k = toks[0]
    rest = toks[1:]
    if k == n_nodes and len(rest) >= n_nodes:
        tour = rest[:n_nodes]
    elif len(toks) >= n_nodes:
        tour = toks[:n_nodes]
    else:
        raise ValueError(
            f"Invalid .sol content in {sol_path}: expected >= {n_nodes} nodes, got {len(toks)} ints"
        )

    tset = set(tour)
    if len(tour) != n_nodes or len(tset) != n_nodes or min(tour) < 0 or max(tour) >= n_nodes:
        raise ValueError(f"Invalid tour permutation in {sol_path}")
    return [int(v) for v in tour]


def _tour_cost_from_dm_or_coord(
    tour: Sequence[int],
    dm: np.ndarray | None,
    coord: np.ndarray,
) -> float:
    n = len(tour)
    if n == 0:
        return float("nan")

    total = 0.0
    if dm is not None and dm.ndim == 2:
        for i in range(n):
            a = int(tour[i])
            b = int(tour[(i + 1) % n])
            total += float(dm[a, b])
        return float(total)

    for i in range(n):
        a = int(tour[i])
        b = int(tour[(i + 1) % n])
        total += float(np.linalg.norm(coord[a] - coord[b]))
    return float(total)


def _solve_one(
    *,
    idx: int,
    coord: np.ndarray,
    dm: np.ndarray | None,
    concorde_exe: Path,
    scale: int,
    seed: int,
) -> dict[str, Any]:
    n_nodes = int(coord.shape[0])
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"concorde_{idx:05d}_") as td:
        tmpdir = Path(td)
        tsp_path = tmpdir / f"inst_{idx:05d}.tsp"
        sol_path = tmpdir / f"inst_{idx:05d}.sol"
        if dm is not None:
            applied_scale = _write_tsplib_explicit_full_matrix(tsp_path, dm, scale=scale)
            edge_weight_type = "EXPLICIT_FULL_MATRIX"
        else:
            _write_tsplib_euc2d(tsp_path, coord, scale=scale)
            applied_scale = int(scale)
            edge_weight_type = "EUC_2D"

        cmd = [str(concorde_exe), "-s", str(int(seed)), "-x", tsp_path.name]
        cp = subprocess.run(
            cmd,
            cwd=str(tmpdir),
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = float(time.perf_counter() - t0)
        stdout = cp.stdout or ""
        stderr = cp.stderr or ""

        m = _RE_OPT.search(stdout)
        opt_int = float(m.group(1)) if m else None

        if not sol_path.exists():
            return {
                "idx": int(idx),
                "status": "error",
                "return_code": int(cp.returncode),
                "time_sec": elapsed,
                "edge_weight_type": edge_weight_type,
                "applied_scale": int(applied_scale),
                "error": "missing .sol output",
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-2000:],
            }

        try:
            tour = _parse_sol(sol_path, n_nodes=n_nodes)
            opt_float = _tour_cost_from_dm_or_coord(tour=tour, dm=dm, coord=coord)
            rec: dict[str, Any] = {
                "idx": int(idx),
                "status": "ok",
                # Cygwin concorde can return 255 even when solved; normalize success to 0.
                "return_code": 0,
                "edge_weight_type": edge_weight_type,
                "opt_int": float(opt_int) if opt_int is not None else None,
                "opt_scaled": (float(opt_int) / float(applied_scale)) if opt_int is not None else None,
                "requested_scale": int(scale),
                "applied_scale": int(applied_scale),
                "opt_float_from_tour": float(opt_float),
                "time_sec": elapsed,
                "tour": [int(x) for x in tour],
            }
            if cp.returncode not in (0, 255):
                rec["raw_return_code"] = int(cp.returncode)
            return rec
        except Exception as e:
            return {
                "idx": int(idx),
                "status": "error",
                "return_code": int(cp.returncode),
                "time_sec": elapsed,
                "edge_weight_type": edge_weight_type,
                "applied_scale": int(applied_scale),
                "error": f"parse_tour_failed: {e}",
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-2000:],
            }


def _merge_costs_from_raw(
    raw_obj: Any,
    n_total: int,
) -> tuple[list[float], list[list[int] | None]]:
    costs = [math.nan] * n_total
    tours: list[list[int] | None] = [None] * n_total
    if not isinstance(raw_obj, dict):
        return costs, tours

    c0 = raw_obj.get("cost_concorde", None)
    if c0 is not None:
        try:
            arr = np.asarray(c0, dtype=np.float64)
            if arr.ndim >= 1 and int(arr.shape[0]) == n_total:
                for i in range(n_total):
                    v = float(arr[i])
                    if np.isfinite(v) and v > 0.0:
                        costs[i] = v
        except Exception:
            pass

    t0 = raw_obj.get("optimal_tour_concorde", None)
    if isinstance(t0, (list, tuple)) and len(t0) == n_total:
        out: list[list[int] | None] = []
        for row in t0:
            if row is None:
                out.append(None)
                continue
            try:
                out.append([int(x) for x in row])
            except Exception:
                out.append(None)
        tours = out

    return costs, tours


def _dump_output_pkl(
    *,
    output_pkl: Path,
    raw_obj: Any,
    coords_rows: list[np.ndarray],
    dist_rows: list[np.ndarray | None],
    cost_concorde: list[float],
    tour_concorde: list[list[int] | None],
) -> None:
    if isinstance(raw_obj, dict):
        out = dict(raw_obj)
    else:
        out = {
            "coordinate": np.asarray(coords_rows, dtype=np.float64),
        }
        if all(dm is not None for dm in dist_rows):
            out["distance_matrix"] = np.asarray([dm for dm in dist_rows if dm is not None], dtype=np.float64)

    out["cost_concorde"] = np.asarray(cost_concorde, dtype=np.float64)
    out["optimal_tour_concorde"] = [
        (None if t is None else [int(x) for x in t]) for t in tour_concorde
    ]

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = Path(args.dataset_path).resolve()
    output_json = Path(args.output_json).resolve()
    output_pkl = Path(args.output_pkl).resolve() if args.output_pkl else None

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    if int(args.scale) <= 0:
        raise ValueError("--scale must be positive")

    concorde_exe = _resolve_concorde_exe(str(args.concorde_exe), repo_root=repo_root)

    raw_obj, coords_rows, dist_rows = _load_dataset(dataset_path)
    n_total = len(coords_rows)
    if n_total <= 0:
        raise ValueError("empty dataset")

    start = max(0, int(args.start_index))
    if start >= n_total:
        raise ValueError(f"--start-index={start} out of range for dataset size {n_total}")
    n_instance = int(args.n_instance)
    if n_instance <= 0:
        end_excl = n_total
    else:
        end_excl = min(n_total, start + n_instance)

    run_indices = list(range(start, end_excl))
    n_run = len(run_indices)

    t_all0 = time.perf_counter()
    records: list[dict[str, Any]] = []
    for k, idx in enumerate(run_indices, start=1):
        rec = _solve_one(
            idx=idx,
            coord=np.asarray(coords_rows[idx], dtype=np.float64),
            dm=(None if dist_rows[idx] is None else np.asarray(dist_rows[idx], dtype=np.float64)),
            concorde_exe=concorde_exe,
            scale=int(args.scale),
            seed=int(args.seed),
        )
        records.append(rec)
        pe = max(1, int(args.print_every))
        if (k % pe == 0) or (k == n_run):
            n_ok_now = sum(1 for x in records if x.get("status") == "ok")
            print(
                f"[concorde] {k}/{n_run} done, ok={n_ok_now}, last_idx={idx}",
                flush=True,
            )
    elapsed_total = float(time.perf_counter() - t_all0)

    n_ok = sum(1 for r in records if r.get("status") == "ok")
    n_error = n_run - n_ok

    out_obj: dict[str, Any] = {
        "scale": int(args.scale),
        "start_index": int(start),
        "end_index_exclusive": int(end_excl),
        "n_total_dataset": int(n_total),
        "n_run": int(n_run),
        "n_ok": int(n_ok),
        "n_error": int(n_error),
        "elapsed_sec_total": float(elapsed_total),
        "records": records,
    }

    ok_vals = [
        float(r["opt_float_from_tour"])
        for r in records
        if r.get("status") == "ok" and r.get("opt_float_from_tour") is not None
    ]
    out_obj["mean_opt_float_from_tour"] = (
        float(np.mean(np.asarray(ok_vals, dtype=np.float64))) if ok_vals else None
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    if output_pkl is not None:
        costs, tours = _merge_costs_from_raw(raw_obj, n_total=n_total)
        for r in records:
            i = int(r["idx"])
            if r.get("status") != "ok":
                costs[i] = math.nan
                tours[i] = None
                continue
            v = float(r.get("opt_float_from_tour"))
            costs[i] = v if np.isfinite(v) and v > 0.0 else math.nan
            t = r.get("tour", None)
            tours[i] = [int(x) for x in t] if isinstance(t, list) else None

        _dump_output_pkl(
            output_pkl=output_pkl,
            raw_obj=raw_obj,
            coords_rows=coords_rows,
            dist_rows=dist_rows,
            cost_concorde=costs,
            tour_concorde=tours,
        )

    print(
        f"[done] json={output_json} "
        + (f"pkl={output_pkl} " if output_pkl is not None else "")
        + f"ok={n_ok}/{n_run}",
        flush=True,
    )


if __name__ == "__main__":
    main()
