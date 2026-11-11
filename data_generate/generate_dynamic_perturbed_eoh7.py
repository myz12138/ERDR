from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class SolverSpec:
    solver_id: str
    solver_path: Path
    update_fn: Any


class _GuideWrapper:
    def __init__(self, fn):
        self._fn = fn

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        return self._fn(edge_distance, local_opt_tour, edge_n_used)


def _load_base_dataset(base_pkl: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with base_pkl.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"base dataset must be dict-format: {base_pkl}")
    if "coordinate" not in raw or "distance_matrix" not in raw:
        raise ValueError("base dataset missing required keys: coordinate/distance_matrix")

    coords = np.asarray(raw["coordinate"], dtype=np.float64)
    dms = np.asarray(raw["distance_matrix"], dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] != 2:
        raise ValueError(f"invalid coordinate shape: {coords.shape}")
    if dms.ndim != 3 or dms.shape[1] != dms.shape[2]:
        raise ValueError(f"invalid distance_matrix shape: {dms.shape}")
    if coords.shape[0] != dms.shape[0] or coords.shape[1] != dms.shape[1]:
        raise ValueError(
            f"coords/dist mismatch: coords={coords.shape}, distance_matrix={dms.shape}"
        )

    cost_concorde = None
    if "cost_concorde" in raw and raw["cost_concorde"] is not None:
        try:
            c = np.asarray(raw["cost_concorde"], dtype=np.float64)
            if c.ndim == 1 and c.shape[0] == coords.shape[0]:
                cost_concorde = c
        except Exception:
            cost_concorde = None

    return coords, dms, cost_concorde


def _load_solver_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"solver file not found: {path}")
    spec = importlib.util.spec_from_file_location(f"solver_mod_{path.stem}", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import solver: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "update_edge_distance", None)
    if fn is None or not callable(fn):
        raise RuntimeError(f"update_edge_distance not found in solver: {path}")
    return fn


def _route_to_sequence(route2end: np.ndarray, n_nodes: int) -> List[int]:
    route2end = np.asarray(route2end, dtype=np.int32)
    seq = [0]
    current = 0
    for _ in range(max(1, n_nodes * 2)):
        nxt = int(route2end[current, 1])
        seq.append(nxt)
        current = nxt
        if current == 0:
            break
    if seq[-1] != 0:
        seq.append(0)
    return seq


def _extract_visit_order(seq: List[int], n_nodes: int) -> List[int]:
    seen = set()
    order: List[int] = []
    for node in seq:
        u = int(node)
        if u < 0 or u >= n_nodes:
            continue
        if u in seen:
            continue
        seen.add(u)
        order.append(u)
        if len(order) >= n_nodes:
            break
    if len(order) != n_nodes:
        remaining = [x for x in range(n_nodes) if x not in seen]
        order.extend(remaining)
    return order[:n_nodes]


def _distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=2).astype(np.float64)


def _make_dummy_closed_tsp(
    *,
    current_coord: np.ndarray,
    origin_coord: np.ndarray,
    remain_coords: np.ndarray,
    big_m: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    # node order: [cur] + remain + [origin] + [dummy]
    cur = np.asarray(current_coord, dtype=np.float64).reshape(1, 2)
    rem = np.asarray(remain_coords, dtype=np.float64)
    ori = np.asarray(origin_coord, dtype=np.float64).reshape(1, 2)
    dummy = np.asarray(origin_coord, dtype=np.float64).reshape(1, 2)

    coords = np.concatenate([cur, rem, ori, dummy], axis=0)
    dist = _distance_matrix(coords)

    idx_cur = 0
    idx_origin = int(coords.shape[0] - 2)
    idx_dummy = int(coords.shape[0] - 1)
    M = float(big_m)
    dist[idx_dummy, :] = M
    dist[:, idx_dummy] = M
    dist[idx_dummy, idx_cur] = 0.0
    dist[idx_cur, idx_dummy] = 0.0
    dist[idx_dummy, idx_origin] = 0.0
    dist[idx_origin, idx_dummy] = 0.0
    dist[idx_dummy, idx_dummy] = 0.0
    np.fill_diagonal(dist, 0.0)

    return coords, dist, {"idx_cur": idx_cur, "idx_origin": idx_origin, "idx_dummy": idx_dummy}


def _run_base_solver(
    *,
    gls_core_mod,
    update_fn,
    coords: np.ndarray,
    dist: np.ndarray,
    neighbor_k: int,
    time_limit_sec: float,
    ite_max: int,
    perturbation_moves: int,
) -> Dict[str, Any]:
    n = int(dist.shape[0])
    dm = np.ascontiguousarray(np.asarray(dist, dtype=np.float32))
    init_tour = np.ascontiguousarray(gls_core_mod.nearest_neighbor_2End(dm, 0), dtype=np.int32)
    init_cost = float(gls_core_mod.tour_cost_2End(dm, init_tour))
    k = max(1, min(int(neighbor_k), n - 1))
    nearest_indices = np.ascontiguousarray(np.argsort(dm, axis=1)[:, 1 : k + 1].astype(np.int32))
    guide = _GuideWrapper(update_fn)
    route, cost, iters = gls_core_mod.guided_local_search(
        coords,
        dm,
        nearest_indices,
        init_tour,
        init_cost,
        time.time() + float(time_limit_sec),
        int(ite_max),
        int(perturbation_moves),
        first_improvement=False,
        guide_algorithm=guide,
    )
    seq = _route_to_sequence(np.asarray(route, dtype=np.int32), n)
    order = _extract_visit_order(seq, n)
    return {
        "route2end": np.asarray(route, dtype=np.int32),
        "cost": float(cost),
        "iters": int(iters),
        "seq": seq,
        "order": order,
    }


def _perturb_remaining(
    *,
    rng: np.random.Generator,
    remain_coords: np.ndarray,
    perturb_type: str,
    jitter_sigma: float,
    node_frac: float,
) -> Dict[str, Any]:
    rem = np.asarray(remain_coords, dtype=np.float64)
    n_rem = int(rem.shape[0])
    if n_rem <= 0:
        return {"coords": rem, "delta_nodes": 0, "detail": {"warning": "empty_remaining"}}

    if perturb_type == "coord_jitter":
        noise = rng.normal(loc=0.0, scale=float(jitter_sigma), size=rem.shape)
        out = np.clip(rem + noise, 0.0, 1.0)
        return {
            "coords": out,
            "delta_nodes": 0,
            "detail": {"sigma": float(jitter_sigma)},
        }

    if perturb_type == "node_add":
        add_n = max(1, int(round(float(node_frac) * n_rem)))
        new_pts = rng.random((add_n, 2), dtype=np.float64)
        out = np.concatenate([rem, new_pts], axis=0)
        return {
            "coords": out,
            "delta_nodes": int(add_n),
            "detail": {"add_n": int(add_n), "node_frac": float(node_frac)},
        }

    if perturb_type == "node_remove":
        remove_n = max(1, int(round(float(node_frac) * n_rem)))
        # Keep at least 2 remaining targets.
        remove_n = min(remove_n, max(0, n_rem - 2))
        if remove_n <= 0:
            return {
                "coords": rem,
                "delta_nodes": 0,
                "detail": {"remove_n": 0, "node_frac": float(node_frac)},
            }
        keep_mask = np.ones(n_rem, dtype=bool)
        drop_idx = rng.choice(n_rem, size=remove_n, replace=False)
        keep_mask[drop_idx] = False
        out = rem[keep_mask]
        return {
            "coords": out,
            "delta_nodes": int(-remove_n),
            "detail": {"remove_n": int(remove_n), "node_frac": float(node_frac)},
        }

    raise ValueError(f"unsupported perturb_type: {perturb_type}")


def _build_dataset_for_setting(
    *,
    coords_all: np.ndarray,
    dms_all: np.ndarray,
    solver: SolverSpec,
    ratio: float,
    perturb_type: str,
    seed: int,
    gls_core_mod,
    neighbor_k: int,
    time_limit_sec: float,
    ite_max: int,
    perturbation_moves: int,
    jitter_sigma: float,
    node_frac: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n_inst = int(coords_all.shape[0])
    n_nodes = int(coords_all.shape[1])
    exec_n = int(math.ceil(float(ratio) * n_nodes))
    exec_n = int(max(1, min(exec_n, n_nodes - 1)))

    new_coords_rows: List[np.ndarray] = []
    new_dist_rows: List[np.ndarray] = []
    meta_rows: List[Dict[str, Any]] = []

    for i in range(n_inst):
        coords = np.asarray(coords_all[i], dtype=np.float64)
        dm = np.asarray(dms_all[i], dtype=np.float64)
        base = _run_base_solver(
            gls_core_mod=gls_core_mod,
            update_fn=solver.update_fn,
            coords=coords,
            dist=dm,
            neighbor_k=neighbor_k,
            time_limit_sec=time_limit_sec,
            ite_max=ite_max,
            perturbation_moves=perturbation_moves,
        )
        order = base["order"]
        prefix = order[:exec_n]
        current = int(prefix[-1])
        remain_idx = order[exec_n:]
        remain_coords = coords[np.asarray(remain_idx, dtype=np.int64)]

        local_seed = int(seed + i * 10007 + int(ratio * 1000) * 101 + len(perturb_type) * 17)
        rng = np.random.default_rng(local_seed)
        pert = _perturb_remaining(
            rng=rng,
            remain_coords=remain_coords,
            perturb_type=perturb_type,
            jitter_sigma=jitter_sigma,
            node_frac=node_frac,
        )
        remain_after = np.asarray(pert["coords"], dtype=np.float64)

        dmax = float(np.max(dm))
        n_tmp = int(1 + remain_after.shape[0] + 1 + 1)
        big_m = float(min(1.0e6, max(1000.0, 20.0 * n_tmp * max(dmax, 1.0e-6))))
        coords_new, dist_new, idx_map = _make_dummy_closed_tsp(
            current_coord=coords[current],
            origin_coord=coords[0],
            remain_coords=remain_after,
            big_m=big_m,
        )

        new_coords_rows.append(coords_new.astype(np.float64, copy=False))
        new_dist_rows.append(dist_new.astype(np.float64, copy=False))
        meta_rows.append(
            {
                "instance_idx": int(i),
                "solver_id": solver.solver_id,
                "ratio": float(ratio),
                "perturb_type": perturb_type,
                "seed": int(local_seed),
                "n_nodes_base": int(n_nodes),
                "n_executed": int(exec_n),
                "n_remaining_before": int(len(remain_idx)),
                "n_remaining_after": int(remain_after.shape[0]),
                "delta_nodes": int(pert["delta_nodes"]),
                "current_node_base_idx": int(current),
                "remaining_base_indices": [int(x) for x in remain_idx],
                "dummy_index_new": int(idx_map["idx_dummy"]),
                "current_index_new": int(idx_map["idx_cur"]),
                "origin_index_new": int(idx_map["idx_origin"]),
                "big_m": float(big_m),
                "base_solver_cost": float(base["cost"]),
                "base_solver_iters": int(base["iters"]),
                "base_visit_order": [int(x) for x in order],
                "perturb_detail": pert["detail"],
            }
        )

    coords_arr = np.asarray(new_coords_rows, dtype=np.float64)
    dist_arr = np.asarray(new_dist_rows, dtype=np.float64)
    payload = {
        "coordinate": coords_arr,
        "distance_matrix": dist_arr,
    }
    return payload, meta_rows


def _run_concorde_for_dataset(
    *,
    python_exe: Path,
    repo_root: Path,
    dataset_pkl: Path,
    concorde_out_dir: Path,
    scale: int,
) -> Path:
    concorde_out_dir.mkdir(parents=True, exist_ok=True)
    script = repo_root / "data_generate" / "compute_concorde_opt.py"
    out_json = concorde_out_dir / f"{dataset_pkl.stem}_concorde_full.json"
    cmd = [
        str(python_exe),
        str(script),
        "--dataset-path",
        str(dataset_pkl),
        "--output-json",
        str(out_json),
        "--output-pkl",
        str(dataset_pkl),
        "--scale",
        str(int(scale)),
        "--print-every",
        "8",
    ]
    subprocess.run(cmd, check=True)
    return out_json


def _dataset_has_cost_concorde(dataset_pkl: Path) -> bool:
    try:
        with dataset_pkl.open("rb") as f:
            raw = pickle.load(f)
        if not isinstance(raw, dict):
            return False
        vals = raw.get("cost_concorde", None)
        if vals is None:
            return False
        arr = np.asarray(vals, dtype=np.float64)
        return bool(arr.ndim == 1 and arr.size > 0 and np.all(np.isfinite(arr)) and np.all(arr > 0.0))
    except Exception:
        return False


def _parse_list_float(text: str) -> List[float]:
    out: List[float] = []
    for part in str(text).split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    if not out:
        raise ValueError("empty float list")
    return out


def _parse_list_str(text: str) -> List[str]:
    out: List[str] = []
    for part in str(text).split(","):
        p = part.strip()
        if not p:
            continue
        out.append(p)
    if not out:
        raise ValueError("empty str list")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dynamic-perturbed TSP datasets from EoH-plan-EoH7 best solvers: "
            "execute prefix, perturb remaining nodes, convert to dummy-closed TSP, then "
            "compute concorde references."
        )
    )
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--base-pkl", type=str, default="data_generate/TSP64_u50/TSP64_u50_concorde.pkl")
    parser.add_argument(
        "--solver-static",
        type=str,
        default="EoH-plan-EoH7/outputs/static/best_solver.py",
    )
    parser.add_argument(
        "--solver-risk2",
        type=str,
        default="EoH-plan-EoH7/outputs/static_risk2/best_solver.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data_generate/dynamic_eoh7_exec_perturb_v1",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="",
        help="Output dataset prefix; empty means auto TSP{base_nodes}_dyn.",
    )
    parser.add_argument("--n-instance", type=int, default=64)
    parser.add_argument("--ratios", type=str, default="0.2,0.5,0.8")
    parser.add_argument("--perturb-types", type=str, default="coord_jitter,node_add,node_remove")
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--neighbor-k", type=int, default=50)
    parser.add_argument("--time-limit-sec", type=float, default=5.0)
    parser.add_argument("--ite-max", type=int, default=100)
    parser.add_argument("--perturbation-moves", type=int, default=1)
    parser.add_argument("--jitter-sigma", type=float, default=0.02)
    parser.add_argument("--node-frac", type=float, default=0.2)
    parser.add_argument("--concorde-scale", type=int, default=10000)
    parser.add_argument(
        "--solver-mode",
        type=str,
        choices=["both", "single"],
        default="both",
        help="both: generate for static + static_risk2; single: generate for one solver only.",
    )
    parser.add_argument(
        "--single-solver-id",
        type=str,
        choices=["static", "static_risk2"],
        default="",
        help="Required when --solver-mode single.",
    )
    parser.add_argument(
        "--single-solver-path",
        type=str,
        default="",
        help="Optional override solver path for --single-solver-id in single mode.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    base_pkl = (repo_root / args.base_pkl).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    concorde_dir = out_dir / "concorde"
    manifest_path = out_dir / "dataset_manifest.jsonl"

    ratios = _parse_list_float(args.ratios)
    perturb_types = _parse_list_str(args.perturb_types)
    n_instance = int(max(int(args.n_instance), 1))

    # Import EoH-plan GLS core implementation.
    src_dir = (repo_root / "EoH-plan-EoH7" / "src").resolve()
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import gls_core as gls_core_mod  # noqa: E402

    gls_core_mod.warmup_numba_kernels()

    coords_all, dms_all, _cost_concorde = _load_base_dataset(base_pkl)
    coords_all = coords_all[:n_instance]
    dms_all = dms_all[:n_instance]
    n_instance_eff = int(coords_all.shape[0])
    n_nodes_base = int(coords_all.shape[1])
    dataset_prefix_raw = str(args.dataset_prefix).strip()
    dataset_prefix = dataset_prefix_raw if dataset_prefix_raw else f"TSP{n_nodes_base}_dyn"
    print(
        f"[base] {base_pkl} | n_instance={n_instance_eff}, n_nodes={coords_all.shape[1]}",
        flush=True,
    )

    solver_mode = str(args.solver_mode).strip().lower()
    if solver_mode == "single":
        single_solver_id = str(args.single_solver_id).strip()
        if single_solver_id not in {"static", "static_risk2"}:
            raise ValueError("single mode requires --single-solver-id in {static, static_risk2}")
        single_solver_path_raw = str(args.single_solver_path).strip()
        if single_solver_path_raw:
            single_solver_path = (repo_root / single_solver_path_raw).resolve()
        elif single_solver_id == "static":
            single_solver_path = (repo_root / args.solver_static).resolve()
        else:
            single_solver_path = (repo_root / args.solver_risk2).resolve()
        solvers = [
            SolverSpec(
                solver_id=single_solver_id,
                solver_path=single_solver_path,
                update_fn=_load_solver_module(single_solver_path),
            )
        ]
    else:
        solvers = [
            SolverSpec(
                solver_id="static",
                solver_path=(repo_root / args.solver_static).resolve(),
                update_fn=_load_solver_module((repo_root / args.solver_static).resolve()),
            ),
            SolverSpec(
                solver_id="static_risk2",
                solver_path=(repo_root / args.solver_risk2).resolve(),
                update_fn=_load_solver_module((repo_root / args.solver_risk2).resolve()),
            ),
        ]

    py_exe = Path(sys.executable).resolve()
    all_rows: List[Dict[str, Any]] = []
    produced = 0

    for solver in solvers:
        for ratio in ratios:
            for perturb_type in perturb_types:
                tag = f"{solver.solver_id}_r{int(round(ratio * 100)):02d}_{perturb_type}"
                dataset_pkl = out_dir / f"{dataset_prefix}_{tag}.pkl"
                meta_json = out_dir / f"{dataset_prefix}_{tag}_meta.json"
                concorde_json = concorde_dir / f"{dataset_pkl.stem}_concorde_full.json"
                if dataset_pkl.exists() and meta_json.exists() and concorde_json.exists() and _dataset_has_cost_concorde(dataset_pkl):
                    row = {
                        "tag": tag,
                        "solver_id": solver.solver_id,
                        "ratio": float(ratio),
                        "perturb_type": perturb_type,
                        "dataset_pkl": str(dataset_pkl),
                        "meta_json": str(meta_json),
                        "concorde_json": str(concorde_json),
                        "skipped_existing": True,
                    }
                    all_rows.append(row)
                    with manifest_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"[skip] existing complete {tag}", flush=True)
                    produced += 1
                    continue
                print(f"[gen] start {tag}", flush=True)
                payload, meta_rows = _build_dataset_for_setting(
                    coords_all=coords_all,
                    dms_all=dms_all,
                    solver=solver,
                    ratio=float(ratio),
                    perturb_type=perturb_type,
                    seed=int(args.seed),
                    gls_core_mod=gls_core_mod,
                    neighbor_k=int(args.neighbor_k),
                    time_limit_sec=float(args.time_limit_sec),
                    ite_max=int(args.ite_max),
                    perturbation_moves=int(args.perturbation_moves),
                    jitter_sigma=float(args.jitter_sigma),
                    node_frac=float(args.node_frac),
                )
                with dataset_pkl.open("wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                meta_obj = {
                    "tag": tag,
                    "solver_id": solver.solver_id,
                    "solver_path": str(solver.solver_path),
                    "base_pkl": str(base_pkl),
                    "n_instance": int(n_instance_eff),
                    "ratio": float(ratio),
                    "perturb_type": perturb_type,
                    "seed": int(args.seed),
                    "neighbor_k": int(args.neighbor_k),
                    "time_limit_sec": float(args.time_limit_sec),
                    "ite_max": int(args.ite_max),
                    "perturbation_moves": int(args.perturbation_moves),
                    "jitter_sigma": float(args.jitter_sigma),
                    "node_frac": float(args.node_frac),
                    "instances": meta_rows,
                }
                meta_json.write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2), encoding="utf-8")

                # Fill `cost_concorde` back into dataset pkl for direct GAP evaluation.
                concorde_json = _run_concorde_for_dataset(
                    python_exe=py_exe,
                    repo_root=repo_root,
                    dataset_pkl=dataset_pkl,
                    concorde_out_dir=concorde_dir,
                    scale=int(args.concorde_scale),
                )
                produced += 1
                row = {
                    "tag": tag,
                    "solver_id": solver.solver_id,
                    "ratio": float(ratio),
                    "perturb_type": perturb_type,
                    "dataset_pkl": str(dataset_pkl),
                    "meta_json": str(meta_json),
                    "concorde_json": str(concorde_json),
                }
                all_rows.append(row)
                with manifest_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[gen] done {tag} -> {dataset_pkl.name}", flush=True)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(out_dir),
        "base_pkl": str(base_pkl),
        "dataset_prefix": dataset_prefix,
        "n_instance": int(n_instance_eff),
        "n_datasets": int(produced),
        "rows": all_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[summary] generated {produced} datasets in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
