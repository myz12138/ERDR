from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm4ad.task.optimization.tsp_gls_2O.gls import guided_local_search


def _load_base_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict) or "coordinate" not in raw or "distance_matrix" not in raw:
        raise ValueError("base dataset must contain coordinate and distance_matrix")
    coordinates = np.asarray(raw["coordinate"], dtype=np.float64)
    distances = np.asarray(raw["distance_matrix"], dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError(f"invalid coordinate shape: {coordinates.shape}")
    if distances.ndim != 3 or distances.shape != (coordinates.shape[0], coordinates.shape[1], coordinates.shape[1]):
        raise ValueError(f"invalid distance matrix shape: {distances.shape}")
    return coordinates, distances


def _load_solver(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"solver not found: {path}")
    spec = importlib.util.spec_from_file_location(f"meoh_solver_{path.stem}", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import solver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "update_edge_distance", None)
    if function is None or not callable(function):
        raise RuntimeError("solver must define update_edge_distance")
    return function


def _run_solver(
    solver: Any,
    coordinates: np.ndarray,
    distances: np.ndarray,
    time_limit_sec: float,
    ite_max: int,
    perturbation_moves: int,
) -> tuple[float, int, list[int]]:
    del coordinates
    tour = guided_local_search(
        np.asarray(distances, dtype=np.float64),
        np.asarray(distances, dtype=np.float64),
        solver,
        perturbation_moves=int(perturbation_moves),
        iter_limit=int(ite_max),
        time_limit_seconds=float(time_limit_sec),
    )
    order = [int(node) for node in np.asarray(tour, dtype=np.int64).tolist()]
    cost = float(distances[tour, np.roll(tour, 1)].sum().item())
    return cost, 0, order


def _perturb(
    remaining: np.ndarray,
    perturb_type: str,
    seed: int,
    jitter_sigma: float,
    node_fraction: float,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    values = np.asarray(remaining, dtype=np.float64)
    if perturb_type == "coord_jitter":
        return np.clip(values + rng.normal(0.0, jitter_sigma, values.shape), 0.0, 1.0), 0
    if perturb_type == "node_add":
        count = max(1, int(round(node_fraction * len(values))))
        return np.concatenate([values, rng.random((count, 2))]), count
    if perturb_type == "node_remove":
        count = min(max(1, int(round(node_fraction * len(values)))), max(0, len(values) - 2))
        if count == 0:
            return values, 0
        keep = np.ones(len(values), dtype=bool)
        keep[rng.choice(len(values), size=count, replace=False)] = False
        return values[keep], -count
    raise ValueError(f"unsupported perturbation type: {perturb_type}")


def _dummy_closed_instance(current: np.ndarray, origin: np.ndarray, remaining: np.ndarray, max_distance: float) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.concatenate([
        np.asarray(current, dtype=np.float64).reshape(1, 2),
        np.asarray(remaining, dtype=np.float64),
        np.asarray(origin, dtype=np.float64).reshape(1, 2),
        np.asarray(origin, dtype=np.float64).reshape(1, 2),
    ])
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(differences, axis=2).astype(np.float64)
    dummy = len(coordinates) - 1
    origin_index = len(coordinates) - 2
    penalty = min(1.0e6, max(1000.0, 20.0 * len(coordinates) * max(max_distance, 1.0e-6)))
    distances[dummy, :] = penalty
    distances[:, dummy] = penalty
    distances[dummy, 0] = distances[0, dummy] = 0.0
    distances[dummy, origin_index] = distances[origin_index, dummy] = 0.0
    np.fill_diagonal(distances, 0.0)
    return coordinates, distances


def _generate_dataset(
    coordinates_all: np.ndarray,
    distances_all: np.ndarray,
    solver: Any,
    ratio: float,
    perturb_type: str,
    seed: int,
    time_limit_sec: float,
    ite_max: int,
    perturbation_moves: int,
    jitter_sigma: float,
    node_fraction: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_nodes = int(coordinates_all.shape[1])
    n_executed = max(1, min(int(math.ceil(ratio * n_nodes)), n_nodes - 1))
    coordinates_out: list[np.ndarray] = []
    distances_out: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for index, (coordinates, distances) in enumerate(zip(coordinates_all, distances_all)):
        base_cost, iterations, order = _run_solver(
            solver,
            coordinates,
            distances,
            time_limit_sec,
            ite_max,
            perturbation_moves,
        )
        current = order[n_executed - 1]
        remaining_indices = order[n_executed:]
        local_seed = int(seed + index * 10007 + int(ratio * 1000) * 101 + len(perturb_type) * 17)
        remaining, delta_nodes = _perturb(
            coordinates[np.asarray(remaining_indices, dtype=np.int64)],
            perturb_type,
            local_seed,
            jitter_sigma,
            node_fraction,
        )
        coordinates_new, distances_new = _dummy_closed_instance(
            coordinates[current],
            coordinates[0],
            remaining,
            float(np.max(distances)),
        )
        coordinates_out.append(coordinates_new)
        distances_out.append(distances_new)
        metadata.append({
            "instance_idx": index,
            "ratio": ratio,
            "perturb_type": perturb_type,
            "seed": local_seed,
            "n_nodes_base": n_nodes,
            "n_executed": n_executed,
            "n_remaining_before": len(remaining_indices),
            "n_remaining_after": len(remaining),
            "delta_nodes": delta_nodes,
            "base_solver_cost": base_cost,
            "base_solver_iterations": iterations,
        })
    return {
        "coordinate": np.asarray(coordinates_out, dtype=np.float64),
        "distance_matrix": np.asarray(distances_out, dtype=np.float64),
    }, metadata


def _add_concorde_costs(repo_root: Path, dataset: Path, output_dir: Path, scale: int) -> None:
    script = repo_root / "data_generate" / "compute_concorde_opt.py"
    if not script.is_file():
        raise FileNotFoundError(f"Concorde evaluator not found: {script}")
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable,
        str(script),
        "--dataset-path", str(dataset),
        "--output-json", str(output_dir / f"{dataset.stem}_concorde_full.json"),
        "--output-pkl", str(dataset),
        "--scale", str(scale),
        "--print-every", "8",
    ], check=True)


def _parse_ratios(value: str) -> list[float]:
    ratios = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not ratios or any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
        raise ValueError("ratios must be values in (0, 1)")
    return ratios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--base-pkl", required=True)
    parser.add_argument("--solver-id", choices=["baseline", "erdr"], required=True)
    parser.add_argument("--solver-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-instance", type=int, default=50)
    parser.add_argument("--ratios", default="0.2,0.5,0.8")
    parser.add_argument("--perturb-types", default="coord_jitter,node_add,node_remove")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--neighbor-k", type=int, default=20)
    parser.add_argument("--time-limit-sec", type=float, default=20.0)
    parser.add_argument("--ite-max", type=int, default=500)
    parser.add_argument("--perturbation-moves", type=int, default=2)
    parser.add_argument("--jitter-sigma", type=float, default=0.02)
    parser.add_argument("--node-frac", type=float, default=0.2)
    parser.add_argument("--concorde-scale", type=int, default=10000)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    base_pkl = Path(args.base_pkl)
    if not base_pkl.is_absolute():
        base_pkl = (repo_root / base_pkl).resolve()
    solver_path = Path(args.solver_path)
    if not solver_path.is_absolute():
        solver_path = (repo_root / solver_path).resolve()
    output_dir = Path(args.out_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    coordinates, distances = _load_base_dataset(base_pkl)
    count = min(max(1, int(args.n_instance)), len(coordinates))
    coordinates = coordinates[:count]
    distances = distances[:count]
    solver = _load_solver(solver_path)
    perturb_types = [item.strip() for item in args.perturb_types.split(",") if item.strip()]
    allowed = {"coord_jitter", "node_add", "node_remove"}
    if not perturb_types or any(item not in allowed for item in perturb_types):
        raise ValueError("perturb-types must use coord_jitter, node_add, node_remove")
    manifest: list[dict[str, Any]] = []
    for ratio in _parse_ratios(args.ratios):
        for perturb_type in perturb_types:
            tag = f"TSP{coordinates.shape[1]}_dyn_{args.solver_id}_r{int(round(ratio * 100)):02d}_{perturb_type}"
            dataset = output_dir / f"{tag}.pkl"
            payload, metadata = _generate_dataset(
                coordinates,
                distances,
                solver,
                ratio,
                perturb_type,
                int(args.seed),
                float(args.time_limit_sec),
                int(args.ite_max),
                int(args.perturbation_moves),
                float(args.jitter_sigma),
                float(args.node_frac),
            )
            with dataset.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            _add_concorde_costs(repo_root, dataset, output_dir / "concorde", int(args.concorde_scale))
            metadata_path = output_dir / f"{tag}_meta.json"
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest.append({"dataset": str(dataset), "metadata": str(metadata_path)})
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
