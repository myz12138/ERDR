from __future__ import annotations

import argparse
import json
import pickle
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _distance_matrix(coords: np.ndarray) -> np.ndarray:
    delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.linalg.norm(delta, axis=2).astype(np.float64, copy=False)


def _nearest_neighbor_tour(dm: np.ndarray) -> np.ndarray:
    n = int(dm.shape[0])
    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=np.int32)
    cur = 0
    tour[0] = 0
    visited[0] = True
    for i in range(1, n):
        row = dm[cur]
        masked = np.where(visited, np.inf, row)
        nxt = int(np.argmin(masked))
        tour[i] = nxt
        visited[nxt] = True
        cur = nxt
    return tour


def _closed_tour_cost(dm: np.ndarray, tour: np.ndarray) -> float:
    idx_from = tour
    idx_to = np.roll(tour, -1)
    return float(np.sum(dm[idx_from, idx_to]))


def _generate_one_dataset(
    *,
    out_path: Path,
    n_nodes: int,
    n_instances: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    coords_all = np.empty((n_instances, n_nodes, 2), dtype=np.float64)
    dms_all = np.empty((n_instances, n_nodes, n_nodes), dtype=np.float64)
    tours_all = np.empty((n_instances, n_nodes), dtype=np.int32)
    costs_all = np.empty((n_instances,), dtype=np.float64)

    tic = time.time()
    for i in range(n_instances):
        coords = rng.random((n_nodes, 2), dtype=np.float64)
        dm = _distance_matrix(coords)
        tour = _nearest_neighbor_tour(dm)
        cost = _closed_tour_cost(dm, tour)

        coords_all[i] = coords
        dms_all[i] = dm
        tours_all[i] = tour
        costs_all[i] = cost

        if (i + 1) % 100 == 0 or (i + 1) == n_instances:
            elapsed = time.time() - tic
            print(
                f"[gen] n_nodes={n_nodes} progress={i + 1}/{n_instances} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    payload = {
        "coordinate": coords_all,
        "optimal_tour": tours_all,
        "distance_matrix": dms_all,
        "cost": costs_all,
    }
    with out_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _parse_sizes(text: str) -> list[int]:
    vals = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("sizes cannot be empty")
    return vals


def _parse_sizes_any(raw: Any) -> list[int]:
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, str):
        return _parse_sizes(raw)
    if isinstance(raw, list):
        return [int(x) for x in raw]
    raise ValueError(f"Unsupported sizes value: {raw!r}")


def _iter_sizes(values: Iterable[int]) -> list[int]:
    out = []
    for v in values:
        n = int(v)
        if n <= 1:
            raise ValueError(f"invalid n_nodes={n}, must be >1")
        out.append(n)
    return out


def _to_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        text = raw.strip().lower()
        return text in {"1", "true", "yes", "y", "on"}
    return bool(raw)


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML is required for YAML config files.") from e
        obj = yaml.safe_load(text)
    if not isinstance(obj, dict):
        raise ValueError(f"Config must parse to mapping: {path}")
    return obj


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"[cmd] {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _run_concorde_labeling(
    *,
    repo_root: Path,
    dataset_pkl: Path,
    output_json: Path,
    output_pkl: Path,
    conc_cfg: dict[str, Any],
) -> None:
    script = repo_root / "data_generate" / "compute_concorde_opt.py"
    if not script.exists():
        raise FileNotFoundError(f"Concorde script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--dataset-path",
        str(dataset_pkl),
        "--output-json",
        str(output_json),
        "--output-pkl",
        str(output_pkl),
        "--concorde-exe",
        str(conc_cfg.get("concorde_exe", "third_party/concorde_cygwin/concorde.exe")),
        "--scale",
        str(int(conc_cfg.get("scale", 1_000_000))),
        "--n-instance",
        str(int(conc_cfg.get("n_instance", 0))),
        "--start-index",
        str(int(conc_cfg.get("start_index", 0))),
        "--seed",
        str(int(conc_cfg.get("seed", 123))),
        "--print-every",
        str(int(conc_cfg.get("print_every", 10))),
    ]
    _run(cmd, cwd=repo_root)


def main() -> None:
    script_path = Path(__file__).resolve()
    default_cfg = script_path.parent / "build_tsp_dataset.yaml"

    parser = argparse.ArgumentParser(
        description=(
            "Generate uniform [0,1]^2 TSP pkl datasets with the same payload keys "
            "as TSP20/TSP50/TSP100."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_cfg),
        help="YAML/JSON config path. Default: data_generate/build_tsp_dataset.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Output directory for generated pkl files.",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="10,20,50,80,120,150,200,500",
        help="Comma-separated node counts.",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=1000,
        help="Number of instances per dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260509,
        help="Base random seed.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="TSP",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_u1000",
        help="Output filename suffix (before .pkl).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dataset-subdir",
        action="store_true",
        help="Store each dataset under its own subdirectory named <prefix><n_nodes><suffix>.",
    )
    parser.add_argument(
        "--keep-raw-pkl",
        action="store_true",
        help="Keep raw <name>.pkl when concorde labeling is enabled.",
    )
    args = parser.parse_args()

    repo_root = script_path.parents[1]

    if str(args.config).strip():
        cfg_path = Path(args.config).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        cfg_dir = cfg_path.parent
        cfg = _load_config(cfg_path)
        paths_cfg = dict(cfg.get("paths", {}))
        gen_cfg = dict(cfg.get("generation", {}))
        conc_cfg = dict(cfg.get("concorde", {}))

        out_dir = Path(str(paths_cfg.get("out_dir", ".")))
        if not out_dir.is_absolute():
            out_dir = (cfg_dir / out_dir).resolve()
        sizes = _iter_sizes(_parse_sizes_any(gen_cfg.get("sizes", [200])))
        n_instances = int(gen_cfg.get("n_instances", 1000))
        seed_base = int(gen_cfg.get("seed", 20260509))
        prefix = str(gen_cfg.get("prefix", "TSP"))
        suffix = str(gen_cfg.get("suffix", "_u1000"))
        overwrite = _to_bool(gen_cfg.get("overwrite", False))
        dataset_subdir = _to_bool(gen_cfg.get("dataset_subdir", True))
        keep_raw_pkl = _to_bool(gen_cfg.get("keep_raw_pkl", False))
        concorde_enabled = _to_bool(conc_cfg.get("enabled", True))
    else:
        out_dir = Path(args.out_dir).resolve()
        sizes = _iter_sizes(_parse_sizes(args.sizes))
        n_instances = int(args.n_instances)
        seed_base = int(args.seed)
        prefix = str(args.prefix)
        suffix = str(args.suffix)
        overwrite = bool(args.overwrite)
        dataset_subdir = bool(args.dataset_subdir)
        keep_raw_pkl = bool(args.keep_raw_pkl)
        concorde_enabled = False
        conc_cfg = {}

    out_dir.mkdir(parents=True, exist_ok=True)
    if n_instances <= 0:
        raise ValueError("n-instances must be positive")

    for idx, n_nodes in enumerate(sizes):
        dataset_tag = f"{prefix}{n_nodes}{suffix}"
        dataset_dir = (out_dir / dataset_tag) if dataset_subdir else out_dir
        dataset_dir.mkdir(parents=True, exist_ok=True)

        raw_pkl = dataset_dir / f"{dataset_tag}.pkl"
        concorde_json = dataset_dir / f"{dataset_tag}_concorde_full.json"
        concorde_pkl = dataset_dir / f"{dataset_tag}_concorde.pkl"

        if concorde_enabled and (not overwrite) and concorde_pkl.exists():
            print(f"[skip] concorde exists: {concorde_pkl}", flush=True)
            continue

        if raw_pkl.exists() and (not overwrite):
            print(f"[skip] raw exists: {raw_pkl}", flush=True)
        else:
            seed = int(seed_base + idx * 100003 + n_nodes * 17)
            print(
                f"[start] n_nodes={n_nodes} n_instances={n_instances} seed={seed} "
                f"-> {raw_pkl}",
                flush=True,
            )
            _generate_one_dataset(
                out_path=raw_pkl,
                n_nodes=n_nodes,
                n_instances=n_instances,
                seed=seed,
            )
            size_mb = raw_pkl.stat().st_size / (1024 * 1024)
            print(f"[done] {raw_pkl.name} size={size_mb:.2f} MB", flush=True)

        if concorde_enabled:
            print(
                f"[concorde] {raw_pkl.name} -> {concorde_pkl.name}",
                flush=True,
            )
            _run_concorde_labeling(
                repo_root=repo_root,
                dataset_pkl=raw_pkl,
                output_json=concorde_json,
                output_pkl=concorde_pkl,
                conc_cfg=conc_cfg,
            )
            if (not keep_raw_pkl) and raw_pkl.exists():
                raw_pkl.unlink()
                print(f"[cleanup] removed raw pkl: {raw_pkl}", flush=True)


if __name__ == "__main__":
    main()
