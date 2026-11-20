from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from result_utils import save_json, save_text


def _normalize_code(code: str) -> str:
    return "\n".join(line.rstrip() for line in str(code).strip().splitlines())


def _safe_float(value: Any, default: float = 1e18) -> float:
    try:
        out = float(value)
        if not math.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _eval_fields(row: Dict[str, Any], source: str | None = None) -> tuple[Any, Dict[str, Any]]:
    if not isinstance(row, dict):
        return None, {}
    if source == "val":
        obj = row.get("val_objective", row.get("objective"))
        other = row.get("val_other_inf", row.get("other_inf"))
    else:
        obj = row.get("objective")
        other = row.get("other_inf")
    return obj, (other if isinstance(other, dict) else {})


def _is_feasible_row(row: Dict[str, Any], source: str | None = None) -> bool:
    obj, other = _eval_fields(row, source=source)
    if obj is None:
        return False
    try:
        obj_f = float(obj)
    except Exception:
        return False
    if not math.isfinite(obj_f):
        return False
    status = str(other.get("status", "")).strip().lower()
    if status in {"invalid", "ok_with_penalty"}:
        return False
    try:
        n_invalid = int(other.get("n_invalid", 0))
    except Exception:
        n_invalid = 0
    return n_invalid <= 0


def _metric_vector(row: Dict[str, Any], source: str | None = None) -> np.ndarray:
    obj_raw, other = _eval_fields(row, source=source)
    mean_eval = _safe_float(other.get("group_mean_eval"), default=np.nan)
    std_eval = _safe_float(other.get("group_std_eval"), default=np.nan)
    worst_eval = _safe_float(other.get("group_worst_eval"), default=np.nan)
    obj = _safe_float(obj_raw, default=1e18)

    if not np.isfinite(mean_eval):
        mean_eval = obj
    if not np.isfinite(std_eval):
        std_eval = 0.0
    if not np.isfinite(worst_eval):
        worst_eval = mean_eval

    return np.asarray([mean_eval, std_eval, worst_eval], dtype=np.float64)


def _dominates(va: np.ndarray, vb: np.ndarray, eps_rel: float) -> bool:
    eps = float(eps_rel) * np.maximum(1.0, np.abs(vb))
    le_all = bool(np.all(va <= (vb + eps)))
    lt_any = bool(np.any(va < (vb - eps)))
    return le_all and lt_any


def _score_population_dds(
    rows: Sequence[Dict[str, Any]],
    *,
    source: str | None = None,
    eps_rel: float = 1e-3,
    novelty_weight: float = 0.05,
) -> List[Dict[str, Any]]:
    valid = [x for x in rows if _is_feasible_row(x, source=source)]
    if not valid:
        return []

    vectors = np.asarray([_metric_vector(x, source=source) for x in valid], dtype=np.float64)
    n = int(vectors.shape[0])
    dominance_counts = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(vectors[j], vectors[i], eps_rel):
                dominance_counts[i] += 1

    novelty = np.zeros(n, dtype=np.float64)
    if n >= 2:
        mu = np.mean(vectors, axis=0)
        sigma = np.std(vectors, axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        normed = (vectors - mu) / sigma
        for i in range(n):
            d = np.sqrt(np.sum((normed - normed[i]) ** 2, axis=1))
            d[i] = np.inf
            novelty[i] = float(np.min(d))

    scored: List[Dict[str, Any]] = []
    for i, row in enumerate(valid):
        item = dict(row)
        item["_dds_dominance_count"] = int(dominance_counts[i])
        item["_dds_novelty"] = float(novelty[i])
        item["_dds_score"] = float(dominance_counts[i] - float(novelty_weight) * novelty[i])
        item["_sort_obj"] = _safe_float(_eval_fields(row, source=source)[0], default=1e18)
        scored.append(item)

    scored.sort(
        key=lambda x: (
            int(x.get("_dds_dominance_count", 10**9)),
            float(x.get("_dds_score", 1e18)),
            float(x.get("_sort_obj", 1e18)),
        )
    )
    return scored


def build_solver_archive(
    search_trace: Sequence[Dict[str, Any]],
    output_dir: str | Path,
    top_k: int = 5,
    *,
    use_val_dds_order: bool = False,
    dds_eps_rel: float = 1e-3,
    dds_novelty_weight: float = 0.05,
) -> Dict[str, Any]:
    out_dir = Path(output_dir).resolve()
    archive_dir = out_dir / "solver_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for stale in archive_dir.glob("solver_*.py"):
        try:
            stale.unlink()
        except Exception:
            pass

    if use_val_dds_order:
        sorted_rows = _score_population_dds(
            search_trace,
            source="val",
            eps_rel=float(dds_eps_rel),
            novelty_weight=float(dds_novelty_weight),
        )
    else:
        sorted_rows = sorted(search_trace, key=lambda x: _safe_float(x.get("objective"), default=1e18))

    unique_rows: List[Dict[str, Any]] = []
    seen = set()
    for row in sorted_rows:
        code = str(row.get("code", "") or "")
        if not code.strip():
            continue
        norm = _normalize_code(code)
        if norm in seen:
            continue
        seen.add(norm)
        unique_rows.append(row)
        if len(unique_rows) >= int(top_k):
            break

    solvers: List[Dict[str, Any]] = []
    for idx, row in enumerate(unique_rows):
        solver_id = f"solver_{idx}"
        code_path = archive_dir / f"{solver_id}.py"
        save_text(code_path, _normalize_code(str(row.get("code", ""))) + "\n")
        obj, other = _eval_fields(row, source=("val" if use_val_dds_order else None))
        solvers.append(
            {
                "solver_id": solver_id,
                "code_path": str(code_path.resolve()),
                "objective": float(_safe_float(obj, default=1e18)),
                "avg_obj": (
                    float(other.get("avg_obj"))
                    if other.get("avg_obj") is not None
                    else (
                        float(other.get("avg_objective"))
                        if other.get("avg_objective") is not None
                        else None
                    )
                ),
                "selection_basis": ("val_dds_order" if use_val_dds_order else "objective"),
                "style_tag": "unknown",
            }
        )

    archive_path = save_json(
        out_dir / "solver_archive.json",
        {
            "selection_basis": ("val_dds_order" if use_val_dds_order else "objective"),
            "solvers": solvers,
        },
    )
    return {
        "archive_path": str(archive_path.resolve()),
        "n_solvers": len(solvers),
        "n_unique_candidates": len(unique_rows),
        "n_candidates": len(sorted_rows),
    }


def load_solver_archive(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path).resolve()
    with p.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        rows = obj.get("solvers", [])
    elif isinstance(obj, list):
        rows = obj
    else:
        raise ValueError("Unsupported solver archive format.")

    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        raw_code_path = str(item.get("code_path", "") or "")
        candidate_paths: List[Path] = []
        if raw_code_path:
            candidate_paths.append(Path(raw_code_path))
            candidate_paths.append(p.parent / raw_code_path)
            candidate_paths.append(p.parent / "solver_archive" / Path(raw_code_path).name)
        if item.get("solver_id"):
            candidate_paths.append(p.parent / "solver_archive" / f"{item['solver_id']}.py")

        resolved_code_path = None
        for cand in candidate_paths:
            cand_resolved = cand.resolve()
            if cand_resolved.exists():
                resolved_code_path = cand_resolved
                break
        if resolved_code_path is None:
            continue

        item["code_path"] = str(resolved_code_path)
        item["code"] = resolved_code_path.read_text(encoding="utf-8-sig")
        out.append(item)

    if not out:
        raise ValueError(f"No usable solvers in archive: {p}")
    return out
