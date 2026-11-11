from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import pickle
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gen_inst import generate_datasets, load_dataset
from gls import guided_local_search, warmup_numba_kernels


LARGE = 1e12
_HEURISTIC_FN_NAMES = ("heuristics_v2", "heuristics_v1", "heuristics")


class _DistOnlyInstance:
    def __init__(self, distmat: np.ndarray) -> None:
        d = np.asarray(distmat, dtype=np.float32)
        if d.ndim != 2 or d.shape[0] != d.shape[1]:
            raise ValueError(f"invalid distance matrix shape: {d.shape}")
        self.distmat = d
        self.n = int(d.shape[0])


class _FallbackKMeans:
    def __init__(
        self,
        n_clusters: int = 8,
        *,
        random_state: int | None = None,
        max_iter: int = 50,
        n_init: int = 1,
        **_: Any,
    ) -> None:
        self.n_clusters = int(max(n_clusters, 1))
        self.random_state = random_state
        self.max_iter = int(max(max_iter, 1))
        self.n_init = int(max(n_init, 1))
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def _fit_once(self, x: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
        n_samples = int(x.shape[0])
        n_clusters = int(max(1, min(self.n_clusters, n_samples)))
        rng = np.random.default_rng(seed)
        init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centers = x[init_idx].copy()

        for _ in range(self.max_iter):
            dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(dist2, axis=1).astype(np.int32)
            new_centers = centers.copy()
            for k in range(n_clusters):
                members = x[labels == k]
                if members.size > 0:
                    new_centers[k] = members.mean(axis=0)
            if np.allclose(new_centers, centers, rtol=1.0e-6, atol=1.0e-6):
                centers = new_centers
                break
            centers = new_centers

        dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dist2, axis=1).astype(np.int32)
        inertia = float(dist2[np.arange(n_samples), labels].sum())
        return centers, labels, inertia

    def fit(self, x: np.ndarray) -> "_FallbackKMeans":
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"KMeans expects 2D input, got {arr.shape}")
        best_centers = None
        best_labels = None
        best_inertia = None
        base_seed = 0 if self.random_state is None else int(self.random_state)
        for offset in range(self.n_init):
            centers, labels, inertia = self._fit_once(arr, base_seed + offset)
            if best_inertia is None or inertia < best_inertia:
                best_centers = centers
                best_labels = labels
                best_inertia = inertia
        self.cluster_centers_ = np.asarray(best_centers, dtype=np.float64)
        self.labels_ = np.asarray(best_labels, dtype=np.int32)
        return self

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        assert self.labels_ is not None
        return self.labels_


def _install_sklearn_kmeans_stub() -> Dict[str, Any]:
    prev = {
        "sklearn": sys.modules.get("sklearn"),
        "sklearn.cluster": sys.modules.get("sklearn.cluster"),
    }
    sklearn_mod = types.ModuleType("sklearn")
    cluster_mod = types.ModuleType("sklearn.cluster")
    cluster_mod.KMeans = _FallbackKMeans
    sklearn_mod.cluster = cluster_mod
    sys.modules["sklearn"] = sklearn_mod
    sys.modules["sklearn.cluster"] = cluster_mod
    return prev


def _restore_sklearn_modules(prev: Dict[str, Any]) -> None:
    for name, mod in prev.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def _import_candidate_module(code_path: Path, module_name: str) -> Any:
    def _load_once() -> Any:
        spec = importlib.util.spec_from_file_location(module_name, str(code_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to import candidate code: {code_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        source = code_path.read_text(encoding="utf-8")
    except Exception:
        source = ""
    force_stub = "from sklearn.cluster import KMeans" in source
    if force_stub:
        prev = _install_sklearn_kmeans_stub()
        try:
            return _load_once()
        finally:
            _restore_sklearn_modules(prev)

    try:
        return _load_once()
    except Exception:  # noqa: BLE001
        prev = _install_sklearn_kmeans_stub()
        try:
            return _load_once()
        finally:
            _restore_sklearn_modules(prev)


def _parse_bool_flag(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _maybe_jit_decorate_heuristics(
    heuristics_fn: Callable[[np.ndarray], np.ndarray],
    enabled: bool,
) -> tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "jit_guide_decorator_enabled": bool(enabled),
        "jit_guide_decorator_active": False,
        "jit_guide_decorator_error": None,
    }
    if not enabled:
        return heuristics_fn, info
    try:
        import numba as nb  # local import to avoid hard dependency in non-JIT runs

        wrapped = nb.njit(cache=True)(heuristics_fn)
        probe = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        _ = np.asarray(wrapped(probe), dtype=np.float32)
        info["jit_guide_decorator_active"] = True
        return wrapped, info
    except Exception as e:  # noqa: BLE001
        info["jit_guide_decorator_error"] = str(e)
        return heuristics_fn, info


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _nearest_neighbor_cost(distmat: np.ndarray) -> float:
    n = int(distmat.shape[0])
    if n <= 1:
        return 0.0
    visited = np.zeros(n, dtype=bool)
    cur = 0
    visited[cur] = True
    total = 0.0
    for _ in range(n - 1):
        candidates = np.where(~visited)[0]
        nxt = int(candidates[np.argmin(distmat[cur, candidates])])
        total += float(distmat[cur, nxt])
        visited[nxt] = True
        cur = nxt
    total += float(distmat[cur, 0])
    return float(total)


def _calculate_cost(distmat: np.ndarray, path: np.ndarray) -> float:
    return float(distmat[path, np.roll(path, 1)].sum().item())


def _load_concorde_sidecar_costs(data_path: Path, expected_len: int) -> np.ndarray | None:
    """
    Sidecar format:
    <dataset_dir>/<dataset_stem>_concorde_full.json
    <dataset_dir>/concorde/<dataset_stem>_concorde_full.json
    """
    sidecar_name = f"{data_path.stem}_concorde_full.json"
    candidates = [
        data_path.with_name(sidecar_name),
        data_path.parent / "concorde" / sidecar_name,
    ]
    seen: set[str] = set()
    for sidecar in candidates:
        key = str(sidecar.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        if not sidecar.exists():
            continue
        try:
            obj = json.loads(sidecar.read_text(encoding="utf-8"))
            recs = obj.get("records", None)
            if not isinstance(recs, list) or len(recs) < int(expected_len):
                continue
            out: list[float] = []
            for i in range(int(expected_len)):
                row = recs[i]
                if not isinstance(row, dict):
                    out = []
                    break
                if str(row.get("status", "")) != "ok":
                    out = []
                    break
                val = float(row.get("opt_float_from_tour"))
                if (not np.isfinite(val)) or val <= 0.0:
                    out = []
                    break
                out.append(float(val))
            if len(out) == int(expected_len):
                return np.asarray(out, dtype=float)
        except Exception:
            continue
    return None


def _pick_heuristic_function(module_obj: Any) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    for name in _HEURISTIC_FN_NAMES:
        candidate = getattr(module_obj, name, None)
        if callable(candidate):
            return candidate, str(name)
    raise RuntimeError("heuristics_v2 / heuristics_v1 / heuristics not found in candidate code.")


def _load_heuristic_function_raw(code_path: Path) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    if not code_path.exists():
        raise FileNotFoundError(f"code file not found: {code_path}")
    module = _import_candidate_module(code_path, "reevo_candidate")
    return _pick_heuristic_function(module)


class _GuideSanitizeTransformer(ast.NodeTransformer):
    def __init__(self, target_fn_name: str) -> None:
        self.target_fn_name = str(target_fn_name)
        self._in_target = 0
        self.changes: List[str] = []
        self.removed_imports: List[str] = []

    def _record(self, tag: str) -> None:
        if tag not in self.changes:
            self.changes.append(tag)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # noqa: ANN401
        if node.name != self.target_fn_name:
            return self.generic_visit(node)
        self._in_target += 1
        new_node = self.generic_visit(node)
        self._in_target -= 1
        return new_node

    def visit_Import(self, node: ast.Import) -> Any:  # noqa: ANN401
        if self._in_target <= 0:
            return node
        names = ", ".join(alias.name for alias in node.names)
        self.removed_imports.append(names)
        self._record("remove_function_level_import")
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:  # noqa: ANN401
        if self._in_target <= 0:
            return node
        mod = str(node.module or "")
        names = ", ".join(alias.name for alias in node.names)
        self.removed_imports.append(f"from {mod} import {names}")
        self._record("remove_function_level_import")
        return None

    def visit_Call(self, node: ast.Call) -> Any:  # noqa: ANN401
        node = self.generic_visit(node)
        if self._in_target <= 0:
            return node

        # Normalize len(distance_matrix) -> distance_matrix.shape[0]
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "distance_matrix"
        ):
            self._record("normalize_len_distance_matrix")
            return ast.Subscript(
                value=ast.Attribute(value=ast.Name(id="distance_matrix", ctx=ast.Load()), attr="shape", ctx=ast.Load()),
                slice=ast.Constant(value=0),
                ctx=ast.Load(),
            )

        # Normalize dtype=float/int for common numpy constructors.
        for kw in node.keywords:
            if kw.arg != "dtype":
                continue
            if isinstance(kw.value, ast.Name) and kw.value.id == "float":
                kw.value = ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="float32", ctx=ast.Load())
                self._record("normalize_dtype_float32")
            elif isinstance(kw.value, ast.Name) and kw.value.id == "int":
                kw.value = ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="int32", ctx=ast.Load())
                self._record("normalize_dtype_int32")
        return node


def _prepare_module_with_numpy_import(tree: ast.Module) -> tuple[ast.Module, List[str], bool]:
    changed = False
    removed_imports: List[str] = []
    kept_body: List[ast.stmt] = []
    has_np_import = False

    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            local_has_np = False
            for alias in stmt.names:
                if alias.name == "numpy":
                    local_has_np = True
                else:
                    removed_imports.append(f"import {alias.name}")
                    changed = True
            if local_has_np:
                has_np_import = True
                changed = True
            continue

        if isinstance(stmt, ast.ImportFrom):
            mod = str(stmt.module or "")
            if mod == "numpy":
                has_np_import = True
                changed = True
            else:
                names = ", ".join(alias.name for alias in stmt.names)
                removed_imports.append(f"from {mod} import {names}")
                changed = True
            continue

        kept_body.append(stmt)

    np_import_stmt = ast.Import(names=[ast.alias(name="numpy", asname="np")])
    tree.body = [np_import_stmt] + kept_body
    if not has_np_import:
        changed = True
    return tree, removed_imports, changed


def _build_function_from_module_ast(
    tree: ast.Module,
    code_path: Path,
    preferred_names: Sequence[str],
) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    ast.fix_missing_locations(tree)
    compiled = compile(tree, filename=str(code_path), mode="exec")
    namespace: Dict[str, Any] = {"np": np, "KMeans": _FallbackKMeans}
    exec(compiled, namespace, namespace)

    for name in preferred_names:
        candidate = namespace.get(name)
        if callable(candidate):
            return candidate, str(name)
    raise RuntimeError("sanitized module does not expose heuristics_v2 / heuristics_v1 / heuristics.")


def _probe_heuristic_runtime(
    heuristics_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[bool, Optional[str]]:
    try:
        probe = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        out = np.asarray(heuristics_fn(probe.copy()), dtype=np.float32)
        if out.shape != probe.shape:
            return False, f"guide runtime probe shape mismatch: got {out.shape}, expected {probe.shape}"
        if not np.isfinite(out).all():
            return False, "guide runtime probe produced non-finite values."
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _probe_heuristic_numba_compile(
    heuristics_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[bool, Optional[str]]:
    try:
        import numba as nb
    except Exception as e:  # noqa: BLE001
        return False, f"numba import failed: {e}"

    try:
        wrapped = nb.njit(cache=True)(heuristics_fn)
        probe = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        out = np.asarray(wrapped(probe.copy()), dtype=np.float32)
        if out.shape != probe.shape:
            return False, f"numba probe shape mismatch: got {out.shape}, expected {probe.shape}"
        if not np.isfinite(out).all():
            return False, "numba probe produced non-finite values."
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _sanitize_and_load_heuristic_function(
    code_path: Path,
    *,
    enabled: bool,
    strict: bool,
) -> tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "jit_guide_sanitize_enabled": bool(enabled),
        "jit_guide_sanitize_strict": bool(strict),
        "jit_guide_sanitize_active": False,
        "jit_guide_sanitize_error": None,
        "jit_guide_sanitize_changes": [],
        "jit_guide_compile_probe_ok": False,
        "jit_guide_compile_probe_error": None,
    }
    if not enabled:
        fn, _ = _load_heuristic_function_raw(code_path)
        return fn, info

    try:
        source = code_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(code_path))

        # Determine target heuristic function in source order preference.
        target_name = None
        for name in _HEURISTIC_FN_NAMES:
            for stmt in tree.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == name:
                    target_name = name
                    break
            if target_name is not None:
                break
        if target_name is None:
            raise RuntimeError("heuristics_v2 / heuristics_v1 / heuristics not found in candidate source.")

        tree, removed_module_imports, module_changed = _prepare_module_with_numpy_import(tree)
        transformer = _GuideSanitizeTransformer(target_fn_name=target_name)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)

        changes: List[str] = []
        if module_changed:
            changes.append("normalize_module_imports_to_numpy_only")
        if removed_module_imports:
            changes.append("remove_non_numpy_module_imports")
        changes.extend(transformer.changes)
        info["jit_guide_sanitize_changes"] = changes

        fn, _resolved_name = _build_function_from_module_ast(tree, code_path=code_path, preferred_names=_HEURISTIC_FN_NAMES)
        runtime_ok, runtime_err = _probe_heuristic_runtime(fn)
        if not runtime_ok:
            raise RuntimeError(f"sanitized runtime probe failed: {runtime_err}")

        compile_ok, compile_err = _probe_heuristic_numba_compile(fn)
        info["jit_guide_compile_probe_ok"] = bool(compile_ok)
        info["jit_guide_compile_probe_error"] = compile_err
        info["jit_guide_sanitize_active"] = True

        if strict and not compile_ok:
            raise RuntimeError(f"sanitized numba compile probe failed: {compile_err}")

        return fn, info
    except Exception as e:  # noqa: BLE001
        info["jit_guide_sanitize_error"] = str(e)
        if strict:
            raise
        # Fallback to original candidate function when sanitization is best-effort.
        fn, _ = _load_heuristic_function_raw(code_path)
        return fn, info


def _load_dataset_with_ref(
    dataset_path: Path,
    n_instance_eval: int = 0,
) -> Tuple[List[_DistOnlyInstance], np.ndarray, str]:
    suffix = dataset_path.suffix.lower()
    if suffix == ".npy":
        raise ValueError(
            "Strict Concorde mode does not support .npy datasets for tsp_gls adapter evaluation. "
            "Use .pkl dataset with `cost_concorde` or sidecar concorde json."
        )

    if suffix == ".pkl":
        with open(dataset_path, "rb") as f:
            packed = pickle.load(f)
        if not isinstance(packed, dict) or "distance_matrix" not in packed:
            raise ValueError(f"unsupported pkl dataset format: {dataset_path}")
        distance_matrices = np.asarray(packed["distance_matrix"])
        if distance_matrices.ndim != 3:
            raise ValueError(f"invalid distance_matrix shape in pkl: {distance_matrices.shape}")
        n_total = int(distance_matrices.shape[0])
        n_eval = n_total if int(n_instance_eval) <= 0 else min(max(int(n_instance_eval), 1), n_total)
        distance_matrices = distance_matrices[:n_eval]
        dataset = [_DistOnlyInstance(dist) for dist in distance_matrices]

        refs: np.ndarray | None = None
        ref_source = "concorde_missing"

        costs_raw = packed.get("cost_concorde", packed.get("reference_cost"))
        if costs_raw is not None:
            arr = np.asarray(costs_raw, dtype=float).reshape(-1)
            if arr.size >= n_eval:
                refs = np.asarray(arr[:n_eval], dtype=float)
                ref_source = "concorde_embedded"

        if refs is None:
            sidecar = _load_concorde_sidecar_costs(dataset_path, expected_len=n_eval)
            if sidecar is not None:
                refs = np.asarray(sidecar, dtype=float)
                ref_source = "concorde_sidecar"

        if refs is None:
            project_name = Path(__file__).resolve().parents[2].name
            raise ValueError(
                "Concorde reference costs are required but missing. "
                "Provide `cost_concorde` in dataset or sidecar "
                f"`{project_name}/outputs/concorde_opt/{dataset_path.stem}_concorde_full.json`."
            )

        bad = np.where((~np.isfinite(refs)) | (refs <= 0.0))[0]
        if bad.size > 0:
            preview = ",".join(str(int(i)) for i in bad[:10])
            raise ValueError(
                f"Invalid concorde reference costs: {int(bad.size)}/{int(n_eval)} bad items "
                f"(first indices: {preview})."
            )
        return dataset, refs, ref_source

    raise ValueError(f"unsupported dataset suffix: {dataset_path.suffix}")


def _population_std(values: Sequence[float]) -> float:
    arr = np.asarray([float(x) for x in values], dtype=float)
    if arr.size <= 0:
        return 0.0
    mean_v = float(np.mean(arr))
    sq_mean = float(np.mean((arr - mean_v) ** 2))
    return float(np.sqrt(max(sq_mean, 0.0)))


def _worst_half_mean(values: Sequence[float]) -> tuple[float, int]:
    arr = np.asarray([float(x) for x in values], dtype=float)
    n = int(arr.size)
    if n <= 0:
        return 0.0, 0
    m = max(1, n // 2)
    sorted_desc = np.sort(arr)[::-1]
    return float(np.mean(sorted_desc[:m])), int(m)


def _risk_weighted_metric(
    mean_eval: float,
    std_eval: float,
    worst_eval: float,
    w_mean: float,
    w_std: float,
    w_worst: float,
) -> float:
    return float(
        float(w_mean) * float(mean_eval)
        + float(w_std) * float(std_eval)
        + float(w_worst) * float(worst_eval)
    )


def _parse_group_indices(raw: str, n_eval: int) -> List[List[int]]:
    if not str(raw or "").strip():
        return [list(range(n_eval))]
    try:
        obj = json.loads(raw)
    except Exception as e:
        raise ValueError(f"invalid group_indices_json: {e}") from e
    if not isinstance(obj, list):
        raise ValueError("group_indices_json must decode to a list.")
    groups: List[List[int]] = []
    for gid, group in enumerate(obj):
        if not isinstance(group, list):
            raise ValueError(f"group {gid} is not a list.")
        cleaned: List[int] = []
        for idx in group:
            i = int(idx)
            if i < 0 or i >= n_eval:
                raise ValueError(f"group index out of range: {i}, n_eval={n_eval}")
            if i not in cleaned:
                cleaned.append(i)
        if cleaned:
            groups.append(cleaned)
    if not groups:
        return [list(range(n_eval))]

    flat = sorted(i for g in groups for i in g)
    expected = list(range(n_eval))
    if flat != expected:
        raise ValueError(
            "group_indices_json must form a full partition of [0, n_eval). "
            f"expected {len(expected)} indices, got {len(flat)}."
        )
    return groups


def _invalid_result(
    error: str,
    n_eval: int,
    metric_basis: str,
    subgroup_mode: str,
    subgroup_tail_basis: str = "time",
    *,
    jit_warmup_enabled: bool = False,
    jit_warmup_active: bool = False,
    jit_guide_decorator_enabled: bool = False,
    jit_guide_decorator_active: bool = False,
    jit_guide_decorator_error: Optional[str] = None,
    jit_guide_sanitize_enabled: bool = False,
    jit_guide_sanitize_strict: bool = False,
    jit_guide_sanitize_active: bool = False,
    jit_guide_sanitize_error: Optional[str] = None,
    jit_guide_sanitize_changes: Optional[Sequence[str]] = None,
    jit_guide_compile_probe_ok: bool = False,
    jit_guide_compile_probe_error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": "invalid",
        "error": str(error),
        "selected_metric": float(LARGE),
        "selected_metric_name": ("mean_gap" if metric_basis == "gap" else "mean_cost"),
        "mean_cost": float(LARGE),
        "mean_gap": float(LARGE),
        "mean_gap_pct": float(LARGE * 100.0),
        "mean_time_sec": 0.0,
        "max_time_sec": 0.0,
        "p95_time_sec": 0.0,
        "n_time_exceed": int(n_eval),
        "time_exceed_ratio": 1.0 if int(n_eval) > 0 else 0.0,
        "n_eval": int(n_eval),
        "n_invalid": int(n_eval),
        "metric_basis": str(metric_basis),
        "subgroup_eval_mode": str(subgroup_mode),
        "subgroup_tail_basis": str(subgroup_tail_basis),
        "group_mean_eval": float(LARGE),
        "group_std_eval": float(LARGE),
        "group_worst_eval": float(LARGE),
        "objective_vector": [float(LARGE), float(LARGE), float(LARGE)],
        "jit_warmup_enabled": bool(jit_warmup_enabled),
        "jit_warmup_active": bool(jit_warmup_active),
        "jit_guide_decorator_enabled": bool(jit_guide_decorator_enabled),
        "jit_guide_decorator_active": bool(jit_guide_decorator_active),
        "jit_guide_decorator_error": jit_guide_decorator_error,
        "jit_guide_sanitize_enabled": bool(jit_guide_sanitize_enabled),
        "jit_guide_sanitize_strict": bool(jit_guide_sanitize_strict),
        "jit_guide_sanitize_active": bool(jit_guide_sanitize_active),
        "jit_guide_sanitize_error": jit_guide_sanitize_error,
        "jit_guide_sanitize_changes": [str(x) for x in (jit_guide_sanitize_changes or [])],
        "jit_guide_compile_probe_ok": bool(jit_guide_compile_probe_ok),
        "jit_guide_compile_probe_error": jit_guide_compile_probe_error,
    }


def evaluate_candidate(
    *,
    dataset: Sequence[_DistOnlyInstance],
    references: np.ndarray,
    heuristics_fn: Callable[[np.ndarray], np.ndarray],
    metric_basis: str,
    subgroup_mode: str,
    subgroup_tail_basis: str,
    group_indices: Sequence[Sequence[int]],
    perturbation_moves: int,
    iter_limit: int,
    time_limit_per_instance: float,
    gap_eps: float,
    risk_weight_mean: float,
    risk_weight_std: float,
    risk_weight_worst: float,
    reference_cost_source: str,
    jit_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    n_eval = int(len(dataset))
    t_cap = float(time_limit_per_instance)
    if (not np.isfinite(t_cap)) or t_cap <= 0.0:
        raise ValueError(
            f"time_limit_per_instance must be > 0 for strict per-instance timing, got {time_limit_per_instance}"
        )

    refs = np.asarray(references, dtype=float).reshape(-1)
    if refs.size < n_eval:
        raise ValueError(
            f"Concorde reference costs are incomplete: expected {n_eval}, got {refs.size}."
        )
    eps = float(max(gap_eps, 1.0e-12))
    bad_ref = np.where((~np.isfinite(refs[:n_eval])) | (refs[:n_eval] <= eps))[0]
    if bad_ref.size > 0:
        preview = ",".join(str(int(i)) for i in bad_ref[:10])
        raise ValueError(
            "Concorde reference costs are required for gap evaluation. "
            f"Missing/invalid {int(bad_ref.size)}/{n_eval} instances "
            f"(first indices: {preview})."
        )

    costs: List[float] = []
    gaps: List[float] = []
    times: List[float] = []
    n_invalid = 0

    for idx, inst in enumerate(dataset):
        inst_t0 = time.perf_counter()
        deadline = inst_t0 + t_cap
        try:
            dist = np.asarray(inst.distmat, dtype=np.float32)
            guide = heuristics_fn(dist.copy())
            guide_arr = np.asarray(guide, dtype=np.float32)
            if guide_arr.shape != dist.shape:
                raise ValueError(f"invalid heuristic matrix shape: {guide_arr.shape}, expected {dist.shape}")

            # Per-instance runtime covers heuristic generation + GLS + post checks.
            tour = guided_local_search(
                dist,
                guide_arr,
                perturbation_moves=max(int(perturbation_moves), 1),
                iter_limit=max(int(iter_limit), 1),
                time_limit_sec=t_cap,
                deadline_perf_counter=deadline,
            )
            obj = _calculate_cost(dist, np.asarray(tour, dtype=np.int64))
            ref = float(max(refs[idx], eps))
            raw_gap = float((obj - ref) / ref)
            if not np.isfinite(obj) or not np.isfinite(raw_gap):
                raise ValueError("non-finite objective or gap")
            # Avoid tiny negative floating drift around the Concorde baseline.
            gap_tol = float(max(gap_eps, 1.0e-7))
            gap = 0.0 if (raw_gap < 0.0 and abs(raw_gap) <= gap_tol) else float(raw_gap)

            costs.append(float(obj))
            gaps.append(float(gap))
            times.append(float(time.perf_counter() - inst_t0))
        except Exception:
            n_invalid += 1
            costs.append(float(LARGE))
            gaps.append(float(LARGE))
            # Keep a finite runtime value for failed cases as well.
            times.append(float(time.perf_counter() - inst_t0))

    mean_cost = float(np.mean(np.asarray(costs, dtype=float)))
    mean_gap = float(np.mean(np.asarray(gaps, dtype=float)))
    arr_times = np.asarray(times, dtype=float)
    mean_time = float(np.mean(arr_times))
    max_time = float(np.max(arr_times)) if arr_times.size > 0 else 0.0
    p95_time = float(np.percentile(arr_times, 95.0)) if arr_times.size > 0 else 0.0
    n_time_exceed = int(np.sum(arr_times > (float(t_cap) + 1.0e-9))) if arr_times.size > 0 else 0
    time_exceed_ratio = float(n_time_exceed / max(int(arr_times.size), 1))

    group_means_cost: List[float] = []
    group_means_gap: List[float] = []
    group_means_time_sec: List[float] = []
    arr_cost = np.asarray(costs, dtype=float)
    arr_gap = np.asarray(gaps, dtype=float)
    for idxs in group_indices:
        idx_arr = np.asarray([int(i) for i in idxs], dtype=int)
        if idx_arr.size <= 0:
            continue
        group_means_cost.append(float(np.mean(arr_cost[idx_arr])))
        group_means_gap.append(float(np.mean(arr_gap[idx_arr])))
        group_means_time_sec.append(float(np.mean(arr_times[idx_arr])))
    if not group_means_cost:
        group_means_cost = [mean_cost]
    if not group_means_gap:
        group_means_gap = [mean_gap]
    if not group_means_time_sec:
        group_means_time_sec = [mean_time]

    group_mean_time_sec = float(np.mean(group_means_time_sec))
    group_std_time_sec = float(_population_std(group_means_time_sec))
    group_worst_time_sec, _ = _worst_half_mean(group_means_time_sec)

    if metric_basis == "gap":
        group_mean_eval = float(np.mean(group_means_gap))
        tail_basis_effective = str(subgroup_tail_basis)
        if tail_basis_effective == "time":
            group_std_eval_raw = float(group_std_time_sec)
            group_worst_eval_raw = float(group_worst_time_sec)
        else:
            group_std_eval_raw = float(_population_std(group_means_gap))
            group_worst_eval_raw, _ = _worst_half_mean(group_means_gap)
        avg_eval = float(mean_gap)
        selected_metric_name = "mean_gap"
    else:
        group_mean_eval = float(np.mean(group_means_cost))
        group_std_eval_raw = float(_population_std(group_means_cost))
        group_worst_eval_raw, _ = _worst_half_mean(group_means_cost)
        tail_basis_effective = "cost"
        avg_eval = float(mean_cost)
        selected_metric_name = "mean_cost"

    if subgroup_mode == "mean_std_worst":
        # Align with EOH7 risk2: optimize mean as scalar metric,
        # while exposing (mean, std, worst) as objective vector.
        group_std_eval = float(group_std_eval_raw)
        group_worst_eval = float(group_worst_eval_raw)
        selected_metric = float(group_mean_eval)
    else:
        group_std_eval = 0.0
        group_worst_eval = float(group_mean_eval)
        selected_metric = float(avg_eval)

    return {
        "status": "ok",
        "error": (None if n_invalid <= 0 else f"n_invalid={n_invalid}"),
        "selected_metric": float(selected_metric),
        "selected_metric_name": str(selected_metric_name),
        "mean_cost": float(mean_cost),
        "mean_gap": float(mean_gap),
        "mean_gap_pct": float(mean_gap * 100.0),
        "mean_time_sec": float(mean_time),
        "max_time_sec": float(max_time),
        "p95_time_sec": float(p95_time),
        "n_time_exceed": int(n_time_exceed),
        "time_exceed_ratio": float(time_exceed_ratio),
        "n_eval": int(n_eval),
        "n_invalid": int(n_invalid),
        "metric_basis": str(metric_basis),
        "subgroup_eval_mode": str(subgroup_mode),
        "subgroup_tail_basis": str(tail_basis_effective),
        "group_mean_eval": float(group_mean_eval),
        "group_std_eval": float(group_std_eval),
        "group_worst_eval": float(group_worst_eval),
        "group_means_cost": [float(x) for x in group_means_cost],
        "group_means_gap": [float(x) for x in group_means_gap],
        "group_means_time_sec": [float(x) for x in group_means_time_sec],
        "group_mean_time_sec": float(group_mean_time_sec),
        "group_std_time_sec": float(group_std_time_sec),
        "group_worst_time_sec": float(group_worst_time_sec),
        "objective_vector": [
            float(group_mean_eval),
            float(group_std_eval),
            float(group_worst_eval),
        ],
        "risk_objective_weights": {
            "mean": float(risk_weight_mean),
            "std": float(risk_weight_std),
            "worst": float(risk_weight_worst),
        },
        "time_limit_per_instance": float(t_cap),
        "reference_cost_source": str(reference_cost_source),
        "n_reference_fallback": 0,
        "jit_warmup_enabled": bool((jit_meta or {}).get("jit_warmup_enabled", False)),
        "jit_warmup_active": bool((jit_meta or {}).get("jit_warmup_active", False)),
        "jit_guide_decorator_enabled": bool((jit_meta or {}).get("jit_guide_decorator_enabled", False)),
        "jit_guide_decorator_active": bool((jit_meta or {}).get("jit_guide_decorator_active", False)),
        "jit_guide_decorator_error": (jit_meta or {}).get("jit_guide_decorator_error"),
        "jit_guide_sanitize_enabled": bool((jit_meta or {}).get("jit_guide_sanitize_enabled", False)),
        "jit_guide_sanitize_strict": bool((jit_meta or {}).get("jit_guide_sanitize_strict", False)),
        "jit_guide_sanitize_active": bool((jit_meta or {}).get("jit_guide_sanitize_active", False)),
        "jit_guide_sanitize_error": (jit_meta or {}).get("jit_guide_sanitize_error"),
        "jit_guide_sanitize_changes": [
            str(x) for x in ((jit_meta or {}).get("jit_guide_sanitize_changes", []) or [])
        ],
        "jit_guide_compile_probe_ok": bool((jit_meta or {}).get("jit_guide_compile_probe_ok", False)),
        "jit_guide_compile_probe_error": (jit_meta or {}).get("jit_guide_compile_probe_error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one ReEvo tsp_gls heuristic on grouped dataset.")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--metric-basis", type=str, choices=["gap", "cost"], default="gap")
    parser.add_argument(
        "--subgroup-mode",
        type=str,
        choices=["mean_only", "mean_std_worst"],
        default="mean_only",
    )
    parser.add_argument("--subgroup-tail-basis", type=str, choices=["gap", "time"], default="time")
    parser.add_argument("--group-indices-json", type=str, default="")
    parser.add_argument("--n-instance-eval", type=int, default=0)
    parser.add_argument("--perturbation-moves", type=int, default=30)
    parser.add_argument("--iter-limit", type=int, default=1200)
    parser.add_argument("--time-limit-per-instance", type=float, default=0.0)
    parser.add_argument("--jit-warmup", type=str, default="1")
    parser.add_argument("--jit-decorate-guide", type=str, default="0")
    parser.add_argument("--jit-sanitize-guide", type=str, default="1")
    parser.add_argument("--jit-sanitize-strict", type=str, default="0")
    parser.add_argument("--gap-eps", type=float, default=1.0e-12)
    parser.add_argument("--risk-weight-mean", type=float, default=1.0)
    parser.add_argument("--risk-weight-std", type=float, default=0.3)
    parser.add_argument("--risk-weight-worst", type=float, default=0.2)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    code_path = Path(args.code_path).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if (not dataset_path.exists()) and dataset_path.suffix.lower() == ".npy":
        generate_datasets(basepath=str(dataset_path.parent))

    jit_warmup_enabled = _parse_bool_flag(args.jit_warmup, default=True)
    jit_guide_decorator_enabled = _parse_bool_flag(args.jit_decorate_guide, default=False)
    jit_guide_sanitize_enabled = _parse_bool_flag(args.jit_sanitize_guide, default=True)
    jit_guide_sanitize_strict = _parse_bool_flag(args.jit_sanitize_strict, default=False)
    jit_meta: Dict[str, Any] = {
        "jit_warmup_enabled": bool(jit_warmup_enabled),
        "jit_warmup_active": False,
        "jit_guide_decorator_enabled": bool(jit_guide_decorator_enabled),
        "jit_guide_decorator_active": False,
        "jit_guide_decorator_error": None,
        "jit_guide_sanitize_enabled": bool(jit_guide_sanitize_enabled),
        "jit_guide_sanitize_strict": bool(jit_guide_sanitize_strict),
        "jit_guide_sanitize_active": False,
        "jit_guide_sanitize_error": None,
        "jit_guide_sanitize_changes": [],
        "jit_guide_compile_probe_ok": False,
        "jit_guide_compile_probe_error": None,
    }

    try:
        if jit_warmup_enabled:
            warmup_numba_kernels()
            jit_meta["jit_warmup_active"] = True

        dataset, refs, ref_source = _load_dataset_with_ref(
            dataset_path=dataset_path,
            n_instance_eval=int(args.n_instance_eval),
        )
        if len(dataset) <= 0:
            raise RuntimeError(f"empty dataset: {dataset_path}")
        group_indices = _parse_group_indices(args.group_indices_json, n_eval=len(dataset))
        heuristics_fn, sanitize_info = _sanitize_and_load_heuristic_function(
            code_path,
            enabled=jit_guide_sanitize_enabled,
            strict=jit_guide_sanitize_strict,
        )
        jit_meta.update(sanitize_info)
        heuristics_fn, decorate_info = _maybe_jit_decorate_heuristics(
            heuristics_fn=heuristics_fn,
            enabled=jit_guide_decorator_enabled,
        )
        jit_meta.update(decorate_info)
        result = evaluate_candidate(
            dataset=dataset,
            references=np.asarray(refs, dtype=float),
            heuristics_fn=heuristics_fn,
            metric_basis=str(args.metric_basis),
            subgroup_mode=str(args.subgroup_mode),
            subgroup_tail_basis=str(args.subgroup_tail_basis),
            group_indices=group_indices,
            perturbation_moves=int(args.perturbation_moves),
            iter_limit=int(args.iter_limit),
            time_limit_per_instance=float(args.time_limit_per_instance),
            gap_eps=float(args.gap_eps),
            risk_weight_mean=float(args.risk_weight_mean),
            risk_weight_std=float(args.risk_weight_std),
            risk_weight_worst=float(args.risk_weight_worst),
            reference_cost_source=str(ref_source),
            jit_meta=jit_meta,
        )
    except Exception as e:
        msg = str(e)
        if (
            ("Concorde reference costs are required" in msg)
            or ("Concorde reference costs are incomplete" in msg)
            or ("time_limit_per_instance must be > 0" in msg)
        ):
            raise
        result = _invalid_result(
            error=str(e),
            n_eval=0,
            metric_basis=str(args.metric_basis),
            subgroup_mode=str(args.subgroup_mode),
            subgroup_tail_basis=str(args.subgroup_tail_basis),
            jit_warmup_enabled=bool(jit_meta.get("jit_warmup_enabled", False)),
            jit_warmup_active=bool(jit_meta.get("jit_warmup_active", False)),
            jit_guide_decorator_enabled=bool(jit_meta.get("jit_guide_decorator_enabled", False)),
            jit_guide_decorator_active=bool(jit_meta.get("jit_guide_decorator_active", False)),
            jit_guide_decorator_error=jit_meta.get("jit_guide_decorator_error"),
            jit_guide_sanitize_enabled=bool(jit_meta.get("jit_guide_sanitize_enabled", False)),
            jit_guide_sanitize_strict=bool(jit_meta.get("jit_guide_sanitize_strict", False)),
            jit_guide_sanitize_active=bool(jit_meta.get("jit_guide_sanitize_active", False)),
            jit_guide_sanitize_error=jit_meta.get("jit_guide_sanitize_error"),
            jit_guide_sanitize_changes=jit_meta.get("jit_guide_sanitize_changes", []),
            jit_guide_compile_probe_ok=bool(jit_meta.get("jit_guide_compile_probe_ok", False)),
            jit_guide_compile_probe_error=jit_meta.get("jit_guide_compile_probe_error"),
        )

    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(float(result.get("selected_metric", LARGE)))


if __name__ == "__main__":
    main()
