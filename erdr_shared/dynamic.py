"""Paired dynamic ILS evaluation from saved Concorde execution tours."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import numpy as np
import yaml

from .support import tour_cost


def snapshot_seed(seed, instance_index, progress, change):
    changes = {"location_change": 0, "request_arrival": 1, "request_cancellation": 2}
    return int(seed) + int(instance_index)*10007 + round(float(progress)*1000)*101 + changes[change]*1009


def load_sources(path):
    with Path(path).open('rb') as handle:
        raw = pickle.load(handle)
    coordinates = raw['coordinate']
    tours = raw.get('optimal_tour_concorde')
    if tours is None:
        sidecar = Path(path).with_name(Path(path).stem + '_full.json')
        if not sidecar.exists():
            sidecar = Path(path).with_name(Path(path).stem + '_concorde_full.json')
        records = json.loads(sidecar.read_text(encoding='utf8'))['records']
        if len(records) != len(coordinates) or any(r['status'] != 'ok' for r in records):
            raise ValueError('Incomplete Concorde execution-tour records')
        tours = [r['tour'] for r in records]
    if len(tours) != len(coordinates):
        raise ValueError('Coordinate/tour count mismatch')
    for coords, tour in zip(coordinates, tours):
        coords = np.asarray(coords, dtype=float)
        tour = np.asarray(tour)
        if len(tour) == len(coords)+1 and tour[0] == tour[-1]:
            tour = tour[:-1]
        if (not np.issubdtype(tour.dtype, np.integer)
                or sorted(tour.tolist()) != list(range(len(coords)))
                or coords.shape != (len(coords), 2) or not np.isfinite(coords).all()):
            raise ValueError('Invalid coordinates or Concorde tour')
        yield coords, tour


def snapshot(coords, tour, progress, change, *, sigma=.02, ratio=.2, seed=1, return_coordinates=False):
    """Construct a residual path in [0,1]^2; IDs follow the repaired warm route."""
    if (not all(math.isfinite(v) for v in (progress, ratio, sigma))
            or not 0 < progress < 1 or not 0 <= ratio < 1 or sigma < 0):
        raise ValueError('Invalid dynamic-event parameters')
    if np.any(coords < 0) or np.any(coords > 1):
        raise ValueError('This generator supports unit-square TSP data; use a TSPLIB-specific constructor for original-scale data')
    rng = np.random.default_rng(seed)
    cut = min(len(tour)-1, max(1, math.ceil(progress*len(tour))))
    start, terminal = coords[tour[cut-1]].copy(), coords[tour[0]].copy()
    remaining = coords[tour[cut:]].copy()
    if change == 'location_change':
        remaining = np.clip(remaining + rng.normal(0, sigma, remaining.shape), 0, 1)
    elif change == 'request_cancellation':
        count = min(max(0, len(remaining)-2), max(1, round(ratio*len(remaining))) if ratio else 0)
        if count:
            remaining = np.delete(remaining, rng.choice(len(remaining), count, replace=False), axis=0)
    elif change != 'request_arrival':
        raise ValueError('Unknown dynamic change: ' + change)
    route_coords = [start, *remaining, terminal]
    if change == 'request_arrival':
        count = max(1, round(ratio*len(remaining))) if ratio else 0
        for point in rng.uniform(0, 1, (count, 2)):
            costs = [np.linalg.norm(a-point)+np.linalg.norm(point-b)-np.linalg.norm(a-b)
                     for a,b in zip(route_coords[:-1], route_coords[1:])]
            route_coords.insert(int(np.argmin(costs))+1, point)
    real = np.asarray(route_coords)
    d = np.linalg.norm(real[:,None]-real[None,:], axis=2)
    upper = float(sum(d[i,i+1] for i in range(len(d)-1)))

    matrix = np.full((len(d)+1, len(d)+1), max(upper, float(d.max()))+1.)
    matrix[:-1,:-1] = d
    np.fill_diagonal(matrix, 0.)
    matrix[-1,0] = matrix[0,-1] = matrix[-1,-2] = matrix[-2,-1] = 0.
    warm = np.arange(len(matrix), dtype=np.int64)
    if return_coordinates:
        return matrix, warm, np.vstack((real, terminal))
    return matrix, warm


def _worker(connection, solver, matrix, warm, budget, iterations, seed):
    try:
        from .ils import solve
        from .support import load_rule
        result = solve(matrix, perturbation_rule=load_rule(solver), initial_tour=warm,
                       time_limit_sec=budget, iter_limit=iterations, seed=seed)
        connection.send(dict(status='ok', obj=result.cost, time_sec=result.elapsed_sec,
                             iterations=result.iterations, tour=result.tour))
    except Exception as exc:
        connection.send(dict(status='failed', error=f'{type(exc).__name__}: {exc}'))
    finally:
        connection.close()


def evaluate_one(solver, matrix, warm, *, budget, iterations, seed, watchdog):
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_worker, args=(child,str(solver),matrix,warm,budget,iterations,seed))
    process.start()
    child.close()
    try:
        if parent.poll(watchdog):
            try:
                result = parent.recv()
            except EOFError:
                result = dict(status='failed', error='evaluation process exited without a result')
        else:
            result = dict(status='failed', error='evaluation watchdog timeout')
    finally:
        if process.is_alive():
            process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
        if process.is_alive():
            process.kill()
        process.join()
        process.close()
        parent.close()
    return result


def main(project):
    project = Path(project).resolve()
    root = project.parents[1]
    parser = argparse.ArgumentParser(description='Paired ILS dynamic evaluation with shared Concorde-derived snapshots')
    parser.add_argument('--config', default=str(project/'configs/dynamic_evaluation_ils.yaml'))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    if cfg.get('search',{}).get('framework') != 'ils':
        parser.error('Requires search.framework: ils')
    resolve = lambda value: (root / value).resolve()
    solvers = {key: resolve(cfg['solvers'][key]) for key in ('baseline','erdr')}
    for solver in solvers.values():
        if not solver.is_file(): raise FileNotFoundError(solver)
    folder = resolve(cfg['output_dir'])
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing evaluation: {folder}')
    source_path = resolve(cfg['dataset_path'])
    count = int(cfg.get('n_instances',50))
    if count <= 0: raise ValueError('n_instances must be positive')
    sources = list(load_sources(source_path))[:count]
    if not sources: raise ValueError('No source instances')
    search = cfg['search']
    budget, iterations = float(search['time_limit_seconds']), int(search['max_iterations'])
    watchdog = float(search.get('watchdog_seconds',300))
    if not math.isfinite(budget) or budget <= 0 or iterations < 0:
        raise ValueError('Invalid search budget')
    if not math.isfinite(watchdog) or watchdog <= 0:
        raise ValueError('watchdog_seconds must be positive and finite')
    folder.mkdir(parents=True, exist_ok=True)
    summaries=[]
    changes=['location_change','request_arrival','request_cancellation']
    for progress in cfg.get('progress',[.2,.5,.8]):
        for change_index,change in enumerate(changes):
            paired=[]
            for index,(coords,tour) in enumerate(sources):
                seed=snapshot_seed(cfg.get('seed',1), index, progress, change)
                matrix,warm=snapshot(coords,tour,progress,change,sigma=cfg.get('sigma',.02),ratio=cfg.get('change_ratio',.2),seed=seed)
                record=dict(instance=index,progress=progress,change=change,seed=seed,
                            snapshot_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
                            initial_obj=tour_cost(matrix,warm))
                for variant,path in solvers.items():
                    record[variant]=evaluate_one(path,matrix,warm,budget=budget,iterations=iterations,
                                                seed=seed,watchdog=watchdog)
                if all(record[v]['status']=='ok' for v in solvers):paired.append(record)
                with (folder/'instances.jsonl').open('a',encoding='utf8') as handle:
                    handle.write(json.dumps(record,allow_nan=False)+'\n')
            for variant in solvers:
                summaries.append(dict(progress=progress,change=change,variant=variant,n_total=len(sources),n_paired=len(paired),
                                      obj=float(np.mean([r[variant]['obj'] for r in paired])) if paired else None,
                                      time_sec=float(np.mean([r[variant]['time_sec'] for r in paired])) if paired else None))
    with (folder/'summary.csv').open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    provenance=dict(config=cfg, dataset_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    solvers={k:dict(path=str(v),sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in solvers.items()},
                    protocol='single run; paired successes; exact supplied warm start; search-only time; fixed 2-opt descent')
    (folder/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    print(folder/'summary.csv')
