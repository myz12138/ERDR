# EoH-TSP

This directory contains the EoH implementation for static TSP, EoH + ERDR, and dynamic TSP re-solving evaluation. It uses the local EoH evolutionary engine in `eoh/` and the TSP-GLS evaluator in `src/`.

## Directory Details

```text
erdr/EoH-tsp/
├── eoh/                         # EoH evolution operators and LLM interfaces
├── src/
│   ├── static_train.py           # Static and ERDR training procedure
│   ├── gls_core.py               # Guided local search implementation
│   ├── gls_operators.py          # TSP local-search operators
│   └── solver_archive.py         # Best-solver and archive export
├── scripts/
│   ├── run_baseline.py
│   ├── run_erdr.py
│   └── run_dynamic_evaluation.py
└── configs/
    ├── baseline.yaml
    ├── erdr.yaml
    └── dynamic_evaluation.yaml
```

## Running Commands

Run the following commands from the `ERDR/` directory.

```bash
python erdr/EoH-tsp/scripts/run_baseline.py --config erdr/EoH-tsp/configs/baseline.yaml
python erdr/EoH-tsp/scripts/run_erdr.py --config erdr/EoH-tsp/configs/erdr.yaml
```

After one of the two training commands has produced a solver, run dynamic evaluation:

```bash
python erdr/EoH-tsp/scripts/run_dynamic_evaluation.py --config erdr/EoH-tsp/configs/dynamic_evaluation.yaml --method baseline
python erdr/EoH-tsp/scripts/run_dynamic_evaluation.py --config erdr/EoH-tsp/configs/dynamic_evaluation.yaml --method erdr
```

## EoH-Specific Configuration

`eoh.ec_pop_size`, `eoh.ec_n_pop`, and `eoh.ec_m` control the EoH population and evolutionary procedure. `eoh.ec_operators` and `eoh.ec_operator_weights` select and weight the EoH generation operators.

For the static baseline, `evaluation.group_evaluation_mode` is `mean_only` and `evaluation.grouped_evaluation.enabled` is `false`. For EoH + ERDR, use `mean_variability_tail` and enable grouped evaluation. The ERDR grouping parameters are `group_size`, `n_groups_total`, `n_groups_train`, `n_groups_val`, and `split_seed`.

The dynamic configuration selects the solver through `paths.baseline_solver` or `paths.erdr_solver`. Set `dynamic_evaluation.change_types`, `ratio_sets`, and `n_instance` to control the dynamic scenarios; the GLS re-solving budget is controlled by the `gls` section.

## Outputs

Training writes the following files to `paths.output_dir`:

- `best_solver.py`: selected EoH solver used by dynamic evaluation.
- `solver_archive/` and `solver_archive.json`: retained solver archive.
- `final_eval.json`: final static-evaluation record.

Dynamic results are written below `paths.run_root` in a separate run directory.
