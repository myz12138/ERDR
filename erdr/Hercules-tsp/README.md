# Hercules-TSP

This directory contains the Hercules implementation for static TSP, Hercules + ERDR, and dynamic TSP re-solving evaluation. The training scripts use the Hercules adapter to run the TSP-GLS problem configuration and export outputs in the common experiment format.

## Directory Details

```text
erdr/Hercules-tsp/
├── cfg/
│   ├── llm_client/               # LLM-client configuration templates
│   └── problem/                  # TSP-GLS baseline and ERDR problem settings
├── problems/                     # Hercules TSP problem definitions
├── prompts/                      # Solver-generation prompts
├── src/                          # Hercules execution and result-export logic
├── scripts/
│   ├── _run_adapter_train_common.py
│   ├── run_baseline.py
│   ├── run_erdr.py
│   └── run_dynamic_evaluation.py
└── configs/
    ├── baseline.yaml
    ├── erdr.yaml
    └── dynamic_evaluation.yaml
```

The baseline entry point selects `tsp_gls_baseline`, whereas the ERDR entry point selects `tsp_gls_erdr` through the adapter.

## Running Commands

Run the following commands from the `ERDR/` directory.

```bash
python erdr/Hercules-tsp/scripts/run_baseline.py --config erdr/Hercules-tsp/configs/baseline.yaml
python erdr/Hercules-tsp/scripts/run_erdr.py --config erdr/Hercules-tsp/configs/erdr.yaml
```

Evaluate either exported solver on dynamic TSP instances:

```bash
python erdr/Hercules-tsp/scripts/run_dynamic_evaluation.py --config erdr/Hercules-tsp/configs/dynamic_evaluation.yaml --method baseline
python erdr/Hercules-tsp/scripts/run_dynamic_evaluation.py --config erdr/Hercules-tsp/configs/dynamic_evaluation.yaml --method erdr
```

## Hercules-Specific Configuration

The `hercules` section controls the evolutionary budget through `max_fe`, `pop_size`, `init_pop_size`, `mutation_rate`, and `diversify_init_pop`.

`evaluation.problem_config` must match the experiment variant: use `tsp_gls_baseline` for the baseline configuration and `tsp_gls_erdr` for the ERDR configuration. ERDR-specific settings are stored under `evaluation.erdr`, including `group_size`, `n_groups_train`, `n_groups_val`, `val_interval`, `train_pool_multiplier`, and the diversity-selection parameters `dds_eps_rel`, `dds_softmax_tau`, and `dds_novelty_weight`.

The dynamic evaluator reads solver paths from `paths.baseline_solver` and `paths.erdr_solver`. `generation.jitter_sigma` controls coordinate perturbation for location changes, while `generation.node_frac` controls the proportion of remaining nodes involved in request arrivals and cancellations.

## Outputs

The adapter writes the following normalized artifacts to `paths.output_dir`:

- `best_solver.py`: selected Hercules solver.
- `solver_archive/` and `solver_archive.json`: retained solver archive.
- `final_eval.json`: final static-evaluation record.

Dynamic evaluation creates an independent directory below `paths.run_root` for each run.
