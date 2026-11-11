# ReEvo-TSP

This directory contains the ReEvo implementation for static TSP, ReEvo + ERDR, and dynamic TSP re-solving evaluation. The TSP workflow is routed through the local ReEvo adapter so that the generated solver and final evaluation use the shared experiment format.

## Directory Details

```text
erdr/reevo-tsp/
├── reevo/                        # ReEvo reflective-evolution implementation
├── cfg/                          # ReEvo problem and LLM configuration templates
├── problems/                     # TSP-GLS problem implementation
├── prompts/                      # TSP solver-generation prompts
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

`_run_adapter_train_common.py` transfers the experiment settings in `configs/` to the ReEvo TSP-GLS implementation and normalizes its outputs.

## Running Commands

Run the following commands from the `ERDR/` directory.

```bash
python erdr/reevo-tsp/scripts/run_baseline.py --config erdr/reevo-tsp/configs/baseline.yaml
python erdr/reevo-tsp/scripts/run_erdr.py --config erdr/reevo-tsp/configs/erdr.yaml
```

Use the generated baseline or ERDR solver for dynamic evaluation:

```bash
python erdr/reevo-tsp/scripts/run_dynamic_evaluation.py --config erdr/reevo-tsp/configs/dynamic_evaluation.yaml --method baseline
python erdr/reevo-tsp/scripts/run_dynamic_evaluation.py --config erdr/reevo-tsp/configs/dynamic_evaluation.yaml --method erdr
```

## ReEvo-Specific Configuration

The `reevo` section controls the reflective-evolution budget: `max_fe`, `pop_size`, `init_pop_size`, `mutation_rate`, `timeout`, and `diversify_init_pop`.

For ReEvo + ERDR, configure the train/validation split in `evaluation.grouped_evaluation`. In addition to the common group settings, ReEvo uses `val_interval` to set the validation frequency and `pop_multiplier` to set the expanded candidate-pool size before reselection.

The dynamic evaluator supports `evaluation.metric_basis` (`gap` or `cost`) and the `jit` section. `jit.enable_kernels`, `jit.numba_cache`, and `jit.warmup` control optional JIT acceleration; they do not change the dynamic scenario definition.

## Outputs

The adapter stores the normalized training artifacts under `paths.output_dir`:

- `best_solver.py`: selected ReEvo solver.
- `solver_archive/` and `solver_archive.json`: retained solver archive.
- `final_eval.json`: final static-evaluation record.

Dynamic evaluation writes each run below `paths.run_root` and records the generated changed instances together with their evaluation results.
