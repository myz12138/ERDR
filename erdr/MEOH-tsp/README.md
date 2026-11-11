# MEoH-TSP

This directory contains the MEoH implementation for static TSP, MEoH + ERDR, and dynamic TSP re-solving evaluation. It uses the local `llm4ad/` implementation for program sampling, solver evaluation, and evolutionary selection.

## Directory Details

```text
erdr/MEOH-tsp/
├── llm4ad/                       # MEoH program-evolution implementation
├── scripts/
│   ├── _run_train_common.py       # Shared baseline and ERDR training procedure
│   ├── run_baseline.py
│   ├── run_erdr.py
│   └── run_dynamic_evaluation.py
├── configs/
│   ├── baseline.yaml
│   ├── erdr.yaml
│   └── dynamic_evaluation.yaml
├── assets/                        # MEoH package assets
└── examples/                      # MEoH usage examples
```

`_run_train_common.py` constructs the TSP-GLS evaluation, calls the configured LLM through `HttpsApi`, evolves programs with MEoH, and reconstructs the selected program as `best_solver.py`.

## Running Commands

Run the following commands from the `ERDR/` directory.

```bash
python erdr/MEOH-tsp/scripts/run_baseline.py --config erdr/MEOH-tsp/configs/baseline.yaml
python erdr/MEOH-tsp/scripts/run_erdr.py --config erdr/MEOH-tsp/configs/erdr.yaml
```

Use the selected solver in the dynamic evaluator:

```bash
python erdr/MEOH-tsp/scripts/run_dynamic_evaluation.py --config erdr/MEOH-tsp/configs/dynamic_evaluation.yaml --method baseline
python erdr/MEOH-tsp/scripts/run_dynamic_evaluation.py --config erdr/MEOH-tsp/configs/dynamic_evaluation.yaml --method erdr
```

## MEoH-Specific Configuration

MEoH uses `llm.llm_api_host`, `llm.llm_api_key`, `llm.llm_model`, `llm.llm_temperature`, and `llm.timeout` for LLM access. Unlike the other baselines, its configuration specifies the API host rather than a full API endpoint.

The `meoh` section controls program evolution through `pop_size`, `selection_num`, `max_sample_nums`, `max_generations`, `num_samplers`, and `num_evaluators`. `debug_mode` enables MEoH debugging output.

For MEoH + ERDR, set `evaluation.group_evaluation_mode` to `mean_variability_tail` and enable `evaluation.grouped_evaluation`. Relative to the original MEoH objective, ERDR adds group-level runtime variability and tail-runtime risk. The same section defines the training/validation split through `group_size`, `n_groups_total`, `n_groups_train`, `n_groups_val`, and `split_seed`.

In MEoH + ERDR, each generated program is evaluated on the training groups during evolution. Once a generation's candidate pool is complete, that pool is evaluated once on the validation groups and reselected before the next generation begins. Validation is therefore generation-level rather than per generated program.

The dynamic configuration chooses `paths.baseline_solver` or `paths.erdr_solver` according to `--method`. Its `dynamic_evaluation` and `gls` sections set the scenario types, progress ratios, number of instances, and re-solving budget.

## Outputs

MEoH writes the following training outputs to `paths.output_dir`:

- `best_solver.py`: selected MEoH solver.
- `final_eval.json`: final training and evaluation record.
- `meoh_run/`: MEoH profiler output.

Each dynamic evaluation is stored under `paths.run_root`.
