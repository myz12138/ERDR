# TSP Data Generation

This initial ICDE 2027 package includes TSP200 with 50 instances and Concorde reference solutions in TSP200_u50/. The scripts in this directory can generate the other supported scales. TSP50, TSP100, TSP500, and TSP1000 are not bundled in this snapshot.

Run the command below from the `ERDR/` directory:

```bash
python data_generate/generate_uniform_tsp_batches.py --config data_generate/build_tsp_dataset.yaml
```

Before running the command, set `concorde.concorde_exe` in `build_tsp_dataset.yaml` to a valid Concorde executable. Concorde is required to produce the `*_concorde.pkl` files used by the baseline configurations.

The default configuration generates 50 uniformly sampled instances for each of TSP50, TSP100, TSP200, TSP500, and TSP1000. The outputs are written as:

```tex
data_generate/TSP{size}_u50/
├── TSP{size}_u50_concorde.pkl
└── TSP{size}_u50_concorde_full.json
```

`*_concorde.pkl` contains the instances and Concorde reference costs. `*_concorde_full.json` records the detailed Concorde solutions. The intermediate raw `.pkl` file is removed after labeling unless `generation.keep_raw_pkl` is set to `true`.

## Configuration

`generation.sizes`, `generation.n_instances`, and `generation.seed` control the dataset scales, number of instances, and reproducible sampling seed. `paths.out_dir` controls the output location. Set `concorde.enabled` to `false` only when raw instances are sufficient; the training scripts require the Concorde-labeled files.

`generation.overwrite: true` regenerates and replaces existing files at the configured output location.
