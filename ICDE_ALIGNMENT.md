# Alignment with the ICDE 2027 manuscript

Snapshot date: 2026-11-11. This package is an initial release derived from the earlier code-and-data archive. It has not been certified to reproduce every result in the current manuscript.

## Included

- GLS training and dynamic-evaluation entry points for EoH, ReEvo, Hercules, and MEoH.
- Baseline and ERDR configurations, including grouped training/validation evaluation.
- TSP200 with 50 synthetic instances and Concorde reference solutions.
- Data generators for additional scales and dynamic request changes.

## Updated in this snapshot

- Paper title and ICDE 2027 submission identification in README.md.
- Dataset descriptions corrected to match the files actually bundled.
- Top-level configs point to bundled TSP200 instead of missing TSP100/TSP1000 files.
- The model field in top-level baseline/ERDR configs is gpt-6-luna, matching the manuscript. This is a configuration alignment, not a claim that new experiments were run.
- EoH no longer silently falls back to a mock LLM when service configuration or API calls fail.

## Items requiring verification or additional artifacts

| Item | Current snapshot | Work before claiming complete reproduction |
| --- | --- | --- |
| ILS | Dedicated ILS experiment workflow not located in this archive | Add the actual ILS implementation, configs, and results |
| Additional sizes | Only TSP200 is bundled | Add the datasets used for TSP100, TSP500, and TSP1000 results, with provenance |
| TSPLIB | Manuscript-specific TSPLIB evaluation assets not located | Add instance lists, dynamic construction/scaling, configs, and results |
| Selection | Earlier DDS/grouped-evaluation implementation retained | Verify nondominated priority, DDS computation, tie-breaking, and final selection against the manuscript |
| Evolution budget | Earlier backbone settings retained | Confirm parameter semantics, including EoH ec_m=2 versus the manuscript's four offspring per operator |
| Dynamic protocol | Earlier generation/evaluation scripts retained | Verify Concorde execution tours, fixed start/end constraints, dummy-node handling, and warm starts for every backbone |
| Runtime metrics | Earlier timing implementation retained | Verify timer boundaries, iteration stopping, failure penalties, and paired-success filtering |
| Evaluation population | Source-side grouping retained | Identify independent test datasets and their overlap with evolution data |
| Supplemental analyses | Not bundled as a complete experiment suite | Add ablations, robustness, common-quality-target results, and supplementary tables |
| Dependencies | Backbone dependencies are incomplete as a unified recipe | Verify supported Python/package versions and document clean-environment installation |
| LLM backend | Provider fields require user configuration | Verify real API compatibility and record provider/model settings for released results |
| Provenance | Earlier framework image and inherited code retained | Check figure alignment and upstream attribution/licensing before public release |

## Validation performed

Only file/configuration consistency checks were performed for this snapshot. No training, model inference, or experimental reproduction was run. No synthetic replacement results were produced.