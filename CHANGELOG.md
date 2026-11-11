# Changes from the earlier archive

## 2026-11-11 — ICDE 2027 initial artifact preparation

- Updated the root README with the current paper title and ICDE 2027 submission status.
- Added an explicit release scope and an alignment checklist.
- Corrected inconsistent dataset claims in both README files. TSP200 is included; other scales are supported by generators.
- Updated top-level training/evaluation data paths to the bundled TSP200 files.
- Set the top-level training model fields to gpt-6-luna, as reported in the manuscript.
- Removed automatic mock fallback from EoH's LLM interface. Missing settings, failed requests, and empty responses now raise errors.
- Added ignore patterns for local credentials, generated outputs, caches, and temporary files.
- Preserved the original input ZIP; this repository contains the unpacked initial artifact snapshot.

The heuristic algorithms, data values, and earlier evolution budgets were retained. Full experiment reproduction remains pending.