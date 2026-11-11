from __future__ import annotations

from _run_adapter_train_common import run_train


def main() -> None:
    run_train(
        default_config_rel="configs/baseline.yaml",
        default_problem_cfg="tsp_gls_baseline",
    )


if __name__ == "__main__":
    main()
