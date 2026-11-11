from __future__ import annotations

from _run_adapter_train_common import run_train


def main() -> None:
    run_train(
        default_config_rel="configs/erdr.yaml",
        default_problem_cfg="tsp_gls_erdr",
    )


if __name__ == "__main__":
    main()
