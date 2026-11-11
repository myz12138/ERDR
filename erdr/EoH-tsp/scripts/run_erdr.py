from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[1]
    repo_root = project_dir.parents[1]

    sys.path.insert(0, str(project_dir))
    sys.path.insert(0, str(project_dir / "src"))

    try:
        from src.config_utils import load_config
    except ModuleNotFoundError:
        from config_utils import load_config
    from static_train import run_static_training

    parser = argparse.ArgumentParser(
        description="Run EoH+ERDR training for TSP-GLS."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_dir / "configs" / "erdr.yaml"),
        help="Path to yaml/json config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_static_training(cfg=cfg, repo_root=repo_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
