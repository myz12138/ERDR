import argparse
import os
from pathlib import Path
import subprocess
import sys
import yaml

def launch(project, mode):
    project = Path(project).resolve()
    root = project.parents[1]
    parser = argparse.ArgumentParser(description="ILS training through the existing AHD/ERDR pipeline")
    parser.add_argument("--config", default=str(project / "configs" / (mode + "_ils.yaml")))
    args = parser.parse_args()
    config = Path(args.config).resolve()
    cfg = yaml.safe_load(config.read_text(encoding="utf8"))
    if cfg.get("search", {}).get("framework") != "ils":
        parser.error("ILS entry point requires search.framework: ils")
    env = os.environ.copy()
    env["ERDR_SEARCH_BACKEND"] = "ils"
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, str(project / "scripts" / ("run_" + mode + ".py")), "--config", str(config)], cwd=root, env=env, check=True)
