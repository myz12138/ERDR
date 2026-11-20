from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1]))
from erdr_shared.entry import launch
if __name__ == "__main__":
    launch(PROJECT, "baseline")
