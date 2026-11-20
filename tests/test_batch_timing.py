import os
import subprocess
import sys
import time
import tempfile
import unittest
from pathlib import Path

from erdr_shared.timing import BatchClock, wait_for_batch, start_batch_monitor, finish_batch_monitor


class BatchTimingTest(unittest.TestCase):
    def test_nested_initialization_is_excluded(self):
        now = [0.0]
        clock = BatchClock(10, clock=lambda: now[0])
        now[0] = 2.0
        with clock.excluded():
            now[0] = 10.0
            with clock.excluded():
                now[0] = 20.0
            now[0] = 30.0
        now[0] = 33.0
        self.assertEqual(clock.elapsed(), 5.0)

    def run_child(self, code, budget, should_timeout=False):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "tests") as folder:
            timing_path = Path(folder) / "timing.json"
            env = os.environ.copy()
            env["ERDR_BATCH_TIMING_PATH"] = str(timing_path)
            proc = subprocess.Popen([sys.executable, "-c", code], cwd=root, env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                if should_timeout:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        wait_for_batch(proc, budget, timing_path, startup_timeout=5)
                else:
                    _, stderr = wait_for_batch(proc, budget, timing_path, startup_timeout=5)
                    self.assertEqual(proc.returncode, 0, stderr)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate()
                wait_for_batch(proc, budget, timing_path, startup_timeout=5)
            self.assertFalse(timing_path.exists())

    def test_initialization_does_not_consume_search_budget(self):
        self.run_child("import time; from erdr_shared.timing import begin_batch, initialization_phase; "
                       "time.sleep(0.3); begin_batch(0.2)\n"
                       "with initialization_phase():\n time.sleep(0.4)\ntime.sleep(0.03)", 0.2)

    def test_monitor_terminates_before_result_collection(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "tests") as folder:
            path = Path(folder) / "timing.json"
            env = dict(os.environ, ERDR_BATCH_TIMING_PATH=str(path))
            process = subprocess.Popen([sys.executable, "-c",
                "import time; from erdr_shared.timing import begin_batch; begin_batch(0.2); time.sleep(60)"],
                cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                start_batch_monitor(process, 0.2, path, startup_timeout=5)
                deadline = time.monotonic()+5
                while process.poll() is None and time.monotonic()<deadline:
                    time.sleep(0.05)
                self.assertIsNotNone(process.poll())
                with self.assertRaises(subprocess.TimeoutExpired):
                    finish_batch_monitor(process)
                self.assertFalse(path.exists())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_nonreturning_candidate_times_out(self):
        self.run_child("import time; from erdr_shared.timing import begin_batch; "
                       "begin_batch(0.2); time.sleep(60)", 0.2, True)


if __name__ == "__main__":
    unittest.main()
