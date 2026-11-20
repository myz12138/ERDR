import json
import math
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path


_clock = None


class BatchClock:
    def __init__(self, budget, publish=None, clock=time.monotonic):
        self.budget = float(budget)
        if not math.isfinite(self.budget) or self.budget <= 0:
            raise ValueError("Batch budget must be positive and finite")
        self.clock = clock
        self.publish = publish
        self.used = 0.0
        self.since = clock()
        self.depth = 0
        self.lock = threading.RLock()
        self.emit()

    def snapshot(self):
        with self.lock:
            return dict(used=self.used, since=self.since, active=self.depth == 0)

    def elapsed(self):
        state = self.snapshot()
        return state["used"] + (self.clock()-state["since"] if state["active"] else 0.0)

    def emit(self):
        if self.publish is not None:
            self.publish(self.snapshot())

    @contextmanager
    def excluded(self):
        with self.lock:
            if self.depth == 0:
                self.used += self.clock()-self.since
                self.since = self.clock()
            self.depth += 1
            self.emit()
        try:
            yield
        finally:
            with self.lock:
                self.depth -= 1
                if self.depth == 0:
                    self.since = self.clock()
                self.emit()


def begin_batch(budget, publish=None):
    global _clock
    destination = os.environ.get("ERDR_BATCH_TIMING_PATH")
    if publish is None and destination:
        path = Path(destination)
        def publish(state):
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(state), encoding="utf-8")
            os.replace(temporary, path)
    _clock = BatchClock(budget, publish)
    return _clock


@contextmanager
def initialization_phase():
    if _clock is None:
        yield
    else:
        with _clock.excluded():
            yield


def wait_for_batch(process, budget, timing_path, startup_timeout=300.0):
    path = Path(timing_path)
    waiting_since = time.monotonic()
    try:
        while process.poll() is None:
            now = time.monotonic()
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                state = None
            if state is None:
                remaining = startup_timeout - (now-waiting_since)
            elif state["active"]:
                remaining = budget - state["used"] - (now-state["since"])
            else:
                remaining = startup_timeout - (now-state["since"])
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, budget)
            try:
                return process.communicate(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                pass
        return process.communicate()
    finally:
        if process.poll() is not None:
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def wait_queue_result(queue, budget, startup_timeout=300.0):
    state = None
    started = time.monotonic()
    while True:
        now = time.monotonic()
        if state is None:
            remaining = startup_timeout-(now-started)
        elif state["active"]:
            remaining = budget-state["used"]-(now-state["since"])
        else:
            remaining = startup_timeout-(now-state["since"])
        if remaining <= 0:
            raise TimeoutError("Candidate evaluation exceeded the batch time budget.")
        item = queue.get(timeout=remaining)
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "erdr_batch_timing":
            state = item[1]
        else:
            return item


def start_batch_monitor(process, budget, timing_path, startup_timeout=300.0):
    state = {}
    def monitor():
        try:
            state["result"] = wait_for_batch(process, budget, timing_path, startup_timeout)
        except BaseException as exc:
            state["error"] = exc
            if process.poll() is None:
                process.kill()
            process.communicate()
            wait_for_batch(process, budget, timing_path, startup_timeout)
    thread = threading.Thread(target=monitor, daemon=True)
    process._erdr_batch_monitor = (thread, state)
    thread.start()


def finish_batch_monitor(process):
    thread, state = process._erdr_batch_monitor
    thread.join()
    if "error" in state:
        raise state["error"]
    return state.get("result")
