import multiprocessing
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from erdr_shared.timing import begin_batch

try:
    from joblib.externals import cloudpickle
except ImportError:
    import cloudpickle


def _evaluate_worker(connection, payload, budget):
    try:
        evaluator, code = cloudpickle.loads(payload)
        begin_batch(budget, publish=lambda state: connection.send(("timing", state)))
        result = evaluator.evaluate(code)
        info = evaluator.get_last_eval_info() if hasattr(evaluator, "get_last_eval_info") else None
        connection.send(("result", True, result, info))
    except BaseException as exc:
        connection.send(("result", False, type(exc).__name__ + ": " + str(exc), None))
    finally:
        connection.close()


def evaluate_in_process(evaluator, code, timeout):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    payload = cloudpickle.dumps((evaluator, code))
    process = context.Process(target=_evaluate_worker, args=(sender, payload, timeout))
    try:
        process.start()
        sender.close()
        waiting_since = time.monotonic()
        state = None
        while True:
            now = time.monotonic()
            if state is None:
                remaining = 300.0 - (now-waiting_since)
            elif state["active"]:
                remaining = timeout-state["used"]-(now-state["since"])
            else:
                remaining = 300.0-(now-state["since"])
            if remaining <= 0 or not receiver.poll(remaining):
                raise TimeoutError("Candidate evaluation exceeded the batch time budget.")
            try:
                message = receiver.recv()
            except EOFError as exc:
                raise RuntimeError("Candidate evaluation process exited without a result.") from exc
            if message[0] == "timing":
                state = message[1]
                continue
            _, success, value, info = message
            break
        if not success:
            raise RuntimeError(value)
        return value, info
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()


def evaluate_metric(evaluator, code):
    count = int(evaluator.n_instance)
    budget = max(300.0, count * float(evaluator.time_limit_per_instance) + 15.0)
    value, info = evaluate_in_process(evaluator, code, budget)
    if isinstance(info, dict):
        evaluator._last_eval_info = info
    return value
