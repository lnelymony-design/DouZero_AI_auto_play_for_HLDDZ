"""Latest-state-wins posterior rollout service.

The live recognition thread never performs IPC or model inference here. It only
updates a short-lock protected pending slot and drains already-decoded results.
A bridge thread owns the Pipe. One spawned child process owns its own rollout
models and evaluator.

Scheduling invariant:
    at most 1 running + 1 newest pending job.
"""

from collections import deque
from dataclasses import asdict, dataclass
import json
import multiprocessing
import threading
import time
import traceback
from typing import Optional


@dataclass(frozen=True)
class RolloutJob:
    session_id: str
    round_id: str
    generation_id: int
    posterior_revision: int
    job_id: str
    suggestion_id: str
    state_hash: str
    created_at_utc: str
    created_monotonic: float
    public_snapshot_json: str
    worlds_json: str
    candidates: tuple
    my_position: str
    posterior_ess_ratio: float
    model_paths: tuple
    max_worlds: int
    min_worlds: int
    max_steps: int
    time_budget_seconds: float
    max_job_age_seconds: float
    device: str
    cpu_threads: int


@dataclass(frozen=True)
class RolloutResult:
    session_id: str
    round_id: str
    generation_id: int
    posterior_revision: int
    job_id: str
    suggestion_id: str
    state_hash: str
    status: str
    payload_json: str
    error: Optional[str] = None


def _rollout_process_main(conn, generation_value, stop_event):
    """Spawn target. Keep this module Qt-free and own models in the child."""
    try:
        first_message = conn.recv()
        if first_message.get("type") != "configure":
            raise RuntimeError("rollout child expected configure message")

        config = first_message["config"]
        device = str(config.get("device", "cpu"))
        cpu_threads = max(1, int(config.get("cpu_threads", 1)))

        import torch

        if device == "cpu":
            torch.set_num_threads(cpu_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

        from douzero.evaluation.deep_agent_new import DeepAgent
        from inference.rollout import PosteriorRolloutEvaluator

        model_paths = dict(config["model_paths"])
        agents = {
            position: DeepAgent(
                position,
                model_path,
                device=device,
            )
            for position, model_path in model_paths.items()
        }

        conn.send({
            "type": "ready",
            "device": device,
        })

        while not stop_event.is_set():
            if not conn.poll(0.1):
                continue

            message = conn.recv()
            message_type = message.get("type")
            if message_type == "stop":
                break
            if message_type != "job":
                continue

            raw_job = message["job"]
            job = RolloutJob(**raw_job)

            if int(generation_value.value) != int(job.generation_id):
                result = RolloutResult(
                    session_id=job.session_id,
                    round_id=job.round_id,
                    generation_id=job.generation_id,
                    posterior_revision=job.posterior_revision,
                    job_id=job.job_id,
                    suggestion_id=job.suggestion_id,
                    state_hash=job.state_hash,
                    status="cancelled",
                    payload_json="{}",
                    error="generation invalidated before dispatch",
                )
                conn.send({"type": "result", "result": asdict(result)})
                continue

            public_snapshot = json.loads(job.public_snapshot_json)
            worlds = json.loads(job.worlds_json)

            evaluator = PosteriorRolloutEvaluator(
                agents=agents,
                max_worlds=job.max_worlds,
                min_worlds=job.min_worlds,
                max_steps=job.max_steps,
                time_budget_seconds=job.time_budget_seconds,
            )

            now = time.monotonic()
            ttl_deadline = job.created_monotonic + job.max_job_age_seconds
            budget_deadline = now + job.time_budget_seconds
            deadline = min(ttl_deadline, budget_deadline)

            def should_cancel():
                return (
                    stop_event.is_set()
                    or int(generation_value.value)
                    != int(job.generation_id)
                )

            payload = evaluator.evaluate_snapshot(
                public_snapshot=public_snapshot,
                worlds=worlds,
                candidates=job.candidates,
                my_position=job.my_position,
                should_cancel=should_cancel,
                deadline_monotonic=deadline,
            )

            status = payload.get("status", "error")
            result = RolloutResult(
                session_id=job.session_id,
                round_id=job.round_id,
                generation_id=job.generation_id,
                posterior_revision=job.posterior_revision,
                job_id=job.job_id,
                suggestion_id=job.suggestion_id,
                state_hash=job.state_hash,
                status=status,
                payload_json=json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                error=None,
            )
            conn.send({"type": "result", "result": asdict(result)})

    except EOFError:
        pass
    except Exception:
        try:
            conn.send({
                "type": "fatal",
                "error": traceback.format_exc(),
            })
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class RolloutService:
    def __init__(
        self,
        model_paths,
        device="cpu",
        cpu_threads=1,
        shutdown_timeout=1.5,
    ):
        self.model_paths = dict(model_paths)
        self.device = str(device)
        self.cpu_threads = max(1, int(cpu_threads))
        self.shutdown_timeout = max(0.25, float(shutdown_timeout))

        self._ctx = multiprocessing.get_context("spawn")
        self._generation = self._ctx.Value("q", 0)
        self._child_stop = self._ctx.Event()
        self._parent_conn, child_conn = self._ctx.Pipe(duplex=True)

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._ready = False
        self._closed = False
        self._running = None
        self._pending = None
        self._results = deque()
        self._superseded_count = 0
        self._fatal_error = None

        self._process = self._ctx.Process(
            target=_rollout_process_main,
            args=(child_conn, self._generation, self._child_stop),
            name="PosteriorRolloutProcess",
            daemon=True,
        )
        self._process.start()
        child_conn.close()

        self._bridge = threading.Thread(
            target=self._bridge_loop,
            name="PosteriorRolloutBridge",
            daemon=True,
        )
        self._bridge.start()

    def _append_result_locked(self, result):
        self._results.append(result)
        while len(self._results) > 32:
            self._results.popleft()

    def submit(self, job):
        if not isinstance(job, RolloutJob):
            raise TypeError("job must be RolloutJob")

        with self._lock:
            if self._closed:
                return False

            self._generation.value = int(job.generation_id)

            if self._pending is not None:
                old = self._pending
                self._superseded_count += 1
                self._append_result_locked(
                    RolloutResult(
                        session_id=old.session_id,
                        round_id=old.round_id,
                        generation_id=old.generation_id,
                        posterior_revision=old.posterior_revision,
                        job_id=old.job_id,
                        suggestion_id=old.suggestion_id,
                        state_hash=old.state_hash,
                        status="superseded",
                        payload_json="{}",
                        error="replaced by newer pending job",
                    )
                )

            self._pending = job

        self._wake.set()
        return True

    def invalidate(self, generation_id, reason="public_state_changed"):
        with self._lock:
            if self._closed:
                return
            self._generation.value = int(generation_id)

            if self._pending is not None:
                old = self._pending
                self._pending = None
                self._append_result_locked(
                    RolloutResult(
                        session_id=old.session_id,
                        round_id=old.round_id,
                        generation_id=old.generation_id,
                        posterior_revision=old.posterior_revision,
                        job_id=old.job_id,
                        suggestion_id=old.suggestion_id,
                        state_hash=old.state_hash,
                        status="cancelled",
                        payload_json="{}",
                        error=str(reason),
                    )
                )

        self._wake.set()

    def drain_results(self, max_items=8):
        out = []
        with self._lock:
            count = min(max(0, int(max_items)), len(self._results))
            for _ in range(count):
                out.append(self._results.popleft())
        return out

    def is_ready(self):
        with self._lock:
            return bool(self._ready and not self._closed)

    def status(self):
        with self._lock:
            return {
                "ready": self._ready,
                "closed": self._closed,
                "running_job_id": (
                    None if self._running is None
                    else self._running.job_id
                ),
                "pending_job_id": (
                    None if self._pending is None
                    else self._pending.job_id
                ),
                "superseded_count": self._superseded_count,
                "fatal_error": self._fatal_error,
                "process_alive": self._process.is_alive(),
            }

    def _take_pending_for_dispatch(self):
        with self._lock:
            if (
                self._closed
                or not self._ready
                or self._running is not None
                or self._pending is None
            ):
                return None

            job = self._pending
            self._pending = None

            age = time.monotonic() - job.created_monotonic
            if age > job.max_job_age_seconds:
                self._append_result_locked(
                    RolloutResult(
                        session_id=job.session_id,
                        round_id=job.round_id,
                        generation_id=job.generation_id,
                        posterior_revision=job.posterior_revision,
                        job_id=job.job_id,
                        suggestion_id=job.suggestion_id,
                        state_hash=job.state_hash,
                        status="expired",
                        payload_json="{}",
                        error="job expired before dispatch",
                    )
                )
                return None

            self._running = job
            return job

    def _handle_message(self, message):
        message_type = message.get("type")
        if message_type == "ready":
            with self._lock:
                self._ready = True
                self._fatal_error = None
            return

        if message_type == "fatal":
            with self._lock:
                self._fatal_error = message.get("error")
                self._ready = False
            return

        if message_type != "result":
            return

        result = RolloutResult(**message["result"])
        with self._lock:
            self._append_result_locked(result)
            if (
                self._running is not None
                and self._running.job_id == result.job_id
            ):
                self._running = None

    def _bridge_loop(self):
        try:
            self._parent_conn.send({
                "type": "configure",
                "config": {
                    "device": self.device,
                    "cpu_threads": self.cpu_threads,
                    "model_paths": self.model_paths,
                },
            })

            while not self._stop.is_set():
                if not self._process.is_alive():
                    with self._lock:
                        if self._fatal_error is None:
                            self._fatal_error = (
                                "rollout child exited unexpectedly"
                            )
                        self._ready = False
                    break

                while self._parent_conn.poll(0.0):
                    self._handle_message(self._parent_conn.recv())

                job = self._take_pending_for_dispatch()
                if job is not None:
                    try:
                        self._parent_conn.send({
                            "type": "job",
                            "job": asdict(job),
                        })
                    except Exception as exc:
                        with self._lock:
                            self._fatal_error = repr(exc)
                            if (
                                self._running is not None
                                and self._running.job_id == job.job_id
                            ):
                                self._running = None

                self._wake.wait(0.05)
                self._wake.clear()

        except EOFError:
            pass
        except Exception:
            with self._lock:
                self._fatal_error = traceback.format_exc()
                self._ready = False

    def stop(self, timeout=None):
        timeout = (
            self.shutdown_timeout
            if timeout is None
            else max(0.0, float(timeout))
        )

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pending = None
            self._generation.value += 1

        self._child_stop.set()
        self._stop.set()
        self._wake.set()

        try:
            self._parent_conn.send({"type": "stop"})
        except Exception:
            pass

        self._bridge.join(timeout=min(timeout, 0.5))
        self._process.join(timeout=max(0.0, timeout - 0.5))

        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=0.5)

        try:
            self._parent_conn.close()
        except Exception:
            pass
