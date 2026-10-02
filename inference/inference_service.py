"""Latest-state-wins posterior inference service.

The live recognition QThread records public actions only. Monte-Carlo posterior
inference runs in one spawned child process from an immutable JSON snapshot.

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
import uuid
from typing import Optional


@dataclass(frozen=True)
class InferenceJob:
    session_id: str
    round_id: str
    generation_id: int
    job_id: str
    created_monotonic: float
    state_json: str
    candidates: tuple
    response_samples: int
    max_worlds: int
    max_job_age_seconds: float


@dataclass(frozen=True)
class InferenceResult:
    service_epoch: str
    dispatch_id: str
    session_id: str
    round_id: str
    generation_id: int
    job_id: str
    status: str
    payload_json: str
    error: Optional[str] = None


def _inference_process_main(
    conn,
    generation_value,
    dispatch_value,
    stop_event,
):
    try:
        conn.send({"type": "ready"})

        while not stop_event.is_set():
            if not conn.poll(0.1):
                continue

            message = conn.recv()
            message_type = message.get("type")
            if message_type == "stop":
                break
            if message_type != "job":
                continue

            job = InferenceJob(**message["job"])
            service_epoch = str(message["service_epoch"])
            dispatch_id = str(message["dispatch_id"])
            dispatch_seq = int(message["dispatch_seq"])

            def cancelled():
                return (
                    stop_event.is_set()
                    or int(generation_value.value)
                    != int(job.generation_id)
                    or int(dispatch_value.value) != dispatch_seq
                )

            if cancelled():
                result = InferenceResult(
                    service_epoch=service_epoch,
                    dispatch_id=dispatch_id,
                    session_id=job.session_id,
                    round_id=job.round_id,
                    generation_id=job.generation_id,
                    job_id=job.job_id,
                    status="cancelled",
                    payload_json="{}",
                    error="generation/dispatch invalidated before execution",
                )
                conn.send({"type": "result", "result": asdict(result)})
                continue

            started = time.monotonic()
            try:
                from inference.engine import HandInferenceEngine

                state = json.loads(job.state_json)
                engine = HandInferenceEngine(
                    my_position=state["my_position"],
                    my_hand_cards=state["initial_my_hand"],
                    three_landlord_cards=state["three_landlord_cards"],
                    sample_count=state["sample_count"],
                    pass_penalty=state["pass_penalty"],
                    friendly_pass_penalty=state["friendly_pass_penalty"],
                    play_behavior_floor=state["play_behavior_floor"],
                    play_behavior_strength=state["play_behavior_strength"],
                    behavior_temperature=state["behavior_temperature"],
                    min_effective_sample_ratio=(
                        state["min_effective_sample_ratio"]
                    ),
                    residual_behavior_floor=(
                        state["residual_behavior_floor"]
                    ),
                    residual_behavior_strength=(
                        state["residual_behavior_strength"]
                    ),
                    residual_behavior_temperature=(
                        state["residual_behavior_temperature"]
                    ),
                    random_seed=state.get("random_seed"),
                )
                for player, action in state["history"]:
                    engine.observe(player, action)

                inference = engine.infer(should_cancel=cancelled)
                if inference.get("cancelled") or cancelled():
                    status = "cancelled"
                    payload = {
                        "elapsed_seconds": time.monotonic() - started,
                    }
                else:
                    profiles = {}
                    for action in job.candidates:
                        if cancelled():
                            break
                        if action == "Pass":
                            profiles[action] = None
                        else:
                            profiles[action] = engine.response_profile(
                                action,
                                max_samples=job.response_samples,
                            )

                    if cancelled():
                        status = "cancelled"
                        payload = {
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    else:
                        worlds = engine.posterior_worlds(
                            max_worlds=job.max_worlds,
                            allow_infer=False,
                        )
                        payload = {
                            "inference": inference,
                            "profiles": profiles,
                            "worlds": worlds,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                        status = "ok"

                result = InferenceResult(
                    service_epoch=service_epoch,
                    dispatch_id=dispatch_id,
                    session_id=job.session_id,
                    round_id=job.round_id,
                    generation_id=job.generation_id,
                    job_id=job.job_id,
                    status=status,
                    payload_json=json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    error=None,
                )
            except Exception:
                result = InferenceResult(
                    service_epoch=service_epoch,
                    dispatch_id=dispatch_id,
                    session_id=job.session_id,
                    round_id=job.round_id,
                    generation_id=job.generation_id,
                    job_id=job.job_id,
                    status="error",
                    payload_json="{}",
                    error=traceback.format_exc(),
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


class InferenceService:
    def __init__(self, shutdown_timeout=1.0):
        self.service_epoch = uuid.uuid4().hex
        self.shutdown_timeout = max(0.25, float(shutdown_timeout))

        self._ctx = multiprocessing.get_context("spawn")
        self._generation = self._ctx.Value("q", 0)
        self._active_dispatch_seq = self._ctx.Value("q", 0)
        self._child_stop = self._ctx.Event()
        self._parent_conn, child_conn = self._ctx.Pipe(duplex=True)

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._ready = False
        self._closed = False
        self._failed = False
        self._running = None
        self._running_dispatch_id = None
        self._pending = None
        self._results = deque()
        self._dispatch_seq = 0
        self._superseded_count = 0
        self._fatal_error = None

        self._process = self._ctx.Process(
            target=_inference_process_main,
            args=(
                child_conn,
                self._generation,
                self._active_dispatch_seq,
                self._child_stop,
            ),
            name="PosteriorInferenceProcess",
            daemon=True,
        )
        self._process.start()
        child_conn.close()

        self._bridge = threading.Thread(
            target=self._bridge_loop,
            name="PosteriorInferenceBridge",
            daemon=True,
        )
        self._bridge.start()

        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="PosteriorInferenceWatchdog",
            daemon=True,
        )
        self._watchdog.start()

    def _append_result_locked(self, result):
        self._results.append(result)

    def is_ready(self):
        with self._lock:
            return bool(
                self._ready
                and not self._closed
                and not self._failed
            )

    def submit(self, job):
        if not isinstance(job, InferenceJob):
            raise TypeError("job must be InferenceJob")

        with self._lock:
            if self._closed or self._failed:
                return None

            self._generation.value = int(job.generation_id)
            self._dispatch_seq += 1
            dispatch_seq = self._dispatch_seq
            self._active_dispatch_seq.value = dispatch_seq
            dispatch_id = f"{self.service_epoch}:{dispatch_seq}"

            if self._pending is not None:
                old, old_dispatch_id, _ = self._pending
                self._superseded_count += 1
                self._append_result_locked(
                    InferenceResult(
                        service_epoch=self.service_epoch,
                        dispatch_id=old_dispatch_id,
                        session_id=old.session_id,
                        round_id=old.round_id,
                        generation_id=old.generation_id,
                        job_id=old.job_id,
                        status="superseded",
                        payload_json="{}",
                        error="replaced by newer pending inference",
                    )
                )

            self._pending = (job, dispatch_id, dispatch_seq)

        self._wake.set()
        return dispatch_id

    def invalidate(self, generation_id, reason="state_changed"):
        with self._lock:
            self._generation.value = int(generation_id)
            self._dispatch_seq += 1
            self._active_dispatch_seq.value = self._dispatch_seq

            if self._pending is not None:
                old, old_dispatch_id, _ = self._pending
                self._pending = None
                self._append_result_locked(
                    InferenceResult(
                        service_epoch=self.service_epoch,
                        dispatch_id=old_dispatch_id,
                        session_id=old.session_id,
                        round_id=old.round_id,
                        generation_id=old.generation_id,
                        job_id=old.job_id,
                        status="cancelled",
                        payload_json="{}",
                        error=str(reason),
                    )
                )
        self._wake.set()

    def drain_results(self, max_items=8):
        out = []
        with self._lock:
            while self._results and len(out) < max_items:
                out.append(self._results.popleft())
        return out

    def status(self):
        with self._lock:
            return {
                "service_epoch": self.service_epoch,
                "ready": self._ready,
                "closed": self._closed,
                "failed": self._failed,
                "running_job_id": (
                    None if self._running is None
                    else self._running.job_id
                ),
                "running_dispatch_id": self._running_dispatch_id,
                "pending_job_id": (
                    None if self._pending is None
                    else self._pending[0].job_id
                ),
                "superseded_count": self._superseded_count,
                "fatal_error": self._fatal_error,
                "process_alive": self._process.is_alive(),
            }

    def _take_pending_for_dispatch(self):
        with self._lock:
            if self._pending is None or self._closed or self._failed:
                return None

            job, dispatch_id, dispatch_seq = self._pending
            self._pending = None
            age = time.monotonic() - job.created_monotonic
            if age > job.max_job_age_seconds:
                self._append_result_locked(
                    InferenceResult(
                        service_epoch=self.service_epoch,
                        dispatch_id=dispatch_id,
                        session_id=job.session_id,
                        round_id=job.round_id,
                        generation_id=job.generation_id,
                        job_id=job.job_id,
                        status="expired",
                        payload_json="{}",
                        error="inference expired before dispatch",
                    )
                )
                return None

            self._running = job
            self._running_dispatch_id = dispatch_id
            return job, dispatch_id, dispatch_seq

    def _handle_message(self, message):
        message_type = message.get("type")
        if message_type == "ready":
            with self._lock:
                self._ready = True
            return
        if message_type == "fatal":
            with self._lock:
                self._fatal_error = (
                    message.get("error")
                    or "inference child fatal error"
                )
                self._ready = False
                self._failed = True
                self._running = None
                self._running_dispatch_id = None
                self._pending = None
            return
        if message_type != "result":
            return

        result = InferenceResult(**message["result"])
        with self._lock:
            self._append_result_locked(result)
            if (
                result.service_epoch == self.service_epoch
                and self._running is not None
                and self._running.job_id == result.job_id
                and self._running_dispatch_id == result.dispatch_id
            ):
                self._running = None
                self._running_dispatch_id = None

    def _bridge_loop(self):
        try:
            while not self._stop.is_set():
                if self._parent_conn.poll(0.02):
                    self._handle_message(self._parent_conn.recv())

                dispatch = self._take_pending_for_dispatch()
                if dispatch is not None:
                    job, dispatch_id, dispatch_seq = dispatch
                    try:
                        self._parent_conn.send({
                            "type": "job",
                            "service_epoch": self.service_epoch,
                            "dispatch_id": dispatch_id,
                            "dispatch_seq": dispatch_seq,
                            "job": asdict(job),
                        })
                    except Exception:
                        with self._lock:
                            if (
                                self._running is not None
                                and self._running.job_id == job.job_id
                            ):
                                self._running = None
                                self._running_dispatch_id = None
                        raise

                if not self._process.is_alive():
                    with self._lock:
                        self._fatal_error = (
                            self._fatal_error
                            or "inference child exited unexpectedly"
                        )
                        self._ready = False
                        self._failed = True
                    break

                self._wake.wait(0.03)
                self._wake.clear()
        except Exception:
            with self._lock:
                self._fatal_error = traceback.format_exc()
                self._ready = False
                self._failed = True

    def _watchdog_loop(self):
        while not self._stop.wait(0.10):
            with self._lock:
                job = self._running
                dispatch_id = self._running_dispatch_id
                closed = self._closed
                failed = self._failed

            if closed or failed or job is None:
                continue

            age = time.monotonic() - job.created_monotonic
            hard_limit = job.max_job_age_seconds + 1.0
            if age <= hard_limit:
                continue

            error = (
                "inference watchdog terminated child after "
                f"{age:.2f}s (limit {hard_limit:.2f}s)"
            )
            try:
                self._child_stop.set()
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=0.5)
            except Exception as exc:
                error += f"; terminate_error={exc!r}"

            with self._lock:
                if (
                    self._running is not None
                    and self._running.job_id == job.job_id
                    and self._running_dispatch_id == dispatch_id
                ):
                    self._append_result_locked(
                        InferenceResult(
                            service_epoch=self.service_epoch,
                            dispatch_id=dispatch_id or "",
                            session_id=job.session_id,
                            round_id=job.round_id,
                            generation_id=job.generation_id,
                            job_id=job.job_id,
                            status="watchdog_terminated",
                            payload_json="{}",
                            error=error,
                        )
                    )
                    self._running = None
                    self._running_dispatch_id = None
                self._pending = None
                self._fatal_error = error
                self._ready = False
                self._failed = True

            self._stop.set()
            self._wake.set()
            break

    def stop(self, timeout=None):
        timeout = (
            self.shutdown_timeout
            if timeout is None
            else max(0.0, float(timeout))
        )
        deadline = time.monotonic() + timeout

        def remaining(cap=None):
            left = max(0.0, deadline - time.monotonic())
            return left if cap is None else min(left, float(cap))

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pending = None
            self._generation.value += 1
            self._dispatch_seq += 1
            self._active_dispatch_seq.value = self._dispatch_seq

        self._child_stop.set()
        self._stop.set()
        self._wake.set()
        try:
            self._parent_conn.send({"type": "stop"})
        except Exception:
            pass

        self._bridge.join(timeout=remaining(0.2))
        if self._watchdog is not threading.current_thread():
            self._watchdog.join(timeout=remaining(0.15))
        self._process.join(timeout=remaining())

        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=0.25)
        if self._process.is_alive() and hasattr(self._process, "kill"):
            self._process.kill()
            self._process.join(timeout=0.25)

        try:
            self._parent_conn.close()
        except Exception:
            pass
