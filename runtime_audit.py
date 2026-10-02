"""Crash-resilient live audit writer.

The live recognition thread only submits detached payload snapshots. A single
background writer performs JSON serialization and atomic replacement in the
same directory. Intermediate revisions may be coalesced; finalization is never
coalesced away.
"""

import copy
from collections import deque
import json
import math
import os
import threading
import time


class LiveAuditWriter:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="LiveAuditWriter",
            daemon=True,
        )

        self._pending_live = {}
        self._pending_final = deque()
        self._persisted_seq = {}
        self._submitted_seq = {}
        self._finalized = set()
        self._finalizing = set()
        self._coalesced = {}
        self._last_error = None
        self._last_reported_error = None
        self._last_final_path = None
        self._inflight = False
        self._thread.start()

    @staticmethod
    def _json_safe(value):
        """Return a JSON-safe detached structure.

        Audit durability is more important than preserving a non-finite model
        diagnostic. NaN/Infinity are normalized to null instead of making the
        writer retry forever under allow_nan=False.
        """
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {
                str(key): LiveAuditWriter._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                LiveAuditWriter._json_safe(item)
                for item in value
            ]
        return value

    @staticmethod
    def _safe_round_id(round_id):
        return "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in str(round_id)
        )

    def live_path(self, round_id):
        safe = self._safe_round_id(round_id)
        return os.path.join(self.root, "live_%s.json" % safe)

    def _atomic_write_json(self, path, payload):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(
                payload,
                fp,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, path)

    def submit_live(self, round_id, event_seq, payload):
        detached = self._json_safe(copy.deepcopy(payload))
        round_id = str(round_id)
        event_seq = int(event_seq)

        with self._lock:
            if (
                round_id in self._finalized
                or round_id in self._finalizing
            ):
                return False

            previous = self._pending_live.get(round_id)
            if previous is not None:
                self._coalesced[round_id] = (
                    self._coalesced.get(round_id, 0) + 1
                )

            self._submitted_seq[round_id] = max(
                event_seq,
                self._submitted_seq.get(round_id, 0),
            )
            self._pending_live[round_id] = (
                event_seq,
                detached,
            )
        self._wake.set()
        return True

    def finalize(self, round_id, event_seq, payload, final_name):
        detached = self._json_safe(copy.deepcopy(payload))
        round_id = str(round_id)
        event_seq = int(event_seq)

        with self._lock:
            if (
                round_id in self._finalized
                or round_id in self._finalizing
            ):
                return False
            self._finalizing.add(round_id)
            self._pending_live.pop(round_id, None)
            self._submitted_seq[round_id] = max(
                event_seq,
                self._submitted_seq.get(round_id, 0),
            )
            self._pending_final.append(
                (round_id, event_seq, detached, str(final_name))
            )
        self._wake.set()
        return True

    def is_persisted(self, round_id, min_seq=1):
        with self._lock:
            return self._persisted_seq.get(str(round_id), 0) >= int(min_seq)

    def wait_persisted(self, round_id, min_seq=1, timeout=2.0):
        """Wait until a submitted revision is durably replaced on disk."""
        round_id = str(round_id)
        min_seq = int(min_seq)
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            with self._lock:
                persisted = self._persisted_seq.get(round_id, 0)
                error = self._last_error
                inflight = self._inflight
                pending_final = len(self._pending_final)
                pending_live = len(self._pending_live)
            if persisted >= min_seq:
                return {
                    "ok": True,
                    "persisted_seq": persisted,
                    "last_error": error,
                    "inflight": inflight,
                    "pending_final": pending_final,
                    "pending_live": pending_live,
                }
            self._wake.set()
            time.sleep(0.02)

        with self._lock:
            return {
                "ok": self._persisted_seq.get(round_id, 0) >= min_seq,
                "persisted_seq": self._persisted_seq.get(round_id, 0),
                "last_error": self._last_error,
                "inflight": self._inflight,
                "pending_final": len(self._pending_final),
                "pending_live": len(self._pending_live),
            }

    def status(self, round_id=None):
        with self._lock:
            if round_id is None:
                return {
                    "last_error": self._last_error,
                    "pending_live": len(self._pending_live),
                    "pending_final": len(self._pending_final),
                    "inflight": self._inflight,
                    "root": self.root,
                    "last_final_path": self._last_final_path,
                }
            key = str(round_id)
            return {
                "submitted_seq": self._submitted_seq.get(key, 0),
                "persisted_seq": self._persisted_seq.get(key, 0),
                "coalesced": self._coalesced.get(key, 0),
                "finalized": key in self._finalized,
                "finalizing": key in self._finalizing,
                "last_error": self._last_error,
                "inflight": self._inflight,
                "root": self.root,
                "last_final_path": self._last_final_path,
            }

    def flush(self, timeout=0.75):
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            with self._lock:
                pending = bool(
                    self._pending_live
                    or self._pending_final
                    or self._inflight
                )
            if not pending:
                return True
            self._wake.set()
            time.sleep(0.02)
        return False

    def close(self, timeout=0.75):
        self.flush(timeout=max(0.0, float(timeout)) / 2.0)
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=max(0.0, float(timeout)) / 2.0)

    def _next_request(self):
        with self._lock:
            if self._pending_final:
                return ("final", self._pending_final.popleft())
            if self._pending_live:
                round_id = next(iter(self._pending_live))
                event_seq, payload = self._pending_live.pop(round_id)
                return (
                    "live",
                    (round_id, event_seq, payload),
                )
        return None

    def _persist_live(self, round_id, event_seq, payload):
        payload["persisted_seq"] = int(event_seq)
        payload["audit_writer"] = {
            "coalesced_revisions": self._coalesced.get(round_id, 0),
            "atomic_replace": True,
        }
        self._atomic_write_json(self.live_path(round_id), payload)
        with self._lock:
            self._persisted_seq[round_id] = max(
                int(event_seq),
                self._persisted_seq.get(round_id, 0),
            )
            self._last_error = None

    def _persist_final(
        self,
        round_id,
        event_seq,
        payload,
        final_name,
    ):
        payload["persisted_seq"] = int(event_seq)
        payload["audit_writer"] = {
            "coalesced_revisions": self._coalesced.get(round_id, 0),
            "atomic_replace": True,
            "finalized": True,
        }

        final_path = os.path.join(self.root, final_name)
        self._atomic_write_json(final_path, payload)

        live_path = self.live_path(round_id)
        try:
            if os.path.exists(live_path):
                os.remove(live_path)
        except OSError:
            # Final is already durable. Failure to remove live is non-fatal.
            pass

        with self._lock:
            self._persisted_seq[round_id] = max(
                int(event_seq),
                self._persisted_seq.get(round_id, 0),
            )
            self._finalizing.discard(round_id)
            self._finalized.add(round_id)
            self._last_error = None
            self._last_reported_error = None
            self._last_final_path = os.path.abspath(final_path)

        print(
            "推牌审计final已落盘 >>> "
            f"{os.path.abspath(final_path)}"
        )

    def _run(self):
        while True:
            request = self._next_request()
            if request is None:
                if self._stop.is_set():
                    break
                self._wake.wait(0.25)
                self._wake.clear()
                continue

            kind, data = request
            with self._lock:
                self._inflight = True

            try:
                if kind == "live":
                    self._persist_live(*data)
                else:
                    self._persist_final(*data)
            except Exception as exc:
                report_error = False
                with self._lock:
                    self._last_error = repr(exc)
                    if self._last_reported_error != self._last_error:
                        self._last_reported_error = self._last_error
                        report_error = True

                    if not self._stop.is_set():
                        if kind == "live":
                            round_id, event_seq, payload = data
                            if (
                                round_id not in self._finalized
                                and round_id not in self._finalizing
                                and round_id not in self._pending_live
                            ):
                                self._pending_live[round_id] = (
                                    event_seq,
                                    payload,
                                )
                        else:
                            self._pending_final.appendleft(data)
                if report_error:
                    print(
                        "推牌审计后台写入失败 >>> "
                        f"root={self.root} error={self._last_error}"
                    )
                # Preserve the previous valid file and retry without spinning.
                time.sleep(0.10)
            finally:
                with self._lock:
                    self._inflight = False
                self._wake.set()
