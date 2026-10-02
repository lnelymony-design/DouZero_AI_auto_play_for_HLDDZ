import json
import os
import tempfile
import unittest

from runtime_audit import LiveAuditWriter


class LiveAuditWriterTests(unittest.TestCase):
    def test_final_persistence_is_absolute_and_sanitizes_non_finite(self):
        with tempfile.TemporaryDirectory() as root:
            writer = LiveAuditWriter(root)
            try:
                final_name = "round-final.json"
                payload = {
                    "event_seq": 5,
                    "nan_value": float("nan"),
                    "inf_value": float("inf"),
                }
                self.assertTrue(
                    writer.finalize(
                        "round-final",
                        5,
                        payload,
                        final_name,
                    )
                )
                status = writer.wait_persisted(
                    "round-final",
                    min_seq=5,
                    timeout=2.0,
                )
                self.assertTrue(status["ok"])
                final_path = os.path.join(
                    writer.root,
                    final_name,
                )
                self.assertTrue(os.path.isabs(final_path))
                self.assertTrue(os.path.exists(final_path))

                with open(final_path, "r", encoding="utf-8") as fp:
                    loaded = json.load(fp)
                self.assertIsNone(loaded["nan_value"])
                self.assertIsNone(loaded["inf_value"])
            finally:
                writer.close(0.5)

    def test_finalize_sync_writes_primary_and_mirror_and_reopens(self):
        with tempfile.TemporaryDirectory() as root:
            writer = LiveAuditWriter(
                os.path.join(root, "primary")
            )
            try:
                mirror = os.path.join(
                    root,
                    "mirror",
                    "round-sync.json",
                )
                result = writer.finalize_sync(
                    "round-sync",
                    7,
                    {
                        "event_seq": 7,
                        "value": "ok",
                        "nan_value": float("nan"),
                    },
                    "round-sync.json",
                    mirror_path=mirror,
                    verify_delay=0,
                )
                self.assertTrue(result["ok"])
                self.assertTrue(os.path.exists(result["path"]))
                self.assertTrue(
                    os.path.exists(result["mirror_path"])
                )
                self.assertGreater(result["size"], 0)
                self.assertGreater(result["mirror_size"], 0)

                with open(
                    result["path"],
                    "r",
                    encoding="utf-8",
                ) as fp:
                    primary = json.load(fp)
                with open(
                    result["mirror_path"],
                    "r",
                    encoding="utf-8",
                ) as fp:
                    mirrored = json.load(fp)

                self.assertEqual(primary["value"], "ok")
                self.assertEqual(mirrored["value"], "ok")
                self.assertIsNone(primary["nan_value"])
            finally:
                writer.close(0.5)

    def test_live_updates_are_rejected_after_finalize_is_queued(self):
        with tempfile.TemporaryDirectory() as root:
            writer = LiveAuditWriter(root)
            try:
                self.assertTrue(
                    writer.submit_live(
                        "round-finalizing",
                        1,
                        {"event_seq": 1},
                    )
                )
                self.assertTrue(writer.flush(1.0))
                self.assertTrue(
                    writer.finalize(
                        "round-finalizing",
                        2,
                        {"event_seq": 2},
                        "round-finalizing.json",
                    )
                )
                self.assertFalse(
                    writer.submit_live(
                        "round-finalizing",
                        3,
                        {"event_seq": 3},
                    )
                )
                self.assertTrue(writer.flush(1.0))
                self.assertFalse(
                    os.path.exists(
                        writer.live_path("round-finalizing")
                    )
                )
            finally:
                writer.close(0.5)

    def test_live_revision_and_finalize_are_atomic_and_parseable(self):
        with tempfile.TemporaryDirectory() as root:
            writer = LiveAuditWriter(root)
            try:
                payload = {
                    "format": "wechat_live_audit_v5",
                    "event_seq": 1,
                    "value": "first",
                }
                self.assertTrue(
                    writer.submit_live("round-1", 1, payload)
                )
                self.assertTrue(writer.flush(1.0))

                live_path = writer.live_path("round-1")
                self.assertTrue(os.path.exists(live_path))
                with open(live_path, "r", encoding="utf-8") as fp:
                    live = json.load(fp)
                self.assertEqual(live["persisted_seq"], 1)
                self.assertEqual(live["value"], "first")

                final_payload = dict(payload)
                final_payload["event_seq"] = 2
                final_payload["value"] = "final"
                self.assertTrue(
                    writer.finalize(
                        "round-1",
                        2,
                        final_payload,
                        "round-1-final.json",
                    )
                )
                self.assertTrue(writer.flush(1.0))

                final_path = os.path.join(
                    root,
                    "round-1-final.json",
                )
                self.assertTrue(os.path.exists(final_path))
                with open(final_path, "r", encoding="utf-8") as fp:
                    final = json.load(fp)
                self.assertEqual(final["persisted_seq"], 2)
                self.assertEqual(final["value"], "final")
                self.assertFalse(os.path.exists(live_path))
            finally:
                writer.close(0.5)


if __name__ == "__main__":
    unittest.main()
