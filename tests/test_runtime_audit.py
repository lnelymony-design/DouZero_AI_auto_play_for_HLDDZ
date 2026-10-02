import json
import os
import tempfile
import unittest

from runtime_audit import LiveAuditWriter


class LiveAuditWriterTests(unittest.TestCase):
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
