import unittest

from inference.play_evidence import OpponentPlayEvidence


class OpponentPlayEvidenceTests(unittest.TestCase):
    def test_single_frame_candidate_survives_until_count_confirmation(self):
        cache = OpponentPlayEvidence(max_age_seconds=6.0)
        cache.observe(
            "left",
            "345678",
            10.0,
            source="single_frame",
        )
        found = cache.best("left", 6, 14.5, max_age_seconds=5.5)
        self.assertIsNotNone(found)
        self.assertEqual(found["cards"], "345678")
        self.assertEqual(found["source"], "single_frame")

    def test_wrong_card_count_is_not_selected(self):
        cache = OpponentPlayEvidence(max_age_seconds=6.0)
        cache.observe(
            "right",
            "T9876",
            20.0,
            source="single_frame",
        )
        self.assertIsNone(
            cache.best("right", 6, 20.5, max_age_seconds=5.5)
        )

    def test_count_trigger_evidence_beats_weaker_single_frame(self):
        cache = OpponentPlayEvidence(max_age_seconds=6.0)
        cache.observe(
            "left",
            "334455",
            30.0,
            source="single_frame",
        )
        cache.observe(
            "left",
            "345678",
            30.2,
            source="count_trigger",
            expected_drop=6,
        )
        found = cache.best("left", 6, 30.3)
        self.assertEqual(found["cards"], "345678")
        self.assertEqual(found["source"], "count_trigger")

    def test_clear_side_removes_only_that_side(self):
        cache = OpponentPlayEvidence(max_age_seconds=6.0)
        cache.observe("left", "AA", 40.0)
        cache.observe("right", "KK", 40.0)
        cache.clear("left")
        self.assertIsNone(cache.best("left", 2, 40.1))
        self.assertEqual(
            cache.best("right", 2, 40.1)["cards"],
            "KK",
        )


if __name__ == "__main__":
    unittest.main()
