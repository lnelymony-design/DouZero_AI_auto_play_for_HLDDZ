import pickle
import unittest
from dataclasses import FrozenInstanceError

from inference.rollout_service import RolloutJob


class RolloutServiceContractTests(unittest.TestCase):
    def _job(self):
        return RolloutJob(
            session_id="session",
            round_id="round",
            generation_id=7,
            posterior_revision=3,
            job_id="job",
            suggestion_id="suggestion",
            state_hash="abc",
            created_at_utc="2026-10-02T00:00:00.000Z",
            created_monotonic=123.0,
            public_snapshot_json='{"my_position":"landlord"}',
            worlds_json="[]",
            candidates=("Pass", "3"),
            my_position="landlord",
            posterior_ess_ratio=0.8,
            model_paths=(("landlord", "a.ckpt"),),
            max_worlds=8,
            min_worlds=3,
            max_steps=72,
            time_budget_seconds=8.0,
            max_job_age_seconds=12.0,
            device="cpu",
            cpu_threads=1,
        )

    def test_job_is_frozen(self):
        job = self._job()
        with self.assertRaises(FrozenInstanceError):
            job.generation_id = 8

    def test_job_pickle_round_trip_preserves_identity(self):
        job = self._job()
        restored = pickle.loads(pickle.dumps(job))
        self.assertEqual(restored, job)
        self.assertEqual(restored.candidates, ("Pass", "3"))
        self.assertIsInstance(restored.public_snapshot_json, str)
        self.assertIsInstance(restored.worlds_json, str)


if __name__ == "__main__":
    unittest.main()
