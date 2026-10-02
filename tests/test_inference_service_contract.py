import pickle
import unittest
from dataclasses import FrozenInstanceError

from inference.inference_service import InferenceJob


class InferenceServiceContractTests(unittest.TestCase):
    def _job(self):
        return InferenceJob(
            session_id="session",
            round_id="round",
            generation_id=11,
            job_id="job",
            created_monotonic=123.0,
            state_json='{"my_position":"landlord","history":[]}',
            candidates=("55", "66", "Pass"),
            response_samples=240,
            max_worlds=8,
            max_job_age_seconds=6.0,
        )

    def test_job_is_frozen(self):
        job = self._job()
        with self.assertRaises(FrozenInstanceError):
            job.generation_id = 12

    def test_job_pickle_round_trip_preserves_snapshot(self):
        job = self._job()
        restored = pickle.loads(pickle.dumps(job))
        self.assertEqual(restored, job)
        self.assertEqual(restored.candidates, ("55", "66", "Pass"))
        self.assertIsInstance(restored.state_json, str)


if __name__ == "__main__":
    unittest.main()
