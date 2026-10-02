import json
import pickle
import time
import unittest
from dataclasses import FrozenInstanceError

from inference.inference_service import InferenceJob, InferenceService


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

    def test_spawned_service_returns_detached_posterior(self):
        state = {
            "my_position": "landlord",
            "initial_my_hand": "333444555666777888DX",
            "three_landlord_cards": "DX8",
            "history": [],
            "sample_count": 200,
            "pass_penalty": 0.62,
            "friendly_pass_penalty": 0.86,
            "play_behavior_floor": 0.18,
            "play_behavior_strength": 1.0,
            "behavior_temperature": 0.75,
            "min_effective_sample_ratio": 0.28,
            "residual_behavior_floor": 0.35,
            "residual_behavior_strength": 1.0,
            "residual_behavior_temperature": 0.55,
            "random_seed": 19,
        }
        service = InferenceService(shutdown_timeout=1.0)
        try:
            job = InferenceJob(
                session_id="session",
                round_id="round",
                generation_id=1,
                job_id="integration-job",
                created_monotonic=time.monotonic(),
                state_json=json.dumps(state),
                candidates=(),
                response_samples=100,
                max_worlds=4,
                max_job_age_seconds=8.0,
            )
            dispatch_id = service.submit(job)
            self.assertIsNotNone(dispatch_id)

            deadline = time.monotonic() + 10.0
            result = None
            while time.monotonic() < deadline and result is None:
                items = service.drain_results(4)
                if items:
                    result = items[-1]
                    break
                time.sleep(0.05)

            self.assertIsNotNone(result)
            self.assertEqual(result.status, "ok")
            payload = json.loads(result.payload_json)
            self.assertEqual(payload["inference"]["samples"], 200)
            self.assertTrue(payload["worlds"])
        finally:
            service.stop(1.0)

    def test_job_pickle_round_trip_preserves_snapshot(self):
        job = self._job()
        restored = pickle.loads(pickle.dumps(job))
        self.assertEqual(restored, job)
        self.assertEqual(restored.candidates, ("55", "66", "Pass"))
        self.assertIsInstance(restored.state_json, str)


if __name__ == "__main__":
    unittest.main()
