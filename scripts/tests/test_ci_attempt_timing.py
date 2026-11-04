import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_ci_attempt import summarize


class CIAttemptTimingTests(unittest.TestCase):
    def test_runner_never_acquired_counts_as_queue(self):
        job = {
            'status': 'completed', 'conclusion': 'cancelled',
            'runner_id': 0, 'steps': [],
            'created_at': '2026-11-04T17:19:27Z',
            'started_at': '2026-11-04T17:19:27Z',
            'completed_at': '2026-11-04T17:34:28Z',
        }
        self.assertEqual(summarize(job), ('never_assigned', 901, None))

    def test_assigned_uses_first_step_instead_of_misleading_start(self):
        job = {
            'status': 'completed', 'conclusion': 'failure', 'runner_id': 17,
            'created_at': '2026-11-04T17:19:27Z',
            'started_at': '2026-11-04T17:19:27Z',
            'completed_at': '2026-11-04T17:22:27Z',
            'steps': [{'started_at': '2026-11-04T17:21:27Z'}],
        }
        self.assertEqual(summarize(job), ('executed', 120, 60))

    def test_unknown_and_skipped_have_no_duration(self):
        self.assertEqual(summarize({'conclusion': 'skipped'}), ('skipped', None, None))
        self.assertEqual(summarize({'status': 'queued', 'runner_id': 0}), ('queued', None, None))
        self.assertEqual(summarize({'runner_id': 19}), ('unknown', None, None))
