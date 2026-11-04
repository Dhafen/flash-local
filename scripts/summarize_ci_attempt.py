#!/usr/bin/env python3
"""Separate GitHub Actions waiting from elapsed time after the first step."""
import argparse
import json
from datetime import datetime
from pathlib import Path


def _elapsed(start, end):
    if not start or not end:
        return None
    delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    seconds = delta.total_seconds()
    return round(seconds, 3) if seconds >= 0 else None


def summarize(job):
    """Return phase, seconds to first step (or cancellation), and execution seconds."""
    if job.get('conclusion') == 'skipped':
        return 'skipped', None, None
    step_start = min(
        (step['started_at'] for step in job.get('steps') or [] if step.get('started_at')),
        default=None,
    )
    created, completed = job.get('created_at'), job.get('completed_at')
    if step_start:
        return 'executed', _elapsed(created, step_start), _elapsed(step_start, completed)
    if job.get('runner_id') == 0 and job.get('status') == 'completed':
        return 'never_assigned', _elapsed(created, completed), None
    if job.get('status') == 'queued':
        return 'queued', None, None
    return 'unknown', None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobs', type=Path, help='GitHub Actions jobs response for one attempt')
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text(encoding='utf-8'))['jobs']
    results = []
    for job in jobs:
        phase, wait, execution = summarize(job)
        results.append({
            'id': job['id'], 'name': job['name'], 'conclusion': job.get('conclusion'),
            'phase': phase, 'wait_or_startup_seconds': wait, 'execution_seconds': execution,
        })
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
