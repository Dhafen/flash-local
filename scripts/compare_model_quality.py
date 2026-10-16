"""Compare paired LOCAL evaluation reports without executing model output.

This checks provenance fields and experimental comparability, not the truth of
a report or broad model equivalence. No models, scores or prompts are fetched.
"""
import argparse
import json
import pathlib
import re
import sys

BASELINE = 'Qwen/Qwen3.8-27B'
BASELINE_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
CATEGORIES = {'coding', 'reasoning', 'agent', 'security_review'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value, length):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None


def integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def sampling_policy(value):
    require(isinstance(value, dict), 'Explicit sampling policy required')
    kind = value.get('kind')
    require(isinstance(kind, str) and kind in ['greedy', 'categorical'], 'Unknown sampling policy')
    keys = {'kind', 'seed'} if kind == 'greedy' else {'kind', 'seed', 'temperature', 'topP', 'topK'}
    require(value.keys() == keys and integer(value.get('seed'), 0, (1 << 32) - 1),
            'Sampling fields and seed must be explicit')
    if kind == 'categorical':
        require(type(value.get('temperature')) in [int, float] and 0 < value['temperature'] <= 10
                and type(value.get('topP')) in [int, float] and 0 < value['topP'] <= 1
                and integer(value.get('topK'), 0, 1_000_000), 'Invalid sampling parameters')


def validate(report):
    require(isinstance(report, dict) and type(report.get('schemaVersion')) is int
            and report['schemaVersion'] == 1, 'Unsupported quality report schema')
    require(report.get('inferenceLocation') in ['local_iPhone', 'local_PC'], 'Only declared local inference reports are eligible')
    model = report.get('model', {})
    require(isinstance(model, dict) and isinstance(model.get('upstreamRepository'), str), 'Missing upstream model identity')
    require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', model['upstreamRepository']), 'Invalid upstream model identity')
    require(fingerprint(model.get('upstreamRevision'), 40) and fingerprint(model.get('artifactSHA256'), 64),
            'Exact upstream revision and evaluated artifact SHA-256 are required')
    require(isinstance(model.get('quantization'), str) and 1 <= len(model['quantization']) <= 100,
            'Evaluated quantization must be declared')
    require(fingerprint(report.get('runtimeSourceRevision'), 40), 'Runtime source revision is required')
    for key in ['datasetSHA256', 'graderSHA256']:
        require(fingerprint(report.get(key), 64), 'Exact dataset and grading harness fingerprints are required')
    policy = report.get('generationPolicy', {})
    require(isinstance(policy, dict) and isinstance(policy.get('thinking'), str)
            and policy['thinking'] in ['off', 'low', 'medium', 'xhigh'], 'Explicit thinking policy required')
    require(policy.keys() == {'thinking', 'contextTokens', 'maximumNewTokens', 'sampling'},
            'Unknown generation policy fields')
    require(integer(policy.get('contextTokens'), 16, 1_000_000)
            and integer(policy.get('maximumNewTokens'), 1, 100_000)
            and policy['maximumNewTokens'] < policy['contextTokens'], 'Invalid generation token budgets')
    sampling_policy(policy.get('sampling'))
    # Also reject NaN/non-JSON values in programmatic callers.
    try:
        json.dumps(policy, allow_nan=False, sort_keys=True)
    except (ValueError, TypeError):
        raise ValueError('Invalid generation policy values') from None
    rows = report.get('tasks')
    require(isinstance(rows, list) and 4 <= len(rows) <= 100_000, 'At least four explicitly graded tasks are required')
    tasks = {}
    for row in rows:
        require(isinstance(row, dict), 'Invalid task record')
        task_id = row.get('id')
        require(isinstance(task_id, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', task_id), 'Invalid task ID')
        require(task_id not in tasks, 'Duplicate task ID')
        require(row.get('category') in CATEGORIES and fingerprint(row.get('promptSHA256'), 64),
                'Task category and exact prompt fingerprint are required')
        require(type(row.get('passed')) is bool and integer(row.get('checksExecuted'), 0, 1_000_000),
                'Every task must have an explicit grading outcome')
        require(row.get('finishReason') in ['stop', 'length', 'cancelled', 'error'], 'Generation finish reason required')
        require(not row['passed'] or row['checksExecuted'] > 0 and row['finishReason'] == 'stop',
                'Unfinished or untested output cannot count as a passing task')
        tasks[task_id] = row
    require({row['category'] for row in tasks.values()} == CATEGORIES, 'All four capability categories are required')
    identity = {key: model[key] for key in ['upstreamRepository', 'upstreamRevision', 'artifactSHA256', 'quantization']}
    return identity, policy, tasks


def compare(baseline, candidate):
    bm, bp, bt = validate(baseline)
    cm, cp, ct = validate(candidate)
    require(bm['upstreamRepository'] == BASELINE and bm['upstreamRevision'] == BASELINE_REVISION,
            'The baseline must identify the pinned Qwen3.8-27B source')
    for key in ['datasetSHA256', 'graderSHA256']:
        require(baseline[key] == candidate[key], 'Dataset or grading harness differs between runs')
    require(bp == cp, 'Thinking, sampling or token budgets differ between runs')
    require(bt.keys() == ct.keys(), 'Task coverage differs; omitted tasks cannot be ignored')
    for task_id in bt:
        require(bt[task_id]['category'] == ct[task_id]['category']
                and bt[task_id]['promptSHA256'] == ct[task_id]['promptSHA256'], 'Task prompt or category differs between runs')
    categories = {}
    for category in sorted(CATEGORIES):
        ids = [key for key in bt if bt[key]['category'] == category]
        b = sum(bt[key]['passed'] for key in ids)
        c = sum(ct[key]['passed'] for key in ids)
        categories[category] = {'tasks': len(ids), 'baselinePassed': b, 'candidatePassed': c,
                                'candidateMeetsObservedFloor': c >= b}
    return {
        'schemaVersion': 1,
        'baseline': bm,
        'candidate': cm,
        'executions': {
            label: {key: report[key] for key in ['inferenceLocation', 'runtimeSourceRevision']}
            for label, report in [('baseline', baseline), ('candidate', candidate)]
        },
        'datasetSHA256': baseline['datasetSHA256'],
        'graderSHA256': baseline['graderSHA256'],
        'generationPolicy': bp,
        'categories': categories,
        'regressedTaskIDs': sorted(key for key in bt if bt[key]['passed'] and not ct[key]['passed']),
        'meetsObservedTaskFloor': all(row['candidateMeetsObservedFloor'] for row in categories.values()),
        'generalQualityEquivalenceEstablished': False,
        'reportAuthenticityVerified': False,
        'speedComparisonPerformed': False,
    }


def read_report(path):
    with pathlib.Path(path).open('rb') as stream:
        data = stream.read((4 << 20) + 1)
    require(len(data) <= 4 << 20, 'Quality report exceeds 4 MiB')
    return json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True, help='Local aggregate JSON; no prompts or generated code are copied')
    args = parser.parse_args()
    try:
        result = compare(read_report(args.baseline), read_report(args.candidate))
        pathlib.Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError, TypeError, KeyError):
        # Reports may contain private prompts/paths; do not print their contents
        # or decoder error excerpts into CI or terminal logs.
        print('Quality comparison refused: invalid, incomplete or incomparable reports.', file=sys.stderr)
        return 2
    print('Compared paired local task outcomes; general model equivalence remains unestablished.')
    return 0 if result['meetsObservedTaskFloor'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
