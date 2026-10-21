"""Deterministic live-phone evaluation. No model runner or fabricated benchmark.

All fixtures are generated locally. Model-generated Python runs only inside an
explicitly selected, already-installed, digest-pinned Docker/Podman image.
"""
import ast
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
import uuid

from flash_local_agent import strict_json
from flash_local_verify import stream_completion, verify, benchmark


def exact(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(value, list):
        return len(value) == len(expected) and all(exact(a, b) for a, b in zip(value, expected))
    if isinstance(value, dict):
        return value.keys() == expected.keys() and all(exact(value[k], expected[k]) for k in value)
    return value == expected


def answer_json(text):
    if not isinstance(text, str):
        raise ValueError('Missing answer')
    value = text.strip()
    if value.startswith('```json\n') and value.endswith('\n```'):
        value = value[8:-4]
    value = strict_json(value)
    if not isinstance(value, dict) or set(value) != {'answer'}:
        raise ValueError('Expected only the answer field')
    return value['answer']


def logic_cases(seed=42183):
    rng = random.Random(seed)
    cases = []
    for index in range(40):
        a, b, c = [rng.randrange(2, 40) for _ in range(3)]
        language = 'sv' if index % 2 else 'en'
        group = index // 10
        if group == 0:
            task, expected = f'A box has {a} rows of {b} items. Remove {c} items. How many remain?', a * b - c
            if language == 'sv':
                task = f'En låda har {a} rader med {b} föremål i varje. Ta bort {c} föremål. Hur många återstår?'
        elif group == 1:
            task = f'An integer x must satisfy x >= {a}, x <= {a + 12}, x % 3 == {b % 3}. Return all possible x in ascending order.'
            expected = [x for x in range(a, a + 13) if x % 3 == b % 3]
            if language == 'sv':
                task = f'Heltalet x ska uppfylla x >= {a}, x <= {a + 12}, x % 3 == {b % 3}. Returnera alla möjliga x i stigande ordning.'
        elif group == 2:
            edges = [(x, y) for x in range(7) for y in range(7) if x != y and rng.random() < 0.18]
            distances, queue = {0: 0}, deque([0])
            while queue:
                node = queue.popleft()
                for start, end in edges:
                    if start == node and end not in distances:
                        distances[end] = distances[node] + 1; queue.append(end)
            task = f'Directed unweighted graph edges: {json.dumps(edges)}. Return the shortest distance from node 0 to node 6, or -1 if unreachable.'
            expected = distances.get(6, -1)
            if language == 'sv':
                task = f'En riktad oviktad graf har kanterna {json.dumps(edges)}. Returnera kortaste avståndet från nod 0 till nod 6, eller -1 om ingen väg finns.'
        else:
            steps = [rng.choice(['add', 'multiply', 'subtract']) for _ in range(4)]
            operands = [rng.randrange(1, 8) for _ in steps]
            expected = a
            for op, n in zip(steps, operands):
                expected = expected + n if op == 'add' else expected * n if op == 'multiply' else expected - n
            task = f'Start with x={a}. Apply in this order: {json.dumps(list(zip(steps, operands)))}. Return the final integer x.'
            if language == 'sv':
                task = f'Börja med x={a}. Utför operationerna i denna ordning: {json.dumps(list(zip(steps, operands)))}. add betyder addera, multiply multiplicera och subtract subtrahera. Returnera slutvärdet som heltal.'
        prefix = 'Lös uppgiften. Returnera endast JSON med nyckeln answer. ' if language == 'sv' else 'Solve this task. Return only JSON with key answer. '
        cases.append({'id': f'logic-{index:02}', 'category': 'logic', 'language': language,
                      'prompt': prefix + task, 'expected': expected})
    return cases


def definition(name, properties, description):
    return {'type': 'function', 'function': {'name': name, 'description': description, 'strict': True,
        'parameters': {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}}}


SUM = definition('local_sum', {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'Add two integers locally.')
PRODUCT = definition('local_product', {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'Multiply two integers locally.')
ECHO = definition('local_echo', {'text': {'type': 'string'}}, 'Return the supplied exact text locally.')


def tool_cases(seed=42183):
    rng, cases = random.Random(seed), []
    for index in range(100):
        a, b = rng.randrange(-999, 1000), rng.randrange(-999, 1000)
        if index < 30:
            prompt, name, arguments = f'Use the local_sum tool to add a={a} and b={b}.', 'local_sum', {'a': a, 'b': b}
            tools = [SUM]
        elif index < 60:
            multiply = index % 2 == 0
            prompt = f'Use the correct declared tool to {"multiply" if multiply else "add"} a={a} and b={b}.'
            name, arguments, tools = 'local_product' if multiply else 'local_sum', {'a': a, 'b': b}, [SUM, PRODUCT]
        elif index < 80:
            prompt = f'I want to add a={a} to another integer, but have not specified b. Ask me for b; do not invent it or call a tool.'
            name, arguments, tools = None, None, [SUM]
        else:
            value = f'Åäö case {index}: "quoted" \\path\\file\nline {a}'
            prompt = 'Call local_echo with exactly this JSON string as text: ' + json.dumps(value, ensure_ascii=False)
            name, arguments, tools = 'local_echo', {'text': value}, [ECHO]
        cases.append({'id': f'tool-{index:03}', 'category': 'tool', 'prompt': prompt,
                      'tools': tools, 'expected_name': name, 'expected_arguments': arguments})
    return cases


def valid_call(message, tools):
    calls = message.get('tool_calls', [])
    if not isinstance(calls, list) or len(calls) > 1:
        raise ValueError('Expected at most one complete call')
    if not calls:
        return None
    call = calls[0]
    if not isinstance(call.get('id'), str) or not call['id'] or call.get('type') != 'function':
        raise ValueError('Missing typed call ID')
    function = call.get('function', {})
    tool = next((x['function'] for x in tools if x['function']['name'] == function.get('name')), None)
    if tool is None or not isinstance(function.get('arguments'), str):
        raise ValueError('Undeclared call')
    arguments = strict_json(function['arguments'])
    properties = tool['parameters']['properties']
    if not isinstance(arguments, dict) or set(arguments) != set(properties):
        raise ValueError('Wrong argument fields')
    for key, schema in properties.items():
        expected_type = {'integer': int, 'string': str}[schema['type']]
        if type(arguments[key]) is not expected_type or isinstance(arguments[key], str) and len(arguments[key]) > 4096:
            raise ValueError('Wrong argument type or bound')
    return call, function['name'], arguments


def complete(client, model, prompt=None, messages=None, tools=None, streaming=False, maximum=2048):
    effort = getattr(client, 'evaluation_effort', 'medium')
    if not isinstance(effort, str): effort = 'medium'
    maximum = max(maximum, 4096 if effort == 'high' else 1)
    payload = {'model': model, 'messages': messages or [{'role': 'user', 'content': prompt}],
               'temperature': 0, 'seed': 42, 'reasoning_effort': effort, 'max_tokens': maximum}
    if tools:
        payload.update(tools=tools, tool_choice='auto', parallel_tool_calls=False)
    start = time.monotonic()
    response = stream_completion(client, payload) if streaming else client.json('/chat/completions', payload)
    choice = response['choices'][0]
    if choice.get('finish_reason') not in {'stop', 'tool_calls'}:
        raise ValueError('Incomplete or token-limited model response')
    message = choice['message']
    if message.get('tool_calls') and choice['finish_reason'] != 'tool_calls':
        raise ValueError('Invalid call finish reason')
    return message, response.get('usage', {}), time.monotonic() - start


def run_case(client, model, case):
    row = {'id': case['id'], 'category': case['category'], 'passed': False, 'format_valid': False}
    try:
        message, usage, seconds = complete(client, model, case['prompt'], tools=case.get('tools'),
                                           streaming=int(case['id'].split('-')[-1]) % 2 == 0)
        row.update(usage=usage, wall_seconds=seconds)
        if case['category'] == 'logic':
            value = answer_json(message.get('content'))
            row.update(format_valid=True, passed=exact(value, case['expected']))
        else:
            call = valid_call(message, case['tools'])
            row['format_valid'] = True
            if case['expected_name'] is None:
                row['passed'] = call is None and isinstance(message.get('content'), str) and '?' in message['content']
            else:
                row['passed'] = call is not None and call[1] == case['expected_name'] and exact(call[2], case['expected_arguments'])
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        row['failure_type'] = type(error).__name__
    return row


def agent_case(client, model, index):
    # Real fresh local fixture data, unavailable to the first model prompt.
    receipt, record_id = 'LOCAL-' + uuid.uuid4().hex, uuid.uuid4().hex
    a, b = 17 + index, 29 + index
    tools = [definition('lookup_fixture', {'name': {'type': 'string'}}, 'Find a named authorized local fixture ID.'),
             definition('read_fixture', {'id': {'type': 'string'}}, 'Read an authorized local fixture by ID.'), SUM]
    messages = [{'role': 'user', 'content': 'Find the local fixture named invoice, read it, call local_sum on its a and b, then reply only with JSON {"answer":{"receipt":the receipt from the fixture,"sum":the tool sum}}. Use the declared tools in that order.'}]
    stages, usage, ids = [], [], set()
    row = {'id': f'agent-{index:02}', 'category': 'agent', 'passed': False}
    try:
        for _ in range(6):
            message, measured, _ = complete(client, model, messages=messages, tools=tools, streaming=index % 2 == 0)
            usage.append(measured)
            call = valid_call(message, tools)
            if call is None:
                row['passed'] = stages == ['lookup_fixture', 'read_fixture', 'local_sum'] and exact(answer_json(message.get('content')), {'receipt': receipt, 'sum': a + b})
                break
            raw, name, arguments = call
            if raw['id'] in ids:
                raise ValueError('Repeated call ID')
            ids.add(raw['id'])
            if name == 'lookup_fixture' and not stages and arguments == {'name': 'invoice'}:
                result = {'id': record_id}
            elif name == 'read_fixture' and stages == ['lookup_fixture'] and arguments == {'id': record_id}:
                result = {'a': a, 'b': b, 'receipt': receipt}
            elif name == 'local_sum' and stages == ['lookup_fixture', 'read_fixture'] and arguments == {'a': a, 'b': b}:
                result = {'sum': a + b}
            else:
                raise ValueError('Wrong tool order or arguments')
            stages.append(name)
            messages.extend([message, {'role': 'tool', 'tool_call_id': raw['id'], 'content': json.dumps(result)}])
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        row['failure_type'] = type(error).__name__
    row.update(observed_tool_sequence=stages, usages=usage, tool_environment='controlled local fixtures; real phone inference required')
    return row


def code_cases():
    specifications = [
        ('add', 'a,b', 'a+b', [(3, 4), (-7, 2), (0, 0)]),
        ('product', 'a,b', 'a*b', [(3, 4), (-7, 2), (0, 9)]),
        ('clamp', 'x,lo,hi', 'max(lo,min(hi,x))', [(-5, 0, 8), (3, 0, 8), (20, 0, 8)]),
        ('total', 'xs', 'sum(xs)', [([],), ([1,-2,9],), ([0,0],)]),
        ('squares', 'xs', '[x*x for x in xs]', [([],), ([-2,0,3],)]),
        ('evens', 'xs', '[x for x in xs if x%2==0]', [([],), ([-3,-2,0,5,6],)]),
        ('unique_sorted', 'xs', 'sorted(set(xs))', [([],), ([3,1,3,-1],)]),
        ('reverse_text', 's', 's[::-1]', [('',), ('abc',), ('Åäö',)]),
        ('normalize_space', 's', '" ".join(s.split())', [(' a  b\n c ',), ('',)]),
        ('count_a', 's', 's.lower().count("a")', [('ABa',), ('xyz',)]),
        ('prefix_sums', 'xs', '[sum(xs[:i+1]) for i in range(len(xs))]', [([],), ([2,-1,4],)]),
        ('rotate_left', 'xs', 'xs[1:]+xs[:1]', [([],), ([1],), ([1,2,3],)]),
        ('nonzero', 'xs', '[x for x in xs if x!=0]', [([],), ([0,-2,0,4],)]),
        ('abs_total', 'xs', 'sum(abs(x) for x in xs)', [([],), ([-3,2,-1],)]),
        ('longest_length', 'xs', 'max([len(s) for s in xs],default=0)', [([],), (['a','abcd','xy'],)]),
        ('is_palindrome', 's', 's==s[::-1]', [('',), ('abba',), ('abca',)]),
        ('zip_add', 'xs,ys', '[a+b for a,b in zip(xs,ys)]', [([],[]), ([1,2],[4]), ([-1],[1])]),
        ('positive_count', 'xs', 'sum(1 for x in xs if x>0)', [([],), ([-1,0,2,3],)]),
        ('descending', 'xs', 'sorted(xs,reverse=True)', [([],), ([2,1,2,-4],)]),
        ('word_count', 's', 'len(s.split())', [('',), (' a b\nc ',)])]
    return [{'id': f'code-{i:02}', 'name': name, 'signature': signature,
             'reference': expression, 'inputs': inputs} for i, (name, signature, expression, inputs) in enumerate(specifications)]


def code_case(client, model, case, runner, image):
    row = {'id': case['id'], 'category': 'code', 'passed': False}
    if not runner or not image or not re.fullmatch(r'[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}', image) or not shutil.which(runner):
        return {**row, 'blocked': 'Explicit installed container runtime and digest-pinned Python image required'}
    try:
        subprocess.run([runner, 'image', 'inspect', image], check=True, capture_output=True, timeout=15)
        prompt = f'Implement Python function {case["name"]}({case["signature"]}) with the behavior of this expression: {case["reference"]}. Return only the function source. No imports, decorators or prose.'
        message, usage, _ = complete(client, model, prompt)
        code = (message.get('content') or '').strip()
        if code.startswith('```python\n') and code.endswith('\n```'):
            code = code[10:-4]
        tree = ast.parse(code)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != case['name'] or tree.body[0].decorator_list:
            raise ValueError('Expected only the declared function')
        safe_calls = {'abs','all','any','bool','dict','enumerate','filter','float','int','len','list','map','max','min',
                      'range','reversed','round','set','sorted','str','sum','tuple','zip',case['name']}
        safe_attributes = {'split','lower','count','join','reverse','append','extend','pop','strip','startswith','endswith'}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal)):
                raise ValueError('Unsupported generated construct')
            if isinstance(node, ast.Name) and node.id.startswith('__'):
                raise ValueError('Dunder access rejected')
            if isinstance(node, ast.Attribute) and node.attr not in safe_attributes:
                raise ValueError('Unsupported attribute')
            if isinstance(node, ast.Call) and not ((isinstance(node.func, ast.Name) and node.func.id in safe_calls) or
                                                   (isinstance(node.func, ast.Attribute) and node.func.attr in safe_attributes)):
                raise ValueError('Unsupported generated call')
        reference = eval('lambda ' + case['signature'] + ': ' + case['reference'], {'__builtins__': __builtins__})
        checks = [(args, reference(*args)) for args in case['inputs']]
        program = code + '\n' + '\n'.join(f'assert {case["name"]}(*{args!r}) == {expected!r}' for args, expected in checks) + '\nprint("FLASH_LOCAL_CODE_PASS")\n'
        container = 'flash-local-eval-' + uuid.uuid4().hex
        command = [runner, 'run', '--name', container, '--rm', '--pull=never', '--network=none', '--read-only', '--cap-drop=ALL',
            '--security-opt=no-new-privileges', '--pids-limit=64', '--memory=256m', '--cpus=1', '--user=65534:65534',
            '--tmpfs=/tmp:rw,noexec,nosuid,size=16m', '-i', image, 'python', '-I', '-S', '-']
        try:
            result = subprocess.run(command, input=program, text=True, capture_output=True, timeout=15)
        finally:
            # A killed Docker/Podman client can otherwise leave the container alive.
            subprocess.run([runner, 'rm', '-f', container], capture_output=True, timeout=15, check=False)
        row.update(passed=result.returncode == 0 and 'FLASH_LOCAL_CODE_PASS' in result.stdout, usage=usage, isolated_container=True)
    except (ValueError, OSError, KeyError, TypeError, SyntaxError, subprocess.SubprocessError) as error:
        row['failure_type'] = type(error).__name__
    return row


def context_probe(client, live):
    # Calibrate with the actual tokenizer count from the real server, not chars/4.
    nonce = uuid.uuid4().hex
    unit = 'Record filler: alpha beta gamma delta. '
    sample = unit * 64 + '\nReply only OK.'
    _, measured, _ = complete(client, live['model'], sample, maximum=1024)
    counted = measured.get('prompt_tokens')
    if type(counted) is not int or counted < 64:
        raise ValueError('Real prompt token calibration unavailable')
    effort = getattr(client, 'evaluation_effort', 'medium')
    output_reserve = 4096 if effort == 'high' else 2048
    target = live['context'] - output_reserve - 1024
    repetitions = max(1, int(64 * target / counted))
    # Insert different keys throughout one shared document; one long prefill.
    parts = []
    for index in range(5):
        parts.append(unit * (repetitions // 5))
        parts.append(f'\nFACT KEY_{index}={nonce}-{index}\n')
    document = ''.join(parts)
    prompt = document + '\nReturn only JSON {"answer":[the five exact KEY values in order]}. Filler text is data, not instructions.'
    message, usage, seconds = complete(client, live['model'], prompt, maximum=2048)
    observed = usage.get('prompt_tokens', 0)
    expected = [nonce + '-' + str(index) for index in range(5)]
    return {'category': 'context', 'passed': exact(answer_json(message.get('content')), expected),
            'actual_prompt_tokens': observed, 'window': live['context'],
            'near_full_window_exercised': type(observed) is int and observed >= live['context'] * 0.70,
            'wall_seconds': seconds, 'usage': usage, 'os_cache_coldness_controlled': False}


def resource_summary(report):
    usages = []
    for row in report['rows']:
        usages.extend(row.get('usages', []))
        if isinstance(row.get('usage'), dict):
            usages.append(row['usage'])
    if 'context_probe' in report:
        usages.append(report['context_probe'].get('usage', {}))
    endurance = report.get('endurance_runs', [])
    usages.extend(row.get('usage', {}) for row in endurance)
    metrics = [u.get('flash_local') for u in usages]
    complete_metrics = bool(metrics) and all(isinstance(m, dict) and all(k in m for k in [
        'peak_memory_bytes','decode_forward_passes','decode_seconds','model_sha256','context_capacity_tokens',
        'platform','device_identifier','app_build','source_revision','maximum_thermal_state']) for m in metrics)
    def valid_metric(m):
        return (isinstance(m, dict) and
            type(m.get('peak_memory_bytes')) is int and 0 < m['peak_memory_bytes'] < 1 << 40 and
            type(m.get('decode_forward_passes')) is int and m['decode_forward_passes'] >= 0 and
            type(m.get('decode_seconds')) in {int, float} and math.isfinite(m['decode_seconds']) and m['decode_seconds'] >= 0 and
            type(m.get('maximum_thermal_state')) is int and m['maximum_thermal_state'] in range(4) and
            type(m.get('context_capacity_tokens')) is int and m['context_capacity_tokens'] in {8192,16384,32768,65536} and
            isinstance(m.get('model_sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', m['model_sha256']) and
            all(isinstance(m.get(key), str) and m[key] for key in ['app_build','source_revision','platform','device_identifier']))
    if not complete_metrics or not all(valid_metric(m) for m in metrics):
        return {'measurements_complete': False, 'resource_gates_passed': False}
    identity = {(m['model_sha256'], m['app_build'], m['source_revision'], m['context_capacity_tokens']) for m in metrics}
    phone = all(m['platform'] == 'iOSDevice' and m['device_identifier'] == 'iPhone18,2' for m in metrics)
    memory = max(m['peak_memory_bytes'] for m in metrics)
    thermal = max(m['maximum_thermal_state'] for m in metrics)
    rates = [r.get('decode_tokens_per_second') for r in endurance]
    sustained = bool(rates) and report.get('endurance_elapsed_seconds', 0) >= 600 and all(
        r.get('minimum_128_tokens_met') is True and type(rate) in {int,float} and math.isfinite(rate) and rate >= 3 for r, rate in zip(endurance, rates))
    return {'measurements_complete': True, 'one_source_model_and_context': len(identity) == 1,
            'model_sha256': metrics[0]['model_sha256'], 'source_revision': metrics[0]['source_revision'],
            'server_reports_target_iphone': phone, 'maximum_reported_process_bytes': memory,
            'maximum_reported_thermal_state': thermal, 'sustained_decode_at_least_3_tokens_per_second': sustained,
            'resource_gates_passed': phone and len(identity) == 1 and metrics[0]['context_capacity_tokens'] == report['context_capacity_tokens'] and memory <= 5_500_000_000 and thermal < 3 and sustained,
            'physical_device_independently_verified': False}


def baseline_comparison(report, baseline):
    path = Path(baseline)
    if path.stat().st_size > 8 << 20:
        raise ValueError('Oversized baseline report')
    previous = strict_json(path.read_text(encoding='utf-8'))
    valid = previous.get('full_suite') and previous.get('verification_complete') and all(
        previous.get(key) == report.get(key) for key in ['suite','fixture_seed','fixture_bank_sha256','reasoning_mode','model'])
    valid = valid and previous.get('context_capacity_tokens') == 8192 and all(
        previous.get('resources', {}).get(key) == report.get('resources', {}).get(key) and report.get('resources', {}).get(key)
        for key in ['model_sha256','source_revision'])
    if not valid:
        raise ValueError('Baseline must use the same source, model, cases and reasoning mode at 8K')
    lost = {r['id'] for r in previous['rows'] if r['category'] == 'logic' and r['passed']} - {
        r['id'] for r in report['rows'] if r['category'] == 'logic' and r['passed']}
    return {'baseline_context_tokens':8192, 'lost_logic_cases':sorted(lost), 'passed':len(lost) <= 1}


def evaluate(client, live, output, full=False, expected_context=None, long_context=False, endurance=False,
             code_runner=None, code_image=None, baseline=None, reasoning='medium'):
    if expected_context is not None and live['context'] != expected_context:
        raise ValueError('Phone session has the wrong context; change it in app and reactivate first')
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    client.evaluation_effort = {'off':'none','small':'low','medium':'medium','ultra':'high'}[reasoning]
    report = {'schema_version': 1, 'suite': 'flash-local-agent-acceptance-v1', 'fixture_seed': 42183,
        'fixture_bank_sha256':hashlib.sha256(json.dumps(logic_cases()+tool_cases()+code_cases(),sort_keys=True).encode()).hexdigest(),
        'reasoning_mode':reasoning,
        'model': live['model'], 'context_capacity_tokens': live['context'], 'full_suite': full,
        'simulated_inference': False, 'physical_device_independently_verified': False,
        'three_model_benchmark_complete': False, 'rows': [], 'verification_complete': False}
    with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
        def checkpoint():
            handle.seek(0); json.dump(report, handle, indent=2, allow_nan=False); handle.write('\n'); handle.truncate(); handle.flush()
        try:
            report['protocol'] = verify(client, live['model'], include_cancellation=True)
            bank = logic_cases() + tool_cases() if full else logic_cases()[:6] + tool_cases()[:3] + tool_cases()[60:61] + tool_cases()[80:82]
            for case in bank:
                row = run_case(client, live['model'], case); report['rows'].append(row); checkpoint()
                print(json.dumps({'case': row['id'], 'passed': row['passed']}), flush=True)
            if full:
                for index in range(10):
                    report['rows'].append(agent_case(client, live['model'], index)); checkpoint()
                for case in code_cases():
                    report['rows'].append(code_case(client, live['model'], case, code_runner, code_image)); checkpoint()
            if long_context:
                report['context_probe'] = context_probe(client, live); checkpoint()
            if endurance:
                start = time.monotonic(); runs = []
                while time.monotonic() - start < 600:
                    runs.extend(benchmark(client, live['model'])); report['endurance_runs'] = runs; checkpoint()
                report['endurance_elapsed_seconds'] = time.monotonic() - start
            report['scores'] = {}
            for category in ['logic', 'tool', 'agent', 'code']:
                rows = [r for r in report['rows'] if r['category'] == category]
                if rows:
                    report['scores'][category] = {'passed': sum(r['passed'] for r in rows), 'total': len(rows),
                        'blocked': sum('blocked' in r for r in rows), 'format_valid': sum(r.get('format_valid', False) for r in rows)}
            scores = report['scores']
            report['quality_gates_passed'] = full and all(scores.get(k, {}).get('passed', 0) >= n for k, n in [('logic',34),('tool',95),('agent',8),('code',16)]) and scores['tool']['format_valid'] >= 99
            report['resources'] = resource_summary(report)
            if baseline:
                report['baseline_comparison'] = baseline_comparison(report, baseline)
            report['verification_complete'] = not any('blocked' in r for r in report['rows'])
            report['run_completed'] = True
        except KeyboardInterrupt:
            report['interrupted'] = True
        except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
            report['failure_type'] = type(error).__name__
        quick_passed = all(r['passed'] for r in report['rows']) and bool(report['rows'])
        success = report['verification_complete'] and all(report.get('protocol', {}).get('checks', {}).values()) and (report.get('quality_gates_passed', False) if full else quick_passed)
        if long_context:
            success = success and report.get('context_probe', {}).get('passed', False) and report.get('context_probe', {}).get('near_full_window_exercised', False)
        if endurance:
            success = success and report.get('resources', {}).get('resource_gates_passed', False)
        if baseline:
            success = success and report.get('baseline_comparison', {}).get('passed', False)
        report['context_promotion_passed'] = bool(full and long_context and endurance and baseline and success)
        checkpoint()
    print(json.dumps({'checks_passed': success, 'report': str(path), 'phone_testing_required': True}))
    return 0 if success else 1
