"""Client/scorer controls. Replies below are synthetic, never phone benchmarks."""
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flash_local as launcher
import flash_local_evaluate as evaluate
from flash_local_agent import LocalClient, MODEL

KEY = 'public-fixture-key-' * 3


class LauncherTests(unittest.TestCase):
    def live(self):
        return {'model': MODEL, 'context': 16384, 'output': 4096,
                'capabilities': {'context_capacity_tokens': 16384, 'max_output_tokens': 4096}}

    def test_real_loopback_discovery_requires_auth_and_uses_live_limits(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                valid = self.headers.get('Authorization') == 'Bearer ' + KEY
                self.send_response(200 if valid else 401); self.end_headers()
                if valid:
                    self.wfile.write(json.dumps({'data': [{'id': MODEL, 'flash_local': {
                        'context_capacity_tokens': 16384, 'max_output_tokens': 4096}}]}).encode())
            def log_message(self, *args): pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            client = LocalClient(f'http://127.0.0.1:{server.server_port}/v1', KEY)
            live = launcher.discover(client)
            self.assertEqual(live['context'], 16384)
            config = launcher.opencode_configuration(client, live)
            self.assertEqual(config['enabled_providers'], ['flashlocal'])
            self.assertEqual(config['small_model'], config['model'])
            self.assertNotIn(KEY, json.dumps(config))
        finally:
            server.shutdown(); server.server_close(); thread.join(5)

    def test_missing_invalid_or_unknown_capacity_never_becomes_guessed_8k(self):
        client = Mock()
        for capabilities in [None, {}, {'context_capacity_tokens': True, 'max_output_tokens': 10},
                             {'context_capacity_tokens': 12345, 'max_output_tokens': 4096}]:
            client.json.return_value = {'data': [{'id': MODEL, 'flash_local': capabilities}]}
            with self.assertRaises(ValueError): launcher.discover(client)
        client.json.return_value = {'data': [{'id': MODEL, 'flash_local': self.live()['capabilities']}]}
        with self.assertRaises(ValueError): launcher.discover(client, 'unknown')

    def test_settings_keep_only_connection_and_child_process_has_no_key_argument(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'connection.json'
            launcher.save_connection(path, 'http://127.0.0.1:8080/v1', MODEL)
            self.assertEqual(set(json.loads(path.read_text())), {'base_url', 'model'})
            self.assertNotIn(KEY, path.read_text())
        client = LocalClient('http://127.0.0.1:8080/v1', KEY)
        with patch('flash_local.shutil.which', return_value='/fixture/opencode'), patch('flash_local.subprocess.run') as run:
            run.return_value.returncode = 0
            with patch.dict(os.environ, {'HTTP_PROXY': 'http://untrusted.invalid', 'ANTHROPIC_API_KEY': 'ambient'}):
                self.assertEqual(launcher.run_opencode(client, self.live()), 0)
            args, kwargs = run.call_args
            self.assertNotIn(KEY, str(args))
            config = json.loads(kwargs['env']['OPENCODE_CONFIG_CONTENT'])
            self.assertEqual(config['permission']['bash'], 'deny')
            self.assertEqual(kwargs['env']['FLASH_LOCAL_API_KEY'], KEY)
            self.assertNotIn('HTTP_PROXY', kwargs['env'])
            self.assertFalse(kwargs.get('shell', False))


class ScorerTests(unittest.TestCase):
    def response(self, text):
        return {'choices': [{'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
                'usage': {'fixture_only': True}}

    def test_exact_grading_never_accepts_boolean_as_number_or_partial_json(self):
        self.assertFalse(evaluate.exact([True], [1]))
        self.assertFalse(evaluate.exact({'x': True}, {'x': 1}))
        for text in ['{"answer":1', '{"answer":1,"answer":2}', '{"answer":NaN}', '1', '{"answer":1,"extra":2}']:
            with self.assertRaises(ValueError): evaluate.answer_json(text)
        self.assertEqual(evaluate.answer_json('```json\n{"answer":5}\n```'), 5)

    def test_fixture_banks_are_reproducible_and_wrong_answers_fail(self):
        cases = evaluate.logic_cases()
        self.assertEqual(len(cases), 40)
        self.assertEqual(cases, evaluate.logic_cases())
        self.assertNotEqual(cases, evaluate.logic_cases(7))
        self.assertEqual(len(evaluate.tool_cases()), 100)
        self.assertEqual(len(evaluate.code_cases()), 20)
        client = Mock()
        case = cases[1]  # JSON rather than SSE in this scorer control.
        client.json.return_value = self.response(json.dumps({'answer': case['expected']}))
        self.assertTrue(evaluate.run_case(client, MODEL, case)['passed'])
        client.json.return_value = self.response('{"answer":999999}')
        row = evaluate.run_case(client, MODEL, case)
        self.assertTrue(row['format_valid']); self.assertFalse(row['passed'])

    def test_tool_parser_keeps_exact_strings_and_rejects_wrong_types(self):
        text = 'Åäö "quote"\\file\nnext'
        def call(arguments):
            return {'tool_calls': [{'id': 'call_fixture', 'type': 'function', 'function': {
                'name': 'local_echo', 'arguments': json.dumps(arguments)}}]}
        self.assertEqual(evaluate.valid_call(call({'text': text}), [evaluate.ECHO])[2]['text'], text)
        for arguments in [{'text': 1}, {'text': text, 'extra': 2}]:
            with self.assertRaises(ValueError): evaluate.valid_call(call(arguments), [evaluate.ECHO])

    def test_code_execution_requires_explicit_sandbox_and_rejects_escape_calls(self):
        client = Mock()
        case = evaluate.code_cases()[0]
        row = evaluate.code_case(client, MODEL, case, None, None)
        self.assertFalse(row['passed']); self.assertIn('blocked', row); client.json.assert_not_called()
        client.json.return_value = self.response('def add(a,b):\n    return __import__("os").system("anything")')
        image = 'python@sha256:' + '0' * 64
        with patch('flash_local_evaluate.shutil.which', return_value='/fixture/docker'), patch('flash_local_evaluate.subprocess.run') as run:
            row = evaluate.code_case(client, MODEL, case, 'docker', image)
            self.assertFalse(row['passed'])
            self.assertEqual(run.call_count, 1)  # image inspect; never a generated execution

    def test_code_container_is_isolated_and_removed_even_after_timeout(self):
        client = Mock(); client.json.return_value = self.response('def add(a,b):\n    return a+b')
        image = 'python@sha256:' + '0' * 64
        with patch('flash_local_evaluate.shutil.which', return_value='/fixture/docker'), patch('flash_local_evaluate.subprocess.run') as run:
            run.side_effect = [Mock(returncode=0), subprocess.TimeoutExpired('docker',15), Mock(returncode=0)]
            row = evaluate.code_case(client, MODEL, evaluate.code_cases()[0], 'docker', image)
            self.assertFalse(row['passed'])
            command = run.call_args_list[1].args[0]
            for flag in ['--network=none','--read-only','--cap-drop=ALL','--memory=256m','--pull=never']:
                self.assertIn(flag, command)
            self.assertEqual(run.call_args_list[-1].args[0][:3], ['docker','rm','-f'])

    def test_missing_measurements_or_short_runs_cannot_promote_context(self):
        report = {'rows': [{'usage': {}}]}
        self.assertFalse(evaluate.resource_summary(report)['resource_gates_passed'])
        metrics = {'peak_memory_bytes': 4_000_000_000, 'decode_forward_passes': 128, 'decode_seconds': 30,
            'model_sha256': 'a' * 64, 'context_capacity_tokens': 16384, 'platform': 'iOSDevice',
            'device_identifier': 'iPhone18,2', 'app_build': 'fixture', 'source_revision': 'fixture', 'maximum_thermal_state': 0}
        report = {'rows': [{'usage': {'flash_local': metrics}}], 'context_capacity_tokens': 16384}
        self.assertFalse(evaluate.resource_summary(report)['resource_gates_passed'])
        report.update(endurance_elapsed_seconds=601, endurance_runs=[{'decode_tokens_per_second':4,
                      'minimum_128_tokens_met':True, 'usage':{'flash_local':metrics}}])
        self.assertTrue(evaluate.resource_summary(report)['resource_gates_passed'])
        report['rows'][0]['usage']['flash_local'] = {**metrics, 'peak_memory_bytes':5_600_000_000}
        self.assertFalse(evaluate.resource_summary(report)['resource_gates_passed'])


if __name__ == '__main__': unittest.main()
