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
    def test_missing_opencode_reports_the_actual_installation_requirement(self):
        with patch('flash_local.shutil.which', return_value=None):
            with self.assertRaisesRegex(launcher.ClientSetupError, 'OpenCode is not installed'):
                launcher.run_opencode(None, self.live())

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
            run.side_effect = [subprocess.CompletedProcess([], 0, '1.19.3\n', ''), subprocess.CompletedProcess([], 0)]
            with patch.dict(os.environ, {'HTTP_PROXY': 'http://untrusted.invalid', 'ANTHROPIC_API_KEY': 'ambient'}):
                self.assertEqual(launcher.run_opencode(client, self.live()), 0)
            args, kwargs = run.call_args
            self.assertNotIn(KEY, str(args))
            config = json.loads(kwargs['env']['OPENCODE_CONFIG_CONTENT'])
            self.assertEqual(config['permission']['bash'], 'deny')
            self.assertEqual(kwargs['env']['FLASH_LOCAL_API_KEY'], KEY)
            self.assertNotIn('HTTP_PROXY', kwargs['env'])
            self.assertFalse(kwargs.get('shell', False))
            self.assertIn('--pure', args[0]); self.assertIn('--model', args[0])

    def test_v2_uses_standalone_inline_model_and_private_config_without_removed_flags(self):
        client = LocalClient('http://127.0.0.1:8080/v1', KEY)
        with tempfile.TemporaryDirectory() as folder, patch('flash_local.settings_path', return_value=Path(folder)/'connection.json'), \
             patch('flash_local.shutil.which', return_value='/fixture/opencode'), patch('flash_local.subprocess.run') as run:
            run.side_effect = [subprocess.CompletedProcess([], 0, 'opencode v2.1.2\n', ''), subprocess.CompletedProcess([], 0)]
            self.assertEqual(launcher.run_opencode(client, self.live()), 0)
            args, kwargs = run.call_args
            self.assertEqual(args[0], ['/fixture/opencode', '--standalone'])
            self.assertNotIn(KEY, str(args))
            config = json.loads(kwargs['env']['OPENCODE_CONFIG_CONTENT'])
            self.assertEqual(config['agents']['title'], {'disabled': True})
            self.assertEqual(config['model'], 'flashlocal/' + MODEL)
            self.assertEqual(config['enabled_providers'], ['flashlocal'])
            self.assertTrue(Path(kwargs['env']['OPENCODE_CONFIG_DIR']).is_dir())
            self.assertIn({'action': 'execute', 'resource': '*', 'effect': 'deny'}, config['permissions'])
            self.assertEqual(config['policies'][1], {'action': 'provider.use', 'resource': 'flashlocal', 'effect': 'allow'})

    def test_unknown_opencode_version_fails_before_any_client_session(self):
        with patch('flash_local.subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'opencode v3.0.0\n', '')) as run:
            with self.assertRaises(launcher.ClientSetupError): launcher.opencode_major('/fixture/opencode', {})
            self.assertEqual(run.call_count, 1)
    def test_connection_errors_distinguish_authentication_and_network_without_leaking_exception_text(self):
        import urllib.error
        for error, expected in [(urllib.error.HTTPError('http://secret.invalid/'+KEY, 401, KEY, {}, None), 'API key was rejected'),
                                (urllib.error.URLError(KEY), 'network connection failed')]:
            with tempfile.TemporaryDirectory() as folder, patch('flash_local.settings_path', return_value=Path(folder)/'connection.json'), \
                 patch('flash_local.getpass.getpass', return_value=KEY), patch('flash_local.discover', side_effect=error), \
                 patch('sys.stderr', new_callable=io.StringIO) as output:
                self.assertEqual(launcher.main(['--base-url', 'http://127.0.0.1:8080/v1', 'status']), 1)
                self.assertIn(expected, output.getvalue()); self.assertNotIn(KEY, output.getvalue())


class ScorerTests(unittest.TestCase):
    def test_retrieval_exercises_8k_even_when_the_logic_suite_uses_ultra(self):
        client = Mock(); client.evaluation_effort = 'high'
        nonce = 'a' * 32
        first = self.response('OK'); first['usage'] = {'prompt_tokens':512}
        second = self.response(json.dumps({'answer':[nonce+'-'+str(i) for i in range(5)]}))
        second['usage'] = {'prompt_tokens':7168}
        client.json.side_effect = [first,second]
        with patch('flash_local_evaluate.uuid.uuid4', return_value=Mock(hex=nonce)):
            result = evaluate.context_probe(client, {'model':MODEL,'context':8192})
        self.assertTrue(result['passed']); self.assertTrue(result['near_full_window_exercised'])
        self.assertEqual(result['reasoning_mode'],'off')
        self.assertEqual(client.evaluation_effort,'high')
        for request in client.json.call_args_list:
            self.assertEqual(request.args[1]['max_tokens'],512)
            self.assertEqual(request.args[1]['reasoning_effort'],'none')

    def test_real_owned_boundary_suite_runs_all_required_guards(self):
        result = evaluate.boundary_controls()
        self.assertTrue(result['passed'])
        self.assertTrue(result['all_required_guards_executed'])
        self.assertFalse(result['phone_inference_evidence'])
        self.assertFalse(result['third_party_client_scope_certified'])

    def test_baseline_rejects_different_source_and_two_lost_logic_cases(self):
        report = {'suite':'fixture', 'fixture_seed':42183, 'fixture_bank_sha256':'bank',
            'harness_sha256':'b'*64,
            'reasoning_mode':'medium', 'model':MODEL, 'context_capacity_tokens':16384,
            'resources':{'model_sha256':'a'*64, 'source_revision':'same'},
            'rows':[{'id':'logic-1','category':'logic','passed':True}]}
        previous = {**report,'context_capacity_tokens':8192,'full_suite':True,'verification_complete':True,
            'rows':[{'id':'logic-'+str(i),'category':'logic','passed':True} for i in range(1,4)]}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'baseline.json'; path.write_text(json.dumps(previous))
            self.assertFalse(evaluate.baseline_comparison(report,path)['passed'])
            report['rows'].append({'id':'logic-2','category':'logic','passed':True})
            self.assertTrue(evaluate.baseline_comparison(report,path)['passed'])
            report['resources'] = {'model_sha256':'a'*64,'source_revision':'changed'}
            with self.assertRaises(ValueError): evaluate.baseline_comparison(report,path)
            report['resources']['source_revision'] = 'same'
            report['harness_sha256'] = 'c'*64
            with self.assertRaises(ValueError): evaluate.baseline_comparison(report,path)

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
