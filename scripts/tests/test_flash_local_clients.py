"""Client protocol controls; all generated replies are labelled synthetic fixtures."""
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flash_local_agent import LocalClient, MODEL
import flash_local_clients as clients
import flash_local_responses as bridge

KEY = 'synthetic-local-fixture-key-00000001'
LIVE = {'model': MODEL, 'context': 16384, 'output': 4096,
        'capabilities': {'context_capacity_tokens': 16384, 'max_output_tokens': 4096}}


class AdapterTests(unittest.TestCase):
    def test_pre_header_phone_request_is_closed_on_cancel(self):
        reserved, disconnected = threading.Event(), threading.Event()
        class PreparingPhone(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.connection.settimeout(5)
                reserved.set()
                if self.connection.recv(1) == b'':
                    disconnected.set()
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), PreparingPhone)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        stop = threading.Event()
        try:
            client = LocalClient(f'http://127.0.0.1:{server.server_port}/v1', KEY)
            chunks = bridge.phone_chunks(client, {'model':MODEL,'stream':True}, stop)
            self.assertIsNone(next(chunks))  # heartbeat while response headers are pending
            self.assertTrue(reserved.wait(2))
            chunks.close()
            self.assertTrue(stop.is_set())
            self.assertTrue(disconnected.wait(3))
        finally:
            stop.set(); server.shutdown(); server.server_close(); worker.join(5)

    def test_stateless_text_tools_and_outputs_are_preserved(self):
        request = {'model': MODEL, 'instructions': 'system contract', 'input': [
            {'role': 'developer', 'content': 'additional contract'},
            {'role': 'user', 'content': [{'type': 'input_text', 'text': 'hello'}]},
            {'role': 'user', 'content': 'world'},
            {'type': 'function_call', 'name': 'lookup', 'call_id': 'call_1', 'arguments': '{"id":7}'},
            {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'private receipt'}],
            'tools': [{'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object'}}]}
        payload, custom = bridge.translate(request, LIVE)
        self.assertEqual(custom, set())
        self.assertEqual(payload['messages'][0]['content'], 'system contract\n\nadditional contract')
        self.assertEqual(payload['messages'][1]['content'], 'hello\nworld')
        self.assertEqual(payload['messages'][-1], {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'private receipt'})
        self.assertEqual(payload['messages'][-2]['tool_calls'][0]['function']['arguments'], '{"id":7}')
        self.assertFalse(payload['parallel_tool_calls'])

    def test_custom_patch_input_has_exact_lossless_roundtrip(self):
        patch_text = '*** Begin Patch\n*** Add File: a.py\n+print("Hej 🐟")\n*** End Patch'
        request = {'model': MODEL, 'input': 'write file', 'tools': [{'type': 'custom', 'name': 'apply_patch'}]}
        _, custom = bridge.translate(request, LIVE)
        chunks = [{'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_2',
            'function': {'name': 'apply_patch', 'arguments': json.dumps({'input': patch_text})}}]}, 'finish_reason': 'tool_calls'}]}]
        events = list(bridge.response_events(chunks, LIVE, custom))
        item = events[-1]['response']['output'][0]
        self.assertEqual(item['type'], 'custom_tool_call')
        self.assertEqual(item['input'], patch_text)
        payload, _ = bridge.translate({**request, 'input': [item,
            {'type': 'custom_tool_call_output', 'call_id': 'call_2', 'output': 'applied'}]}, LIVE)
        self.assertEqual(json.loads(payload['messages'][0]['tool_calls'][0]['function']['arguments']), {'input': patch_text})

    def test_unsupported_modalities_history_tools_and_capacity_fail_before_inference(self):
        for extra in [{'previous_response_id': 'resp_x'}, {'store': True}, {'background': True},
                      {'input': [{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'x'}]}]},
                      {'tools': [{'type': 'web_search'}]}, {'max_output_tokens': 8192},
                      {'text': {'format': {'type': 'json_schema'}}},
                      {'input': [{'type': 'reasoning', 'encrypted_content': 'opaque'}]}]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                bridge.translate({'model': MODEL, 'input': 'test', **extra}, LIVE)

    def test_streamed_text_completes_but_missing_finish_or_bad_tool_never_does(self):
        chunks = [{'choices': [{'delta': {'content': 'Hej '}, 'finish_reason': None}]},
                  {'choices': [{'delta': {'content': '🐟'}, 'finish_reason': 'stop'}]},
                  {'choices': [], 'usage': {'prompt_tokens': 17, 'completion_tokens': 3, 'total_tokens': 20}}]
        events = list(bridge.response_events(chunks, LIVE, set()))
        self.assertEqual(''.join(e.get('delta', '') for e in events), 'Hej 🐟')
        self.assertEqual(events[-1]['response']['usage']['input_tokens'], 17)
        self.assertEqual(events[-1]['type'], 'response.completed')
        with self.assertRaises(ValueError): list(bridge.response_events(chunks[:1], LIVE, set()))
        bad = [{'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '{'}}]}, 'finish_reason': 'length'}]}]
        with self.assertRaises(ValueError): list(bridge.response_events(bad, LIVE, set()))
        limited = list(bridge.response_events([{'choices': [{'delta': {'content': 'partial'}, 'finish_reason': 'length'}]}], LIVE, set()))
        self.assertEqual(limited[-1]['type'], 'response.incomplete')

    def test_real_loopback_authenticated_sse_bridge_and_unknown_path(self):
        observed = []
        class Phone(BaseHTTPRequestHandler):
            def do_POST(self):
                observed.append((self.path, self.headers.get('Authorization'), json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
                body = {'choices': [{'delta': {'content': 'SYNTHETIC TRANSPORT FIXTURE'}, 'finish_reason': 'stop'}]}
                self.wfile.write(('data: ' + json.dumps(body) + '\n\ndata: [DONE]\n\n').encode())
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Phone)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        try:
            phone = LocalClient(f'http://127.0.0.1:{server.server_port}/v1', KEY)
            with bridge.gateway(phone, LIVE) as (base, key):
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                def post(auth, path='/responses', stream=True, origin=None):
                    headers = {'Authorization': 'Bearer ' + auth, 'Content-Type': 'application/json'}
                    if origin: headers['Origin'] = origin
                    return opener.open(urllib.request.Request(base + path,
                        json.dumps({'model': MODEL, 'input': 'fixture', 'stream': stream}).encode(), headers), timeout=5)
                with self.assertRaises(urllib.error.HTTPError) as error: post('wrong')
                self.assertEqual(error.exception.code, 401)
                with self.assertRaises(urllib.error.HTTPError): post(key, origin='https://untrusted.invalid')
                with self.assertRaises(urllib.error.HTTPError): post(key, '/responses/compact')
                self.assertEqual(observed, [])
                with post(key) as response:
                    data = response.read().decode()
                self.assertIn('response.output_text.delta', data)
                self.assertIn('response.completed', data)
                self.assertNotIn(KEY, data)
                with post(key, stream=False) as response:
                    self.assertEqual(json.load(response)['status'], 'completed')
                self.assertEqual(observed[0][0], '/v1/chat/completions')
                self.assertEqual(observed[0][1], 'Bearer ' + KEY)
        finally:
            server.shutdown(); server.server_close(); worker.join(5)


class ProfileTests(unittest.TestCase):
    def test_hermes_does_not_lie_about_context_to_bypass_its_64k_floor(self):
        with patch('flash_local_clients.subprocess.run') as run:
            with self.assertRaisesRegex(clients.ClientSetupError, '64,000'):
                clients.run_hermes(None, LIVE, Path('/unused'))
            run.assert_not_called()

    def test_hermes_and_all_pentagi_roles_use_phone_without_saved_keys(self):
        phone = LocalClient('http://192.168.1.12:8080/v1', KEY)
        hermes = clients.hermes_configuration(phone, LIVE)
        self.assertEqual(hermes['model']['context_length'], 16384)
        self.assertEqual(hermes['model']['api_mode'], 'chat_completions')
        self.assertEqual(hermes['fallback_model'], [])
        for task in hermes['auxiliary'].values():
            self.assertEqual(task['model'], MODEL)
            self.assertEqual(task['base_url'], phone.base)
        roles, compose = clients.pentagi_configuration(phone, LIVE, Path('/fixture/profile'))
        self.assertEqual(len(roles), 13)
        self.assertTrue(all(role['model'] == MODEL and role['max_tokens'] == 4096 and not role['json'] for role in roles.values()))
        env = compose['services']['pentagi']['environment']
        self.assertEqual(env['ASK_USER'], 'true')
        self.assertEqual(env['OPEN_AI_KEY'], '')
        self.assertNotIn(KEY, json.dumps([hermes, roles, compose]))

    def test_unique_profiles_never_overwrite_and_codex_key_is_only_ephemeral_env(self):
        with tempfile.TemporaryDirectory() as root:
            first = clients.private_profile(root, 'hermes'); second = clients.private_profile(root, 'hermes')
            self.assertNotEqual(first, second)
            file = first / 'config.yaml'; clients.write_private(file, 'original')
            with self.assertRaises(FileExistsError): clients.write_private(file, 'overwritten')
            self.assertEqual(file.read_text(), 'original')
        args = clients.codex_arguments(LIVE, 'http://127.0.0.1:12345/v1')
        self.assertIn('model_provider="flashlocal"', args)
        self.assertIn('model_providers.flashlocal.wire_api="responses"', args)
        self.assertIn('model_context_window=16384', args)
        self.assertNotIn(KEY, str(args))
        self.assertNotIn('--dangerously-bypass-approvals-and-sandbox', args)
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'cloud-key', 'HTTPS_PROXY': 'untrusted',
                                      'AUXILIARY_VISION_BASE_URL': 'cloud-url'}):
            env = clients.environment()
        self.assertNotIn('OPENAI_API_KEY', env); self.assertNotIn('HTTPS_PROXY', env)
        self.assertNotIn('AUXILIARY_VISION_BASE_URL', env)


if __name__ == '__main__': unittest.main()
