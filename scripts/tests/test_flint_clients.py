"""Capability/transport controls with synthetic replies; no model benchmark."""
import io
import json
import unittest
from unittest.mock import Mock, patch

from flash_local import discover, opencode_configuration
from flash_local_agent import LocalClient
from flash_local_clients import codex_arguments, hermes_configuration
from flash_local_responses import translate

MODEL = 'recursive-flint-40b-a2.5b-int4'
KEY = 'public-synthetic-fixture-key-12345678'
LIVE = dict(model=MODEL, context=32768, output=4096,
    capabilities=dict(context_capacity_tokens=32768, max_output_tokens=4096, thinking_budget_control=False, reasoning_presets=[]))


class FlintClientTests(unittest.TestCase):
    def test_discovery_binds_native_reasoning_policy_to_the_authenticated_model(self):
        client = LocalClient('http://127.0.0.1:8080/v1', KEY)
        with patch.object(client, 'json', return_value=dict(data=[dict(id=MODEL, flash_local=LIVE['capabilities'])])):
            self.assertEqual(discover(client, MODEL), LIVE)
        self.assertEqual(client.selected_model, MODEL)
        self.assertFalse(client.capabilities['thinking_budget_control'])

    def test_transport_omits_only_unsupported_reasoning_defaults_without_mutating_source(self):
        client = LocalClient('http://127.0.0.1:8080/v1', KEY)
        client.selected_model, client.capabilities = MODEL, LIVE['capabilities']
        source = dict(model=MODEL, messages=[dict(role='user', content='fixture')], temperature=0, max_tokens=256,
            enable_thinking=True, thinking_budget=256, reasoning_effort='medium')
        with patch.object(client.opener, 'open', return_value=io.BytesIO(b'{}')) as network:
            client.json('/chat/completions', source)
            sent = json.loads(network.call_args.args[0].data)
        self.assertEqual(sent['model'], MODEL); self.assertEqual(sent['max_tokens'], 256)
        self.assertFalse(any(k in sent for k in ['enable_thinking', 'thinking_budget', 'reasoning_effort']))
        self.assertEqual(source['thinking_budget'], 256)
        client.capabilities = {}
        with patch.object(client.opener, 'open', return_value=io.BytesIO(b'{}')) as network:
            client.json('/chat/completions', source)
            self.assertEqual(json.loads(network.call_args.args[0].data)['thinking_budget'], 256)

    def test_responses_bridge_and_agent_profiles_preserve_native_flint_reasoning(self):
        request = dict(model=MODEL, input='fixture', stream=True, reasoning=dict(effort='medium'))
        payload, _ = translate(request, LIVE)
        self.assertNotIn('reasoning_effort', payload); self.assertEqual(payload['model'], MODEL)
        client = Mock(base='http://127.0.0.1:8080/v1')
        profile = opencode_configuration(client, LIVE, 2)
        self.assertEqual(profile['model'], 'flashlocal/' + MODEL)
        self.assertEqual(profile['provider']['flashlocal']['models'][MODEL]['limit']['context'], 32768)
        self.assertNotIn('reasoning_effort', hermes_configuration(client, LIVE)['agent'])
        with patch('flash_local_clients.Path.is_file', return_value=False):
            arguments = codex_arguments(LIVE, 'http://127.0.0.1:12345/v1')
        self.assertFalse(any('model_reasoning_effort=' in arg for arg in arguments))
        self.assertTrue(any(MODEL in arg for arg in arguments))


if __name__ == '__main__': unittest.main()
