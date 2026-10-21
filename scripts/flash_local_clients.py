"""Isolated client launch profiles. These clients execute tools, not the phone.

Profiles contain no keys. No dependencies, images or models are installed.
Client approval policies are distinct from flash_local_agent.py's ROE guard.
"""
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile

PENTAGI_ROLES = ['simple', 'simple_json', 'primary_agent', 'assistant', 'generator',
    'refiner', 'adviser', 'reflector', 'searcher', 'enricher', 'coder', 'installer', 'pentester']


class ClientSetupError(ValueError):
    """Only fixed, credential-free operator instructions belong in this error."""


def environment():
    env = dict(os.environ)
    for key in list(env):
        upper = key.upper()
        if (upper.endswith(('_API_KEY', '_AUTH_TOKEN')) or upper.startswith('AUXILIARY_') or
                upper in {'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE',
                          'HERMES_INFERENCE_MODEL', 'HERMES_MODEL', 'OPENAI_API_KEY'}):
            del env[key]
    env['NO_PROXY'] = env['no_proxy'] = '*'
    return env


def private_profile(root, client):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError('Profile directory must not be a symlink')
    return Path(tempfile.mkdtemp(prefix=client + '-', dir=root))


def write_private(path, content):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
        handle.write(content)


def hermes_configuration(client, live):
    return {'model': {'default': live['model'], 'provider': 'custom', 'base_url': client.base,
                      'api_mode': 'chat_completions', 'context_length': live['context']},
        'agent': {'max_turns': 32, 'max_tokens': live['output'], 'reasoning_effort': 'medium'},
        'compression': {'enabled': True, 'threshold': 0.55, 'target_ratio': 0.20},
        'fallback_providers': [], 'fallback_model': [],
        'auxiliary': {name: {'provider': 'main', 'model': live['model'], 'base_url': client.base,
                            'fallback_chain': [], 'max_concurrency': 1,
                            **({'enabled': False} if name == 'title_generation' else {})}
            for name in ['compression', 'approval', 'title_generation', 'vision']},
        'delegation': {'provider': 'custom', 'model': live['model'], 'base_url': client.base,
                       'max_iterations': 16},
        'display': {'streaming': True}, 'checkpoints': {'enabled': True}}


def run_hermes(client, live, root, project=None):
    if live['context'] < 64000:
        raise ClientSetupError('Current Hermes requires at least 64,000 context tokens. '
            'The phone advertises a smaller real window. Use OpenCode/Codex now; '
            'Hermes remains gated until the phone can admit and validate 65K. No false context limit was sent.')
    executable = shutil.which('hermes')
    if not executable:
        raise ClientSetupError('Install the official Hermes CLI first: https://hermes-agent.nousresearch.com/')
    profile = private_profile(root, 'hermes')
    # JSON is a YAML subset; no YAML dependency or credential interpolation needed.
    write_private(profile / 'config.yaml', json.dumps(hermes_configuration(client, live), indent=2))
    env = environment()
    env.update(HERMES_HOME=str(profile), OPENAI_BASE_URL=client.base, OPENAI_API_KEY=client.key)
    print('Hermes local profile (sessions retained): ' + str(profile), flush=True)
    return subprocess.run([executable, 'chat', '--provider', 'custom', '--model', live['model'],
                           '--toolsets', 'terminal,file', '--reasoning', 'medium'],
                          cwd=project, env=env, check=False).returncode


def codex_arguments(live, base, project=None):
    values = {'model': live['model'], 'review_model': live['model'], 'model_provider': 'flashlocal',
        'model_context_window': live['context'],
        'model_reasoning_effort': 'medium', 'web_search': 'disabled',
        'model_providers.flashlocal.name': 'Flash Local iPhone',
        'model_providers.flashlocal.base_url': base,
        'model_providers.flashlocal.env_key': 'FLASH_LOCAL_BRIDGE_KEY',
        'model_providers.flashlocal.wire_api': 'responses',
        'model_providers.flashlocal.requires_openai_auth': False,
        'model_providers.flashlocal.supports_websockets': False,
        'model_providers.flashlocal.request_max_retries': 0,
        'model_providers.flashlocal.stream_max_retries': 0,
        'model_providers.flashlocal.stream_idle_timeout_ms': 10800000,
        'features.apps': False, 'features.plugins': False, 'features.multi_agent': False,
        'features.multi_agent_v2': False, 'features.goals': False, 'features.tool_suggest': False}
    # Disable configured MCP servers for this invocation only. Read section names,
    # never copy credentials or rewrite the operator's config. Python >=3.11.
    import tomllib
    config_root = Path(os.environ.get('CODEX_HOME', Path.home() / '.codex'))
    candidates = [config_root / 'config.toml']
    if project:
        directory = Path(project).resolve()
        candidates += [parent / '.codex/config.toml' for parent in [directory, *directory.parents]]
    for path in candidates:
        if path.is_file():
            if path.stat().st_size > 1 << 20:
                raise ValueError('Oversized Codex config')
            with path.open('rb') as handle:
                config = tomllib.load(handle)
            for name in config.get('mcp_servers', {}):
                if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
                    raise ValueError('Use simple MCP server names for per-launch disabling')
                values['mcp_servers.' + name + '.enabled'] = False
    arguments = ['--no-daemon', '--sandbox', 'read-only', '--ask-for-approval', 'on-request']
    for key, value in values.items():
        arguments += ['--config', key + '=' + json.dumps(value)]
    return arguments


def run_codex(client, live, project=None):
    executable = shutil.which('codex')
    if not executable:
        raise ClientSetupError('Install the official Codex CLI first')
    from flash_local_responses import gateway
    with gateway(client, live) as (base, key):
        env = environment()
        # Only the ephemeral loopback key reaches Codex; the phone key stays in the adapter.
        env['FLASH_LOCAL_BRIDGE_KEY'] = key
        print('Codex uses the local Responses adapter; text/tool inference stays on iPhone. '
              'Read-only sandbox; no hosted search or stored responses.', flush=True)
        return subprocess.run([executable] + codex_arguments(live, base, project),
                              cwd=project, env=env, check=False).returncode


def pentagi_configuration(client, live, profile):
    # The phone does not promise grammar-constrained JSON. PentAGI's JSON-oriented
    # role uses its prompt/parser, not an unsupported response_format=json_object.
    roles = {role: {'model': live['model'], 'temperature': 0.2, 'top_p': 0.9, 'n': 1,
                    'max_tokens': live['output'], 'json': False} for role in PENTAGI_ROLES}
    cleared = {name: '' for name in ['OPEN_AI_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY',
        'BEDROCK_BEARER_TOKEN', 'BEDROCK_ACCESS_KEY_ID', 'BEDROCK_SECRET_ACCESS_KEY',
        'DEEPSEEK_API_KEY', 'GLM_API_KEY', 'KIMI_API_KEY', 'QWEN_API_KEY', 'MINIMAX_API_KEY',
        'MISTRAL_API_KEY', 'XAI_API_KEY', 'OLLAMA_SERVER_URL']}
    compose = {'services': {'pentagi': {'environment': {**cleared,
        'ASK_USER': 'true', 'LLM_SERVER_URL': client.base, 'LLM_SERVER_MODEL': live['model'],
        'LLM_SERVER_KEY': '${FLASH_LOCAL_API_KEY:?Set the phone API key}',
        'LLM_SERVER_PROVIDER': 'openai', 'LLM_SERVER_API_TYPE': 'openai',
        'LLM_SERVER_CONFIG_PATH': '/opt/pentagi/conf/flash-local.provider.yml'},
        'volumes': [{'type': 'bind', 'source': str(profile / 'provider.json'),
                     'target': '/opt/pentagi/conf/flash-local.provider.yml', 'read_only': True}]}}}
    return roles, compose


def run_pentagi(client, live, root, project):
    executable = shutil.which('docker')
    if not executable or not project:
        raise ClientSetupError('PentAGI needs an existing Docker Compose installation; pass --project PATH')
    project = Path(project).resolve(strict=True)
    compose = next((project / name for name in ['docker-compose.yml', 'compose.yml', 'compose.yaml']
                    if (project / name).is_file()), None)
    if not compose:
        raise ValueError('No existing PentAGI Compose file in the project')
    profile = private_profile(root, 'pentagi')
    roles, override = pentagi_configuration(client, live, profile)
    write_private(profile / 'provider.json', json.dumps(roles, indent=2))
    override_path = profile / 'compose.json'
    write_private(override_path, json.dumps(override, indent=2))
    env = environment(); env['FLASH_LOCAL_API_KEY'] = client.key
    command = [executable, 'compose', '--project-directory', str(project), '-f', str(compose), '-f', str(override_path)]
    # Validate without printing resolved secrets. Never download/build/start targets.
    validation = subprocess.run(command + ['config', '--format', 'json'], cwd=project,
                                env=env, capture_output=True, check=False)
    if validation.returncode:
        raise ValueError('PentAGI Compose validation failed; existing services were not changed')
    services = json.loads(validation.stdout).get('services', {})
    if 'pentagi' not in services:
        raise ValueError('Expected the official pentagi service')
    phone = services['pentagi'].get('environment', {})
    if phone.get('LLM_SERVER_URL') != client.base or phone.get('LLM_SERVER_MODEL') != live['model']:
        raise ValueError('Compose did not retain the selected phone provider')
    print('PentAGI local profile: ' + str(profile) + '\n'
          'Starting existing services only; ASK_USER enabled. Select the custom/local provider in PentAGI. '
          'Its JSON/tool reliability still needs the live-phone acceptance test.', flush=True)
    return subprocess.run(command + ['up', '-d', '--no-build', '--pull', 'never'],
                          cwd=project, env=env, check=False).returncode
