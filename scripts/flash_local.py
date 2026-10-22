#!/usr/bin/env python3
"""Connect a local agent to Flash Local. Python stdlib; no cloud fallback."""
import argparse
import getpass
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.error

from flash_local_agent import LocalClient, MODEL, strict_json
from flash_local_clients import ClientSetupError, request_timeout_seconds, environment, private_profile


def settings_path():
    if os.name == 'nt':
        return Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local')) / 'FlashLocal/connection.json'
    return Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'flash-local/connection.json'


def save_connection(path, base, model):
    # Credentials are deliberately absent from the saved connection.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('Connection settings must not be symlinks')
    descriptor, temporary = tempfile.mkstemp(prefix='.connection-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump({'base_url': base, 'model': model}, handle)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def discover(client, requested=None):
    models = client.json('/models')
    entries = models.get('data', [])
    if not isinstance(entries, list) or not entries:
        raise ValueError('The phone did not advertise a model')
    selected = next((item for item in entries if item.get('id') == requested), None) if requested else entries[0]
    if selected is None:
        raise ValueError('Selected model is not available on this phone')
    name = selected.get('id')
    if not isinstance(name, str) or not name or len(name) > 128 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._' for c in name):
        raise ValueError('Unexpected local model identity')
    capabilities = selected.get('flash_local')
    if not isinstance(capabilities, dict):
        raise ValueError('Update Flash Local: live context capabilities are required; no guessed limits')
    context, output = capabilities.get('context_capacity_tokens'), capabilities.get('max_output_tokens')
    if type(context) is not int or context not in [8192, 16384, 32768, 65536] or type(output) is not int or not 1 <= output < context:
        raise ValueError('Invalid live context/output budget')
    return {'model': name, 'context': context, 'output': output, 'capabilities': capabilities}


def opencode_configuration(client, live, major=1):
    model = live['model']
    timeout = request_timeout_seconds(live) * 1000
    config = {
        '$schema': 'https://opencode.ai/config.json',
        'enabled_providers': ['flashlocal'],
        'model': 'flashlocal/' + model, 'small_model': 'flashlocal/' + model,
        'share': 'disabled', 'autoupdate': False,
        'permission': {'*': 'ask', 'bash': 'deny', 'webfetch': 'deny', 'external_directory': 'deny'},
        'provider': {'flashlocal': {'name': 'Flash Local iPhone', 'npm': '@ai-sdk/openai-compatible',
            'options': {'baseURL': client.base, 'apiKey': '{env:FLASH_LOCAL_API_KEY}',
                'timeout': timeout, 'headerTimeout': timeout, 'chunkTimeout': timeout},
            'models': {model: {'name': 'Flash Local / ' + model, 'tool_call': True,
                'limit': {'context': live['context'], 'output': live['output']}}}}}}
    if major == 2:
        # A background title request would replace the single phone KV session
        # between an assistant tool call and its result. Keep that context warm.
        config['agents'] = {'title': {'disabled': True}}
        # v2 renamed execution/delegation tools. Keep the existing ask/deny
        # policy instead of accidentally granting a newly named executor.
        config['permissions'] = [{'action': name, 'resource': '*', 'effect': 'deny'}
            for name in ['execute', 'shell', 'bash', 'webfetch', 'websearch', 'subagent', 'external_directory']]
        config['policies'] = [{'action': 'provider.use', 'resource': '*', 'effect': 'deny'},
            {'action': 'provider.use', 'resource': 'flashlocal', 'effect': 'allow'},
            {'action': 'integration.use', 'resource': '*', 'effect': 'deny'}]
    return config


def run_opencode(client, live, project=None):
    executable = shutil.which('opencode')
    if executable is None:
        raise ClientSetupError('OpenCode is not installed. Install its official CLI: https://opencode.ai/docs/')
    env = environment()
    major = opencode_major(executable, env)
    config = opencode_configuration(client, live, major)
    env['FLASH_LOCAL_API_KEY'] = client.key
    env['OPENCODE_CONFIG_CONTENT'] = json.dumps(config, separators=(',', ':'))
    if major == 2:
        # v2 removed root --pure/--model. Its original ConfigNormalize migrates
        # our v1 provider/limits/permissions document. The inline default model
        # and enabled_providers become a deny-all/allow-flashlocal policy.
        # Standalone prevents an existing daemon from retaining other settings.
        profile = private_profile(settings_path().parent / 'profiles', 'opencode-v2')
        env['OPENCODE_CONFIG_DIR'] = str(profile)
    # Do not let ambient proxy configuration forward a LAN credential.
    for key in list(env):
        if key.lower() in {'http_proxy', 'https_proxy', 'all_proxy'}:
            del env[key]
    env['NO_PROXY'] = env['no_proxy'] = '*'
    directory = str(Path(project).resolve(strict=True)) if project else None
    if directory and not Path(directory).is_dir():
        raise ValueError('Project must be a directory')
    arguments = [executable, '--pure', '--model', 'flashlocal/' + live['model']] if major == 1 else [executable, '--standalone']
    print('Phone connected; starting OpenCode v' + str(major) + ' with the phone model and live context ' + str(live['context']) + '.', flush=True)
    return subprocess.run(arguments,
                          cwd=directory, env=env, check=False).returncode


def opencode_major(executable, env):
    result = subprocess.run([executable, '--version'], env=env, capture_output=True, text=True,
                            timeout=15, check=False)
    match = re.search(r'(?m)^\s*(?:opencode\s+)?v?([12])\.\d+\.\d+(?:\s|$)', result.stdout or '')
    if result.returncode != 0 or match is None:
        raise ClientSetupError('Unsupported or unreadable OpenCode version. Run opencode --version; this launcher supports v1 and v2. No client was started.')
    return int(match.group(1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', help='Explicit phone IPv4 URL ending in /v1')
    parser.add_argument('--model')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('connect', help='Save a verified phone address; never saves the API key')
    commands.add_parser('status', help='Read authenticated model and live capacity')
    run = commands.add_parser('run', help='Start a client using the phone only')
    run.add_argument('client', choices=['opencode', 'hermes', 'codex', 'pentagi'])
    run.add_argument('--project')
    test = commands.add_parser('test', help='Real model acceptance; no simulated model')
    mode = test.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true')
    mode.add_argument('--full', action='store_true')
    test.add_argument('--output', required=True)
    test.add_argument('--expected-context', type=int, choices=[8192, 16384, 32768, 65536])
    test.add_argument('--baseline', help='Completed same-model 8K full report for regression comparison')
    test.add_argument('--reasoning', choices=['off','small','medium','ultra'], default='medium')
    test.add_argument('--long-context', action='store_true', help='Separate potentially slow near-full-window retrieval test')
    test.add_argument('--endurance', action='store_true', help='At least ten minutes of real generation')
    test.add_argument('--code-runner', choices=['docker', 'podman'])
    test.add_argument('--code-image', help='Already-installed Python image pinned by sha256 digest; no image pulls')
    args = parser.parse_args(argv)
    try:
        path = settings_path()
        if path.is_file() and path.stat().st_size > 4096:
            raise ValueError('Oversized connection settings')
        saved = strict_json(path.read_text(encoding='utf-8')) if path.is_file() else {}
        base = args.base_url or saved.get('base_url')
        if not base and args.command == 'connect':
            base = input('Phone Base URL (shown in Ornith → API): ').strip()
        if not base:
            raise ValueError('Run connect first, or supply --base-url before the subcommand')
        key = os.environ.get('FLASH_LOCAL_API_KEY') or getpass.getpass('Phone API key (not saved): ')
        client = LocalClient(base, key, timeout=10800 if getattr(args, 'long_context', False) else 1800)
        live = discover(client, args.model or saved.get('model'))
        client.timeout = request_timeout_seconds(live)
        if args.command == 'connect':
            save_connection(path, client.base, live['model'])
        if args.command in {'connect', 'status'}:
            print(json.dumps({'connected': True, 'base_url': client.base, **live}, ensure_ascii=False))
            return 0
        if args.command == 'run':
            if args.client == 'opencode':
                return run_opencode(client, live, args.project)
            from flash_local_clients import run_hermes, run_codex, run_pentagi
            directory = str(Path(args.project).resolve(strict=True)) if args.project else None
            if directory and not Path(directory).is_dir():
                raise ValueError('Project must be a directory')
            root = path.parent / 'profiles'
            if args.client == 'hermes':
                return run_hermes(client, live, root, directory)
            if args.client == 'codex':
                return run_codex(client, live, directory)
            return run_pentagi(client, live, root, directory)
        from flash_local_evaluate import evaluate
        return evaluate(client, live, args.output, full=args.full, expected_context=args.expected_context,
                        long_context=args.long_context, endurance=args.endurance,
                        code_runner=args.code_runner, code_image=args.code_image, baseline=args.baseline, reasoning=args.reasoning)
    except KeyboardInterrupt:
        print('Cancelled; no new request started.', file=sys.stderr)
        return 130
    except ClientSetupError as error:
        print(str(error), file=sys.stderr)
        return 1
    except urllib.error.HTTPError as error:
        reason = {401: 'Phone reachable, but the API key was rejected. Copy the current key from Flash Local.',
                  403: 'Phone reachable, but access was denied.',
                  429: 'Phone is busy with another request.',
                  503: 'Phone API reachable, but the model is not ready. Load it and keep the app foreground.'}
        print(reason.get(error.code, 'Phone returned HTTP ' + str(error.code) + '. Check its server status.') + ' No cloud fallback.', file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        print('Phone network connection failed. Check the current IP/port, same LAN, local-network permission and that Flash Local is foreground with its API started. No cloud fallback.', file=sys.stderr)
        return 1
    except (ValueError, OSError, KeyError, TypeError):
        # Exception text may contain a key-bearing request URL or private config.
        print('Connection/test failed. Check phone foreground state, Wi-Fi, API key, model and advertised capacity. No cloud fallback.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
