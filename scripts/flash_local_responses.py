"""Loopback-only, stateless Responses adapter. Inference stays on the phone.

Text and function/custom-tool items are supported; images, hosted tools, stored
response IDs and encrypted reasoning are explicitly rejected. No tool executes
here. Completed tool arguments come from the phone's validated Chat response.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import http.client
import json
import queue
import secrets
import socket
import threading
import time
import uuid
import urllib.parse

from flash_local_agent import strict_json


def text_content(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError('Expected text content')
    parts = []
    for part in value:
        if not isinstance(part, dict) or part.get('type') not in {'input_text', 'output_text'} or not isinstance(part.get('text'), str):
            raise ValueError('Only text input is supported by this model')
        parts.append(part['text'])
    return '\n'.join(parts)


def translate(request, live):
    if not isinstance(request, dict) or request.get('model') != live['model']:
        raise ValueError('Select the phone model advertised by /v1/models')
    supported = {'model','instructions','input','tools','tool_choice','parallel_tool_calls','reasoning',
        'store','stream','include','prompt_cache_key','client_metadata','metadata','temperature','top_p',
        'max_output_tokens','text','previous_response_id','conversation','background','truncation',
        'service_tier','safety_identifier','user'}
    if set(request) - supported or request.get('truncation', 'disabled') != 'disabled':
        raise ValueError('Unsupported request controls; no automatic truncation')
    if 'stream' in request and type(request['stream']) is not bool:
        raise ValueError('stream must be boolean')
    for key in ['previous_response_id', 'conversation', 'background']:
        if request.get(key):
            raise ValueError('Stateless adapter requires complete text history')
    if request.get('store') is True:
        raise ValueError('Stored responses are not supported')
    if any(value != 'reasoning.encrypted_content' for value in request.get('include', [])):
        raise ValueError('Unsupported response include')
    # No opaque reasoning state is emitted or needed: replay complete text/tool history.
    output_format = request.get('text', {}).get('format', {'type': 'text'})
    if output_format != {'type': 'text'}:
        raise ValueError('Constrained output formats are not implemented')
    definitions, custom = [], set()
    for tool in request.get('tools', []):
        kind = tool.get('type')
        if kind == 'function':
            function = {k: tool[k] for k in ['name', 'description', 'parameters', 'strict'] if k in tool}
        elif kind == 'custom':
            name = tool['name']
            custom.add(name)
            function = {'name': name, 'description': tool.get('description', '') +
                '\nReturn the exact raw tool input as the input string. The caller validates its grammar.',
                'parameters': {'type': 'object', 'properties': {'input': {'type': 'string'}},
                               'required': ['input'], 'additionalProperties': False}}
        else:
            raise ValueError('Hosted, namespace and search tools are not supported')
        definitions.append({'type': 'function', 'function': function})
    messages, system = [], []
    if request.get('instructions'):
        system.append(text_content(request['instructions']))
    history = request.get('input', [])
    if isinstance(history, str):
        history = [{'role': 'user', 'content': history}]
    if not isinstance(history, list):
        raise ValueError('Expected complete input history')
    for item in history:
        kind = item.get('type', 'message')
        if kind == 'message':
            role, text = item.get('role'), text_content(item.get('content', []))
            if role in {'system', 'developer'}:
                system.append(text)
            elif role in {'user', 'assistant'}:
                if messages and messages[-1]['role'] == role and not messages[-1].get('tool_calls'):
                    messages[-1]['content'] += '\n' + text
                else:
                    messages.append({'role': role, 'content': text})
            else:
                raise ValueError('Unexpected message role')
        elif kind in {'function_call', 'custom_tool_call'}:
            if not messages or messages[-1]['role'] != 'assistant':
                messages.append({'role': 'assistant', 'content': None})
            arguments = item.get('arguments') if kind == 'function_call' else json.dumps({'input': item['input']})
            if not isinstance(arguments, str):
                raise ValueError('Tool arguments must be JSON strings')
            strict_json(arguments)
            messages[-1].setdefault('tool_calls', []).append({'id': item['call_id'], 'type': 'function',
                'function': {'name': item['name'], 'arguments': arguments}})
        elif kind in {'function_call_output', 'custom_tool_call_output'}:
            messages.append({'role': 'tool', 'tool_call_id': item['call_id'], 'content': text_content(item['output'])})
        elif kind == 'reasoning' and not item.get('encrypted_content'):
            # Reasoning summaries are presentation, never substituted for model state.
            continue
        else:
            raise ValueError('Unsupported history item; no content was silently discarded')
    if system:
        messages.insert(0, {'role': 'system', 'content': '\n\n'.join(system)})
    maximum = request.get('max_output_tokens', live['output'])
    if type(maximum) is not int or not 1 <= maximum <= live['output']:
        raise ValueError('Output limit exceeds the actual phone capacity')
    payload = {'model': live['model'], 'messages': messages, 'stream': True,
        'stream_options': {'include_usage': True}, 'max_tokens': maximum,
        'parallel_tool_calls': False}
    if definitions:
        payload['tools'] = definitions
    choice = request.get('tool_choice', 'auto')
    if isinstance(choice, dict):
        choice = {'type': 'function', 'function': {'name': choice['name']}}
    if definitions:
        payload['tool_choice'] = choice
    for key in ['temperature', 'top_p']:
        if key in request:
            payload[key] = request[key]
    if request.get('reasoning', {}).get('effort'):
        payload['reasoning_effort'] = request['reasoning']['effort']
    return payload, custom


def response_events(chunks, live, custom):
    identifier = 'resp_' + uuid.uuid4().hex
    response = {'id': identifier, 'object': 'response', 'created_at': int(time.time()),
        'status': 'in_progress', 'error': None, 'incomplete_details': None,
        'model': live['model'], 'output': [], 'parallel_tool_calls': False}
    yield {'type': 'response.created', 'response': dict(response)}
    yield {'type': 'response.in_progress', 'response': dict(response)}
    message = None
    text = ''
    calls, usage, finish = {}, {}, None
    for chunk in chunks:
        if chunk is None:  # heartbeat, also observes client disconnect during prefill
            yield None
            continue
        if chunk.get('error'):
            raise ValueError('Phone generation failed')
        usage = chunk.get('usage') or usage
        for choice in chunk.get('choices', []):
            finish = choice.get('finish_reason') or finish
            delta = choice.get('delta', {})
            content = delta.get('content')
            if content:
                if not isinstance(content, str):
                    raise ValueError('Invalid content delta')
                if message is None:
                    message = {'id': 'msg_' + uuid.uuid4().hex, 'type': 'message', 'status': 'in_progress',
                               'role': 'assistant', 'content': []}
                    yield {'type': 'response.output_item.added', 'output_index': 0, 'item': dict(message)}
                    yield {'type': 'response.content_part.added', 'item_id': message['id'], 'output_index': 0,
                           'content_index': 0, 'part': {'type': 'output_text', 'text': '', 'annotations': [], 'logprobs': []}}
                text += content
                if len(text.encode()) > 1 << 20:
                    raise ValueError('Oversized generated text')
                yield {'type': 'response.output_text.delta', 'item_id': message['id'], 'output_index': 0,
                       'content_index': 0, 'delta': content, 'logprobs': []}
            for call in delta.get('tool_calls', []):
                index = call['index']
                if type(index) is not int or not 0 <= index < 32:
                    raise ValueError('Invalid function index')
                entry = calls.setdefault(index, {'id': '', 'name': '', 'arguments': ''})
                entry['id'] += call.get('id', '')
                entry['name'] += call.get('function', {}).get('name', '')
                entry['arguments'] += call.get('function', {}).get('arguments', '')
                if len(entry['arguments'].encode()) > 65536:
                    raise ValueError('Oversized function arguments')
    if finish not in {'stop', 'tool_calls', 'length'} or (calls and finish != 'tool_calls'):
        raise ValueError('Missing or truncated terminal generation status')
    if message:
        part = {'type': 'output_text', 'text': text, 'annotations': [], 'logprobs': []}
        message.update(status='completed', content=[part])
        yield {'type': 'response.output_text.done', 'item_id': message['id'], 'output_index': 0,
               'content_index': 0, 'text': text, 'logprobs': []}
        yield {'type': 'response.content_part.done', 'item_id': message['id'], 'output_index': 0,
               'content_index': 0, 'part': part}
        yield {'type': 'response.output_item.done', 'output_index': 0, 'item': message}
        response['output'].append(message)
    for index, call in sorted(calls.items()):
        args = strict_json(call['arguments'])
        item = {'id': 'fc_' + uuid.uuid4().hex, 'call_id': call['id'], 'name': call['name'], 'status': 'in_progress'}
        if call['name'] in custom:
            if not isinstance(args, dict) or set(args) != {'input'} or not isinstance(args['input'], str):
                raise ValueError('Custom tool input differs from its declared schema')
            item.update(type='custom_tool_call', input='')
            field, prefix, value = 'input', 'response.custom_tool_call_input', args['input']
        else:
            item.update(type='function_call', arguments='')
            field, prefix, value = 'arguments', 'response.function_call_arguments', call['arguments']
        output_index = len(response['output'])
        yield {'type': 'response.output_item.added', 'output_index': output_index, 'item': dict(item)}
        yield {'type': prefix + '.delta', 'item_id': item['id'], 'output_index': output_index, 'delta': value}
        yield {'type': prefix + '.done', 'item_id': item['id'], 'output_index': output_index, field: value}
        item.update(status='completed', **{field: value})
        response['output'].append(item)
        yield {'type': 'response.output_item.done', 'output_index': output_index, 'item': item}
    response['status'] = 'incomplete' if finish == 'length' else 'completed'
    if finish == 'length':
        response['incomplete_details'] = {'reason': 'max_output_tokens'}
    # Missing measurements are unknown, never fabricated zero-token inference.
    response['usage'] = None
    if usage:
        for field in ['prompt_tokens','completion_tokens','total_tokens']:
            if type(usage.get(field)) is not int or usage[field] < 0:
                raise ValueError('Invalid phone usage counts')
        if usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
            raise ValueError('Inconsistent phone usage counts')
        response['usage'] = {'input_tokens': usage['prompt_tokens'], 'output_tokens': usage['completion_tokens'],
                             'total_tokens': usage['total_tokens']}
        metrics = usage.get('flash_local', {})
        cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', metrics.get('reused_prompt_tokens'))
        reasoning = usage.get('completion_tokens_details', {}).get('reasoning_tokens')
        for count, maximum, group, key in [(cached, usage['prompt_tokens'], 'input_tokens_details', 'cached_tokens'),
                (reasoning, usage['completion_tokens'], 'output_tokens_details', 'reasoning_tokens')]:
            if count is not None:
                if type(count) is not int or not 0 <= count <= maximum:
                    raise ValueError('Invalid phone usage details')
                response['usage'][group] = {key: count}
        if metrics:
            response['flash_local'] = metrics
    yield {'type': 'response.' + response['status'], 'response': response}


def phone_chunks(client, payload, cancelled):
    # A bounded reader permits heartbeat/disconnect detection while the phone prefills.
    inbox = queue.Queue(maxsize=4)
    holder = []
    def put(value):
        while not cancelled.is_set():
            try:
                inbox.put(value, timeout=0.2)
                return
            except queue.Full:
                pass
    def read():
        url = urllib.parse.urlsplit(client.base)
        connection_type = http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection
        connection = connection_type(url.hostname, url.port, timeout=min(15, client.timeout))
        # Retain the connection before waiting for response headers, so disconnect
        # can shut down an already-connected request during owner preparation.
        holder.append(connection)
        try:
            encoded = json.dumps(payload, allow_nan=False).encode()
            if len(encoded) > 262144:
                raise ValueError('Translated input exceeds phone request bound')
            connection.connect()
            if cancelled.is_set(): return
            connection.sock.settimeout(client.timeout)
            connection.request('POST', url.path + '/chat/completions', encoded,
                {'Authorization': 'Bearer ' + client.key, 'Content-Type': 'application/json'})
            with connection.getresponse() as upstream:
                if upstream.status != 200 or upstream.headers.get_content_type() != 'text/event-stream':
                    raise ValueError('Phone did not accept a streaming completion')
                total, data = 0, []
                while not cancelled.is_set():
                    line = upstream.readline(65538)
                    total += len(line)
                    if not line or len(line) > 65537 or total > 4 << 20:
                        raise ValueError('Truncated or oversized phone stream')
                    if line in {b'\n', b'\r\n'} and data:
                        block = b'\n'.join(data); data = []
                        if block == b'[DONE]':
                            put(('done', None)); return
                        put(('chunk', strict_json(block)))
                    elif line.startswith(b'data:'):
                        data.append(line[5:].strip())
        except Exception:
            put(('error', None))
        finally:
            connection.close()
    worker = threading.Thread(target=read, daemon=True); worker.start()
    try:
        while not cancelled.is_set():
            try:
                kind, value = inbox.get(timeout=1)
            except queue.Empty:
                yield None; continue
            if kind == 'done':
                return
            if kind == 'error':
                raise ValueError('Phone stream failed; no cloud retry')
            yield value
    finally:
        cancelled.set()
        for connection in holder:
            # Shutdown the socket before buffered close, which otherwise waits for a reader.
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):
                pass
        worker.join(timeout=2)


@contextmanager
def gateway(client, live):
    key = secrets.token_urlsafe(36)
    single = threading.BoundedSemaphore(1)
    active = set()
    active_lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *args):
            pass  # Never log conversation bodies, tokens or request paths.
        def reply(self, status, obj):
            body = json.dumps(obj, allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.send_header('Connection', 'close'); self.end_headers()
            self.wfile.write(body); self.close_connection = True
        def authorized(self):
            self.connection.settimeout(15)
            valid = hmac.compare_digest(self.headers.get('Authorization', '').encode(), ('Bearer ' + key).encode())
            if not valid or self.headers.get('Origin'):
                self.reply(401, {'error': {'message': 'Loopback bearer authentication required'}})
                return False
            return True
        def do_GET(self):
            if not self.authorized(): return
            if self.path == '/v1/models':
                self.reply(200, {'object': 'list', 'data': [{'id': live['model'], 'object': 'model',
                    'owned_by': 'flash-local', 'flash_local': live['capabilities']}]})
            else:
                self.reply(404, {'error': {'message': 'Unsupported path'}})
        def do_POST(self):
            if not self.authorized(): return
            if self.path != '/v1/responses':
                self.reply(404, {'error': {'message': 'Only stateless /v1/responses is supported'}}); return
            try:
                if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length', [])) != 1:
                    raise ValueError('Use one explicit Content-Length')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 262144: raise ValueError('Request exceeds 256 KiB')
                raw = self.rfile.read(length)
                if len(raw) != length: raise ValueError('Incomplete request')
                request = strict_json(raw)
                payload, custom = translate(request, live)
            except (ValueError, TypeError, KeyError, AttributeError):
                self.reply(400, {'error': {'message': 'Unsupported or malformed text/tool request; no inference started'}}); return
            if not single.acquire(blocking=False):
                self.reply(429, {'error': {'message': 'Phone is busy; retry after the current turn'}}); return
            cancelled = threading.Event()
            with active_lock: active.add(cancelled)
            chunks = phone_chunks(client, payload, cancelled)
            events = response_events(chunks, live, custom)
            streaming = request.get('stream') is True
            try:
                if streaming:
                    self.send_response(200); self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache'); self.send_header('Connection', 'close'); self.end_headers()
                    for sequence, event in enumerate(events):
                        if event is None:
                            self.wfile.write(b': keepalive\n\n')
                        else:
                            event['sequence_number'] = sequence
                            self.wfile.write(('event: ' + event['type'] + '\ndata: ' + json.dumps(event, allow_nan=False) + '\n\n').encode())
                        self.wfile.flush()
                    self.close_connection = True
                else:
                    final = None
                    for event in events:
                        if event and event['type'] in {'response.completed', 'response.incomplete'}:
                            final = event['response']
                    if final is None: raise ValueError('No final response')
                    self.reply(200, final)
            except (OSError, ValueError, TypeError, KeyError):
                if streaming:
                    try:
                        self.wfile.write(b'event: error\ndata: {"type":"error","code":"phone_generation_failed","message":"Phone generation stopped; no fallback"}\n\n'); self.wfile.flush()
                    except OSError: pass
                else:
                    self.reply(502, {'error': {'message': 'Phone generation failed; no fallback'}})
            finally:
                cancelled.set(); chunks.close(); events.close()
                with active_lock: active.discard(cancelled)
                single.release(); self.close_connection = True
    class Server(ThreadingHTTPServer):
        daemon_threads = True
    server = Server(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', key
    finally:
        with active_lock:
            for cancelled in active: cancelled.set()
        server.shutdown(); server.server_close(); thread.join(timeout=3)
