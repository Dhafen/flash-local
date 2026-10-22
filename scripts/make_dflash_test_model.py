"""Tiny deterministic BF16 DFlash graph. Untrained; no model downloads.

Six draft layers, five sliding + one full, feature fusion and shared target
embedding/head, matching the production graph's operations with reduced widths.
The target fixture is generated separately by make_qwen4exp_test_model --ornith.
"""
import math
import pathlib
import random
import struct


def generate():
    rng = random.Random(20261022)
    meta = {'general.architecture': (8, 'dflash'), 'general.name': (8, 'synthetic-dflash-graph-test'),
            'tokenizer.ggml.model': (8, 'none')}
    for key, item in {'block_count': 6, 'embedding_length': 16, 'vocab_size': 16,
                      'context_length': 32768, 'feed_forward_length': 32,
                      'attention.head_count': 2, 'attention.head_count_kv': 1,
                      'attention.key_length': 128, 'attention.value_length': 128,
                      'rope.dimension_count': 128, 'attention.sliding_window': 4096,
                      'block_size': 16}.items():
        meta['dflash.' + key] = (4, item)
    meta['dflash.attention.layer_norm_rms_epsilon'] = (6, 1e-6)
    meta['dflash.rope.freq_base'] = (6, 10000.)
    meta['dflash.target_layers'] = (9, (5, [1, 2, 3]))
    meta['dflash.attention.sliding_window_pattern'] = (9, (4, [1] * 5 + [0]))
    meta['dflash.rope.dimension_sections'] = (9, (4, [64, 0, 0, 0]))
    tensors = []

    def tensor(name, shape, norm=False):
        if norm:
            raw = struct.pack('<' + 'f' * math.prod(shape), *([1.] * math.prod(shape)))
            kind = 0
        else:
            # Round to nearest even BF16, retaining the canonical packed bytes.
            words = []
            for _ in range(math.prod(shape)):
                bits = struct.unpack('<I', struct.pack('<f', rng.uniform(-.08, .08)))[0]
                words.append((bits + 0x7fff + ((bits >> 16) & 1)) >> 16)
            raw = struct.pack('<' + 'H' * len(words), *words)
            kind = 30
        tensors.append((name, shape, kind, raw))

    tensor('fc.weight', [48, 16])
    tensor('enc.output_norm.weight', [16], True)
    tensor('output_norm.weight', [16], True)
    for i in range(6):
        prefix = f'blk.{i}.'
        for name, shape in [('attn_norm', [16]), ('attn_q_norm', [128]),
                            ('attn_k_norm', [128]), ('ffn_norm', [16])]:
            tensor(prefix + name + '.weight', shape, True)
        for name, shape in [('attn_q', [16, 256]), ('attn_k', [16, 128]),
                            ('attn_v', [16, 128]), ('attn_output', [256, 16]),
                            ('ffn_gate', [16, 32]), ('ffn_up', [16, 32]), ('ffn_down', [32, 16])]:
            tensor(prefix + name + '.weight', shape)

    def string(value):
        raw = value.encode('utf-8')
        return struct.pack('<Q', len(raw)) + raw

    def value(kind, item):
        if kind == 8:
            return string(item)
        if kind == 9:
            element, items = item
            return struct.pack('<IQ', element, len(items)) + b''.join(value(element, v) for v in items)
        return struct.pack({4: '<I', 5: '<i', 6: '<f', 7: '<?'}[kind], item)

    header = bytearray(struct.pack('<4sIQQ', b'GGUF', 3, len(tensors), len(meta)))
    for key, (kind, item) in meta.items():
        header += string(key) + struct.pack('<I', kind) + value(kind, item)
    payload = bytearray()
    for name, shape, kind, raw in tensors:
        payload += bytes((-len(payload)) % 32)
        header += string(name) + struct.pack('<I', len(shape))
        header += b''.join(struct.pack('<Q', d) for d in shape) + struct.pack('<IQ', kind, len(payload))
        payload += raw
    header += bytes((-len(header)) % 32)
    root = pathlib.Path(__file__).resolve().parents[1] / '.test-fixtures'
    root.mkdir(exist_ok=True)
    data = header + payload
    (root / 'synthetic-dflash.gguf').write_bytes(data)
    print(f'Generated {len(tensors)} DFlash tensors, {len(data)} bytes; syntheticOnly=true')


if __name__ == '__main__':
    generate()
