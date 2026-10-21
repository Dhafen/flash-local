#!/usr/bin/env python3
"""Inspect pinned public JSON only. Never fetch weights or execute remote code.

The output is a compatibility/budget calculation, not an inference benchmark.
Existing installed GGUF weights, app settings and inference remain untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

DRAFT = 'ornith-ai/Ornith-1.5-35B-A3B-DFlash'
DRAFT_REVISION = '942d20fee84d04c9ecbf18e9c6fc4144914a1d5d'
TARGET = 'ornith-ai/Ornith-1.5-35B-A3B'
TARGET_REVISION = '10fbf86fed7ecee4a061f8b499a618f46001cac1'
DRAFT_CONFIG_SHA256 = '20784b063040f7871efec4bf63342e5cb941907925943540176dfbb8aaf186dc'
TARGET_CONFIG_SHA256 = 'af198ccbe6c17dd973da98bbf6c92ce1c3ec7261c2e2fbc4c4f5150018418ea8'
TARGET_GGUF_SHA256 = '42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f'


def integer(value, field, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Invalid ' + field)
    return value


def inspect(draft, target, context=8192, verify_width=4, draft_weight_bytes=771819680):
    """Compute explicit unquantized-F16 cache bounds; no admission or timing."""
    context = integer(context, 'context', 1, 262144)
    target = target.get('text_config', target)
    if draft.get('architectures') != ['DFlashDraftModel']:
        raise ValueError('Only the inspected DFlash v1 architecture is supported')
    spec = draft.get('dflash_config', {})
    block = integer(spec.get('block_size'), 'block_size', 2, 64)
    verify_width = integer(verify_width, 'verify_width', 2, block)
    hidden = integer(draft.get('hidden_size'), 'hidden_size', 1, 65536)
    if hidden != target.get('hidden_size') or draft.get('num_target_layers') != target.get('num_hidden_layers'):
        raise ValueError('Draft and target hidden/layer geometry do not match')
    vocab = integer(draft.get('vocab_size'), 'vocab_size', 1, 1000000)
    if vocab != target.get('vocab_size'):
        raise ValueError('Vocabulary dimensions do not match; token ID equivalence still needs verification')
    integer(spec.get('mask_token_id'), 'mask_token_id', 0, vocab - 1)
    ids = spec.get('target_layer_ids')
    if not isinstance(ids, list) or not ids or len(ids) > 64 or len(ids) != len(set(ids)):
        raise ValueError('Invalid extracted target layers')
    for layer in ids:
        integer(layer, 'target_layer_id', 0, target['num_hidden_layers'] - 2)
    for config in [draft, target]:
        if context > integer(config.get('max_position_embeddings'), 'max_position_embeddings', 1, 1048576):
            raise ValueError('Context exceeds declared metadata; no RoPE scaling is applied')
    layers = integer(draft.get('num_hidden_layers'), 'draft layers', 1, 256)
    types = draft.get('layer_types')
    if not isinstance(types, list) or len(types) != layers or set(types) - {'sliding_attention', 'full_attention'}:
        raise ValueError('Unsupported draft attention layout')
    window = integer(draft.get('sliding_window'), 'sliding_window', 1, 1048576) if 'sliding_attention' in types else context
    head_dim = integer(draft.get('head_dim'), 'head_dim', 1, 1024)
    kv_heads = integer(draft.get('num_key_value_heads'), 'KV heads', 1, 256)
    draft_weight_bytes = integer(draft_weight_bytes, 'draft weight bytes', 1, 8000000000)
    # K + V, two bytes each. This excludes allocator padding and GPU scratch.
    kv_per_layer_token = 2 * 2 * kv_heads * head_dim
    draft_kv = kv_per_layer_token * sum(min(context, window) if t == 'sliding_attention' else context for t in types)
    features_per_token_f32 = len(ids) * hidden * 4
    # This particular Python MLX implementation retains all target features
    # through prefill when at least one draft layer uses full attention.
    retained = context if 'full_attention' in types else min(context, window - 1)
    feature_bytes = [retained * features_per_token_f32 // 2, retained * features_per_token_f32]
    return {
        'schema_version': 1, 'inference_executed': False, 'runtime_enabled': False,
        'quantized_target_pair_verified': False, 'context_verified_on_iPhone': False,
        'target_GGUF_SHA256': TARGET_GGUF_SHA256,
        'context_tokens': context, 'verify_width_including_anchor': verify_width,
        'maximum_proposals_in_this_block': verify_width - 1,
        'target_post_layer_ids': ids,
        # llama extracts layer inputs, while this checkpoint uses layer outputs.
        'llama_input_layer_ids': [i + 1 for i in ids],
        'draft_weight_file_bytes': draft_weight_bytes,
        'draft_F16_KV_payload_bytes_estimate': draft_kv,
        'naive_MLX_retained_target_features_F16_F32_bytes': feature_bytes,
        'bounded_streaming_feature_block_F32_bytes': features_per_token_f32 * verify_width,
        'weights_plus_draft_KV_payload_bytes_estimate': draft_weight_bytes + draft_kv,
        'excluded_from_estimate': ['target weights and KV', 'recurrent rollback snapshots',
            'Metal graph and allocator scratch', 'output logits', 'app memory and OS file cache'],
        'decision': 'Native Metal prototype is plausible; production activation requires rollback, sampling, memory and phone validation',
    }


def fetch_json(url, expected_sha=None, limit=2000000):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Flash-Local-metadata-audit'}), timeout=30) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Public metadata size bound exceeded')
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise ValueError('Pinned public metadata digest mismatch')
    return json.loads(raw), {'url': url, 'bytes': len(raw), 'sha256': digest}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context', type=int, choices=[8192, 16384, 32768, 65536], default=8192)
    p.add_argument('--verify-width', type=int, choices=[2, 4, 5, 8, 16], default=4)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    draft, a = fetch_json(f'https://huggingface.co/{DRAFT}/resolve/{DRAFT_REVISION}/config.json', DRAFT_CONFIG_SHA256)
    target, b = fetch_json(f'https://huggingface.co/{TARGET}/resolve/{TARGET_REVISION}/config.json', TARGET_CONFIG_SHA256)
    metadata, c = fetch_json(f'https://huggingface.co/api/models/{DRAFT}/revision/{DRAFT_REVISION}?blobs=true')
    if metadata.get('sha') != DRAFT_REVISION:
        raise ValueError('Unexpected checkpoint revision')
    weight = next(x for x in metadata['siblings'] if x['rfilename'] == 'model.safetensors')
    if weight.get('lfs', {}).get('sha256') != 'b9d55100f43e1a2bd56e287ac11309086ccb827ae5d6f546dc9197c2c75b34f6':
        raise ValueError('Unexpected draft weight identity')
    report = inspect(draft, target, args.context, args.verify_width, weight['size'])
    report.update(metadata_downloads=[a, b, c], downloaded_weight_bytes=0,
        checkpoint_parameters=metadata.get('safetensors', {}).get('total'),
        draft_revision=DRAFT_REVISION, target_revision=TARGET_REVISION)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Metadata only: {report["checkpoint_parameters"]} draft parameters; no weights downloaded; no inference executed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
