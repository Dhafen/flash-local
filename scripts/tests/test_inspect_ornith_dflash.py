import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspect_ornith_dflash import inspect


class DFlashInspectionTests(unittest.TestCase):
    def setUp(self):
        # Public checkpoint geometry; no weights or generated model outputs.
        self.draft = {'architectures': ['DFlashDraftModel'], 'hidden_size': 2048,
            'num_hidden_layers': 6, 'num_target_layers': 40, 'vocab_size': 248320,
            'num_key_value_heads': 8, 'head_dim': 128, 'sliding_window': 4096,
            'max_position_embeddings': 262144,
            'layer_types': ['sliding_attention'] * 5 + ['full_attention'],
            'dflash_config': {'block_size': 16, 'mask_token_id': 248077,
                'target_layer_ids': [1, 6, 11, 16, 22, 27, 32, 37]}}
        self.target = {'text_config': {'hidden_size': 2048, 'num_hidden_layers': 40,
            'vocab_size': 248320, 'max_position_embeddings': 262144}}

    def test_real_checkpoint_geometry_does_not_require_equal_attention_heads(self):
        r = inspect(self.draft, self.target)
        self.assertEqual(r['llama_input_layer_ids'], [2, 7, 12, 17, 23, 28, 33, 38])
        self.assertEqual(r['maximum_proposals_in_this_block'], 3)
        self.assertFalse(r['quantized_target_pair_verified']); self.assertFalse(r['runtime_enabled'])
        self.assertFalse(r['inference_executed'])

    def test_65k_budget_exposes_naive_mlx_feature_cost_and_separate_draft_KV(self):
        r = inspect(self.draft, self.target, 65536)
        self.assertEqual(r['draft_F16_KV_payload_bytes_estimate'], 352321536)
        self.assertEqual(r['naive_MLX_retained_target_features_F16_F32_bytes'], [2147483648, 4294967296])
        self.assertEqual(r['bounded_streaming_feature_block_F32_bytes'], 262144)
        self.assertEqual(r['weights_plus_draft_KV_payload_bytes_estimate'], 1124141216)

    def test_rejects_invalid_layers_masks_and_changed_target(self):
        for field, value in [('hidden_size', 4096), ('num_target_layers', 39), ('vocab_size', 123)]:
            changed = copy.deepcopy(self.draft); changed[field] = value
            with self.assertRaises(ValueError): inspect(changed, self.target)
        for field, value in [('target_layer_ids', [1, 1]), ('target_layer_ids', [40]),
                             ('target_layer_ids', [True]), ('mask_token_id', 248320), ('block_size', 1)]:
            changed = copy.deepcopy(self.draft); changed['dflash_config'][field] = value
            with self.assertRaises(ValueError): inspect(changed, self.target)

    def test_never_extends_rope_or_claims_larger_context(self):
        self.target['text_config']['max_position_embeddings'] = 32768
        with self.assertRaises(ValueError): inspect(self.draft, self.target, 65536)
        with self.assertRaises(ValueError): inspect(self.draft, self.target, True)
        with self.assertRaises(ValueError): inspect(self.draft, self.target, 8192, 32)

    def test_refuses_unreviewed_attention_layout_and_dflash2(self):
        for field, value in [('layer_types', ['full_attention']), ('layer_types', ['linear_attention'] * 6),
                             ('architectures', ['DFlash2DraftModel'])]:
            changed = copy.deepcopy(self.draft); changed[field] = value
            with self.assertRaises(ValueError): inspect(changed, self.target)
