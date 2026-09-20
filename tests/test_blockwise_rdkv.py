import math
import unittest

import torch

from obkv_fast import apply_score_pooling
from obkv_accel.blockwise_decode import (
    BlockwiseRDKVConfig,
    _fixed_slot_targets,
    _logical_layer_bits,
    _score_block,
    plan_block_compressions,
)
from obkv_accel.packing import (
    _pad_and_stack_per_head_layers,
    build_packed_layer,
    pack_k_mixed,
    pack_k_mixed_per_head,
    pack_v_segments,
)


class TestBlockwiseSchedule(unittest.TestCase):
    def test_fixed_slot_capacity_scales_with_block_budget(self):
        targets_128 = _fixed_slot_targets(
            128, block_size=128, block_budget_tokens=1
        )
        targets_256 = _fixed_slot_targets(
            128, block_size=256, block_budget_tokens=2
        )
        self.assertEqual(targets_128, ((8, 8, 8), (128, 128, 128), 8))
        self.assertEqual(
            targets_256, ((16, 16, 16), (128, 128, 128), 16)
        )

    def test_512_tokens_has_seven_lookahead_events_and_final_flush(self):
        plan = plan_block_compressions(512, 64)
        self.assertEqual(len(plan), 8)
        self.assertEqual(
            sum(not event["is_final_flush"] for event in plan), 7
        )
        self.assertEqual(plan[0]["token_start"], 0)
        self.assertEqual(plan[-1]["token_end"], 511)
        self.assertEqual(plan[-1]["query_mode"], "self_causal")

    def test_128_and_192_smoke_schedules_are_contiguous(self):
        for n_tokens, expected_blocks in ((128, 2), (192, 3)):
            plan = plan_block_compressions(n_tokens, 64)
            self.assertEqual(len(plan), expected_blocks)
            flattened = [
                token
                for event in plan
                for token in range(
                    event["token_start"], event["token_end"] + 1
                )
            ]
            self.assertEqual(flattened, list(range(n_tokens)))
            self.assertTrue(plan[-1]["is_final_flush"])

    def test_partial_successor_drains_without_overlap(self):
        plan = plan_block_compressions(127, 64)
        self.assertEqual(
            [(event["token_start"], event["token_end"]) for event in plan],
            [(0, 63), (64, 126)],
        )
        self.assertTrue(all(event["is_final_flush"] for event in plan))


class TestBlockwiseScoring(unittest.TestCase):
    def test_next_block_attention_scores_match_explicit_reference(self):
        torch.manual_seed(7)
        h_kv, groups, q_len, t_len, dim = 2, 2, 5, 4, 8
        q = torch.randn(h_kv * groups, q_len, dim)
        k = torch.randn(1, h_kv, t_len, dim)
        token, channel = _score_block(
            q, k, num_kv_groups=groups, causal=False
        )
        qg = q.view(h_kv, groups, q_len, dim)
        logits = torch.einsum("hgqd,htd->hgqt", qg, k[0]) / math.sqrt(dim)
        expected_token = logits.softmax(-1).sum((1, 2))
        expected_channel = qg.square().mean((1, 2)) * k[0].square().mean(1)
        torch.testing.assert_close(token, expected_token)
        torch.testing.assert_close(channel, expected_channel)

    def test_final_flush_is_causal(self):
        q = torch.ones(1, 4, 2)
        k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0],
                            [1.0, 1.0], [2.0, 0.0]]]])
        token, _ = _score_block(q, k, num_kv_groups=1, causal=True)
        # The last key is visible only to the last query; without the causal
        # mask it would receive mass from every query.
        self.assertGreater(float(token[0, 0]), float(token[0, 3]))
        self.assertTrue(torch.isfinite(token).all())

    def test_raw_bit_accounting_uses_kept_v_positions_for_k(self):
        class Allocation:
            allocation_v_bits = torch.tensor([[0, 2, 16], [4, 0, 8]])
            allocation_k_bits = torch.tensor([[2, 4], [8, 16]])

        # V: (18 + 12) * D=2 = 60. K: (2+4)*2 + (8+16)*2 = 60.
        self.assertEqual(_logical_layer_bits(Allocation(), 2), 120)

    def test_protocol_config_validation(self):
        config = BlockwiseRDKVConfig(
            block_size=64,
            block_budget_tokens=8,
            k_budget_ratio=0.5,
            obs_window=32,
            pool_kernel_size=5,
            pool_padding="reflect",
            v_bit_options=torch.tensor([0, 2, 4, 8, 16]),
            k_bit_options=torch.tensor([0, 2, 4, 8, 16]),
            epsilon_v={},
            epsilon_k={},
        )
        config.validate()


class TestPackingOptimizations(unittest.TestCase):
    def test_precomputed_segment_counts_preserve_packed_tensors(self):
        torch.manual_seed(1)
        keys = torch.randn(1, 1, 7, 8, dtype=torch.float16)
        k_bits = torch.tensor([0, 2, 4, 8, 16, 2, 4, 8])
        expected_k = pack_k_mixed(keys, k_bits)
        actual_k = pack_k_mixed(
            keys, k_bits, segment_counts=(2, 2, 3)
        )

        values = torch.randn(1, 1, 6, 8, dtype=torch.float16)
        v_bits = torch.tensor([2, 4, 8, 2, 8, 4])
        expected_v = pack_v_segments(values, v_bits)
        actual_v = pack_v_segments(
            values, v_bits, segment_counts=(2, 2, 2)
        )

        for expected, actual in zip(expected_k + expected_v, actual_k + actual_v):
            if torch.is_tensor(expected):
                torch.testing.assert_close(actual, expected)
            else:
                self.assertEqual(actual, expected)

    def test_deferred_k_stack_matches_single_pass_reference(self):
        torch.manual_seed(11)
        h_kv, t_len, head_dim, groups = 2, 7, 8, 2
        keys = torch.randn(
            1, h_kv, t_len, head_dim, dtype=torch.float16
        )
        values = torch.randn_like(keys)
        v_bits = (
            torch.tensor([2, 4, 8, 16, 0, 2, 4]),
            torch.tensor([4, 16, 2, 8, 0, 4, 2]),
        )
        k_bits = torch.tensor(
            [
                [2, 4, 8, 0, 16, 2, 4, 8],
                [4, 2, 0, 8, 16, 4, 2, 8],
            ]
        )

        layers = []
        v16_layers = []
        for head in range(h_kv):
            kept = (v_bits[head] > 0).nonzero(as_tuple=True)[0]
            kept_bits = v_bits[head].index_select(0, kept)
            counts = tuple(
                int((kept_bits == bit).sum()) for bit in (2, 4, 8)
            )
            layer, _, v16 = build_packed_layer(
                keys[:, head:head + 1].index_select(2, kept),
                values[:, head:head + 1].index_select(2, kept),
                kept_bits,
                k_bits[head],
                v_segment_counts=counts,
                defer_k_for_per_head_stack=True,
            )
            self.assertIsNotNone(layer.deferred_k_for_stack)
            layers.append(layer)
            v16_layers.append(v16)

        packed, _ = _pad_and_stack_per_head_layers(
            layers,
            v16_layers,
            H_q=h_kv * groups,
            gqa_factor=groups,
            head_dim=head_dim,
            device=keys.device,
            dtype=keys.dtype,
            k_bits_per_head=k_bits,
        )

        max_t_v = max(layer.T_eff for layer in layers)
        max_v16 = max(layer.n_v16 for layer in layers)
        canvas = torch.zeros(
            1,
            h_kv,
            max_t_v + max_v16,
            head_dim,
            dtype=keys.dtype,
        )
        for head, layer in enumerate(layers):
            raw = layer.deferred_k_for_stack
            canvas[:, head:head + 1, :layer.T_eff] = (
                raw[:, :, :layer.T_eff]
            )
            canvas[
                :,
                head:head + 1,
                max_t_v:max_t_v + layer.n_v16,
            ] = raw[:, :, layer.T_eff:layer.T_eff + layer.n_v16]

        reference = pack_k_mixed_per_head(canvas, k_bits)
        actual = (
            packed.K_2bit,
            packed.K_4bit,
            packed.K_8bit,
            packed.K_ch_scale,
            packed.K_ch_zp,
            packed.K_ch_sort_idx_per_head,
            packed.K_ch_perm_padded_idx_per_head,
            packed.K_ch_seg_bounds_per_head,
            packed.K_ch_seg_bounds,
        )
        for expected, result in zip(reference, actual):
            if torch.is_tensor(expected):
                torch.testing.assert_close(result, expected)
            else:
                self.assertEqual(result, expected)
        self.assertIsNone(packed.deferred_k_for_stack)


class TestScorePooling(unittest.TestCase):
    def test_reflect_pooling_is_identity_for_one_token(self):
        scores = torch.tensor([[1.5], [3.0]])
        pooled = apply_score_pooling(
            scores, pool_type="avg", kernel_size=5, padding_mode="reflect"
        )
        torch.testing.assert_close(pooled, scores)

    def test_reflect_pooling_uses_kernel_three_for_two_tokens(self):
        scores = torch.tensor([[1.0, 4.0], [2.0, 8.0]])
        pooled = apply_score_pooling(
            scores, pool_type="avg", kernel_size=5, padding_mode="reflect"
        )
        expected = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(
                scores.unsqueeze(1), (1, 1), mode="reflect"
            ),
            kernel_size=3,
            stride=1,
        ).squeeze(1)
        torch.testing.assert_close(pooled, expected)

    def test_reflect_pooling_keeps_configured_kernel_for_normal_blocks(self):
        scores = torch.arange(16, dtype=torch.float32).view(2, 8)
        pooled = apply_score_pooling(
            scores, pool_type="avg", kernel_size=5, padding_mode="reflect"
        )
        expected = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(
                scores.unsqueeze(1), (2, 2), mode="reflect"
            ),
            kernel_size=5,
            stride=1,
        ).squeeze(1)
        torch.testing.assert_close(pooled, expected)


if __name__ == "__main__":
    unittest.main()
