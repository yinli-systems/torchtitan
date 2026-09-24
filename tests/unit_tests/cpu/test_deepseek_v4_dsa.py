# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the DeepSeek-V4 DSA attention core.

The DSA core expresses the fixed patterns (sliding window, compress-block
causality) as a ``mask_mod`` predicate and only the CSA top-k selection as
indices. These tests verify that the resulting attended (q, kv) pairs match
the previous index-based formulation for every compression ratio.
"""

import inspect
import unittest

import torch
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.deepseek_v4.attention import (
    CompressedSparseAttention,
    DSV4FlexInnerAttention,
)
from torchtitan.models.deepseek_v4.compressor import Indexer


def window_idxs(window_size, bsz, seqlen, device):
    window = min(seqlen, window_size)
    base = torch.arange(seqlen, device=device).unsqueeze(1)
    idxs = (base - window + 1).clamp(0) + torch.arange(window, device=device)
    idxs = torch.where(idxs > base, -1, idxs)
    return idxs.unsqueeze(0).expand(bsz, -1, -1)


def compress_idxs(ratio, bsz, seqlen, device, offset):
    """Old HCA index formulation: causal compressed positions + offset."""
    compress_len = seqlen // ratio
    if compress_len == 0:
        return torch.empty((bsz, seqlen, 0), dtype=torch.int64, device=device)
    idxs = torch.arange(compress_len, device=device).repeat(seqlen, 1)
    causal_limit = torch.arange(1, seqlen + 1, device=device).unsqueeze(1)
    causal_limit = causal_limit // ratio
    idxs = torch.where(idxs >= causal_limit, -1, idxs + offset)
    return idxs.unsqueeze(0).expand(bsz, -1, -1)


def old_indexer_topk(bsz, seqlen, ratio, topk, device, seed):
    """Old CSA top-k selection: causal-masked scores, then topk + offset."""
    g = torch.Generator(device).manual_seed(seed)
    scores = torch.randn(bsz, seqlen, seqlen // ratio, generator=g, device=device)
    causal_limit = torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
    mask = torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1)
    mask = mask >= causal_limit
    scores = scores + torch.where(mask, torch.finfo(torch.float32).min, 0)
    _, topk_idxs = scores.topk(min(topk, seqlen // ratio), dim=-1)
    topk_idxs = torch.where(topk_idxs >= causal_limit, -1, topk_idxs)
    return topk_idxs


def old_attended(topk_idxs, sink_idx):
    """Attended kv positions per (b, q) from the old index formulation."""
    out = []
    for b in range(topk_idxs.size(0)):
        for q in range(topk_idxs.size(1)):
            s = {int(i) for i in topk_idxs[b, q] if i >= 0}
            s.add(sink_idx)
            out.append(s)
    return out


def new_attended(block_mask, seqlen, n_cmp):
    """Attended kv positions per (b, q) from the new block mask's mask_mod."""
    kv_len = seqlen + n_cmp + 1
    B = block_mask.kv_num_blocks.size(0)
    out = []
    for b in range(B):
        q_idx = torch.arange(seqlen).unsqueeze(0).unsqueeze(-1)
        kv_idx = torch.arange(kv_len).unsqueeze(0).unsqueeze(-2)
        m = block_mask.mask_mod(torch.tensor(b), torch.tensor(0), q_idx, kv_idx)
        m = m[0]
        for q in range(seqlen):
            out.append(set(m[q].nonzero().flatten().tolist()))
    return out


def build_dsa(ratio, window_size, block_size=128):
    cfg = DSV4FlexInnerAttention.Config(
        block_size=block_size,
        window_size=window_size,
        compress_ratio=ratio,
        softmax_scale=0.1,
        index_topk=16,
    )
    return DSV4FlexInnerAttention(cfg)


class TestDSABlockMask(unittest.TestCase):
    def test_csa_gather_attention_reference_forward_and_backward(self):
        torch.manual_seed(0)
        seqlen, n_heads, head_dim = 16, 2, 8
        ratio, index_topk = 4, 3
        cfg = CompressedSparseAttention.Config(
            block_size=8,
            window_size=5,
            compress_ratio=ratio,
            softmax_scale=head_dim**-0.5,
            index_topk=index_topk,
        )
        attention = CompressedSparseAttention(cfg)
        inputs = [
            torch.randn(seqlen, n_heads, head_dim, requires_grad=True),
            torch.randn(seqlen, head_dim, requires_grad=True),
            torch.randn(seqlen // ratio, head_dim, requires_grad=True),
            torch.randn(seqlen, 3, 6, requires_grad=True),
            torch.randn(seqlen // ratio, 6, requires_grad=True),
            torch.randn(seqlen, 3, requires_grad=True),
            torch.randn(n_heads, requires_grad=True),
        ]

        output = attention(*inputs)
        self.assertEqual(output.shape, (seqlen, n_heads, head_dim))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        for tensor in (*inputs[:3], inputs[-1]):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_static_mask_does_not_capture_token_pair_tensor(self):
        seqlen = 32768
        dsa = build_dsa(ratio=128, window_size=2048, block_size=128)
        block_mask = dsa.build_block_mask(seqlen=seqlen, device="cpu")

        closure = inspect.getclosurevars(block_mask.mask_mod)
        captured_tensors = [
            value
            for value in closure.nonlocals.values()
            if isinstance(value, torch.Tensor)
        ]
        self.assertEqual(captured_tensors, [])

        metadata = [
            block_mask.kv_num_blocks,
            block_mask.kv_indices,
            block_mask.q_num_blocks,
            block_mask.q_indices,
        ]
        metadata_bytes = sum(x.numel() * x.element_size() for x in metadata)
        self.assertLess(metadata_bytes, 1024 * 1024)

    def test_static_masks_match_index_formulation(self):
        device = torch.device("cpu")
        cases = [
            (128, 17, 0, 32),
            (128, 63, 1, (32, 16)),
            (256, 64, 128, 32),
            (512, 128, 128, (64, 32)),
        ]
        for seqlen, window_size, ratio, block_size in cases:
            with self.subTest(
                seqlen=seqlen,
                window_size=window_size,
                ratio=ratio,
                block_size=block_size,
            ):
                n_cmp = seqlen // ratio if ratio > 1 else 0
                sink_idx = seqlen + n_cmp
                win = window_idxs(window_size, 1, seqlen, device)
                compress = (
                    compress_idxs(ratio, 1, seqlen, device, seqlen)
                    if ratio > 1
                    else torch.empty((1, seqlen, 0), dtype=torch.int64)
                )
                selected = (
                    torch.cat([win, compress], dim=-1) if compress.size(-1) else win
                )

                dsa = build_dsa(ratio, window_size, block_size)
                block_mask = dsa.build_block_mask(seqlen=seqlen, device=device)
                expected = old_attended(selected, sink_idx)
                actual = new_attended(block_mask, seqlen, n_cmp)
                self.assertEqual(expected, actual)

                bq, bk = (
                    block_size
                    if isinstance(block_size, tuple)
                    else (block_size, block_size)
                )
                for q_block in range(seqlen // bq):
                    q_slice = expected[q_block * bq : (q_block + 1) * bq]
                    expected_blocks = {
                        kv_idx // bk for attended in q_slice for kv_idx in attended
                    }
                    num_blocks = int(block_mask.kv_num_blocks[0, 0, q_block])
                    actual_blocks = set(
                        block_mask.kv_indices[0, 0, q_block, :num_blocks].tolist()
                    )
                    self.assertEqual(expected_blocks, actual_blocks)

    def test_attended_sets_match_old_formulation(self):
        torch.manual_seed(0)
        device = torch.device("cpu")
        bsz, seqlen, window_size = 2, 512, 16
        for ratio, topk in [(0, 0), (1, 0), (4, 16), (128, 0)]:
            n_cmp = seqlen // ratio if ratio > 1 else 0
            sink_idx = seqlen + n_cmp
            with self.subTest(ratio=ratio):
                if ratio == 4:
                    topk_sel = old_indexer_topk(bsz, seqlen, ratio, topk, device, 7)
                    compress = torch.where(topk_sel < 0, -1, topk_sel + seqlen)
                elif ratio > 1:
                    topk_sel = None
                    compress = compress_idxs(ratio, bsz, seqlen, device, seqlen)
                else:
                    topk_sel = None
                    compress = torch.empty(
                        (bsz, seqlen, 0), dtype=torch.int64, device=device
                    )
                win = window_idxs(window_size, bsz, seqlen, device)
                topk_idxs = (
                    torch.cat([win, compress], dim=-1) if compress.size(-1) else win
                )
                selected_indices = torch.cat(
                    [
                        topk_idxs,
                        torch.full(
                            (bsz, seqlen, 1),
                            sink_idx,
                            dtype=torch.int64,
                            device=device,
                        ),
                    ],
                    dim=-1,
                )

                dsa = build_dsa(ratio, window_size)
                bm = dsa._build_block_mask(
                    bsz,
                    seqlen,
                    sink_idx + 1,
                    selected_indices,
                    device,
                )
                self.assertIsInstance(bm, BlockMask)
                expected = old_attended(topk_idxs, sink_idx)
                actual = new_attended(bm, seqlen, n_cmp)
                self.assertEqual(expected, actual)

    def test_indexer_select_matches_old_topk(self):
        torch.manual_seed(0)
        device = torch.device("cpu")
        seqlen, ratio, topk = 512, 4, 16
        n_cmp = seqlen // ratio
        g = torch.Generator(device).manual_seed(11)
        idx_q = torch.randn(seqlen, 8, 32, generator=g, device=device)
        idx_k = torch.randn(n_cmp, 32, generator=g, device=device)
        idx_w = torch.randn(seqlen, 8, generator=g, device=device)

        selected = Indexer.select(
            idx_q, idx_k, idx_w, seqlen=seqlen, ratio=ratio, topk=topk
        )

        # Old formulation: causal-masked scores -> topk -> map invalid to -1.
        scores = torch.einsum("shd,td->sht", idx_q, idx_k)
        scores = scores.relu_() * idx_w.unsqueeze(-1)
        scores = scores.sum(dim=1)
        causal_limit = torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
        mask = torch.arange(n_cmp, device=device).repeat(seqlen, 1) >= causal_limit
        scores = scores + torch.where(mask, torch.finfo(idx_q.dtype).min, 0)
        _, topk_idxs = scores.topk(min(topk, n_cmp), dim=-1)
        old = torch.where(topk_idxs >= causal_limit, -1, topk_idxs)

        # Valid (non-masked) entries must agree exactly; the new formulation
        # keeps raw indices and gates causality in the mask instead of -1.
        self.assertEqual(selected.shape, (seqlen, topk))
        self.assertTrue((selected >= 0).all())
        self.assertTrue(torch.equal(selected.masked_fill(old < 0, -1), old))


if __name__ == "__main__":
    unittest.main()
