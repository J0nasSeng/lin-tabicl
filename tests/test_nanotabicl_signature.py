import torch

from src.tabicl._model.nanotabicl import (
    NanoTabICLv2,
    InducedTransformerBlock,
    TransformerBlock,
    _safe_standardize_with_train_stats,
)


def test_nanotabicl_forward_signature_compatibility_without_repr():
    torch.manual_seed(0)
    model = NanoTabICLv2(max_classes=5, out_dim=5, embed_dim=16, col_num_blocks=1, row_num_blocks=1, icl_num_blocks=1)

    x = torch.randn(2, 10, 6)
    y = torch.randint(0, 5, size=(2, 6))
    d = torch.tensor([4, 6], dtype=torch.long)

    pred = model(x, y, d=d)
    assert pred.shape == (2, 4, 5)
    assert torch.isfinite(pred).all()


def test_nanotabicl_forward_signature_compatibility_with_repr_and_extra_kwargs():
    torch.manual_seed(0)
    model = NanoTabICLv2(max_classes=4, out_dim=4, embed_dim=16, col_num_blocks=1, row_num_blocks=1, icl_num_blocks=1)

    x = torch.randn(3, 12, 5)
    y = torch.randint(0, 4, size=(3, 7))

    pred, repr_test = model(
        x,
        y,
        d=None,
        return_pre_decoder_repr=True,
        unused_kwarg=True,
    )

    assert pred.shape == (3, 5, 4)
    assert repr_test.shape == (3, 5, 16 * model.row_cls_tokens.size(2))
    assert torch.isfinite(pred).all()
    assert torch.isfinite(repr_test).all()


def test_nanotabicl_constant_input_fp16_stays_finite():
    torch.manual_seed(0)
    x_half = torch.ones(2, 12, 5, dtype=torch.float16)
    x_norm = _safe_standardize_with_train_stats(x_half, n_train=7)
    assert torch.isfinite(x_norm).all()

    model = NanoTabICLv2(max_classes=3, out_dim=3, embed_dim=16, col_num_blocks=1, row_num_blocks=1, icl_num_blocks=1)

    x = torch.ones(2, 12, 5, dtype=torch.float32)
    y = torch.randint(0, 3, size=(2, 7))

    pred = model(x, y)
    assert pred.shape == (2, 5, 3)
    assert torch.isfinite(pred).all()


def test_transformer_block_handles_empty_kv_without_nan():
    torch.manual_seed(0)
    block = TransformerBlock(embed_dim=16, num_heads=4, ssmax=True)

    q = torch.randn(2, 7, 16)
    kv = torch.randn(2, 5, 16)

    out = block(q, kv, kv_max_idx=0)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_induced_transformer_block_tfm2_path_stays_finite_for_large_inputs():
    torch.manual_seed(0)
    block = InducedTransformerBlock(embed_dim=16, num_heads=4, n_inducing=8, ssmax=True)

    # Large-but-finite inputs to stress the tfm1 -> tfm2 handoff path.
    q = torch.randn(3, 9, 16) * 1e3

    out = block(q, kv_max_idx=4)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
