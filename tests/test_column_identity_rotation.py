import torch
import torch.nn.functional as F
import pytest

from src.tabicl._model.embedding import ColEmbedding


def _build_embedder(*, embed_dim: int = 8, reserve_cls_tokens: int = 2, max_features: int = 8) -> ColEmbedding:
    return ColEmbedding(
        embed_dim=embed_dim,
        num_blocks=1,
        nhead=2,
        dim_feedforward=16,
        num_inds=4,
        affine=False,
        feature_group=False,
        target_aware=True,
        max_classes=4,
        max_features=max_features,
        reserve_cls_tokens=reserve_cls_tokens,
        enable_column_identity_rotation=True,
    )


def test_project_input_rotation_preserves_shape_and_skips_cls_slots():
    torch.manual_seed(0)

    model = _build_embedder(embed_dim=8, reserve_cls_tokens=2, max_features=6)
    X = torch.ones(1, 4, 4)

    out = model.project_input(X)

    X_pad = F.pad(X, (2, 0), value=-100.0)
    features = X_pad.transpose(1, 2).unsqueeze(-1)
    base = model.in_linear(features).transpose(1, 2)

    assert out.shape == base.shape
    assert torch.allclose(out[:, :, :2, :], base[:, :, :2, :], atol=1e-6)

    # Same scalar inputs across columns become distinct after per-column rotation.
    assert not torch.allclose(out[:, :, 2, :], out[:, :, 3, :])


def test_project_input_with_d_keeps_invalid_columns_zero():
    torch.manual_seed(0)

    model = _build_embedder(embed_dim=8, reserve_cls_tokens=1, max_features=6)
    X = torch.randn(2, 3, 5)
    d = torch.tensor([3, 5], dtype=torch.long)

    out = model.project_input(X, d=d)

    assert out.shape == (2, 3, 6, 8)
    assert torch.allclose(out[0, :, 4:, :], torch.zeros_like(out[0, :, 4:, :]))
    assert torch.any(out[1, :, 4:, :] != 0)


def test_column_identity_rotation_requires_even_embed_dim():
    with pytest.raises(ValueError, match="even embed_dim"):
        _build_embedder(embed_dim=7, reserve_cls_tokens=1, max_features=4)
