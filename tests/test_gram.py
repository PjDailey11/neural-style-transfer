from __future__ import annotations

import torch

from nst.utils.gram import gram_matrix


def test_gram_matrix_shape() -> None:
    x = torch.randn(2, 16, 8, 8)
    g = gram_matrix(x)
    assert g.shape == (2, 16, 16)
