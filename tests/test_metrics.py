import pytest
import torch

from saliency.metrics import compute_metrics, kld_loss, total_loss


def test_identical_maps():
    torch.manual_seed(0)
    m = torch.rand(2, 1, 24, 32)
    r = compute_metrics(m, m.clone())
    assert r["cc"] == pytest.approx(1.0, abs=1e-5)
    assert r["sim"] == pytest.approx(1.0, abs=1e-5)
    # the notebook formula has eps terms, so KLD is only ~0 (slightly negative)
    assert r["kld"] == pytest.approx(0.0, abs=1e-3)
    assert float(total_loss(m, m, m, m, m)) == pytest.approx(0.0, abs=2e-3)


def test_different_maps():
    torch.manual_seed(0)
    a, b = torch.rand(1, 1, 24, 32), torch.rand(1, 1, 24, 32)
    r = compute_metrics(a, b)
    assert set(r) == {"cc", "kld", "sim"}
    assert -1 <= r["cc"] < 0.5 and 0 < r["sim"] < 1 and r["kld"] > 0
    assert float(kld_loss(a, b)) == pytest.approx(r["kld"])
