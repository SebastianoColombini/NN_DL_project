import pytest
import torch

from saliency.models import SUBMODULES
from saliency.registry import REGISTRY


@pytest.mark.parametrize("model_id", ["v1.1", "v2.0"])
def test_random_init_shapes(model_id):
    spec = REGISTRY[model_id]
    torch.manual_seed(0)
    model = spec.builder().eval()
    assert [n for n, _ in model.named_children()] == list(SUBMODULES)
    x = torch.rand(2, 3, *spec.input_size)
    with torch.inference_mode():
        outs = model.forward_all(x)
        single = model(x)
    assert len(outs) == 4
    for o in outs:
        assert o.shape == (2, 1, *spec.input_size)
    assert torch.equal(single, outs[0])
    assert 0 <= float(single.min()) and float(single.max()) <= 1
