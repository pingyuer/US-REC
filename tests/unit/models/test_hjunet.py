import pytest
pytestmark = pytest.mark.unit

import torch

from models.hjunet import HJUNet


def _forward(model: HJUNet, batch_shape=(1, 3, 128, 128), device="cpu"):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        input_tensor = torch.randn(batch_shape, device=device)
        return model(input_tensor)


def test_hjunet_output_shape_default():
    model = HJUNet(num_classes=3)
    output = _forward(model)
    assert output.shape == (1, 3, 128, 128)


def test_hjunet_probabilistic_output_shape():
    model = HJUNet(num_classes=2, probabilistic={
        "strategy": "mix_add",
        "fusion_type": "add_norm",
        "layers": [2, 3],
        "prior": "cbl_linear",
        "posterior": "cbl_linear",
        "use_layernorm": True,
        "sample_channels": 128,
    })
    output = _forward(model, batch_shape=(2, 3, 96, 96))
    assert output.shape == (2, 2, 96, 96)
