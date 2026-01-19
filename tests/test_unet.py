import torch

from models.unet import UNet


def _forward(model: UNet, batch_shape=(1, 3, 128, 128), device="cpu"):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        input_tensor = torch.randn(batch_shape, device=device)
        return model(input_tensor)


def test_unet_output_shape():
    model = UNet(num_classes=3, input_channels=3, bilinear=True)
    output = _forward(model)
    assert output.shape == (1, 3, 128, 128)


def test_unet_non_bilinear_forward():
    model = UNet(num_classes=2, input_channels=3, bilinear=False)
    output = _forward(model, batch_shape=(2, 3, 96, 96))
    assert output.shape == (2, 2, 96, 96)
