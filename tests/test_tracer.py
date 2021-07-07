import pytest


def test_tracer():
    import torch
    from torchvision.models.resnet import resnet18
    from torch_db import Tracer

    x = torch.randn(1, 3, 224, 224)
    model = resnet18()
    with Tracer().trace(model) as tracer:
        y = model(x)
        y.sum().backward()


if __name__ == '__main__':
    # test_tracer()
    pytest.main()