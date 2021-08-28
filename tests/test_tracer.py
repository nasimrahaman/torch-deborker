import pytest


def test_tracer():
    import torch
    import torch.nn as nn
    from torch_db import Tracer

    @Tracer.register_tracer_probes("simple_output_x2")
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.simple_module = nn.Linear(1, 32)
            self.activation = nn.ReLU()
            self.complicated_module = nn.Sequential(
                nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)
            )

        def forward(self, x):
            simple_output = self.simple_module(x)
            simple_output_x2 = Tracer.track(self, "simple_output_x2", simple_output * 2)
            activated = self.activation(simple_output_x2)
            return self.complicated_module(activated)

    x = torch.randn(2, 1)
    model = Model()
    with Tracer().trace(model) as tracer:
        y = model(x)  # noqa
        y.sum().backward()

    assert "simple_module" in tracer.find("").keys()
    assert "complicated_module.1" in tracer.find("comp").keys()


if __name__ == "__main__":
    # test_tracer()
    pytest.main()
