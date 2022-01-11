import pytest


def test_tracer_recurrent():
    import torch
    import torch.nn as nn
    from torch_db import Tracer

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.simple_module = nn.Linear(1, 32)
            self.activation = nn.ReLU()
            self.reused_module = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
            self.final_module = nn.Linear(32, 1)

        def forward(self, x, timesteps=3):
            output = self.simple_module(x)
            output = self.activation(output)
            for _ in range(timesteps):
                output = self.reused_module(output)
            output = self.final_module(output)
            return output

    x = torch.randn(2, 1)
    model = Model()
    with Tracer().trace(model, clear_records=True) as tracer:
        y = model(x)  # noqa
        y.sum().backward()
        assert "simple_module" in tracer.find("").keys()
        assert (['input.0', 'output.0', 'input.1', 'output.1', 'input.2', 'output.2', 'grad_input.0', 'grad_output.0',
                 'grad_input.1', 'grad_output.1', 'grad_input.2', 'grad_output.2']) == list(
            tracer.records["reused_module"].keys())
        assert not tracer.records['reused_module']['grad_input.0'][0].eq(tracer.records['reused_module']['grad_input.1'][0]).all()
    with Tracer().trace(model, clear_records=True) as tracer:
        y = model(x, timesteps=4)  # noqa
        y.sum().backward()
        assert 'grad_output.3' in tracer.records["reused_module"].keys()

if __name__ == "__main__":
    pytest.main()
