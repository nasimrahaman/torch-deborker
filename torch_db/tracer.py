from contextlib import contextmanager
from typing import Callable, Sequence, Mapping, Optional, Tuple

import torch
import torch.nn as nn


class Tracer(object):
    def __init__(self, detach: bool = True):
        self.records = dict()
        self.handles = dict()
        self.detach = detach

    def _make_hook(self, name: str) -> Tuple[Callable, Callable]:
        def detach(x):
            if self.detach:
                if torch.is_tensor(x):
                    return x.detach()
                elif isinstance(x, Sequence):
                    return [detach(_x) for _x in x]
                elif isinstance(x, Mapping):
                    return {key: detach(x[key]) for key in x}
                else:
                    # x might be something undetachable,
                    # in which case this is a fallback
                    return x
            else:
                return x

        def forward_hook(module, input, output):
            if self.records.get(name) is None:
                self.records[name] = {"input": detach(input), "output": detach(output)}
            else:
                timesteps = [0]
                for x in self.records[name].keys():
                    if not x.startswith("grad") and "." in x:
                        timesteps.append(int(x.split(".")[1]))
                self.records[name].update(
                    {
                        f"input.{max(timesteps) + 1}": detach(input),
                        f"output.{max(timesteps) + 1}": detach(output),
                    }
                )

        def backward_hook(module, grad_input, grad_output):
            if name not in self.records:
                return
            assert isinstance(self.records[name], dict)

            if self.records[name].get("grad_input") is None:
                self.records[name].update(
                    {
                        "grad_input": detach(grad_input),
                        "grad_output": detach(grad_output),
                    }
                )
            else:
                timesteps = [0]
                for x in self.records[name].keys():
                    if x.startswith("grad") and "." in x:
                        timesteps.append(int(x.split(".")[1]))
                self.records[name].update(
                    {
                        f"grad_input.{max(timesteps) + 1}": detach(grad_input),
                        f"grad_output.{max(timesteps) + 1}": detach(grad_output),
                    }
                )

        return forward_hook, backward_hook

    def attach_probes(self, model: "torch.nn.Module") -> "Tracer":
        def _attach_probes(m):
            probe_names = getattr(m, "TRACER_PROBES", None)
            if probe_names is not None:
                tracer_probes = {}
                for probe_name in probe_names:
                    tracer_probes[probe_name] = nn.Identity()
                m.tracer_probes = nn.ModuleDict(tracer_probes)

        for name, module in model.named_modules():
            _attach_probes(module)
        return self

    def detach_probes(self, model: "torch.nn.Module") -> "Tracer":
        def _detach_probes(m):
            if hasattr(m, "tracer_probes"):
                del m.tracer_probes

        for name, module in model.named_modules():
            _detach_probes(module)
        return self

    def bind(self, model: "torch.nn.Module") -> "Tracer":
        self.unbind()
        self.attach_probes(model)
        for name, module in model.named_modules():
            forward_hook, backward_hook = self._make_hook(name)
            self.handles[name] = module.register_forward_hook(forward_hook)
            self.handles[f"{name}/grad"] = module.register_full_backward_hook(
                backward_hook
            )
        return self

    def get(self, name: str = "", key: Optional[str] = None):
        record = self.records.get(name)
        if key is None:
            return record
        else:
            return record[key]

    def find(self, name: str = "", mode: str = "startswith"):
        hit_criteria = {
            "startswith": lambda key: key.startswith(name),
            "endswith": lambda key: key.endswith(name),
            "in": lambda key: name in key,
        }
        assert mode in hit_criteria
        hit_criterion = hit_criteria[mode]
        return {key: self.records[key] for key in self.records if hit_criterion(key)}

    def unbind(
        self,
        model: Optional["torch.nn.Module"] = None,
        clear_records: bool = True,
        detach_probes: bool = False,
    ) -> "Tracer":
        if clear_records:
            self.records.clear()
        if detach_probes:
            assert model is not None
            self.detach_probes(model)
        for handle in self.handles.values():
            handle.remove()
        return self

    @contextmanager
    def trace(self, model, clear_records: bool = False, detach_probes: bool = True):
        self.bind(model)
        yield self
        self.unbind(model, clear_records=clear_records, detach_probes=detach_probes)

    @staticmethod
    def track(
        module: "torch.nn.Module", name: str, tensor: "torch.Tensor"
    ) -> "torch.Tensor":
        if not hasattr(module, "tracer_probes"):
            return tensor
        if name in module.tracer_probes.keys():
            tensor = module.tracer_probes[name](tensor)
        return tensor

    @staticmethod
    def register_tracer_probes(*tracer_probe_names):
        def decorator(cls):
            cls.TRACER_PROBES = list(tracer_probe_names)
            return cls

        return decorator

    def keys(self):
        return list(self.records.keys())
