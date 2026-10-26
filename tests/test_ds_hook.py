"""MOTUS_OPT_DS_HOOK on CPU: the cached parameter count DeepSpeed's ZeRO-1/2 gradient hooks see.

DeepSpeed calls `count_used_parameters_in_backward(all_params)` from every parameter's post-accumulate-grad hook and
keeps the maximum it has seen (`max_expected_hooks_seen`). The cached version must return, at every hook of a
backward, the maximum the uncached helper reports during that backward, recomputed for every new backward (graph
task) and for every parameter list.

  python -m pytest tests/test_ds_hook.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
s12 = pytest.importorskip("deepspeed.runtime.zero.stage_1_and_2")
from deepspeed.runtime.utils import count_used_parameters_in_backward as uncached  # noqa: E402
from deepspeed.utils.torch import register_grad_hook  # noqa: E402

from utils.fastpath import patch_deepspeed_hook_count  # noqa: E402


def _model():
    torch.manual_seed(0)
    return torch.nn.ModuleDict({name: torch.nn.Linear(8, 8) for name in ("a", "b", "unused")})


def _hooked(model, params, other):
    patch_deepspeed_hook_count()
    cached = s12.count_used_parameters_in_backward
    assert getattr(cached, "_motus_cached", False)
    log = []

    def hook(*_):
        log.append((torch._C._current_graph_task_id(), cached(params), uncached(params), cached(other), uncached(other)))

    for p in params:
        register_grad_hook(p, hook)
    return log


def test_cached_count_is_the_per_backward_maximum():
    model = _model()
    params = list(model.parameters())
    other = list(model["b"].parameters())
    log = _hooked(model, params, other)
    x = torch.randn(4, 8)
    tasks = set()
    for used, expected, expected_other in ((("a", "b"), 4, 2), (("b",), 2, 2), (("a",), 2, 0), (("a", "b"), 4, 2)):
        log.clear()
        y = x
        for name in used:
            y = model[name](y)
        y.sum().backward()
        assert len(log) == expected                     # one hook per parameter that received a gradient
        assert len({entry[0] for entry in log}) == 1    # one graph task per backward
        tasks.add(log[0][0])
        assert all(entry[1] == max(e[2] for e in log) == expected for entry in log)
        assert all(entry[3] == max(e[4] for e in log) == expected_other for entry in log)
    assert len(tasks) == 4


def test_outside_backward_is_not_cached():
    patch_deepspeed_hook_count()
    with pytest.raises(RuntimeError):
        s12.count_used_parameters_in_backward(list(_model().parameters()))
