"""MOTUS_OPT_LANG_MMAP on CPU: the mapped, per-worker cached umt5 list equals torch.load's list bitwise.

A synthetic file in the converter's format (a list of bf16 [tokens, 4096] tensors, one per prompt line) is read with
the switch off and on.

  python -m pytest tests/test_lang_mmap.py
"""
import importlib
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _module(monkeypatch, on):
    monkeypatch.setenv("MOTUS_OPT_LANG_MMAP", "1" if on else "0")
    import data.robotwin2.robotwin_agilex_dataset as m
    return importlib.reload(m)


@pytest.fixture
def umt5_file(tmp_path):
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randn(int(n), 4096, generator=g).to(torch.bfloat16) for n in torch.randint(40, 90, (100,), generator=g)]
    path = tmp_path / "0.pt"
    torch.save(prompts, path)
    return str(path)


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x.dtype, x.shape, x.stride()) == (y.dtype, y.shape, y.stride())
        assert torch.equal(x.view(torch.int16), y.view(torch.int16))


def test_mapped_list_equals_loaded_list(monkeypatch, umt5_file):
    off = _module(monkeypatch, False)._load_embedding_list(umt5_file)
    on_module = _module(monkeypatch, True)
    first = on_module._load_embedding_list(umt5_file)
    _same(off, first)
    assert on_module._load_embedding_list(umt5_file) is first   # parsed once per worker, then reused


def test_switch_off_reads_the_file_every_time(monkeypatch, umt5_file):
    m = _module(monkeypatch, False)
    assert m._load_embedding_list(umt5_file) is not m._load_embedding_list(umt5_file)
    assert not m._LANG_CACHE
