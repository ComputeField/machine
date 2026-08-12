# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Hugging Face bundles are prepared without executing bundled code."""

import json
from types import SimpleNamespace
import zipfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from model_artifact import materialize_hf_model, prepare_model_artifact


def _bundle(path: Path) -> None:
    source = path.parent / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"architectures": ["BertForSequenceClassification"]}),
        encoding="utf-8",
    )
    (source / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    shared = torch.arange(4, dtype=torch.float32)
    torch.save({"first": shared, "second": shared}, source / "pytorch_model.bin")
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for item in source.iterdir():
            archive.write(item, f"downloaded-model/{item.name}")


def test_bundle_becomes_canonical_state_and_keeps_metadata(tmp_path):
    archive = tmp_path / "model.zip"
    _bundle(archive)

    canonical, root = prepare_model_artifact(str(archive), str(tmp_path / "work"))

    assert not archive.exists()
    assert root is not None
    assert Path(root, "config.json").is_file()
    assert Path(root, "tokenizer.json").is_file()
    assert not Path(root, "pytorch_model.bin").exists()
    state = load_file(canonical)
    assert torch.equal(state["first"], torch.arange(4, dtype=torch.float32))
    assert torch.equal(state["second"], state["first"])


def test_bundle_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside", "bad")

    with pytest.raises(ValueError, match="unsafe path"):
        prepare_model_artifact(str(archive), str(tmp_path / "work"))


def test_materialization_uses_transformers_checkpoint_conversion(tmp_path, monkeypatch):
    class ConvertedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            assert source is None
            assert kwargs["local_files_only"] is True
            assert kwargs["trust_remote_code"] is False
            assert set(kwargs["state_dict"]) == {"legacy.weight"}
            model = cls()
            model.weight.data.copy_(kwargs["state_dict"]["legacy.weight"])
            return model, {"missing_keys": [], "unexpected_keys": []}

    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(architectures=["ConvertedModel"]),
    )
    monkeypatch.setattr(transformers, "ConvertedModel", ConvertedModel, raising=False)

    model, missing, unexpected = materialize_hf_model(str(tmp_path), {"legacy.weight": torch.ones(1)})

    assert model.weight.item() == 1
    assert missing == []
    assert unexpected == []


def test_materialization_rejects_mostly_unmatched_checkpoint(tmp_path, monkeypatch):
    class IncompatibleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for index in range(10):
                self.register_parameter(f"weight_{index}", torch.nn.Parameter(torch.zeros(1)))

        @classmethod
        def from_pretrained(cls, _source, **_kwargs):
            model = cls()
            return model, {
                "missing_keys": [f"weight_{index}" for index in range(9)],
                "unexpected_keys": ["legacy.weight"],
            }

    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(architectures=["IncompatibleModel"]),
    )
    monkeypatch.setattr(transformers, "IncompatibleModel", IncompatibleModel, raising=False)

    with pytest.raises(ValueError, match="only 1 of 10 tensors matched"):
        materialize_hf_model(str(tmp_path), {"legacy.weight": torch.ones(1)})
