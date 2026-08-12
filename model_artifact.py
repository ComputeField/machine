# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""Safe preparation and automatic materialization of Hugging Face bundles."""

import gc
import json
import os
import shutil
import zipfile
from pathlib import Path

import torch


def prepare_model_artifact(path: str, work_dir: str) -> tuple[str, str | None]:
    """Return a canonical state file and optional local Hugging Face config root."""
    if not path.endswith(".zip"):
        return path, None

    bundle_dir = os.path.join(work_dir, "hf-bundle")
    _extract_zip(path, bundle_dir)
    os.unlink(path)  # avoid retaining archive + extracted weights + canonical state
    root = _find_hf_root(bundle_dir)
    state = _load_bundle_state(root)
    canonical = os.path.join(work_dir, "model.safetensors")
    from safetensors.torch import save_file

    save_file(
        _safetensors_state(state),
        canonical,
    )
    del state
    gc.collect()
    _remove_bundle_weights(root)
    return canonical, root


def materialize_hf_model(root: str, state: dict) -> tuple[torch.nn.Module, list[str], list[str]]:
    """Build a built-in architecture and apply Transformers' key conversions."""
    import transformers
    from transformers import AutoConfig

    # `root` is an already extracted local bundle; network and remote code are
    # both explicitly disabled, so revision pinning does not apply here.
    config = AutoConfig.from_pretrained(  # nosec B615
        root,
        local_files_only=True,
        trust_remote_code=False,
    )
    architectures = list(getattr(config, "architectures", None) or [])
    if not architectures:
        raise ValueError("Hugging Face config.json does not declare an architecture")
    model_class = getattr(transformers, architectures[0], None)
    if model_class is None or not isinstance(model_class, type) or not issubclass(model_class, torch.nn.Module):
        raise ValueError(
            f"Architecture {architectures[0]!r} requires unsupported custom code; "
            "only built-in Transformers architectures are accepted"
        )
    model, loading = model_class.from_pretrained(
        None,
        config=config,
        state_dict=_validate_state(state),
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    missing = list(loading.get("missing_keys", ()))
    unexpected = list(loading.get("unexpected_keys", ()))
    expected = len(model.state_dict())
    matched = expected - len(missing)
    if expected and matched / expected < 0.9:
        raise ValueError(
            f"Model checkpoint is incompatible with {architectures[0]}: only {matched} of {expected} tensors matched"
        )
    return model, missing, unexpected


def _extract_zip(archive: str, destination: str) -> None:
    os.makedirs(destination, mode=0o700, exist_ok=True)
    root = Path(destination).resolve()
    free = shutil.disk_usage(destination).free
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > 20_000:
            raise ValueError("Model bundle contains too many files")
        declared = sum(item.file_size for item in members)
        if declared > int(free * 0.8):
            raise ValueError("Model bundle is larger than the available safe extraction space")
        written = 0
        for item in members:
            mode = item.external_attr >> 16
            if mode & 0o170000 == 0o120000:
                raise ValueError("Model bundle may not contain symbolic links")
            target = (root / item.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError("Model bundle contains an unsafe path")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, target.open("wb") as sink:
                while chunk := source.read(4 * 1024 * 1024):
                    written += len(chunk)
                    if written > declared or written > int(free * 0.8):
                        raise ValueError("Model bundle exceeded its safe extraction budget")
                    sink.write(chunk)


def _find_hf_root(directory: str) -> str:
    configs = sorted(Path(directory).rglob("config.json"), key=lambda path: len(path.parts))
    if not configs:
        raise ValueError("Hugging Face bundle requires config.json")
    root = configs[0].parent
    _bundle_layout(root)
    return str(root)


def _bundle_layout(root: str | Path) -> tuple[str, Path, dict | None]:
    root = Path(root)
    for kind, index_name in (
        ("safetensors", "model.safetensors.index.json"),
        ("bin", "pytorch_model.bin.index.json"),
    ):
        index_path = root / index_name
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
                raise ValueError(f"Invalid Hugging Face weight index: {index_name}")
            return kind, index_path, index
    for kind, name in (("safetensors", "model.safetensors"), ("bin", "pytorch_model.bin")):
        path = root / name
        if path.is_file():
            return kind, path, None
    raise ValueError("Hugging Face bundle contains no supported model weights")


def _load_bundle_state(root: str) -> dict:
    kind, path, index = _bundle_layout(root)
    if index is None:
        return _load_state(path, kind)
    state: dict = {}
    root_path = Path(root).resolve()
    for filename in dict.fromkeys(index["weight_map"].values()):
        shard = (root_path / filename).resolve()
        if shard.parent != root_path or not shard.is_file():
            raise ValueError("Hugging Face weight index references an unsafe or missing shard")
        state.update(_load_state(shard, kind))
    return state


def _load_state(path: Path, kind: str) -> dict:
    if kind == "safetensors":
        from safetensors.torch import load_file

        loaded = load_file(str(path), device="cpu")
        return _validate_state(loaded)
    loaded = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError("Hugging Face weight file must contain a state dict")
    return _validate_state(loaded.get("state_dict", loaded))


def _validate_state(state) -> dict:
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items())
    ):
        raise ValueError("Hugging Face weight file must contain only named tensors")
    return state


def _safetensors_state(state: dict) -> dict:
    """Make shared-storage tensors safe for the canonical safetensors file."""
    prepared: dict = {}
    storages: set[int] = set()
    for key, value in _validate_state(state).items():
        tensor = value.detach().cpu()
        storage = tensor.untyped_storage().data_ptr()
        if storage in storages:
            tensor = tensor.clone()
        elif not tensor.is_contiguous():
            tensor = tensor.contiguous()
        storages.add(tensor.untyped_storage().data_ptr())
        prepared[key] = tensor
    return prepared


def _remove_bundle_weights(root: str) -> None:
    kind, path, index = _bundle_layout(root)
    paths = [path]
    if index:
        paths.extend(Path(root) / name for name in set(index["weight_map"].values()))
    for target in paths:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
