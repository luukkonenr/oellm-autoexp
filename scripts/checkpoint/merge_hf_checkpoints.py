#!/usr/bin/env python3
"""Weighted average of Hugging Face safetensors checkpoints, on CPU.

Each shared floating-point tensor is combined element by element:

    output = w1 * checkpoint_1 + w2 * checkpoint_2 + ...

The weights must sum to 1, or pass --normalize. Tensors are accumulated in
float32 and stored in the dtype of the highest-weight checkpoint. A tensor
that is missing from some inputs is copied from the highest-weight input that
has it. Tokenizer and config files are copied from that same checkpoint.

This reads and writes Hugging Face exports. It does not read Megatron
distributed checkpoints.

Examples
--------
python scripts/checkpoint/merge_hf_checkpoints.py \\
    --inputs /path/iter_0047684:0.75 /path/iter_0004768:0.25 \\
    --output /path/merge_75late_25early
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import torch
from safetensors import safe_open
from safetensors.torch import save_file


class MergeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    weight: float
    weight_map: dict[str, Path]


def _parse_size(text: str) -> int:
    value = text.strip().lower().replace(" ", "")
    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }
    for suffix, scale in sorted(units.items(), key=lambda item: -len(item[0])):
        if value.endswith(suffix) and value[: -len(suffix)] != "":
            return int(float(value[: -len(suffix)]) * scale)
    return int(value)


def _load_weight_map(model_dir: Path) -> dict[str, Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text())
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise MergeError(f"{index_path} has no weight_map")
        resolved = {}
        for key, shard_name in weight_map.items():
            shard_path = model_dir / shard_name
            if not shard_path.is_file():
                raise MergeError(f"{index_path} names missing shard {shard_name}")
            resolved[key] = shard_path
        return resolved

    single = model_dir / "model.safetensors"
    candidates = [single] if single.is_file() else sorted(model_dir.glob("*.safetensors"))
    if not candidates:
        raise MergeError(f"{model_dir} has no safetensors checkpoint")
    resolved = {}
    for shard_path in candidates:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in resolved:
                    raise MergeError(f"duplicate tensor {key} in {model_dir}")
                resolved[key] = shard_path
    if not resolved:
        raise MergeError(f"{model_dir} safetensors files contain no tensors")
    return resolved


def _parse_inputs(entries: list[str]) -> list[tuple[Path, float]]:
    parsed = []
    for entry in entries:
        path_text, separator, weight_text = entry.rpartition(":")
        if not separator or path_text == "":
            raise MergeError(f"expected PATH:WEIGHT, got {entry!r}")
        path = Path(path_text).expanduser().resolve()
        if not path.is_dir():
            raise MergeError(f"checkpoint directory does not exist: {path}")
        try:
            weight = float(weight_text)
        except ValueError as exc:
            raise MergeError(f"weight for {path} is not a number: {weight_text}") from exc
        parsed.append((path, weight))
    if len(parsed) < 2:
        raise MergeError("at least two checkpoints are required")
    if len({path for path, _ in parsed}) != len(parsed):
        raise MergeError("the same checkpoint was passed more than once")
    return parsed


def _prepare_weights(parsed: list[tuple[Path, float]], normalize: bool) -> list[Checkpoint]:
    total = sum(weight for _, weight in parsed)
    if normalize:
        if total == 0:
            raise MergeError("cannot normalize weights that sum to 0")
        parsed = [(path, weight / total) for path, weight in parsed]
    elif abs(total - 1.0) > 1e-4:
        raise MergeError(
            f"weights sum to {total:.8g}, not 1. Pass --normalize to rescale them."
        )
    checkpoints = []
    for path, weight in parsed:
        checkpoints.append(Checkpoint(path, weight, _load_weight_map(path)))
    return checkpoints


class _ShardReader:
    def __init__(self) -> None:
        self._handles = {}

    def get(self, shard_path: Path, key: str) -> torch.Tensor:
        handle = self._handles.get(shard_path)
        if handle is None:
            handle = safe_open(shard_path, framework="pt", device="cpu")
            self._handles[shard_path] = handle
        tensor = handle.get_tensor(key)
        if tensor.device.type != "cpu":
            raise MergeError(f"{key} was loaded on {tensor.device}, not CPU")
        return tensor

    def close(self) -> None:
        self._handles.clear()


def _primary(checkpoints: list[Checkpoint]) -> Checkpoint:
    return max(checkpoints, key=lambda item: (item.weight, str(item.path)))


def _merge_tensor(
    key: str,
    checkpoints: list[Checkpoint],
    reader: _ShardReader,
) -> torch.Tensor:
    present = [item for item in checkpoints if key in item.weight_map and item.weight != 0]
    if not present:
        present = [item for item in checkpoints if key in item.weight_map]
    source = max(present, key=lambda item: (item.weight, str(item.path)))
    reference = reader.get(source.weight_map[key], key)
    weighted = [item for item in checkpoints if key in item.weight_map and item.weight != 0]
    missing = [item for item in checkpoints if key not in item.weight_map and item.weight != 0]
    if not reference.is_floating_point() or len(weighted) <= 1 or missing:
        return reference

    shapes = {}
    for item in weighted:
        tensor = reader.get(item.weight_map[key], key)
        shapes[item.path] = tuple(tensor.shape)
    if len(set(shapes.values())) != 1:
        detail = ", ".join(f"{path}: {shape}" for path, shape in shapes.items())
        raise MergeError(f"shape mismatch for {key}: {detail}")

    merged = torch.zeros(reference.shape, dtype=torch.float32, device="cpu")
    for item in weighted:
        tensor = reader.get(item.weight_map[key], key)
        merged.add_(tensor.float(), alpha=item.weight)
    if not torch.isfinite(merged).all():
        raise MergeError(f"non-finite values in merged tensor {key}")
    return merged.to(dtype=reference.dtype)


def _iter_keys(checkpoints: list[Checkpoint]) -> list[str]:
    keys = set()
    for checkpoint in checkpoints:
        keys.update(checkpoint.weight_map)
    return sorted(keys)


def _copy_sidecars(source: Path, output: Path) -> list[str]:
    copied = []
    for path in sorted(source.iterdir()):
        if path.suffix == ".safetensors" or path.name == "model.safetensors.index.json":
            continue
        target = output / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        elif path.is_file():
            shutil.copy2(path, target)
        else:
            continue
        copied.append(path.name)
    return copied


def _shard_name(index: int, count: int) -> str:
    width = max(5, len(str(count)))
    return f"model.safetensors-{index:0{width}d}-of-{count:0{width}d}.safetensors"


def merge_checkpoints(
    entries: list[str],
    output: Path,
    *,
    normalize: bool = False,
    max_shard_size: int = 5 * 1000**3,
) -> dict:
    if max_shard_size <= 0:
        raise MergeError("--max-shard-size must be positive")
    output = output.expanduser().resolve()
    if output.exists():
        raise MergeError(f"output already exists: {output}")

    checkpoints = _prepare_weights(_parse_inputs(entries), normalize)
    for checkpoint in checkpoints:
        if output == checkpoint.path or output.is_relative_to(checkpoint.path):
            raise MergeError("output must not be inside an input checkpoint")

    reader = _ShardReader()
    output.mkdir(parents=True)
    try:
        weight_map: dict[str, str] = {}
        total_size = 0
        shard_tensors: dict[str, torch.Tensor] = {}
        shard_bytes = 0
        temporary_shards: list[Path] = []
        copied_keys = []
        averaged_keys = 0

        def flush() -> None:
            nonlocal shard_tensors, shard_bytes
            if not shard_tensors:
                return
            temporary = output / f".shard-{len(temporary_shards) + 1:05d}.safetensors"
            save_file({key: tensor.cpu() for key, tensor in shard_tensors.items()}, temporary)
            temporary_shards.append(temporary)
            shard_tensors = {}
            shard_bytes = 0

        active = [item for item in checkpoints if item.weight != 0]
        for number, key in enumerate(_iter_keys(checkpoints), start=1):
            holders = [item for item in active if key in item.weight_map]
            tensor = _merge_tensor(key, checkpoints, reader)
            shared_float = (
                tensor.is_floating_point()
                and len(holders) == len(active)
                and len(holders) > 1
            )
            if shared_float:
                averaged_keys += 1
            else:
                copied_keys.append(key)
            tensor_bytes = tensor.nbytes
            if shard_tensors and shard_bytes + tensor_bytes > max_shard_size:
                flush()
            shard_tensors[key] = tensor
            shard_bytes += tensor_bytes
            total_size += tensor_bytes
            if number % 100 == 0:
                print(f"merged {number} tensors", file=sys.stderr)
        flush()

        count = len(temporary_shards)
        if count == 0:
            raise MergeError("no tensors were written")
        final_names = []
        for index, temporary in enumerate(temporary_shards, start=1):
            final_name = _shard_name(index, count) if count > 1 else "model.safetensors"
            temporary.rename(output / final_name)
            final_names.append(final_name)

        for final_name in final_names:
            with safe_open(output / final_name, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    weight_map[key] = final_name
        if count > 1:
            index_payload = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
            (output / "model.safetensors.index.json").write_text(json.dumps(index_payload, indent=2) + "\n")

        primary = _primary(checkpoints)
        sidecars = _copy_sidecars(primary.path, output)
        manifest = {
            "inputs": [
                {"path": str(item.path), "weight": item.weight} for item in checkpoints
            ],
            "metadata_from": str(primary.path),
            "averaged_tensors": averaged_keys,
            "copied_tensors": copied_keys,
            "sidecars": sidecars,
            "total_size": total_size,
            "device": "cpu",
        }
        (output / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"wrote {len(weight_map)} tensors to {output} "
            f"({averaged_keys} averaged, {len(copied_keys)} copied)",
            file=sys.stderr,
        )
        return manifest
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    finally:
        reader.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="PATH:WEIGHT",
        help="Checkpoint directory and its merge weight. Weights must sum to 1 unless --normalize is set.",
    )
    parser.add_argument("--output", type=Path, required=True, help="New Hugging Face checkpoint directory.")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Divide the given weights by their sum.",
    )
    parser.add_argument(
        "--max-shard-size",
        type=_parse_size,
        default=5 * 1000**3,
        help="Maximum safetensors shard size. Default: 5GB.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        merge_checkpoints(
            args.inputs,
            args.output,
            normalize=args.normalize,
            max_shard_size=args.max_shard_size,
        )
    except MergeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
