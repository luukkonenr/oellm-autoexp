#!/usr/bin/env python3
"""CPU checks for the Hugging Face weighted checkpoint merge."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from scripts.checkpoint.merge_hf_checkpoints import MergeError, merge_checkpoints


def _write_checkpoint(root: Path, tensors: dict[str, torch.Tensor], marker: str) -> None:
    root.mkdir()
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps({"marker": marker}))


class MergeHfCheckpointsTest(unittest.TestCase):
    def test_weighted_average_and_copied_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            late = root / "late"
            early = root / "early"
            _write_checkpoint(
                late,
                {
                    "shared": torch.tensor([1.0, 3.0]),
                    "late_only": torch.tensor([4.0]),
                },
                "late",
            )
            _write_checkpoint(
                early,
                {"shared": torch.tensor([5.0, 7.0])},
                "early",
            )
            output = root / "merged"
            manifest = merge_checkpoints(
                [f"{late}:0.75", f"{early}:0.25"],
                output,
            )

            merged = load_file(output / "model.safetensors", device="cpu")
            self.assertTrue(torch.equal(merged["shared"], torch.tensor([2.0, 4.0])))
            self.assertTrue(torch.equal(merged["late_only"], torch.tensor([4.0])))
            self.assertEqual(merged["shared"].device.type, "cpu")
            self.assertEqual(json.loads((output / "config.json").read_text())["marker"], "late")
            self.assertEqual(manifest["averaged_tensors"], 1)
            self.assertEqual(manifest["copied_tensors"], ["late_only"])

    def test_normalize_and_reject_unnormalized_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left"
            right = root / "right"
            _write_checkpoint(left, {"w": torch.tensor([2.0])}, "left")
            _write_checkpoint(right, {"w": torch.tensor([8.0])}, "right")
            with self.assertRaises(MergeError):
                merge_checkpoints([f"{left}:3", f"{right}:1"], root / "bad")
            self.assertFalse((root / "bad").exists())

            output = root / "normalized"
            merge_checkpoints([f"{left}:3", f"{right}:1"], output, normalize=True)
            merged = load_file(output / "model.safetensors", device="cpu")
            self.assertTrue(torch.equal(merged["w"], torch.tensor([3.5])))

    def test_shape_mismatch_removes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left"
            right = root / "right"
            _write_checkpoint(left, {"w": torch.tensor([1.0, 2.0])}, "left")
            _write_checkpoint(right, {"w": torch.tensor([1.0])}, "right")
            with self.assertRaises(MergeError):
                merge_checkpoints([f"{left}:0.5", f"{right}:0.5"], root / "bad")
            self.assertFalse((root / "bad").exists())

    def test_shards_stay_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left"
            right = root / "right"
            _write_checkpoint(
                left,
                {
                    "a": torch.ones(8, dtype=torch.bfloat16),
                    "b": torch.ones(8, dtype=torch.bfloat16),
                },
                "left",
            )
            _write_checkpoint(
                right,
                {
                    "a": torch.full((8,), 3, dtype=torch.bfloat16),
                    "b": torch.full((8,), 3, dtype=torch.bfloat16),
                },
                "right",
            )
            output = root / "merged"
            merge_checkpoints(
                [f"{left}:0.5", f"{right}:0.5"],
                output,
                max_shard_size=16,
            )
            shards = sorted(output.glob("model.safetensors-*"))
            self.assertEqual(len(shards), 2)
            index = json.loads((output / "model.safetensors.index.json").read_text())
            self.assertEqual(set(index["weight_map"]), {"a", "b"})
            for shard in shards:
                tensors = load_file(shard, device="cpu")
                for tensor in tensors.values():
                    self.assertEqual(tensor.device.type, "cpu")
                    self.assertEqual(tensor.dtype, torch.bfloat16)
                    self.assertTrue(torch.equal(tensor, torch.full_like(tensor, 2)))


if __name__ == "__main__":
    unittest.main()
