"""Check full-video SAM agreement on frozen approved seeds; no Qwen or human ground truth."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.sam3_adapter import Sam3Adapter
from scripts import benchmark_umi_fast
from scripts.benchmark_umi_fast import agreement, sha256, timed


@dataclass(frozen=True)
class FrozenRole:
    frames: Path
    seed: Path
    shape: tuple[int, int]
    tracking_window: tuple[int, int]


def prepare(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import json

    cases = json.loads(args.manifest.read_text())["cases"]
    if len(cases) != 10 or len({case["task"] for case in cases}) != 10:
        raise ValueError("expected ten fixed, distinct video tasks")
    rows, inputs = [], {}
    for case in cases:
        task, count = case["task"], case["frame_count"]
        if Path(task).name != task or task in {".", ".."} or count < 1:
            raise ValueError("invalid task or frame count")
        episode = (
            args.seed_run / "runs" / task / task / f"episode_{case['episode']:06d}" / "cam_high"
        )
        manifest_path = episode / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["frame_count"] != count
            or manifest["episode"]["episode_index"] != case["episode"]
        ):
            raise ValueError(f"historical episode mismatch: {task}")
        frames = args.frames_root / task
        paths = [frames / f"{index:06d}.jpg" for index in range(count)]
        if sorted(frames.glob("*.jpg")) != paths:
            raise ValueError(f"frames must contain exactly {count} contiguous JPEGs: {task}")
        shapes = set()
        for path in paths:
            with Image.open(path) as frame:
                shapes.add((frame.height, frame.width))
        if len(shapes) != 1:
            raise ValueError(f"inconsistent frame dimensions: {task}")
        shape = shapes.pop()
        inputs[task] = {
            "episode": case["episode"],
            "frame_count": count,
            "shape": shape,
            "run_manifest_sha256": sha256(manifest_path),
            "frame_sha256": {path.name: sha256(path) for path in paths},
        }
        roles = {role["role"]: role for role in manifest["roles"]}
        for name in manifest["required_object_roles"]:
            role = roles[name]
            row = {
                "task": task,
                "episode": case["episode"],
                "role": name,
                "frame_count": count,
                "historical_status": role["status"],
                "seed_frame": role["seed_frame_id"],
                "output_window": role["output_window"],
                "status": "pending",
            }
            rows.append(row)
            seed = episode / role["seed_mask_path"] if role.get("seed_mask_path") else None
            if role["qc_status"] != "passed" or seed is None or not seed.is_file():
                row.update(status="skipped", reason=f"no approved seed: QC={role['qc_status']}")
                continue
            with Image.open(seed) as image:
                mask = np.asarray(image) > 0
            start, end = role["output_window"]
            seed_frame = role["seed_frame_id"]
            if mask.shape != shape or not mask.any() or not 0 <= start <= end < count:
                raise ValueError(f"invalid frozen seed/window: {task}/{name}")
            if not isinstance(seed_frame, int) or not start <= seed_frame <= end:
                raise ValueError(f"invalid frozen seed frame: {task}/{name}")
            row["seed_sha256"] = sha256(seed)
            row["fixture"] = FrozenRole(frames, seed, shape, (start, end))
    return rows, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "seed-run", "frames-root", "baseline-adapter", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    rows, inputs = prepare(args)
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1] / "src/robotwin_annotation_v2/adapters"
    metadata = {
        "started_utc": datetime.now(UTC).isoformat(),
        "gpu": args.gpu,
        "cpu_threads": 8,
        "tf32": {"cuda_matmul": True, "cudnn": True},
        "scope": "One paired propagation per approved frozen seed; no human ground truth. "
        "One run and rotated order do not establish steady-state speedup.",
        "inputs": inputs,
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "manifest_sha256": sha256(args.manifest),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_sha256": {
            str(path): sha256(path)
            for path in (
                Path(__file__),
                Path(benchmark_umi_fast.__file__),
                args.baseline_adapter,
                source / "sam3_adapter.py",
                source / "sam3_tracking_features.py",
            )
        },
    }
    ArtifactStore.write_json(args.output / "metadata.json", metadata)

    def persist() -> None:
        completed = [row for row in rows if row["status"] == "completed"]
        summary = {
            "unique_rgb_frames": sum(item["frame_count"] for item in inputs.values()),
            "required_roles": len(rows),
            "compared_roles": len(completed),
            "skipped_roles": sum(row["status"] == "skipped" for row in rows),
            "compared_role_frames": sum(row["frame_count"] for row in completed),
            "all_compared_exact": bool(completed)
            and all(row["agreement"]["changed_pixels"] == 0 for row in completed),
            "changed_pixels": sum(row["agreement"]["changed_pixels"] for row in completed),
            "dual_empty_frames": sum(row["agreement"]["dual_empty_frames"] for row in completed),
            "mean_iou": float(
                np.mean([value for row in completed for value in row["agreement"]["frame_iou"]])
            )
            if completed
            else None,
            "records": [
                {key: value for key, value in row.items() if key != "fixture"} for row in rows
            ],
        }
        ArtifactStore.write_json(args.output / "summary.json", summary)

    persist()
    baseline = None
    try:
        import torch

        torch.set_num_threads(8)
        torch.cuda.set_device(args.gpu)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        metadata.update(torch=torch.__version__, gpu_name=torch.cuda.get_device_name(args.gpu))
        spec = importlib.util.spec_from_file_location(
            "frozen_sam_full_baseline", args.baseline_adapter
        )
        if spec is None or spec.loader is None:
            raise ValueError("cannot import frozen baseline")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        baseline, metadata["model_load_s"] = timed(
            lambda: module.Sam3Adapter(checkpoint_path=args.checkpoint, gpus=(args.gpu,)), args.gpu
        )
        adapters = {
            "baseline": baseline,
            "fast": Sam3Adapter(
                checkpoint_path=args.checkpoint,
                gpus=(args.gpu,),
                predictor=baseline.predictor,
                seed_image_only=True,
                tracking_batch_size=1,
            ),
        }
        for index, row in enumerate(rows):
            if row["status"] == "skipped":
                continue
            fixture = row["fixture"]
            with Image.open(fixture.seed) as image:
                seed = np.asarray(image) > 0
            directory = args.output / row["task"] / row["role"]
            order = ("baseline", "fast") if index % 2 == 0 else ("fast", "baseline")
            row["execution_order"] = order
            outputs = {}
            for name in order:
                masks, elapsed = timed(
                    partial(
                        adapters[name].propagate_mask,
                        fixture.frames,
                        seed,
                        seed_frame=row["seed_frame"],
                        frame_count=row["frame_count"],
                        frame_shape=fixture.shape,
                        tracking_window=fixture.tracking_window,
                    ),
                    args.gpu,
                )
                if masks.shape != (row["frame_count"], *fixture.shape) or masks.dtype != bool:
                    raise ValueError(
                        f"invalid full-video track: {row['task']}/{row['role']}/{name}"
                    )
                ArtifactStore.write_npz(directory / f"{name}.npz", masks=masks)
                row[f"{name}_propagation_s"] = elapsed
                outputs[name] = masks
            row["agreement"] = agreement(outputs["baseline"], outputs["fast"])
            changed = (outputs["baseline"] != outputs["fast"]).sum(axis=(1, 2))
            baseline_pixels = outputs["baseline"].sum(axis=(1, 2))
            fast_pixels = outputs["fast"].sum(axis=(1, 2))
            with (directory / "per_frame.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ("frame_id", "iou", "baseline_pixels", "fast_pixels", "changed_pixels")
                )
                writer.writerows(
                    zip(
                        range(row["frame_count"]),
                        row["agreement"]["frame_iou"],
                        baseline_pixels,
                        fast_pixels,
                        changed,
                        strict=True,
                    )
                )
            row["status"] = "completed"
            persist()
            print(
                f"{row['task']}/{row['role']}: IoU={row['agreement']['mean_iou']:.8f}", flush=True
            )
    except BaseException as exc:
        for row in rows:
            if row["status"] == "pending":
                row.update(status="not_completed", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if baseline is not None:
            baseline.shutdown()
        metadata["ended_utc"] = datetime.now(UTC).isoformat()
        ArtifactStore.write_json(args.output / "metadata.json", metadata)
        persist()


if __name__ == "__main__":
    main()
