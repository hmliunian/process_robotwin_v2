"""Compare SAM implementations on exactly ten frozen RGB frames, without API calls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.sam3_adapter import Sam3Adapter


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def agreement(reference: NDArray[np.bool_], actual: NDArray[np.bool_]) -> dict[str, Any]:
    if reference.shape != actual.shape or reference.dtype != bool or actual.dtype != bool:
        raise ValueError("comparison requires equally shaped boolean masks")
    axes = tuple(range(1, reference.ndim))
    union = (reference | actual).sum(axis=axes)
    intersection = (reference & actual).sum(axis=axes)
    iou = np.divide(intersection, union, out=np.ones(len(union)), where=union > 0)
    return {
        "frame_iou": iou.tolist(),
        "mean_iou": float(iou.mean()),
        "min_iou": float(iou.min()),
        "nonempty_mean_iou": float(iou[union > 0].mean()) if np.any(union) else None,
        "dual_empty_frames": int((union == 0).sum()),
        "presence_changes": int((reference.any(axis=axes) != actual.any(axis=axes)).sum()),
        "changed_pixels": int((reference != actual).sum()),
    }


def timed(call: Callable[[], Any], gpu: int) -> tuple[Any, float]:
    import torch

    torch.cuda.synchronize(gpu)
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize(gpu)
    return result, time.perf_counter() - started


def monitor_gpu(gpu: int, stop: threading.Event, output: Path) -> None:
    with output.open("w") as handle:
        while not stop.is_set():
            sample = subprocess.run(
                ["nvidia-smi", "pmon", "-i", str(gpu), "-s", "um", "-c", "1"],
                capture_output=True,
                text=True,
                check=False,
            )
            handle.write(datetime.now(UTC).isoformat() + "\n" + sample.stdout)
            handle.flush()
            stop.wait(3)


def review(frames: Path, baseline: NDArray[np.bool_], fast: NDArray[np.bool_], path: Path) -> None:
    width, height = 240, 180
    sheet = Image.new("RGB", (10 * width, 4 * (height + 24)), "#101823")
    draw = ImageDraw.Draw(sheet)
    for frame in range(10):
        original = Image.open(frames / f"{frame:06d}.jpg").convert("RGB")
        for row, masks in enumerate((None, baseline, fast, baseline != fast)):
            rgb = np.asarray(original).copy()
            if masks is not None:
                for role, color in enumerate(((60, 230, 90), (65, 145, 255))):
                    mask = masks[role, frame]
                    rgb[mask] = (0.45 * rgb[mask] + 0.55 * np.asarray(color)).astype(np.uint8)
            image = Image.fromarray(rgb).resize((width, height))
            x, y = frame * width, row * (height + 24)
            sheet.paste(image, (x, y + 24))
            label = ("RGB", "baseline", "fast", "difference")[row]
            draw.text((x + 4, y + 4), f"f{frame}: {label}", fill="white")
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--seed-artifacts", type=Path, required=True)
    parser.add_argument("--baseline-adapter", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three paired repetitions are required")
    args.output.mkdir(parents=True, exist_ok=False)
    frames = args.output / "frames"
    frames.mkdir()
    for frame in range(10):
        shutil.copyfile(args.frames / f"{frame:06d}.jpg", frames / f"{frame:06d}.jpg")
    run = json.loads((args.seed_artifacts / "run_manifest.json").read_text())
    semantic = json.loads((args.seed_artifacts / "semantic_plan.json").read_text())["roles"]
    roles = [role for role in run["roles"] if role["role"] in run["required_object_roles"]]
    shape = tuple(reversed(Image.open(frames / "000000.jpg").size))
    seeds = []
    requests = []
    for role in roles:
        if role["seed_frame_id"] != 0 or role["qc_status"] != "passed":
            raise ValueError("ten-frame fixture requires historical approved seeds on frame zero")
        seed = np.asarray(Image.open(args.seed_artifacts / role["seed_mask_path"])) > 0
        if seed.shape != shape or not seed.any():
            raise ValueError("invalid historical seed")
        seeds.append(seed)
        ys, xs = np.nonzero(seed)
        box = (
            xs.min() / shape[1],
            ys.min() / shape[0],
            (xs.max() + 1) / shape[1],
            (ys.max() + 1) / shape[0],
        )
        plan = semantic[role["role"]]
        queries = [plan[key] for key in plan["recommended_order"] if plan.get(key)]
        requests.append({"role": role["role"], "queries": queries, "box": list(map(float, box))})
    spec = importlib.util.spec_from_file_location("frozen_sam_baseline", args.baseline_adapter)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import frozen baseline")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import torch

    torch.set_num_threads(8)
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    metadata = {
        "started_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "scope": "SAM seed and fixed-mask propagation; Qwen/save/render/model load excluded",
        "frame_count": 10,
        "source_frame_ids": list(range(10)),
        "shape": shape,
        "episode": run["episode"],
        "roles": [role["role"] for role in roles],
        "gpu": args.gpu,
        "gpu_name": torch.cuda.get_device_name(args.gpu),
        "cpu_threads": 8,
        "torch": torch.__version__,
        "requests": requests,
        "candidate_frame_ids": [0, 5, 9],
        "bbox_origin": "historical approved seed extent, controlled prompt test",
        "frame_sha256": [sha256(frames / f"{i:06d}.jpg") for i in range(10)],
        "checkpoint_sha256": sha256(args.checkpoint),
        "baseline_sha256": sha256(args.baseline_adapter),
        "source_sha256": {
            str(path): sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1]
                / "src/robotwin_annotation_v2/adapters/sam3_adapter.py",
                Path(__file__).resolve().parents[1]
                / "src/robotwin_annotation_v2/adapters/sam3_tracking_features.py",
            )
        },
    }
    ArtifactStore.write_json(args.output / "metadata.json", metadata)
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(args.gpu, stop, args.output / "gpu.log"))
    monitor.start()
    baseline = None
    records: list[dict[str, Any]] = []
    saved: dict[str, NDArray[np.bool_]] = {}
    try:
        baseline, load_s = timed(
            lambda: module.Sam3Adapter(checkpoint_path=args.checkpoint, gpus=(args.gpu,)), args.gpu
        )
        metadata["model_load_s"] = load_s
        adapters = {"baseline": baseline}
        for name, batch in (("fast_t1", 1), ("fast_t4", 4)):
            adapters[name] = Sam3Adapter(
                checkpoint_path=args.checkpoint,
                gpus=(args.gpu,),
                predictor=baseline.predictor,
                seed_image_only=True,
                tracking_batch_size=batch,
            )

        def seed_calls(adapter: Any) -> NDArray[np.bool_]:
            masks = []
            for request in requests:
                for frame in (0, 5, 9):
                    common = {"frame_id": frame, "frame_count": 10, "frame_shape": shape}
                    masks.extend(
                        adapter.text_query_masks(frames, request["queries"], **common).values()
                    )
                    masks.append(adapter.box_mask(frames, request["box"], **common))
                    masks.append(
                        adapter.text_box_mask(
                            frames, request["queries"][0], request["box"], **common
                        )
                    )
            return np.stack(masks)

        def propagate(adapter: Any) -> NDArray[np.bool_]:
            return np.stack(
                [
                    adapter.propagate_mask(
                        frames,
                        seed,
                        seed_frame=0,
                        frame_count=10,
                        frame_shape=shape,
                        tracking_window=(0, 9),
                    )
                    for seed in seeds
                ]
            )

        warmup = {}
        for name, adapter in adapters.items():
            _, seed_warmup = timed(partial(seed_calls, adapter), args.gpu)
            _, propagation_warmup = timed(partial(propagate, adapter), args.gpu)
            warmup[name] = seed_warmup + propagation_warmup
        metadata["warmup_s"] = warmup
        names = list(adapters)
        for repetition in range(args.repeats):
            outputs = {}
            for name in names[repetition % 3 :] + names[: repetition % 3]:
                candidate, seed_s = timed(partial(seed_calls, adapters[name]), args.gpu)
                masks, propagation_s = timed(partial(propagate, adapters[name]), args.gpu)
                outputs[name] = (candidate, masks)
                records.append(
                    {
                        "variant": name,
                        "repeat": repetition,
                        "seed_s": seed_s,
                        "propagation_s": propagation_s,
                        "sam_total_s": seed_s + propagation_s,
                    }
                )
                print(json.dumps(records[-1]), flush=True)
            for name in names:
                row = next(
                    row for row in records if row["variant"] == name and row["repeat"] == repetition
                )
                row["seed_agreement"] = agreement(outputs["baseline"][0], outputs[name][0])
                row["role_agreement"] = {
                    role["role"]: agreement(outputs["baseline"][1][i], outputs[name][1][i])
                    for i, role in enumerate(roles)
                }
                if repetition == 0:
                    saved[name] = outputs[name][1]
                    np.savez_compressed(
                        args.output / f"{name}.npz",
                        candidates=outputs[name][0],
                        masks=outputs[name][1],
                    )
            ArtifactStore.write_json(args.output / "records.json", {"records": records})
        summary: dict[str, Any] = {}
        for name in names:
            rows = [row for row in records if row["variant"] == name]
            summary[name] = {
                key: {
                    "mean_s": statistics.mean(row[key] for row in rows),
                    "stddev_s": statistics.pstdev(row[key] for row in rows),
                }
                for key in ("seed_s", "propagation_s", "sam_total_s")
            }
            summary[name]["role_agreement"] = rows[0]["role_agreement"]
            summary[name]["all_repetitions_exact"] = all(
                row["seed_agreement"]["changed_pixels"] == 0
                and all(value["changed_pixels"] == 0 for value in row["role_agreement"].values())
                for row in rows
            )
        for name in names:
            summary[name]["speedup"] = (
                summary["baseline"]["sam_total_s"]["mean_s"]
                / summary[name]["sam_total_s"]["mean_s"]
            )
        ArtifactStore.write_json(args.output / "summary.json", summary)
        review(frames, saved["baseline"], saved["fast_t1"], args.output / "review_t1.jpg")
        review(frames, saved["baseline"], saved["fast_t4"], args.output / "review_t4.jpg")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        if baseline is not None:
            baseline.shutdown()
        stop.set()
        monitor.join()
        metadata["ended_utc"] = datetime.now(UTC).isoformat()
        ArtifactStore.write_json(args.output / "metadata.json", metadata)


if __name__ == "__main__":
    main()
