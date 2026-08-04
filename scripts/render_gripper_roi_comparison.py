#!/usr/bin/env python3
"""Render sharp per-episode contact-sheet comparisons for gripper ROI runs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


CANVAS_WIDTH = 1280
GLOBAL_HEADER_HEIGHT = 76
RUN_HEADER_HEIGHT = 108
CONTACT_HEIGHT = 720
SEED_STRIP_HEIGHT = 286
RUN_GAP = 18
BACKGROUND = (18, 21, 26)
PANEL_BACKGROUND = (29, 34, 41)
SUBTLE_BACKGROUND = (38, 44, 53)
TEXT = (242, 245, 248)
MUTED_TEXT = (184, 194, 205)
MISSING_TEXT = (255, 176, 87)
LABEL_COLORS = {
    "A": (65, 154, 255),
    "C": (60, 201, 139),
    "D": (255, 174, 66),
    "S": (191, 122, 255),
}
LEGACY_BACK_M = 0.025
LEGACY_FRONT_M = 0.060


@dataclass(frozen=True)
class RunSpec:
    """One labeled batch output root."""

    label: str
    root: Path
    fallback_roots: tuple[Path, ...] = ()
    is_baseline: bool = False
    batch_roi_policy: dict[str, Any] | None = None

    @property
    def roots(self) -> tuple[Path, ...]:
        return (self.root, *self.fallback_roots)


@dataclass(frozen=True)
class EpisodeVisual:
    """Metadata and source images for one run/episode panel."""

    run: RunSpec
    episode: int
    status: str
    status_detail: str | None
    roi_label: str
    seed_label: str
    seed_flag: str | None
    contact_sheet: Image.Image | None
    seed_panel: Image.Image | None
    episode_dir: Path
    manifest_path: Path
    contact_sheet_path: Path
    seed_panel_path: Path | None
    review_video_path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help="A/baseline batch root",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        required=True,
        metavar="LABEL=PATH",
        help="Variant batch root; repeat for C/D/S or other labels",
    )
    parser.add_argument("--episode-ids", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_variants(values: Sequence[str]) -> dict[str, tuple[Path, ...]]:
    """Parse variants, grouping repeated labels into ordered output shards."""

    grouped: dict[str, list[Path]] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        label = label.strip()
        raw_path = raw_path.strip()
        if not separator or not label or not raw_path:
            raise ValueError(f"variant must use non-empty LABEL=PATH syntax: {value!r}")
        if label.upper() == "A" or label.lower() == "baseline":
            raise ValueError(f"variant label {label!r} is reserved for the baseline")
        grouped.setdefault(label, []).append(Path(raw_path).expanduser().resolve())
    if not grouped:
        raise ValueError("at least one --variant LABEL=PATH is required")
    return {label: tuple(paths) for label, paths in grouped.items()}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON manifest is not an object: {path}")
    return value


def _batch_roi_policy(root: Path) -> dict[str, Any] | None:
    path = root / "batch_manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    value = manifest.get("roi_policy")
    return dict(value) if isinstance(value, dict) else None


def _variant_roots(value: Path | Sequence[Path]) -> tuple[Path, ...]:
    raw_roots = (value,) if isinstance(value, Path) else tuple(value)
    if not raw_roots:
        raise ValueError("each variant must have at least one output root")
    return tuple(Path(root).expanduser().resolve() for root in raw_roots)


def build_run_specs(
    baseline_root: Path,
    variants: Mapping[str, Path | Sequence[Path]],
) -> list[RunSpec]:
    """Create ordered run specifications and cache batch-level ROI policy."""

    baseline = baseline_root.expanduser().resolve()
    runs = [
        RunSpec(
            label="A",
            root=baseline,
            is_baseline=True,
            batch_roi_policy=_batch_roi_policy(baseline),
        )
    ]
    for label, value in variants.items():
        roots = _variant_roots(value)
        policy = next(
            (policy for root in roots if (policy := _batch_roi_policy(root)) is not None),
            None,
        )
        runs.append(
            RunSpec(
                label=label,
                root=roots[0],
                fallback_roots=roots[1:],
                batch_roi_policy=policy,
            )
        )
    return runs


def _geometry_pair(policy: Mapping[str, Any], key: str) -> tuple[float, float] | None:
    role = policy.get(key)
    if not isinstance(role, dict):
        return None
    geometry = role.get("geometry", role)
    if not isinstance(geometry, dict):
        return None
    try:
        return float(geometry["axial_back_m"]), float(geometry["axial_front_m"])
    except (KeyError, TypeError, ValueError):
        return None


def format_roi_label(
    manifest: Mapping[str, Any] | None,
    run: RunSpec,
) -> str:
    """Format prompt/hard axial parameters, including legacy baseline defaults."""

    raw_policy = manifest.get("roi_policy") if manifest is not None else None
    policy = raw_policy if isinstance(raw_policy, dict) else run.batch_roi_policy
    prompt = _geometry_pair(policy, "prompt") if policy is not None else None
    hard = _geometry_pair(policy, "hard") if policy is not None else None
    if prompt is None and hard is None and run.is_baseline:
        prompt = hard = (LEGACY_BACK_M, LEGACY_FRONT_M)
        source = "legacy roi_track"
    elif prompt is None and hard is None:
        return "ROI parameters unavailable"
    else:
        prompt = prompt or hard
        hard = hard or prompt
        source = "explicit policy"
    assert prompt is not None and hard is not None
    return (
        f"prompt back/front={prompt[0]:.3f}/{prompt[1]:.3f} m  |  "
        f"hard back/front={hard[0]:.3f}/{hard[1]:.3f} m  |  {source}"
    )


def _selected_seed(
    manifest: Mapping[str, Any],
) -> tuple[str | None, int | None, int | None, str | None]:
    raw_seed = manifest.get("seed", {})
    seed = raw_seed if isinstance(raw_seed, dict) else {}
    selected = seed.get("selected_candidate")
    frame = seed.get("frame")
    clean = seed.get("clean_pixels")
    prompt_mode = seed.get("prompt_mode")
    selection_source = seed.get("selection_source")
    raw_qc = seed.get("qc", {})
    qc = raw_qc if isinstance(raw_qc, dict) else {}
    flags: list[str] = []
    if selection_source == "forced_fallback" or qc.get("forced_fallback") is True:
        flags.append("forced_fallback")
    if prompt_mode == "box_only":
        flags.append("box_only")

    def optional_int(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return (
        None if selected is None else str(selected),
        optional_int(frame),
        optional_int(clean),
        "/".join(dict.fromkeys(flags)) or None,
    )


def _seed_panel_path(
    manifest: Mapping[str, Any],
    episode_dir: Path,
    selected: str | None,
) -> Path | None:
    if selected is None:
        return None
    raw_seed = manifest.get("seed", {})
    seed = raw_seed if isinstance(raw_seed, dict) else {}
    raw_artifacts = seed.get("candidate_artifacts", {})
    artifacts = raw_artifacts if isinstance(raw_artifacts, dict) else {}
    raw_panels = artifacts.get("panels", {})
    panels = raw_panels if isinstance(raw_panels, dict) else {}
    relative = panels.get(selected)
    if isinstance(relative, str) and relative:
        return episode_dir / relative
    return episode_dir / "seed_candidates" / f"candidate_{selected}.png"


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_episode_visual(run: RunSpec, episode: int) -> EpisodeVisual:
    """Load one run's contact sheet, selected seed panel, and title metadata."""

    episode_dirs = [root / f"episode_{episode:06d}" for root in run.roots]
    episode_dir = next((path for path in episode_dirs if path.is_dir()), episode_dirs[0])
    manifest_path = episode_dir / "manifest.json"
    contact_path = episode_dir / "episode_contact_sheet.jpg"
    review_path = episode_dir / "episode_gripper_review.mp4"
    manifest: dict[str, Any] | None = None
    details: list[str] = []
    if manifest_path.is_file():
        try:
            manifest = _read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            details.append(f"manifest unreadable: {error}")
    else:
        details.append("manifest missing")

    selected: str | None = None
    frame: int | None = None
    clean: int | None = None
    seed_flag: str | None = None
    seed_path: Path | None = None
    if manifest is not None:
        selected, frame, clean, seed_flag = _selected_seed(manifest)
        seed_path = _seed_panel_path(manifest, episode_dir, selected)
    if selected is None:
        details.append("selected seed unavailable")

    contact: Image.Image | None = None
    if contact_path.is_file():
        try:
            contact = _load_rgb(contact_path)
        except (OSError, UnidentifiedImageError) as error:
            details.append(f"contact sheet unreadable: {error}")
    else:
        details.append("contact sheet missing")

    seed_panel: Image.Image | None = None
    if seed_path is not None and seed_path.is_file():
        try:
            seed_panel = _load_rgb(seed_path)
        except (OSError, UnidentifiedImageError) as error:
            details.append(f"seed panel unreadable: {error}")
    else:
        details.append("selected seed panel missing")

    if manifest is None and contact is None and seed_panel is None:
        status = "missing"
    elif manifest is not None and contact is not None and seed_panel is not None:
        status = "complete"
    else:
        status = "partial"
    frame_label = "?" if frame is None else str(frame)
    clean_label = "?" if clean is None else f"{clean:,}"
    seed_label = (
        f"selected seed={selected or '?'}  |  frame={frame_label}  |  "
        f"clean={clean_label} px"
    )
    return EpisodeVisual(
        run=run,
        episode=episode,
        status=status,
        status_detail=";\n".join(details) if details else None,
        roi_label=format_roi_label(manifest, run),
        seed_label=seed_label,
        seed_flag=seed_flag,
        contact_sheet=contact,
        seed_panel=seed_panel,
        episode_dir=episode_dir,
        manifest_path=manifest_path,
        contact_sheet_path=contact_path,
        seed_panel_path=seed_path,
        review_video_path=review_path,
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def _label_color(label: str) -> tuple[int, int, int]:
    return LABEL_COLORS.get(label.upper(), (89, 196, 212))


def _paste_contained_without_upscale(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Fit an image into a box, preserving native pixels whenever possible."""

    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top
    scale = min(1.0, box_width / image.width, box_height / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    rendered = image
    if (width, height) != image.size:
        rendered = image.resize((width, height), Image.Resampling.LANCZOS)
    x = left + (box_width - width) // 2
    y = top + (box_height - height) // 2
    canvas.paste(rendered, (x, y))
    return x, y, x + width, y + height


def _draw_placeholder(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    draw.rectangle(box, fill=SUBTLE_BACKGROUND)
    bounds = draw.multiline_textbbox((0, 0), text, font=font, spacing=8, align="center")
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    left, top, right, bottom = box
    draw.multiline_text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        text,
        font=font,
        fill=MISSING_TEXT,
        spacing=8,
        align="center",
    )


def _render_run_block(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    visual: EpisodeVisual,
    top: int,
) -> int:
    color = _label_color(visual.run.label)
    header_box = (0, top, CANVAS_WIDTH, top + RUN_HEADER_HEIGHT)
    draw.rectangle(header_box, fill=PANEL_BACKGROUND)
    draw.rectangle((0, top, 12, top + RUN_HEADER_HEIGHT), fill=color)
    run_name = "baseline" if visual.run.is_baseline else "variant"
    status_color = TEXT if visual.status == "complete" else MISSING_TEXT
    draw.text(
        (28, top + 12),
        f"{visual.run.label} / {run_name}  ·  {visual.status.upper()}  ·  {visual.seed_label}",
        font=_font(29, bold=True),
        fill=status_color,
    )
    draw.text((28, top + 60), visual.roi_label, font=_font(23), fill=MUTED_TEXT)

    contact_top = top + RUN_HEADER_HEIGHT
    contact_box = (0, contact_top, CANVAS_WIDTH, contact_top + CONTACT_HEIGHT)
    draw.rectangle(contact_box, fill=(8, 10, 13))
    if visual.contact_sheet is None:
        _draw_placeholder(
            draw,
            contact_box,
            f"{visual.run.label}: episode contact sheet missing",
            _font(34, bold=True),
        )
    else:
        _paste_contained_without_upscale(canvas, visual.contact_sheet, contact_box)

    seed_top = contact_top + CONTACT_HEIGHT
    seed_box = (0, seed_top, CANVAS_WIDTH, seed_top + SEED_STRIP_HEIGHT)
    draw.rectangle(seed_box, fill=PANEL_BACKGROUND)
    candidate_box = (24, seed_top + 23, 344, seed_top + 263)
    if visual.seed_panel is None:
        _draw_placeholder(draw, candidate_box, "selected seed\npanel missing", _font(24, bold=True))
    else:
        _paste_contained_without_upscale(canvas, visual.seed_panel, candidate_box)
    draw.text(
        (374, seed_top + 44),
        "Selected seed candidate panel",
        font=_font(30, bold=True),
        fill=TEXT,
    )
    draw.text((374, seed_top + 94), visual.seed_label, font=_font(25), fill=MUTED_TEXT)
    detail_top = seed_top + 146
    if visual.seed_flag:
        draw.text(
            (374, seed_top + 140),
            f"QC FLAG: {visual.seed_flag}",
            font=_font(24, bold=True),
            fill=MISSING_TEXT,
        )
        detail_top = seed_top + 190
    if visual.status_detail:
        draw.multiline_text(
            (374, detail_top),
            visual.status_detail,
            font=_font(20),
            fill=MISSING_TEXT,
            spacing=7,
        )
    else:
        draw.text(
            (374, detail_top),
            "Source contact sheet is preserved at native 1280×720 resolution.",
            font=_font(20),
            fill=MUTED_TEXT,
        )
    return seed_top + SEED_STRIP_HEIGHT


def render_episode_overview(
    visuals: Sequence[EpisodeVisual],
    output_path: Path,
) -> Path:
    """Render one vertically stacked, native-resolution episode comparison."""

    if not visuals:
        raise ValueError("at least one run is required for an episode overview")
    episode = visuals[0].episode
    if any(visual.episode != episode for visual in visuals):
        raise ValueError("all comparison panels must belong to the same episode")
    height = (
        GLOBAL_HEADER_HEIGHT
        + len(visuals) * (RUN_HEADER_HEIGHT + CONTACT_HEIGHT + SEED_STRIP_HEIGHT)
        + (len(visuals) - 1) * RUN_GAP
    )
    canvas = Image.new("RGB", (CANVAS_WIDTH, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    labels = "/".join(visual.run.label for visual in visuals)
    draw.text(
        (24, 17),
        f"Episode {episode:06d} — Gripper ROI {labels} comparison",
        font=_font(34, bold=True),
        fill=TEXT,
    )
    top = GLOBAL_HEADER_HEIGHT
    for index, visual in enumerate(visuals):
        top = _render_run_block(canvas, draw, visual, top)
        if index + 1 < len(visuals):
            draw.rectangle((0, top, CANVAS_WIDTH, top + RUN_GAP), fill=BACKGROUND)
            top += RUN_GAP
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(
        output_path,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=True,
    )
    return output_path


def _markdown_escape(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def _markdown_link(label: str, target: Path, output_dir: Path) -> str:
    relative = Path(os.path.relpath(target.resolve(), start=output_dir.resolve()))
    destination = quote(relative.as_posix(), safe="/._-")
    return f"[{label}]({destination})"


def _artifact_links(visual: EpisodeVisual, output_dir: Path) -> str:
    links: list[str] = []
    for label, path in (
        ("review mp4", visual.review_video_path),
        ("contact sheet", visual.contact_sheet_path),
        ("manifest", visual.manifest_path),
        ("seed panel", visual.seed_panel_path),
    ):
        if path is not None and path.is_file():
            links.append(_markdown_link(label, path, output_dir))
    if not links:
        return f"**{visual.status}**"
    suffix = "" if visual.status == "complete" else f" — **{visual.status}**"
    if visual.seed_flag:
        suffix += f" — **QC FLAG: {_markdown_escape(visual.seed_flag)}**"
    return " · ".join(links) + suffix


def write_index(
    output_dir: Path,
    runs: Sequence[RunSpec],
    episode_visuals: Mapping[int, Sequence[EpisodeVisual]],
    overview_paths: Mapping[int, Path],
) -> Path:
    """Write a Markdown index linking comparisons and every source artifact."""

    lines = [
        "# Gripper ROI " + "/".join(run.label for run in runs) + " comparison",
        "",
        (
            "Each overview preserves the 1280×720 contact sheets at native resolution and "
            "shows the selected seed candidate panel below each run."
        ),
        "",
        "| Episode | Overview | " + " | ".join(_markdown_escape(run.label) for run in runs) + " |",
        "| ---: | --- | " + " | ".join("---" for _ in runs) + " |",
    ]
    for episode, visuals in episode_visuals.items():
        overview = _markdown_link("JPEG", overview_paths[episode], output_dir)
        cells = [_artifact_links(visual, output_dir) for visual in visuals]
        lines.append(
            f"| {episode} | {overview} | " + " | ".join(cells) + " |"
        )
    lines.extend(["", "## Run roots", ""])
    for run in runs:
        batch_links = []
        for shard_index, root in enumerate(run.roots, start=1):
            batch = root / "batch_manifest.json"
            label = "batch manifest" if len(run.roots) == 1 else f"shard {shard_index}"
            batch_links.append(
                _markdown_link(label, batch, output_dir)
                if batch.is_file()
                else f"{label} manifest missing"
            )
        kind = "baseline" if run.is_baseline else "variant"
        lines.append(
            f"- **{_markdown_escape(run.label)} / {kind}:** "
            + " · ".join(batch_links)
        )
    index_path = output_dir / "index.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def render_comparisons(
    baseline_root: Path,
    variants: Mapping[str, Path | Sequence[Path]],
    episode_ids: Sequence[int],
    output_dir: Path,
) -> tuple[list[Path], Path]:
    """Render all requested episode overviews and their artifact index."""

    unique_episodes = tuple(dict.fromkeys(int(value) for value in episode_ids))
    if not unique_episodes:
        raise ValueError("at least one episode id is required")
    if any(value < 0 for value in unique_episodes):
        raise ValueError("episode ids must be non-negative")
    resolved_output = output_dir.expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    runs = build_run_specs(baseline_root, variants)
    episode_visuals: dict[int, list[EpisodeVisual]] = {}
    overview_paths: dict[int, Path] = {}
    for episode in unique_episodes:
        visuals = [load_episode_visual(run, episode) for run in runs]
        path = resolved_output / f"episode_{episode:06d}_roi_comparison.jpg"
        render_episode_overview(visuals, path)
        episode_visuals[episode] = visuals
        overview_paths[episode] = path
        for visual in visuals:
            if visual.contact_sheet is not None:
                visual.contact_sheet.close()
            if visual.seed_panel is not None:
                visual.seed_panel.close()
    index_path = write_index(resolved_output, runs, episode_visuals, overview_paths)
    return [overview_paths[episode] for episode in unique_episodes], index_path


def main() -> None:
    args = _parse_args()
    variants = parse_variants(args.variant)
    overviews, index = render_comparisons(
        args.baseline_root,
        variants,
        args.episode_ids,
        args.output_dir,
    )
    for path in overviews:
        print(path)
    print(index)


if __name__ == "__main__":
    main()
