from __future__ import annotations

import hashlib
import json
from pathlib import Path

from robotwin_annotation_v2.urdf_gripper_renderer import (
    ALOHA_RENDER_LINKS,
    load_urdf,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "configs" / "assets" / "aloha-agilex"
URDF_PATH = ASSET_ROOT / "arx5_description_isaac_gripper.urdf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_bundled_aloha_asset_matches_manifest() -> None:
    manifest = json.loads((ASSET_ROOT / "asset_manifest.json").read_text())

    assert manifest["format_version"] == "robotwin_aloha_gripper_render_asset_v1"
    for record in manifest["files"]:
        path = ASSET_ROOT / record["path"]
        assert path.is_file()
        assert path.stat().st_size == record["bytes"]
        assert _sha256(path) == record["sha256"]


def test_bundled_aloha_urdf_has_only_required_render_tree() -> None:
    model = load_urdf(URDF_PATH)

    assert model.root_link == "footprint"
    assert set(model.links) == {"footprint", *ALOHA_RENDER_LINKS}
    assert len(model.joints) == len(ALOHA_RENDER_LINKS)

    referenced_meshes = set()
    for link_name in ALOHA_RENDER_LINKS:
        visuals = model.visuals_by_link[link_name]
        assert visuals
        for visual in visuals:
            mesh_path = URDF_PATH.parent / visual.mesh_filename
            assert mesh_path.is_file()
            referenced_meshes.add(mesh_path.name)

    assert referenced_meshes == {
        "base_arm.dae",
        *(f"link{index}.dae" for index in range(1, 9)),
    }
