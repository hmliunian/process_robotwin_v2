"""Application use case: Review and approve keyframes."""

from ..domain import ApprovedSeed, MaskArtifactRef
from ..ports import ArtifactRepository


class ReviewKeyframes:
    """Use case: Approve/reject keyframe candidates."""

    def __init__(self, artifact_repo: ArtifactRepository) -> None:
        self.artifact_repo = artifact_repo

    def approve(
        self,
        run_id: str,
        episode_id: str,
        slot_name: str,
        candidate_id: str,
        reviewer: str,
        note: str = "",
    ) -> ApprovedSeed:
        """Approve a candidate as the official seed."""

        # Load request data
        data = self.artifact_repo.load_request(run_id, episode_id, slot_name)
        request = data["request"]

        # Find the candidate
        candidate = next(
            (c for c in data["candidates"] if c["candidate_id"] == candidate_id),
            None,
        )
        if candidate is None:
            raise ValueError(f"Candidate {candidate_id} not found")

        # Create approved seed
        seed = ApprovedSeed(
            request_id=request["request_id"],
            candidate_id=candidate_id,
            frame_index=candidate["frame_index"],
            slot=slot_name,  # type: ignore
            mask_artifact=MaskArtifactRef(
                sha256="",  # TODO: compute from mask file
                relative_path=f"{episode_id}/{slot_name}/{candidate_id}.mask.png",
            ),
            approval_revision=request["revision"],
            reviewer=reviewer,
            note=note,
        )

        # Save approval
        self.artifact_repo.save_approved_seed(run_id, seed)

        return seed
