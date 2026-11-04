import sqlite3
from dataclasses import replace

import pytest

from robotwin_annotation_v2.application.umi_job_store import UmiJobStore
from tests.unit.test_umi_pipeline import jobs_at


def test_dedupe_ownership_restart_and_conflicting_ack(tmp_path):
    job = replace(jobs_at(tmp_path, 1)[0], namespace="a", metadata={"id": "source-a"})
    path = tmp_path / "state.sqlite3"
    store = UmiJobStore(path)
    store.register("a", tmp_path / "a", {"hash": "fixed"}, [job])
    store.lease(job.key, 0)
    with pytest.raises(RuntimeError, match="live owner"):
        store.reclaim(0, owner_exited=False)
    with pytest.raises(ValueError, match="unowned"):
        store.finish(job, {"objects_ready": True}, 1)
    store.close()
    store = UmiJobStore(path)
    assert store.leases("a")[0]["worker"] == 0
    store.register("a", tmp_path / "a", {"hash": "fixed"}, [job])
    with pytest.raises(ValueError, match="identity changed"):
        store.register("a", tmp_path / "a", {"hash": "changed"}, [job])
    store.finish(job, {"objects_ready": True}, 0)
    store.finish(job, {"objects_ready": True}, 0)
    with pytest.raises(ValueError, match="conflicting"):
        store.finish(job, {"objects_ready": False}, 0)
    assert store.finalizable() == ["a"]
    duplicate = replace(job, namespace="b")
    with pytest.raises(sqlite3.IntegrityError):
        store.register("b", tmp_path / "b", {}, [duplicate])
    assert store.db.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 1
    assert store.counts() == {"terminal": 1}
    store.close()
