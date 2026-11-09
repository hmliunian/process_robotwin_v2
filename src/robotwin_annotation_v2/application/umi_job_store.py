"""Durable ownership for streamed UMI jobs; one coordinator writes this SQLite store."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .umi_pipeline import UmiJob


class UmiJobStore:
    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS batches (
                name TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, identity TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'inference', error TEXT
            );
            CREATE TABLE IF NOT EXISTS jobs (
                key TEXT PRIMARY KEY, source_id TEXT UNIQUE NOT NULL, batch TEXT NOT NULL,
                identity TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'ready', worker INTEGER,
                attempts INTEGER NOT NULL DEFAULT 0, result TEXT, updated REAL NOT NULL,
                FOREIGN KEY(batch) REFERENCES batches(name)
            );
            CREATE INDEX IF NOT EXISTS jobs_state_batch ON jobs(state,batch,key);
            CREATE INDEX IF NOT EXISTS jobs_batch_state ON jobs(batch,state);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, key TEXT NOT NULL, state TEXT NOT NULL, at REAL NOT NULL
            );
        """)

    @staticmethod
    def signature(job: UmiJob) -> str:
        return hashlib.sha256(json.dumps(job.identity, sort_keys=True).encode()).hexdigest()

    def register(self, name: str, path: Path, identity: dict[str, Any], jobs: list[UmiJob]) -> None:
        encoded = json.dumps(identity, sort_keys=True)
        existing = self.db.execute("SELECT * FROM batches WHERE name=?", (name,)).fetchone()
        if existing:
            if existing["path"] != str(path) or existing["identity"] != encoded:
                raise ValueError("batch selection or identity changed")
            saved = {
                r["key"]: r["identity"]
                for r in self.db.execute("SELECT key,identity FROM jobs WHERE batch=?", (name,))
            }
            if saved != {job.key: self.signature(job) for job in jobs}:
                raise ValueError("registered job identities changed")
            return
        if any(not job.namespace for job in jobs):
            raise ValueError("streamed jobs require a batch namespace")
        with self.db:
            self.db.execute(
                "INSERT INTO batches(name,path,identity) VALUES(?,?,?)", (name, str(path), encoded)
            )
            self.db.executemany(
                "INSERT INTO jobs(key,source_id,batch,identity,updated) VALUES(?,?,?,?,?)",
                [
                    (j.key, str(j.metadata["id"]), name, self.signature(j), time.time())
                    for j in jobs
                ],
            )

    def ready(self, limit: int) -> list[str]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT key FROM jobs WHERE state='ready' ORDER BY batch,key LIMIT ?", (limit,)
            )
        ]

    def lease(self, key: str, worker: int) -> None:
        with self.db:
            changed = self.db.execute(
                "UPDATE jobs SET state='leased',worker=?,attempts=attempts+1,updated=? "
                "WHERE key=? AND state='ready'",
                (worker, time.time(), key),
            )
            if changed.rowcount != 1:
                raise RuntimeError("job already leased or terminal")
            self.db.execute(
                "INSERT INTO events(key,state,at) VALUES(?,'leased',?)", (key, time.time())
            )

    def finish(self, job: UmiJob, row: dict[str, Any], worker: int) -> None:
        saved = self.db.execute("SELECT * FROM jobs WHERE key=?", (job.key,)).fetchone()
        if saved is None or saved["identity"] != self.signature(job):
            raise ValueError("unknown or changed completion identity")
        encoded = json.dumps(row, sort_keys=True)
        if saved["state"] == "terminal":
            if saved["result"] != encoded:
                raise ValueError("conflicting duplicate completion")
            return
        if saved["state"] != "leased" or saved["worker"] != worker:
            raise ValueError("completion from unowned worker")
        with self.db:
            self.db.execute(
                "UPDATE jobs SET state='terminal',result=?,updated=? WHERE key=?",
                (encoded, time.time(), job.key),
            )
            self.db.execute(
                "INSERT INTO events(key,state,at) VALUES(?,'terminal',?)", (job.key, time.time())
            )

    def reclaim(self, worker: int, *, owner_exited: bool) -> None:
        if not owner_exited:
            raise RuntimeError("cannot reclaim a live owner")
        with self.db:
            self.db.execute(
                "UPDATE jobs SET state='ready',worker=NULL,updated=? "
                "WHERE state='leased' AND worker=?",
                (time.time(), worker),
            )

    def finalizable(self) -> list[str]:
        return [
            r[0]
            for r in self.db.execute("""
            SELECT name FROM batches WHERE state='inference' AND NOT EXISTS (
                SELECT 1 FROM jobs WHERE batch=name AND state!='terminal') ORDER BY name
        """)
        ]

    def results(self, batch: str) -> list[dict[str, Any]]:
        return [
            json.loads(r[0])
            for r in self.db.execute(
                "SELECT result FROM jobs WHERE batch=? AND state='terminal' ORDER BY key", (batch,)
            )
        ]

    def batch_state(self, batch: str) -> str:
        return str(
            self.db.execute("SELECT state FROM batches WHERE name=?", (batch,)).fetchone()[0]
        )

    def set_batch_state(self, batch: str, state: str, error: str | None = None) -> None:
        with self.db:
            self.db.execute(
                "UPDATE batches SET state=?,error=? WHERE name=?", (state, error, batch)
            )

    def leases(self, batch: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM jobs WHERE batch=? AND state='leased'", (batch,)
            )
        ]

    def counts(self) -> dict[str, int]:
        return {
            r[0]: r[1] for r in self.db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state")
        }

    def result_counts(self) -> dict[str, int]:
        return {
            r[0]: r[1]
            for r in self.db.execute(
                "SELECT json_extract(result,'$.status'),COUNT(*) FROM jobs WHERE state='terminal' GROUP BY 1"
            )
        }

    def close(self) -> None:
        self.db.close()
