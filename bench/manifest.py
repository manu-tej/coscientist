from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import aiosqlite

_CREATE = """
CREATE TABLE IF NOT EXISTS manifest (
    goal_id TEXT NOT NULL,
    variant TEXT NOT NULL,
    system_version TEXT NOT NULL,
    seed INTEGER NOT NULL,
    db_path TEXT NOT NULL,
    n_llm_calls INTEGER NOT NULL,
    wall_clock_s REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'complete',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (goal_id, variant, system_version, seed)
);
"""


def system_version() -> str:
    """Git SHA of HEAD — used to invalidate the cache when core/ changes.
    (Cache key includes this so a code change forces fresh runs.)"""
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


class Manifest:
    def __init__(self, path: str):
        self.path = path

    async def init(self) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(_CREATE)
            await db.commit()

    async def get(
        self, goal_id: str, variant: str, system_version: str, seed: int
    ) -> Optional[dict]:
        """Return the cached cell, or None on miss. A row whose db_path no
        longer exists is treated as a miss (stale)."""
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM manifest WHERE goal_id=? AND variant=?
                   AND system_version=? AND seed=? AND status='complete'""",
                (goal_id, variant, system_version, seed),
            ) as cur:
                row = await cur.fetchone()
        if row is None:
            return None
        if not Path(row["db_path"]).exists():
            return None
        return dict(row)

    async def record(
        self, goal_id: str, variant: str, system_version: str, seed: int,
        *, db_path: str, n_llm_calls: int, wall_clock_s: float,
        status: str = "complete",
    ) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO manifest
                   (goal_id, variant, system_version, seed, db_path,
                    n_llm_calls, wall_clock_s, status)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (goal_id, variant, system_version, seed, db_path,
                 n_llm_calls, wall_clock_s, status),
            )
            await db.commit()
