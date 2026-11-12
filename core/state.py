import dataclasses
import hashlib
import json
import math
import secrets
import time
import aiosqlite
from dataclasses import dataclass, field
from typing import Literal
from typing import Optional
from core.models import (
    AbsoluteScore,
    Hypothesis,
    Review,
    TournamentMatch,
    ResearchPlanConfig,
)


CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS configs (
    run_id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    preferences TEXT NOT NULL,
    attributes TEXT NOT NULL,
    constraints TEXT NOT NULL,
    safety_approved BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    text TEXT NOT NULL,
    summary TEXT NOT NULL,
    category TEXT,
    generation_method TEXT NOT NULL,
    evolved_from TEXT,
    source TEXT NOT NULL,
    elo_rating REAL DEFAULT 1200.0,
    annotations TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    grounding_refs TEXT DEFAULT NULL,
    FOREIGN KEY (run_id) REFERENCES configs(run_id)
);

CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    tier INTEGER NOT NULL,
    verdict TEXT,
    critique TEXT NOT NULL,
    web_citations TEXT DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

CREATE TABLE IF NOT EXISTS claim_verdicts (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    claim TEXT NOT NULL,
    status TEXT NOT NULL,
    rationale TEXT DEFAULT '',
    records TEXT DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

CREATE TABLE IF NOT EXISTS tournament_matches (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    h1_id TEXT NOT NULL,
    h2_id TEXT NOT NULL,
    winner_id TEXT NOT NULL,
    match_type TEXT NOT NULL,
    debate_transcript TEXT,
    elo_before_h1 REAL,
    elo_before_h2 REAL,
    elo_after_h1 REAL,
    elo_after_h2 REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS absolute_scores (
    run_id TEXT NOT NULL,
    hypothesis_id TEXT PRIMARY KEY,
    scorer_id TEXT NOT NULL,
    axis_scores TEXT NOT NULL,
    total REAL NOT NULL CHECK (total >= 1.0 AND total <= 5.0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES configs(run_id),
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

CREATE TABLE IF NOT EXISTS absolute_scorers (
    run_id TEXT PRIMARY KEY,
    scorer_id TEXT NOT NULL,
    scorer_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES configs(run_id)
);

CREATE TABLE IF NOT EXISTS absolute_score_claims (
    hypothesis_id TEXT PRIMARY KEY,
    scorer_id TEXT NOT NULL,
    owner_token_digest TEXT NOT NULL,
    expires_at REAL NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

CREATE TABLE IF NOT EXISTS proximity_graph (
    h1_id TEXT NOT NULL,
    h2_id TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (h1_id, h2_id)
);

CREATE TABLE IF NOT EXISTS meta_reviews (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    meta_critique TEXT NOT NULL,
    research_overview TEXT,
    research_contacts TEXT,
    tick INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS review_claims (
    hypothesis_id TEXT PRIMARY KEY,
    claimed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass(frozen=True)
class AbsoluteScoreClaim:
    """Atomic claim result for one external absolute-judge call."""

    state: Literal["ready", "acquired", "waiting"]
    owner_token: Optional[str] = field(default=None, repr=False)
    expires_at: Optional[float] = None


class StateStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self):
        # Concurrent agent workers use separate short-lived connections. A longer
        # busy timeout plus WAL avoids treating normal write contention as a task
        # failure when worker counts grow.
        return aiosqlite.connect(self.db_path, timeout=30.0)

    # Columns added to existing tables after the first release. CREATE TABLE
    # IF NOT EXISTS is a no-op on a table that already exists, so a new *table*
    # lands on old databases but a new *column* never does — every pre-existing
    # DB would raise "no such column" at the first write. New tables need no
    # entry here; only columns added to tables that already shipped.
    _ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
        ("hypotheses", "grounding_refs", "TEXT DEFAULT NULL"),
    )

    async def _migrate_columns(self, db) -> None:
        """Add post-release columns that an older database is missing."""
        for table, column, decl in self._ADDED_COLUMNS:
            async with db.execute(f"PRAGMA table_info({table})") as cursor:
                existing = {row[1] for row in await cursor.fetchall()}
            if column not in existing:
                await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    async def init_db(self) -> None:
        async with self._connect() as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=30000")
            await db.executescript(CREATE_TABLES)
            await self._migrate_columns(db)
            await db.commit()

    # ── Hypothesis ──────────────────────────────────────────────────────────

    async def save_hypothesis(self, h: Hypothesis) -> None:
        async with self._connect() as db:
            await db.execute(
                """INSERT INTO hypotheses
                   (id, run_id, text, summary, category, generation_method,
                    evolved_from, source, elo_rating, annotations, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                       run_id=excluded.run_id,
                       text=excluded.text,
                       summary=excluded.summary,
                       category=excluded.category,
                       generation_method=excluded.generation_method,
                       evolved_from=excluded.evolved_from,
                       source=excluded.source,
                       elo_rating=excluded.elo_rating,
                       annotations=excluded.annotations,
                       status=excluded.status""",
                (h.id, h.run_id, h.text, h.summary, h.category,
                 h.generation_method, h.evolved_from, h.source,
                 h.elo_rating, json.dumps(h.annotations), h.status),
            )
            await db.commit()

    async def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return Hypothesis(
            id=row["id"], run_id=row["run_id"], text=row["text"],
            summary=row["summary"], category=row["category"],
            generation_method=row["generation_method"],
            evolved_from=row["evolved_from"], source=row["source"],
            elo_rating=row["elo_rating"],
            annotations=json.loads(row["annotations"]),
            status=row["status"],
        )

    async def list_hypotheses(self, run_id: str, status: str = "active") -> list[Hypothesis]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM hypotheses WHERE run_id = ? AND status = ? "
                "ORDER BY elo_rating DESC, created_at ASC, id ASC",
                (run_id, status),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            Hypothesis(
                id=r["id"], run_id=r["run_id"], text=r["text"],
                summary=r["summary"], category=r["category"],
                generation_method=r["generation_method"],
                evolved_from=r["evolved_from"], source=r["source"],
                elo_rating=r["elo_rating"],
                annotations=json.loads(r["annotations"]),
                status=r["status"],
            )
            for r in rows
        ]

    async def update_elo(self, hypothesis_id: str, new_rating: float) -> None:
        async with self._connect() as db:
            await db.execute(
                "UPDATE hypotheses SET elo_rating = ? WHERE id = ?",
                (new_rating, hypothesis_id),
            )
            await db.commit()

    async def append_annotation(self, hypothesis_id: str, observation: str) -> None:
        h = await self.get_hypothesis(hypothesis_id)
        if h is None:
            return
        h.annotations.append(observation)
        async with self._connect() as db:
            await db.execute(
                "UPDATE hypotheses SET annotations = ? WHERE id = ?",
                (json.dumps(h.annotations), hypothesis_id),
            )
            await db.commit()

    async def set_hypothesis_status(self, hypothesis_id: str, status: str) -> None:
        async with self._connect() as db:
            await db.execute(
                "UPDATE hypotheses SET status = ? WHERE id = ?",
                (status, hypothesis_id),
            )
            await db.commit()

    async def set_grounding_refs(self, hypothesis_id: str, refs: set[str]) -> None:
        """Record the exact ref set shown to this hypothesis at generation time.

        An explicit (possibly empty) `refs` here is distinct from never calling
        this at all -- see `get_grounding_refs`, which returns `None` only for
        the latter.
        """
        async with self._connect() as db:
            await db.execute(
                "UPDATE hypotheses SET grounding_refs = ? WHERE id = ?",
                (json.dumps(sorted(refs)), hypothesis_id),
            )
            await db.commit()

    async def get_grounding_refs(self, hypothesis_id: str) -> Optional[set[str]]:
        """The ref set shown to this hypothesis at generation time.

        Returns `None` when it was never recorded -- the column default, so
        every pre-existing hypothesis and every hypothesis generated by a
        non-literature strategy reads as `None` -- and only `None` in that
        case. A hypothesis explicitly recorded with no evidence shown returns
        an empty set, not `None`; callers must not collapse the two, since
        `None` means "unknown" while `set()` means "nothing was shown."
        """
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT grounding_refs FROM hypotheses WHERE id = ?",
                (hypothesis_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None or row["grounding_refs"] is None:
            return None
        return set(json.loads(row["grounding_refs"]))

    # ── Absolute benchmark scores ───────────────────────────────────────────

    @staticmethod
    def _score_owner_digest(owner_token: str) -> str:
        return hashlib.sha256(owner_token.encode("utf-8")).hexdigest()

    @staticmethod
    def _absolute_score_from_row(row) -> AbsoluteScore:
        try:
            axis_scores = json.loads(row["axis_scores"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("persisted absolute score has invalid axis JSON") from exc
        if not isinstance(axis_scores, dict):
            raise ValueError("persisted absolute score axes must be a mapping")
        return AbsoluteScore(
            run_id=row["run_id"],
            hypothesis_id=row["hypothesis_id"],
            scorer_id=row["scorer_id"],
            axis_scores=axis_scores,
            total=float(row["total"]),
        )

    async def get_absolute_score(
        self, hypothesis_id: str,
    ) -> Optional[AbsoluteScore]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM absolute_scores WHERE hypothesis_id=?",
                (hypothesis_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return self._absolute_score_from_row(row) if row is not None else None

    async def save_absolute_scorer(
        self,
        run_id: str,
        descriptor: dict[str, object],
    ) -> None:
        scorer_id = descriptor.get("scorer_id")
        if not isinstance(scorer_id, str) or not scorer_id:
            raise ValueError("absolute scorer descriptor must contain scorer_id")
        scorer_json = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        async with self._connect() as db:
            await db.execute(
                """INSERT INTO absolute_scorers (run_id, scorer_id, scorer_json)
                   VALUES (?,?,?) ON CONFLICT(run_id) DO NOTHING""",
                (run_id, scorer_id, scorer_json),
            )
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT scorer_id, scorer_json FROM absolute_scorers
                   WHERE run_id=?""",
                (run_id,),
            ) as cursor:
                persisted = await cursor.fetchone()
            if (
                persisted is None
                or persisted["scorer_id"] != scorer_id
                or persisted["scorer_json"] != scorer_json
            ):
                await db.rollback()
                raise ValueError(
                    "run already uses a different absolute scorer identity"
                )
            await db.commit()

    async def get_absolute_scorer(
        self, run_id: str,
    ) -> Optional[dict[str, object]]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT scorer_json FROM absolute_scorers WHERE run_id=?",
                (run_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        try:
            descriptor = json.loads(row["scorer_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("persisted absolute scorer metadata is invalid") from exc
        if not isinstance(descriptor, dict):
            raise ValueError("persisted absolute scorer metadata must be a mapping")
        return descriptor

    async def list_absolute_scores(self, run_id: str) -> dict[str, AbsoluteScore]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM absolute_scores
                   WHERE run_id=? ORDER BY hypothesis_id""",
                (run_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return {
            row["hypothesis_id"]: self._absolute_score_from_row(row)
            for row in rows
        }

    async def claim_absolute_score(
        self,
        hypothesis_id: str,
        scorer_id: str,
        *,
        lease_seconds: float = 300.0,
        owner_token: Optional[str] = None,
    ) -> AbsoluteScoreClaim:
        """Atomically reuse, claim, or wait for one external judge call."""
        if not isinstance(scorer_id, str) or not scorer_id.strip():
            raise ValueError("scorer_id must be nonempty")
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be finite and > 0")
        now = time.time()
        expires_at = now + lease_seconds
        if owner_token is not None and not owner_token:
            raise ValueError("owner_token must be nonempty when provided")
        owner_token = owner_token or secrets.token_urlsafe(32)
        owner_digest = self._score_owner_digest(owner_token)
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT scorer_id FROM absolute_scores WHERE hypothesis_id=?",
                    (hypothesis_id,),
                ) as cursor:
                    score_row = await cursor.fetchone()
                if score_row is not None:
                    if score_row["scorer_id"] != scorer_id:
                        raise ValueError(
                            "hypothesis already has an absolute score from a "
                            "different scorer"
                        )
                    await db.commit()
                    return AbsoluteScoreClaim(state="ready")

                async with db.execute(
                    """SELECT scorer_id, expires_at FROM absolute_score_claims
                       WHERE hypothesis_id=?""",
                    (hypothesis_id,),
                ) as cursor:
                    claim_row = await cursor.fetchone()
                if claim_row is not None and claim_row["scorer_id"] != scorer_id:
                    raise ValueError(
                        "hypothesis score is claimed by a different scorer"
                    )
                if claim_row is not None and claim_row["expires_at"] > now:
                    if owner_token and self._score_owner_digest(owner_token) == owner_digest:
                        # The equality above is intentionally paired with the
                        # owner-guarded SQL below; the local digest alone cannot
                        # prove ownership of the persisted row.
                        extended = await db.execute(
                            """UPDATE absolute_score_claims
                               SET expires_at=?, updated_at=?
                               WHERE hypothesis_id=? AND scorer_id=?
                                 AND owner_token_digest=?""",
                            (
                                expires_at,
                                now,
                                hypothesis_id,
                                scorer_id,
                                owner_digest,
                            ),
                        )
                        if extended.rowcount == 1:
                            await db.commit()
                            return AbsoluteScoreClaim(
                                state="acquired",
                                owner_token=owner_token,
                                expires_at=expires_at,
                            )
                    await db.commit()
                    return AbsoluteScoreClaim(
                        state="waiting", expires_at=float(claim_row["expires_at"])
                    )

                if claim_row is None:
                    await db.execute(
                        """INSERT INTO absolute_score_claims (
                               hypothesis_id, scorer_id, owner_token_digest,
                               expires_at, created_at, updated_at
                           ) VALUES (?,?,?,?,?,?)""",
                        (
                            hypothesis_id,
                            scorer_id,
                            owner_digest,
                            expires_at,
                            now,
                            now,
                        ),
                    )
                else:
                    updated = await db.execute(
                        """UPDATE absolute_score_claims
                           SET scorer_id=?, owner_token_digest=?, expires_at=?,
                               updated_at=?
                           WHERE hypothesis_id=? AND expires_at<=?""",
                        (
                            scorer_id,
                            owner_digest,
                            expires_at,
                            now,
                            hypothesis_id,
                            now,
                        ),
                    )
                    if updated.rowcount != 1:
                        await db.rollback()
                        return AbsoluteScoreClaim(state="waiting")
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
        return AbsoluteScoreClaim(
            state="acquired", owner_token=owner_token, expires_at=expires_at
        )

    async def renew_absolute_score_claim(
        self,
        hypothesis_id: str,
        scorer_id: str,
        owner_token: str,
        *,
        lease_seconds: float = 300.0,
    ) -> bool:
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be finite and > 0")
        now = time.time()
        async with self._connect() as db:
            updated = await db.execute(
                """UPDATE absolute_score_claims
                   SET expires_at=?, updated_at=?
                   WHERE hypothesis_id=? AND scorer_id=?
                     AND owner_token_digest=?""",
                (
                    now + lease_seconds,
                    now,
                    hypothesis_id,
                    scorer_id,
                    self._score_owner_digest(owner_token),
                ),
            )
            await db.commit()
        return updated.rowcount == 1

    async def release_absolute_score_claim(
        self,
        hypothesis_id: str,
        scorer_id: str,
        owner_token: str,
    ) -> bool:
        async with self._connect() as db:
            deleted = await db.execute(
                """DELETE FROM absolute_score_claims
                   WHERE hypothesis_id=? AND scorer_id=?
                     AND owner_token_digest=?""",
                (
                    hypothesis_id,
                    scorer_id,
                    self._score_owner_digest(owner_token),
                ),
            )
            await db.commit()
        return deleted.rowcount == 1

    async def publish_absolute_score(
        self,
        score: AbsoluteScore,
        *,
        owner_token: str,
    ) -> None:
        """Atomically publish a score and release its still-live owner claim."""
        if (
            not math.isfinite(score.total)
            or score.total < 1.0
            or score.total > 5.0
        ):
            raise ValueError("absolute score total must be finite and within [1, 5]")
        if not score.axis_scores:
            raise ValueError("absolute score axes must be nonempty")
        axes_json = json.dumps(
            score.axis_scores,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        digest = self._score_owner_digest(owner_token)
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT run_id FROM hypotheses WHERE id=?",
                    (score.hypothesis_id,),
                ) as cursor:
                    hypothesis_row = await cursor.fetchone()
                if hypothesis_row is None:
                    raise ValueError("cannot score a missing hypothesis")
                if hypothesis_row["run_id"] != score.run_id:
                    raise ValueError("absolute score run_id does not match hypothesis")
                async with db.execute(
                    """SELECT 1 FROM absolute_score_claims
                       WHERE hypothesis_id=? AND scorer_id=?
                         AND owner_token_digest=?""",
                    (
                        score.hypothesis_id,
                        score.scorer_id,
                        digest,
                    ),
                ) as cursor:
                    owns_claim = await cursor.fetchone()
                if owns_claim is None:
                    raise ValueError(
                        "cannot publish absolute score without its active claim"
                    )
                await db.execute(
                    """INSERT INTO absolute_scores (
                           run_id, hypothesis_id, scorer_id, axis_scores, total
                       ) VALUES (?,?,?,?,?)""",
                    (
                        score.run_id,
                        score.hypothesis_id,
                        score.scorer_id,
                        axes_json,
                        score.total,
                    ),
                )
                deleted = await db.execute(
                    """DELETE FROM absolute_score_claims
                       WHERE hypothesis_id=? AND scorer_id=?
                         AND owner_token_digest=?""",
                    (score.hypothesis_id, score.scorer_id, digest),
                )
                if deleted.rowcount != 1:
                    raise RuntimeError("absolute score claim changed during publication")
                await db.commit()
            except BaseException:
                await db.rollback()
                raise

    # ── Review ───────────────────────────────────────────────────────────────

    async def save_review(self, review: Review) -> None:
        async with self._connect() as db:
            await db.execute(
                """INSERT OR REPLACE INTO reviews
                   (id, hypothesis_id, tier, verdict, critique, web_citations)
                   VALUES (?,?,?,?,?,?)""",
                (review.id, review.hypothesis_id, review.tier,
                 review.verdict, review.critique,
                 json.dumps(review.web_citations)),
            )
            await db.commit()

    async def list_reviews(self, hypothesis_id: str) -> list[Review]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM reviews WHERE hypothesis_id = ? ORDER BY tier",
                (hypothesis_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            Review(
                id=r["id"], hypothesis_id=r["hypothesis_id"], tier=r["tier"],
                verdict=r["verdict"], critique=r["critique"],
                web_citations=json.loads(r["web_citations"]),
            )
            for r in rows
        ]

    # ── Claim verdicts ──────────────────────────────────────────────────────

    async def save_claim_verdict(self, hypothesis_id: str, verdict) -> None:
        """Persist one verified claim. Provenance is queryable, not prose."""
        import uuid as _uuid
        from dataclasses import asdict
        async with self._connect() as db:
            await db.execute(
                """INSERT INTO claim_verdicts
                   (id, hypothesis_id, claim, status, rationale, records)
                   VALUES (?,?,?,?,?,?)""",
                (str(_uuid.uuid4()), hypothesis_id, verdict.claim, verdict.status,
                 verdict.rationale, json.dumps([asdict(r) for r in verdict.records])),
            )
            await db.commit()

    async def list_claim_verdicts(self, hypothesis_id: str) -> list:
        from tools.evidence import ClaimVerdict, EvidenceRecord
        fields = {f.name for f in dataclasses.fields(EvidenceRecord)}
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT claim, status, rationale, records FROM claim_verdicts
                   WHERE hypothesis_id = ? ORDER BY created_at, rowid""",
                (hypothesis_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            ClaimVerdict(
                claim=row["claim"], status=row["status"],
                rationale=row["rationale"] or "",
                # Ignore unknown keys so a future schema change cannot make old
                # rows unreadable.
                records=[EvidenceRecord(**{k: v for k, v in r.items() if k in fields})
                         for r in json.loads(row["records"] or "[]")],
            )
            for row in rows
        ]

    async def try_claim_review(self, hypothesis_id: str) -> bool:
        """Atomically claim a hypothesis for reflection. Returns True if THIS
        caller won the claim, False if another worker already claimed it.

        Uses INSERT OR IGNORE on a hypothesis_id primary key: the first insert
        succeeds (rowcount 1), concurrent inserts are ignored (rowcount 0).
        This is a lock-free compare-and-swap that prevents two workers from
        running the reflection pipeline on the same hypothesis."""
        async with self._connect() as db:
            cursor = await db.execute(
                "INSERT OR IGNORE INTO review_claims (hypothesis_id) VALUES (?)",
                (hypothesis_id,),
            )
            await db.commit()
            return cursor.rowcount == 1

    async def release_review_claim(self, hypothesis_id: str) -> None:
        """Release a failed/cancelled reflection claim so another worker can retry."""
        async with self._connect() as db:
            await db.execute(
                "DELETE FROM review_claims WHERE hypothesis_id = ?",
                (hypothesis_id,),
            )
            await db.commit()

    # ── Tournament ────────────────────────────────────────────────────────────

    async def save_match(self, match: TournamentMatch) -> None:
        async with self._connect() as db:
            await db.execute(
                """INSERT OR REPLACE INTO tournament_matches
                   (id, run_id, h1_id, h2_id, winner_id, match_type,
                    debate_transcript, elo_before_h1, elo_before_h2,
                    elo_after_h1, elo_after_h2)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (match.id, match.run_id, match.h1_id, match.h2_id,
                 match.winner_id, match.match_type, match.debate_transcript,
                 match.elo_before_h1, match.elo_before_h2,
                 match.elo_after_h1, match.elo_after_h2),
            )
            await db.commit()

    async def save_match_and_elos(self, match: TournamentMatch) -> None:
        """Atomically persist a tournament match and update both hypotheses' Elo
        ratings in a single transaction, so a mid-write failure cannot leave one
        hypothesis updated and the other not."""
        async with self._connect() as db:
            await db.execute(
                """INSERT OR REPLACE INTO tournament_matches
                   (id, run_id, h1_id, h2_id, winner_id, match_type,
                    debate_transcript, elo_before_h1, elo_before_h2,
                    elo_after_h1, elo_after_h2)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (match.id, match.run_id, match.h1_id, match.h2_id,
                 match.winner_id, match.match_type, match.debate_transcript,
                 match.elo_before_h1, match.elo_before_h2,
                 match.elo_after_h1, match.elo_after_h2),
            )
            await db.execute(
                "UPDATE hypotheses SET elo_rating = ? WHERE id = ?",
                (match.elo_after_h1, match.h1_id),
            )
            await db.execute(
                "UPDATE hypotheses SET elo_rating = ? WHERE id = ?",
                (match.elo_after_h2, match.h2_id),
            )
            await db.commit()

    async def list_matches(self, run_id: str) -> list[TournamentMatch]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tournament_matches WHERE run_id = ? ORDER BY created_at",
                (run_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            TournamentMatch(
                id=r["id"], run_id=r["run_id"], h1_id=r["h1_id"],
                h2_id=r["h2_id"], winner_id=r["winner_id"],
                match_type=r["match_type"],
                debate_transcript=r["debate_transcript"],
                elo_before_h1=r["elo_before_h1"],
                elo_before_h2=r["elo_before_h2"],
                elo_after_h1=r["elo_after_h1"],
                elo_after_h2=r["elo_after_h2"],
            )
            for r in rows
        ]

    # ── Proximity graph ───────────────────────────────────────────────────────

    async def save_proximity(self, h1_id: str, h2_id: str, score: float) -> None:
        async with self._connect() as db:
            await db.execute(
                """INSERT OR REPLACE INTO proximity_graph (h1_id, h2_id, similarity_score)
                   VALUES (?,?,?)""",
                (h1_id, h2_id, score),
            )
            await db.commit()

    async def get_similar_pairs(
        self, run_id: str, threshold: float = 0.5
    ) -> list[tuple[str, str, float]]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT p.h1_id, p.h2_id, p.similarity_score
                   FROM proximity_graph p
                   JOIN hypotheses h1 ON p.h1_id = h1.id
                   JOIN hypotheses h2 ON p.h2_id = h2.id
                   WHERE h1.run_id = ? AND h2.run_id = ?
                     AND p.similarity_score >= ?
                   ORDER BY p.similarity_score DESC""",
                (run_id, run_id, threshold),
            ) as cursor:
                rows = await cursor.fetchall()
        return [(r["h1_id"], r["h2_id"], r["similarity_score"]) for r in rows]

    # ── Meta-review ───────────────────────────────────────────────────────────

    async def save_meta_review(
        self,
        id: str,
        run_id: str,
        meta_critique: str,
        research_overview: Optional[str],
        research_contacts: Optional[str],
        tick: int,
    ) -> None:
        async with self._connect() as db:
            await db.execute(
                """INSERT OR REPLACE INTO meta_reviews
                   (id, run_id, meta_critique, research_overview, research_contacts, tick)
                   VALUES (?,?,?,?,?,?)""",
                (id, run_id, meta_critique, research_overview, research_contacts, tick),
            )
            await db.commit()

    async def get_latest_meta_review(self, run_id: str) -> Optional[dict]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM meta_reviews WHERE run_id = ?
                   ORDER BY tick DESC LIMIT 1""",
                (run_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    # ── Config ────────────────────────────────────────────────────────────────

    async def save_config(self, config: ResearchPlanConfig) -> None:
        async with self._connect() as db:
            await db.execute(
                """INSERT OR REPLACE INTO configs
                   (run_id, goal, preferences, attributes, constraints, safety_approved)
                   VALUES (?,?,?,?,?,?)""",
                (config.run_id, config.goal, config.preferences,
                 json.dumps(config.attributes), config.constraints,
                 config.safety_approved),
            )
            await db.commit()

    async def get_config(self, run_id: str) -> Optional[ResearchPlanConfig]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM configs WHERE run_id = ?", (run_id,)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return ResearchPlanConfig(
            run_id=row["run_id"],
            goal=row["goal"],
            preferences=row["preferences"],
            attributes=json.loads(row["attributes"]),
            constraints=row["constraints"],
            safety_approved=bool(row["safety_approved"]),
        )
