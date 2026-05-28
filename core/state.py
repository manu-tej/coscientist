import json
import aiosqlite
from typing import Optional
from core.models import Hypothesis, Review, TournamentMatch, ResearchPlanConfig


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
"""


class StateStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init_db(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(CREATE_TABLES)
            await db.commit()

    # ── Hypothesis ──────────────────────────────────────────────────────────

    async def save_hypothesis(self, h: Hypothesis) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO hypotheses
                   (id, run_id, text, summary, category, generation_method,
                    evolved_from, source, elo_rating, annotations, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (h.id, h.run_id, h.text, h.summary, h.category,
                 h.generation_method, h.evolved_from, h.source,
                 h.elo_rating, json.dumps(h.annotations), h.status),
            )
            await db.commit()

    async def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        async with aiosqlite.connect(self.db_path) as db:
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
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM hypotheses WHERE run_id = ? AND status = ? ORDER BY elo_rating DESC",
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
        async with aiosqlite.connect(self.db_path) as db:
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
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE hypotheses SET annotations = ? WHERE id = ?",
                (json.dumps(h.annotations), hypothesis_id),
            )
            await db.commit()

    async def set_hypothesis_status(self, hypothesis_id: str, status: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE hypotheses SET status = ? WHERE id = ?",
                (status, hypothesis_id),
            )
            await db.commit()

    # ── Review ───────────────────────────────────────────────────────────────

    async def save_review(self, review: Review) -> None:
        async with aiosqlite.connect(self.db_path) as db:
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
        async with aiosqlite.connect(self.db_path) as db:
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

    # ── Tournament ────────────────────────────────────────────────────────────

    async def save_match(self, match: TournamentMatch) -> None:
        async with aiosqlite.connect(self.db_path) as db:
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

    async def list_matches(self, run_id: str) -> list[TournamentMatch]:
        async with aiosqlite.connect(self.db_path) as db:
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
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO proximity_graph (h1_id, h2_id, similarity_score)
                   VALUES (?,?,?)""",
                (h1_id, h2_id, score),
            )
            await db.commit()

    async def get_similar_pairs(
        self, run_id: str, threshold: float = 0.5
    ) -> list[tuple[str, str, float]]:
        async with aiosqlite.connect(self.db_path) as db:
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
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO meta_reviews
                   (id, run_id, meta_critique, research_overview, research_contacts, tick)
                   VALUES (?,?,?,?,?,?)""",
                (id, run_id, meta_critique, research_overview, research_contacts, tick),
            )
            await db.commit()

    async def get_latest_meta_review(self, run_id: str) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
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
        async with aiosqlite.connect(self.db_path) as db:
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
        async with aiosqlite.connect(self.db_path) as db:
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
