import pytest
from pathlib import Path
from bench.manifest import Manifest, system_version


@pytest.mark.asyncio
async def test_manifest_miss_then_hit(tmp_path: Path):
    mpath = tmp_path / "manifest.sqlite"
    m = Manifest(str(mpath))
    await m.init()
    key = ("g1", "full", "abc123", 0)
    assert await m.get(*key) is None                 # miss

    run_db = tmp_path / "run_g1.db"
    run_db.write_text("")                            # the run's sqlite exists
    await m.record(*key, db_path=str(run_db), n_llm_calls=42, wall_clock_s=3.0)

    hit = await m.get(*key)
    assert hit is not None
    assert hit["db_path"] == str(run_db)
    assert hit["n_llm_calls"] == 42
    assert hit["status"] == "complete"


@pytest.mark.asyncio
async def test_manifest_hit_requires_existing_db(tmp_path: Path):
    m = Manifest(str(tmp_path / "manifest.sqlite"))
    await m.init()
    await m.record("g1", "full", "v1", 0, db_path=str(tmp_path / "missing.db"),
                   n_llm_calls=1, wall_clock_s=1.0)
    # row exists but the db file does not → treated as a miss
    assert await m.get("g1", "full", "v1", 0) is None


def test_system_version_is_git_sha():
    sha = system_version()
    assert isinstance(sha, str) and len(sha) >= 7
