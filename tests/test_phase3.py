"""Phase 3 invariants (5)."""
import json
from pathlib import Path
import pandas as pd
import pytest

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"


@pytest.fixture(scope="module")
def leaders() -> pd.DataFrame:
    path = CHECKPOINTS / "leaderboards.parquet"
    if not path.exists():
        pytest.skip("leaderboards.parquet missing — run `python src/phase3_story.py` first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def story() -> dict:
    return json.loads((CHECKPOINTS / "story.json").read_text())


def test_leaders_nonempty(leaders):
    assert len(leaders) > 0


def test_leaders_direction_values(leaders):
    assert set(leaders["direction"].unique()).issubset({"improving", "worsening"})


def test_leaders_all_significant(leaders):
    assert (leaders["p_value"] < 0.05).all(), "leaderboards must be stat-sig filtered"


def test_story_shape(story):
    assert "paragraph" in story and len(story["paragraph"]) > 20
    assert "quotables" in story and isinstance(story["quotables"], list)


def test_story_quotables_nonempty(story):
    assert len(story["quotables"]) >= 1
