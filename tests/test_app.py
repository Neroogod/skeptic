"""Phase 7: the UI script renders headlessly in every mode (no API key needed)."""
import pytest

from skeptic import config

pytestmark = pytest.mark.skipif(not config.forge_available(), reason="FORGE data not found")


def test_app_renders_in_every_mode():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    for mode in ("Benchmark task", "Replay trace"):
        at.sidebar.radio[0].set_value(mode).run()
        assert not at.exception, (mode, at.exception)


def test_replay_offers_the_committed_runs_and_renders_a_demo_trace():
    """The network-free fallback: replay works from results/ even in a fresh clone (runs/ is git-ignored)."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=120).run()
    at.sidebar.radio[0].set_value("Replay trace").run()
    run = next(s for s in at.selectbox if s.label == "Run")
    demo = next(o for o in run.options if o.startswith("results/demo"))
    run.select(demo).run()
    assert not at.exception, at.exception
    assert any(m.label == "Verdict" for m in at.metric)
