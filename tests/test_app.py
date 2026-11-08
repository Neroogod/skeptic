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
