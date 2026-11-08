"""Paths and settings. No data lives inside the repo: the data folders are found via
environment variables (or a .env file) and default to ../Drilling_Project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # the skeptic/ repo folder
CACHE = ROOT / "cache"                              # parsed data (git-ignored)
RUNS = ROOT / "runs"                                # agent traces (git-ignored)
BENCH = ROOT / "bench"


def _load_dotenv(path=ROOT / ".env"):
    """Tiny .env reader: KEY=VALUE lines, '#' comments. Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

_DRILLING = ROOT.parent / "Drilling_Project"
LSU_DIR = Path(os.environ.get("SKEPTIC_LSU_DIR", _DRILLING))
FORGE_DIR = Path(os.environ.get("SKEPTIC_FORGE_DIR", _DRILLING / "FORGE_16A"))

# LSU H-001 files (LSU-only data: never commit, never publish)
LSU_FILES = {
    "pason": LSU_DIR / "PETE7103_Victory_Oghale_A5_LSU_H_001_Pason" / "PETE7103_Victory_Oghale_A5_LSU_H_001_Pason.csv",
    "tour": LSU_DIR / "PETE7103_Victory_Oghale_A5_LSU_H_001_Pason" / "LSU_H_001_Tour_Sheets_IADC_May10-Jun13_2026.xlsx",
    "halliburton": LSU_DIR / "PETE7103_Victory_Oghale_A5_LSU_H_001_Pason" / "PETE7103_Victory_Oghale_A5_LSU_H_001_Halliburton_Project_Data.xlsx",
    "supplement": LSU_DIR / "OneDrive_1_10-1-2026" / "PETE7103_Victory_Oghale_A5_LSU_H_001_Supplement.xlsx",
    "eow": LSU_DIR / "OneDrive_1_10-1-2026" / "PETE7103_Victory_Oghale_A5_Final_DD_EOW_Report.pdf",
}

# FORGE 16A(78)-32 (public, DOI 10.15121/1776602, CC BY 4.0)
FORGE_FILES = {
    "edr": FORGE_DIR / "raw" / "16A(78)-32_time_data_10s_intervals.csv",
    "survey": FORGE_DIR / "raw" / "16A(78)-32 Survey.xlsx",
    "reports": FORGE_DIR / "reports_text",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("SKEPTIC_MODEL", "anthropic/claude-haiku-5.5")
CRITIC_MODEL = os.environ.get("SKEPTIC_CRITIC_MODEL")      # None = same model as the agent


def lsu_available():
    return all(p.exists() for p in LSU_FILES.values())


def forge_available():
    return FORGE_FILES["edr"].exists() and FORGE_FILES["reports"].exists()
