"""Fixed locations. Everything else in the pipeline derives its paths from here."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("PDI_ROOT", Path(__file__).resolve().parents[2]))

# Untouched responses, with retrieval metadata. Git-ignored and never published.
RAW = ROOT / "data" / "raw"
# Tidy derived tables, committed by the scheduled workflows. This is the pipeline's memory:
# anything that cannot be re-downloaded later (daily prices) lives here.
STORE = ROOT / "data" / "store"
# Model artefacts, git-ignored.
MODELS = ROOT / "data" / "models"
# The contract the site reads. Fixed paths, fixed column names.
SITE_DATA = ROOT / "site" / "data" / "v1"
REGISTRY = ROOT / "sources" / "registry.yaml"
