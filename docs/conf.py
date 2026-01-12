from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))

project = "projector_utils"
author = "EarlMilktea"

release = importlib.metadata.version(project)

extensions: list[str] = ["sphinx.ext.autodoc"]
templates_path = ["_templates"]
exclude_patterns: list[str] = []

root_doc = "index"
html_theme = "furo"
