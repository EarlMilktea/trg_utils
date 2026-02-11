from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))

project = "trg_utils"
author = "EarlMilktea"
copyright = f"2025, {author}"  # noqa: A001

release = importlib.metadata.version(project)

extensions: list[str] = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.intersphinx"]
templates_path = ["_templates"]
exclude_patterns: list[str] = []

root_doc = "index"
html_theme = "furo"
default_role = "any"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
