from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))

project = "trg_utils"
author = "EarlMilktea"
copyright = f"2025, {author}"  # noqa: A001

release = importlib.metadata.version(project)

extensions: list[str] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
templates_path = ["_templates"]
exclude_patterns: list[str] = []

autodoc_member_order = "bysource"
autosummary_generate = True
default_role = "any"
html_theme = "furo"
root_doc = "index"
toc_object_entries = False
viewcode_line_numbers = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
