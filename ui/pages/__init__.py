"""
UI Pages for GENESIS LAB Streamlit application.
"""

from .generate import render_generate_page
from .validate import render_validate_page
from .registry import render_registry_page
from .compare import render_compare_page

__all__ = [
    "render_generate_page",
    "render_validate_page",
    "render_registry_page",
    "render_compare_page",
]
