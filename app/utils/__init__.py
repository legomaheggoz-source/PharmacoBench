"""
PharmacoBench App Utilities

Shared utility functions for the Streamlit application.
Note: Imports are lazy to avoid circular import issues on HuggingFace.
"""

# Don't import anything at module level to avoid import errors
# Users should import directly from submodules:
#   from app.utils.page_utils import load_aurora_css
#   from app.utils.data_loader import get_ic50_data

__all__ = [
    "page_utils",
    "data_loader",
]
