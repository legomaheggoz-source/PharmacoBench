"""
PharmacoBench Reusable UI Components

This module provides consistent, styled UI components for the Streamlit application.
All components follow the Aurora Solar-inspired design system.

Note: Imports are lazy to avoid issues on HuggingFace.
Import directly from submodules when needed.
"""

# Don't import at module level to avoid import errors
# Users should import directly:
#   from app.components.metric_card import MetricCard
#   from app.components.loading_skeleton import render_loading_skeleton

__all__ = [
    "metric_card",
    "filter_panel",
    "error_boundary",
    "loading_skeleton",
]
