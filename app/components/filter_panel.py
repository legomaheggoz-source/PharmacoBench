"""
FilterPanel Component

Provides reusable filter widget generator from dataframe columns.
Supports automatic type detection and filter creation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class FilterConfig:
    """Configuration for a single filter."""
    column: str
    label: Optional[str] = None
    filter_type: Optional[str] = None  # "multiselect", "slider", "date_range", "text"
    default: Optional[Any] = None
    help_text: Optional[str] = None

    def __post_init__(self):
        if self.label is None:
            # Convert column name to readable label
            self.label = self.column.replace("_", " ").title()


@dataclass
class FilterPanel:
    """
    A panel of filters generated from dataframe columns.

    Attributes:
        df: The dataframe to create filters for
        filters: List of FilterConfig objects
        location: Where to render ("sidebar" or "main")
    """
    df: pd.DataFrame
    filters: List[FilterConfig] = field(default_factory=list)
    location: str = "sidebar"

    def __post_init__(self):
        self._filter_values: Dict[str, Any] = {}

    def add_filter(
        self,
        column: str,
        label: Optional[str] = None,
        filter_type: Optional[str] = None,
        default: Optional[Any] = None,
        help_text: Optional[str] = None,
    ) -> "FilterPanel":
        """Add a filter configuration. Returns self for chaining."""
        self.filters.append(FilterConfig(
            column=column,
            label=label,
            filter_type=filter_type,
            default=default,
            help_text=help_text,
        ))
        return self

    def render(self) -> Dict[str, Any]:
        """
        Render all filters and return the selected values.

        Returns:
            Dictionary mapping column names to selected filter values
        """
        container = st.sidebar if self.location == "sidebar" else st

        for config in self.filters:
            if config.column not in self.df.columns:
                continue

            col_data = self.df[config.column]
            filter_type = config.filter_type or self._infer_filter_type(col_data)

            if filter_type == "multiselect":
                self._filter_values[config.column] = self._render_multiselect(
                    container, config, col_data
                )
            elif filter_type == "slider":
                self._filter_values[config.column] = self._render_slider(
                    container, config, col_data
                )
            elif filter_type == "date_range":
                self._filter_values[config.column] = self._render_date_range(
                    container, config, col_data
                )
            elif filter_type == "text":
                self._filter_values[config.column] = self._render_text(
                    container, config, col_data
                )

        return self._filter_values

    def apply(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply all filters to the dataframe.

        Args:
            df: Optional dataframe to filter. Uses self.df if not provided.

        Returns:
            Filtered dataframe
        """
        result = df if df is not None else self.df.copy()

        for column, value in self._filter_values.items():
            if column not in result.columns:
                continue

            if isinstance(value, list):
                # Multiselect filter
                if len(value) > 0:
                    result = result[result[column].isin(value)]
            elif isinstance(value, tuple) and len(value) == 2:
                # Range filter (slider or date_range)
                result = result[result[column].between(value[0], value[1])]
            elif isinstance(value, str) and value:
                # Text filter
                result = result[result[column].str.contains(value, case=False, na=False)]

        return result

    def _infer_filter_type(self, col_data: pd.Series) -> str:
        """Infer the appropriate filter type from the column data."""
        if pd.api.types.is_datetime64_any_dtype(col_data):
            return "date_range"
        elif pd.api.types.is_numeric_dtype(col_data):
            # Use multiselect for low cardinality, slider for high
            if col_data.nunique() <= 10:
                return "multiselect"
            return "slider"
        elif pd.api.types.is_categorical_dtype(col_data) or col_data.nunique() <= 50:
            return "multiselect"
        else:
            return "text"

    def _render_multiselect(
        self,
        container,
        config: FilterConfig,
        col_data: pd.Series,
    ) -> List[Any]:
        """Render a multiselect filter."""
        options = sorted(col_data.dropna().unique().tolist())
        default = config.default if config.default is not None else options

        return container.multiselect(
            config.label,
            options=options,
            default=default,
            help=config.help_text,
            key=f"filter_{config.column}",
        )

    def _render_slider(
        self,
        container,
        config: FilterConfig,
        col_data: pd.Series,
    ) -> Tuple[float, float]:
        """Render a range slider filter."""
        min_val = float(col_data.min())
        max_val = float(col_data.max())

        if config.default is not None:
            default = config.default
        else:
            # Default to 5th-95th percentile
            default = (
                float(col_data.quantile(0.05)),
                float(col_data.quantile(0.95)),
            )

        return container.slider(
            config.label,
            min_value=min_val,
            max_value=max_val,
            value=default,
            help=config.help_text,
            key=f"filter_{config.column}",
        )

    def _render_date_range(
        self,
        container,
        config: FilterConfig,
        col_data: pd.Series,
    ) -> Tuple[Any, Any]:
        """Render a date range filter."""
        min_date = col_data.min()
        max_date = col_data.max()
        default = config.default if config.default is not None else (min_date, max_date)

        return container.date_input(
            config.label,
            value=default,
            min_value=min_date,
            max_value=max_date,
            help=config.help_text,
            key=f"filter_{config.column}",
        )

    def _render_text(
        self,
        container,
        config: FilterConfig,
        col_data: pd.Series,
    ) -> str:
        """Render a text search filter."""
        default = config.default if config.default is not None else ""

        return container.text_input(
            config.label,
            value=default,
            help=config.help_text,
            key=f"filter_{config.column}",
        )


def create_dataframe_filters(
    df: pd.DataFrame,
    columns: List[str],
    location: str = "sidebar",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Quick helper to create and apply filters for specified columns.

    Args:
        df: The dataframe to filter
        columns: List of column names to create filters for
        location: Where to render filters ("sidebar" or "main")

    Returns:
        Tuple of (filtered_df, filter_values)

    Example:
        filtered_df, filters = create_dataframe_filters(
            df,
            columns=["SOURCE", "TISSUE", "LN_IC50"],
            location="sidebar"
        )
    """
    panel = FilterPanel(df=df, location=location)

    for col in columns:
        if col in df.columns:
            panel.add_filter(col)

    filter_values = panel.render()
    filtered_df = panel.apply()

    return filtered_df, filter_values
