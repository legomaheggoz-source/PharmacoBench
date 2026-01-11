"""
MetricCard Component

Provides consistent styled metric display with optional delta, icon, and help text.
Follows Aurora Solar-inspired design.
"""

import streamlit as st
from typing import Optional, Union, List
from dataclasses import dataclass


@dataclass
class MetricCard:
    """
    A styled metric card component.

    Attributes:
        label: The metric label (displayed above the value)
        value: The metric value (can be string, int, or float)
        delta: Optional delta value showing change
        delta_color: Color for delta ("normal", "inverse", or "off")
        help_text: Optional help text shown on hover
        icon: Optional icon to display (emoji or unicode)
    """
    label: str
    value: Union[str, int, float]
    delta: Optional[Union[str, int, float]] = None
    delta_color: str = "normal"
    help_text: Optional[str] = None
    icon: Optional[str] = None

    def render(self, container=None):
        """
        Render the metric card.

        Args:
            container: Optional Streamlit container to render in.
                      If None, renders in current context.
        """
        target = container if container else st

        # Format value for display
        display_value = self._format_value(self.value)

        # Add icon to label if provided
        display_label = f"{self.icon} {self.label}" if self.icon else self.label

        # Render the metric
        target.metric(
            label=display_label,
            value=display_value,
            delta=self.delta,
            delta_color=self.delta_color,
            help=self.help_text,
        )

    def _format_value(self, value: Union[str, int, float]) -> str:
        """Format the value for display with appropriate formatting."""
        if isinstance(value, str):
            return value
        elif isinstance(value, int):
            return f"{value:,}"
        elif isinstance(value, float):
            # Use appropriate precision based on magnitude
            if abs(value) >= 1000:
                return f"{value:,.0f}"
            elif abs(value) >= 1:
                return f"{value:,.2f}"
            else:
                return f"{value:.4f}"
        return str(value)


def render_metric_row(
    metrics: List[MetricCard],
    columns: Optional[int] = None,
) -> None:
    """
    Render a row of metric cards in columns.

    Args:
        metrics: List of MetricCard objects to render
        columns: Number of columns (defaults to len(metrics))

    Example:
        metrics = [
            MetricCard("Users", 1500, delta="+50", icon="👥"),
            MetricCard("Revenue", 25000.50, delta="-2%", delta_color="inverse"),
            MetricCard("Growth", "15%", help_text="Month over month"),
        ]
        render_metric_row(metrics)
    """
    num_cols = columns if columns else len(metrics)
    cols = st.columns(num_cols)

    for i, metric in enumerate(metrics):
        col_idx = i % num_cols
        metric.render(container=cols[col_idx])


def render_kpi_card(
    title: str,
    value: Union[str, int, float],
    subtitle: Optional[str] = None,
    trend: Optional[str] = None,
    trend_positive: bool = True,
) -> None:
    """
    Render a styled KPI card with custom HTML/CSS.

    Args:
        title: Card title
        value: Main value to display
        subtitle: Optional subtitle text
        trend: Optional trend indicator (e.g., "+5%")
        trend_positive: Whether trend is positive (affects color)
    """
    trend_color = "#10b981" if trend_positive else "#ef4444"
    trend_html = f'<span style="color: {trend_color}; font-size: 0.875rem;">{trend}</span>' if trend else ""
    subtitle_html = f'<p style="color: #6b7280; font-size: 0.875rem; margin: 0.5rem 0 0 0;">{subtitle}</p>' if subtitle else ""

    # Format value
    if isinstance(value, (int, float)):
        display_value = f"{value:,}" if isinstance(value, int) else f"{value:,.2f}"
    else:
        display_value = str(value)

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    ">
        <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
            {title}
        </div>
        <div style="display: flex; align-items: baseline; gap: 0.5rem; margin-top: 0.5rem;">
            <span style="color: #1e293b; font-size: 1.75rem; font-weight: 700;">{display_value}</span>
            {trend_html}
        </div>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)
