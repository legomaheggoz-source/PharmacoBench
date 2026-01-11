"""
LoadingSkeleton Component

Provides animated placeholder while data loads.
Follows Aurora Solar-inspired design with subtle animations.
"""

import streamlit as st
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class LoadingSkeleton:
    """
    Animated loading skeleton component.

    Attributes:
        height: Height in pixels
        width: Width (CSS value, e.g., "100%", "200px")
        variant: Type of skeleton ("text", "rect", "circle", "chart")
        lines: Number of text lines (for "text" variant)
    """
    height: int = 20
    width: str = "100%"
    variant: str = "rect"
    lines: int = 3

    def render(self):
        """Render the loading skeleton."""
        if self.variant == "text":
            self._render_text_skeleton()
        elif self.variant == "circle":
            self._render_circle_skeleton()
        elif self.variant == "chart":
            self._render_chart_skeleton()
        else:
            self._render_rect_skeleton()

    def _render_rect_skeleton(self):
        """Render a rectangular skeleton."""
        st.markdown(f"""
        <div style="
            height: {self.height}px;
            width: {self.width};
            background: linear-gradient(90deg, #f1f5f9 0%, #e2e8f0 50%, #f1f5f9 100%);
            background-size: 200% 100%;
            animation: shimmer 1.5s ease-in-out infinite;
            border-radius: 8px;
        "></div>
        <style>
            @keyframes shimmer {{
                0% {{ background-position: 200% 0; }}
                100% {{ background-position: -200% 0; }}
            }}
        </style>
        """, unsafe_allow_html=True)

    def _render_text_skeleton(self):
        """Render text line skeletons."""
        widths = ["100%", "80%", "90%", "70%", "85%"]

        lines_html = ""
        for i in range(self.lines):
            width = widths[i % len(widths)]
            lines_html += f"""
            <div style="
                height: {self.height}px;
                width: {width};
                background: linear-gradient(90deg, #f1f5f9 0%, #e2e8f0 50%, #f1f5f9 100%);
                background-size: 200% 100%;
                animation: shimmer 1.5s ease-in-out infinite;
                border-radius: 4px;
                margin-bottom: 8px;
            "></div>
            """

        st.markdown(f"""
        <div>
            {lines_html}
        </div>
        <style>
            @keyframes shimmer {{
                0% {{ background-position: 200% 0; }}
                100% {{ background-position: -200% 0; }}
            }}
        </style>
        """, unsafe_allow_html=True)

    def _render_circle_skeleton(self):
        """Render a circular skeleton."""
        size = self.height
        st.markdown(f"""
        <div style="
            height: {size}px;
            width: {size}px;
            background: linear-gradient(90deg, #f1f5f9 0%, #e2e8f0 50%, #f1f5f9 100%);
            background-size: 200% 100%;
            animation: shimmer 1.5s ease-in-out infinite;
            border-radius: 50%;
        "></div>
        <style>
            @keyframes shimmer {{
                0% {{ background-position: 200% 0; }}
                100% {{ background-position: -200% 0; }}
            }}
        </style>
        """, unsafe_allow_html=True)

    def _render_chart_skeleton(self):
        """Render a chart placeholder skeleton."""
        st.markdown(f"""
        <div style="
            height: {self.height}px;
            width: {self.width};
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.5) 50%, transparent 100%);
                animation: shimmer 1.5s ease-in-out infinite;
            "></div>
            <div style="
                display: flex;
                align-items: flex-end;
                gap: 8px;
                height: 60%;
            ">
                <div style="width: 24px; height: 40%; background: #e2e8f0; border-radius: 4px 4px 0 0;"></div>
                <div style="width: 24px; height: 70%; background: #e2e8f0; border-radius: 4px 4px 0 0;"></div>
                <div style="width: 24px; height: 55%; background: #e2e8f0; border-radius: 4px 4px 0 0;"></div>
                <div style="width: 24px; height: 85%; background: #e2e8f0; border-radius: 4px 4px 0 0;"></div>
                <div style="width: 24px; height: 60%; background: #e2e8f0; border-radius: 4px 4px 0 0;"></div>
            </div>
        </div>
        <style>
            @keyframes shimmer {{
                0% {{ transform: translateX(-100%); }}
                100% {{ transform: translateX(100%); }}
            }}
        </style>
        """, unsafe_allow_html=True)


def render_loading_skeleton(
    height: int = 200,
    variant: str = "chart",
    message: Optional[str] = None,
) -> None:
    """
    Quick helper to render a loading skeleton.

    Args:
        height: Height in pixels
        variant: Type of skeleton ("text", "rect", "circle", "chart")
        message: Optional message to display below skeleton

    Example:
        placeholder = st.empty()
        with placeholder:
            render_loading_skeleton(300, "chart", "Loading visualization...")

        data = load_data()
        placeholder.empty()
        render_chart(data)
    """
    LoadingSkeleton(height=height, variant=variant).render()

    if message:
        st.markdown(f"""
        <p style="
            text-align: center;
            color: #94a3b8;
            font-size: 0.875rem;
            margin-top: 1rem;
        ">{message}</p>
        """, unsafe_allow_html=True)


def render_metric_skeleton(count: int = 4) -> None:
    """
    Render skeleton placeholders for metric cards.

    Args:
        count: Number of metric placeholders to render
    """
    cols = st.columns(count)

    for col in cols:
        with col:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 1rem;
            ">
                <div style="
                    height: 14px;
                    width: 60%;
                    background: linear-gradient(90deg, #e2e8f0 0%, #cbd5e1 50%, #e2e8f0 100%);
                    background-size: 200% 100%;
                    animation: shimmer 1.5s ease-in-out infinite;
                    border-radius: 4px;
                    margin-bottom: 8px;
                "></div>
                <div style="
                    height: 28px;
                    width: 80%;
                    background: linear-gradient(90deg, #e2e8f0 0%, #cbd5e1 50%, #e2e8f0 100%);
                    background-size: 200% 100%;
                    animation: shimmer 1.5s ease-in-out infinite;
                    border-radius: 4px;
                "></div>
            </div>
            <style>
                @keyframes shimmer {
                    0% { background-position: 200% 0; }
                    100% { background-position: -200% 0; }
                }
            </style>
            """, unsafe_allow_html=True)


def render_table_skeleton(rows: int = 5, cols: int = 4) -> None:
    """
    Render skeleton placeholder for a table.

    Args:
        rows: Number of row placeholders
        cols: Number of column placeholders
    """
    # Header
    header_cols = st.columns(cols)
    for col in header_cols:
        with col:
            st.markdown("""
            <div style="
                height: 16px;
                background: linear-gradient(90deg, #e2e8f0 0%, #cbd5e1 50%, #e2e8f0 100%);
                background-size: 200% 100%;
                animation: shimmer 1.5s ease-in-out infinite;
                border-radius: 4px;
                margin-bottom: 16px;
            "></div>
            """, unsafe_allow_html=True)

    # Rows
    for _ in range(rows):
        row_cols = st.columns(cols)
        for col in row_cols:
            with col:
                st.markdown("""
                <div style="
                    height: 14px;
                    background: linear-gradient(90deg, #f1f5f9 0%, #e2e8f0 50%, #f1f5f9 100%);
                    background-size: 200% 100%;
                    animation: shimmer 1.5s ease-in-out infinite;
                    border-radius: 4px;
                    margin-bottom: 12px;
                "></div>
                """, unsafe_allow_html=True)

    # Add keyframes once
    st.markdown("""
    <style>
        @keyframes shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
    </style>
    """, unsafe_allow_html=True)
