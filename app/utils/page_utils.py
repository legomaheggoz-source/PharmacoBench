"""
Page Utilities

Shared utilities for Streamlit pages including CSS loading,
error handling, loading states, and refresh functionality.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Any
import functools


def load_aurora_css() -> None:
    """
    Load the Aurora Solar-inspired CSS theme from external file.

    This function loads the CSS from app/styles/aurora.css and injects
    it into the Streamlit app. Falls back to inline CSS if file not found.
    """
    css_path = Path(__file__).parent.parent / "styles" / "aurora.css"

    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        # Fallback to minimal inline styles
        st.markdown("""
        <style>
        :root {
            --aurora-blue: #0ea5e9;
            --aurora-light: #f9fafb;
        }
        .stApp { background-color: #ffffff; }
        </style>
        """, unsafe_allow_html=True)


def render_page_header(
    title: str,
    icon: str,
    description: str,
    show_refresh: bool = False,
    refresh_callback: Optional[Callable] = None,
) -> None:
    """
    Render a consistent page header with optional refresh button.

    Args:
        title: Page title
        icon: Emoji icon
        description: Brief description
        show_refresh: Whether to show refresh button
        refresh_callback: Function to call on refresh
    """
    col1, col2 = st.columns([5, 1]) if show_refresh else (st, None)

    with col1 if show_refresh else st.container():
        st.title(f"{icon} {title}")
        st.markdown(description)

    if show_refresh and col2:
        with col2:
            if st.button("Refresh", key=f"refresh_{title.lower().replace(' ', '_')}"):
                if refresh_callback:
                    refresh_callback()
                st.cache_data.clear()
                st.rerun()


def render_data_refresh_button(
    cache_key: str,
    last_updated: Optional[datetime] = None,
) -> bool:
    """
    Render a data refresh section with timestamp and button.

    Args:
        cache_key: Key for the cached data
        last_updated: Optional timestamp of last data update

    Returns:
        True if refresh was clicked, False otherwise
    """
    col1, col2 = st.columns([3, 1])

    with col1:
        if last_updated:
            time_str = last_updated.strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"Last updated: {time_str}")
        else:
            st.caption("Data loaded from cache")

    with col2:
        if st.button("Refresh Data", key=f"refresh_btn_{cache_key}"):
            st.cache_data.clear()
            return True

    return False


def get_last_updated_timestamp(session_key: str = "last_data_update") -> Optional[datetime]:
    """
    Get the last updated timestamp from session state.

    Args:
        session_key: Key to look up in session state

    Returns:
        datetime if found, None otherwise
    """
    return st.session_state.get(session_key)


def set_last_updated_timestamp(session_key: str = "last_data_update") -> datetime:
    """
    Set the current timestamp as last updated in session state.

    Args:
        session_key: Key to store in session state

    Returns:
        The timestamp that was set
    """
    now = datetime.now()
    st.session_state[session_key] = now
    return now


def with_error_handling(
    fallback_message: str = "An error occurred",
    show_traceback: bool = False,
):
    """
    Decorator to add error handling to a function.

    Args:
        fallback_message: Message to show on error
        show_traceback: Whether to show technical details

    Example:
        @with_error_handling("Could not load data")
        def load_data():
            return pd.read_csv("data.csv")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                st.error(f"**{fallback_message}**\n\n{str(e)}")
                if show_traceback:
                    import traceback
                    with st.expander("Technical Details"):
                        st.code(traceback.format_exc(), language="python")
                return None
        return wrapper
    return decorator


def render_loading_placeholder(height: int = 200, message: str = "Loading...") -> st.empty:
    """
    Render a loading placeholder with animation.

    Args:
        height: Height of the placeholder in pixels
        message: Loading message to display

    Returns:
        Streamlit empty container for replacement
    """
    placeholder = st.empty()

    with placeholder.container():
        st.markdown(f"""
        <div style="
            height: {height}px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-radius: 12px;
            border: 1px solid #e2e8f0;
        ">
            <div style="
                width: 40px;
                height: 40px;
                border: 3px solid #e2e8f0;
                border-top: 3px solid #0ea5e9;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            "></div>
            <p style="color: #64748b; margin-top: 1rem; font-size: 0.875rem;">{message}</p>
        </div>
        <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
        """, unsafe_allow_html=True)

    return placeholder


def render_empty_state(
    message: str = "No data available",
    suggestion: Optional[str] = None,
    icon: str = "",
) -> None:
    """
    Render an empty state placeholder.

    Args:
        message: Main message to display
        suggestion: Optional suggestion for the user
        icon: Optional icon/emoji to display
    """
    suggestion_html = f'<p class="suggestion">{suggestion}</p>' if suggestion else ""

    st.markdown(f"""
    <div class="empty-state">
        <div class="icon">{icon}</div>
        <p class="message">{message}</p>
        {suggestion_html}
    </div>
    """, unsafe_allow_html=True)


def create_accessible_button(
    label: str,
    key: str,
    aria_label: Optional[str] = None,
    help_text: Optional[str] = None,
    **kwargs,
) -> bool:
    """
    Create a button with accessibility enhancements.

    Args:
        label: Button label text
        key: Unique key for the button
        aria_label: Optional ARIA label (defaults to label)
        help_text: Optional help tooltip
        **kwargs: Additional arguments passed to st.button

    Returns:
        True if button was clicked
    """
    return st.button(
        label,
        key=key,
        help=help_text or aria_label or label,
        **kwargs,
    )


def format_metric_value(value: Any, precision: int = 2) -> str:
    """
    Format a metric value for display.

    Args:
        value: The value to format
        precision: Decimal precision for floats

    Returns:
        Formatted string
    """
    if isinstance(value, int):
        return f"{value:,}"
    elif isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        elif abs(value) >= 1:
            return f"{value:,.{precision}f}"
        else:
            return f"{value:.{precision + 2}f}"
    else:
        return str(value)
