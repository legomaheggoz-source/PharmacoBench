"""
ErrorBoundary Component

Provides graceful error handling for Streamlit components.
Includes decorators and context managers for safe rendering.
"""

import streamlit as st
import traceback
import functools
from typing import Callable, Optional, Any, TypeVar
from dataclasses import dataclass

T = TypeVar("T")


@dataclass
class ErrorDisplay:
    """
    Styled error display component.

    Attributes:
        title: Error title
        message: Error message
        show_traceback: Whether to show full traceback
        error_type: Type of error for styling ("error", "warning", "info")
    """
    title: str = "An error occurred"
    message: str = ""
    show_traceback: bool = False
    error_type: str = "error"
    traceback_text: Optional[str] = None

    def render(self):
        """Render the error display."""
        if self.error_type == "error":
            st.error(f"**{self.title}**\n\n{self.message}")
        elif self.error_type == "warning":
            st.warning(f"**{self.title}**\n\n{self.message}")
        else:
            st.info(f"**{self.title}**\n\n{self.message}")

        if self.show_traceback and self.traceback_text:
            with st.expander("Technical Details"):
                st.code(self.traceback_text, language="python")


def error_boundary(
    fallback_message: str = "Something went wrong",
    show_traceback: bool = False,
    error_type: str = "error",
    reraise: bool = False,
) -> Callable:
    """
    Decorator that wraps a function with error handling.

    Args:
        fallback_message: Message to display on error
        show_traceback: Whether to show technical details
        error_type: Type of error display ("error", "warning", "info")
        reraise: Whether to re-raise the exception after displaying

    Returns:
        Decorated function with error handling

    Example:
        @error_boundary(fallback_message="Could not load data")
        def load_data():
            # ... code that might fail
            return data
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                tb = traceback.format_exc()

                ErrorDisplay(
                    title=fallback_message,
                    message=str(e),
                    show_traceback=show_traceback,
                    error_type=error_type,
                    traceback_text=tb,
                ).render()

                if reraise:
                    raise
                return None

        return wrapper
    return decorator


def safe_render(
    render_func: Callable,
    fallback_message: str = "Could not render component",
    fallback_component: Optional[Callable] = None,
    **kwargs,
) -> Any:
    """
    Safely render a component with error handling.

    Args:
        render_func: The function to render
        fallback_message: Message to show if rendering fails
        fallback_component: Optional alternative component to render on error
        **kwargs: Additional arguments to pass to render_func

    Returns:
        Result of render_func or fallback_component

    Example:
        safe_render(
            lambda: expensive_chart(data),
            fallback_message="Chart could not be generated",
            fallback_component=lambda: st.info("No data to display")
        )
    """
    try:
        return render_func(**kwargs)
    except Exception as e:
        st.error(f"{fallback_message}: {str(e)}")

        if fallback_component:
            try:
                return fallback_component()
            except Exception:
                pass

        return None


class ErrorBoundaryContext:
    """
    Context manager for error handling in Streamlit.

    Example:
        with ErrorBoundaryContext("Loading data", show_traceback=True):
            data = load_expensive_data()
            render_chart(data)
    """

    def __init__(
        self,
        operation_name: str = "Operation",
        fallback_message: Optional[str] = None,
        show_traceback: bool = False,
        error_type: str = "error",
        suppress: bool = True,
    ):
        """
        Initialize error boundary context.

        Args:
            operation_name: Name of the operation (for error messages)
            fallback_message: Custom fallback message
            show_traceback: Whether to show technical details
            error_type: Type of error display
            suppress: Whether to suppress the exception
        """
        self.operation_name = operation_name
        self.fallback_message = fallback_message or f"{operation_name} failed"
        self.show_traceback = show_traceback
        self.error_type = error_type
        self.suppress = suppress
        self.error: Optional[Exception] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.error = exc_val
            tb = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))

            ErrorDisplay(
                title=self.fallback_message,
                message=str(exc_val),
                show_traceback=self.show_traceback,
                error_type=self.error_type,
                traceback_text=tb,
            ).render()

            return self.suppress
        return False

    @property
    def had_error(self) -> bool:
        """Check if an error occurred."""
        return self.error is not None


def render_error_fallback(
    message: str = "No data available",
    suggestion: Optional[str] = None,
    icon: str = "info",
) -> None:
    """
    Render a styled fallback message when data is unavailable.

    Args:
        message: Main message to display
        suggestion: Optional suggestion for the user
        icon: Icon type ("info", "warning", "error")
    """
    icon_map = {
        "info": "",
        "warning": "",
        "error": "",
    }

    icon_color = {
        "info": "#0ea5e9",
        "warning": "#f59e0b",
        "error": "#ef4444",
    }

    color = icon_color.get(icon, icon_color["info"])

    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        border: 1px dashed #cbd5e1;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;">
            {icon_map.get(icon, icon_map["info"])}
        </div>
        <p style="color: #64748b; font-size: 1rem; margin: 0;">
            {message}
        </p>
        {"<p style='color: #94a3b8; font-size: 0.875rem; margin-top: 0.5rem;'>" + suggestion + "</p>" if suggestion else ""}
    </div>
    """, unsafe_allow_html=True)
