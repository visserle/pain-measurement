import ast
import tkinter as tk

import polars as pl
from screeninfo import get_monitors


def ensure_list(to_list: str | list[str]) -> list[str]:
    """Convert str to list idempotently."""
    return [to_list] if isinstance(to_list, str) else to_list


def convert_str_to_list(str_list: str) -> list:
    """
    Convert a string representation of a list to a list.

    Returns an empty list if the conversion fails.
    """
    try:
        return ast.literal_eval(str_list)
    except ValueError:
        return []


def to_describe(col, prefix=""):
    """
    Helper function for having the describe expression for use in groupby-agg.
    From https://github.com/pola-rs/polars/issues/8066#issuecomment-1794144838
    """
    prefix = prefix or f"{col}_"
    return [
        pl.col(col).count().alias(f"{prefix}count"),
        pl.col(col).is_null().sum().alias(f"{prefix}null_count"),
        pl.col(col).mean().alias(f"{prefix}mean"),
        pl.col(col).std().alias(f"{prefix}std"),
        pl.col(col).min().alias(f"{prefix}min"),
        pl.col(col).quantile(0.25).alias(f"{prefix}25%"),
        pl.col(col).quantile(0.5).alias(f"{prefix}50%"),
        pl.col(col).quantile(0.75).alias(f"{prefix}75%"),
        pl.col(col).max().alias(f"{prefix}max"),
    ]


def center_tk_window(
    window: tk.Tk,
    primary_screen: bool = False,
) -> None:
    """
    Center a window, by default on the first available non-primary screen if available,
    otherwise on the primary screen.
    """
    # Get sorted list of monitors
    monitors = sorted(get_monitors(), key=lambda m: m.is_primary)
    # non-primary monitor comes first (False < True)
    monitor = monitors[0] if not primary_screen else monitors[-1]

    # Get window size
    window.update_idletasks()
    window_width = window.winfo_width()
    window_height = window.winfo_height()

    # Calculate center coordinates
    center_x = int(monitor.x + (monitor.width / 2 - window_width / 2))
    center_y = int(monitor.y + (monitor.height / 2 - window_height / 2))

    # Move window to the center of the chosen monitor
    window.geometry(f"+{center_x}+{center_y}")
