import tkinter as tk

from screeninfo import get_monitors


def ensure_list(to_list: str | list[str]) -> list[str]:
    """Convert str to list idempotently."""
    return [to_list] if isinstance(to_list, str) else to_list


def center_tk_window(window: tk.Tk, primary_screen: bool = False) -> tuple[int, int]:
    """Center a window, by default on the first available non-primary screen if available, otherwise on the primary screen."""
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()

    # Get list of monitors and sort them
    monitors = sorted(get_monitors(), key=lambda m: m.is_primary)
    # Non-primary monitor comes first (False < True)
    if primary_screen:
        target_monitor = monitors[-1]
    else:
        target_monitor = monitors[0]

    screen_width = target_monitor.width
    screen_height = target_monitor.height
    center_x = int(target_monitor.x + (screen_width / 2 - width / 2))
    center_y = int(target_monitor.y + (screen_height / 2 - height / 2))

    # Move window to the center of the chosen monitor
    window.geometry(f"+{center_x}+{center_y}")

    return (center_x, center_y)
