"""Logging configuration for the root logger (Python >= 3.10)."""

import logging
import platform
import sys
from pathlib import Path


def configure_logging(
    stream_level: int = logging.INFO,
    stream: bool = True,
    file_level: int = logging.DEBUG,
    file_path: Path | str | None = None,
    ignore_libs: list[str] | None = None,
    stream_milliseconds: bool = False,
) -> None:
    """
    Configures the root logger for logging messages to the console and optionally to a file.
    Supports ignoring logs from specified libraries and colored output.

    Parameters:
    - stream_level: The logging level for the stream handler.
    - stream: Whether to enable the stream handler for console logging.
    - file_level: The logging level for the file handler.
    - file_path: The path to the debug log file for the file handler, logs are only saved to a file if this is provided.
    - ignore_libs: A list of library names whose logs should be ignored.
    - stream_milliseconds: Whether to include milliseconds in the stream log timestamps (performance impact).

    Example usage:
    >>> import logging
    >>> configure_logging(stream_level=logging.DEBUG, ignore_libs=["matplotlib"])
    >>> logging.debug("This is a debug message.")
    """

    handlers = []

    # StreamHandler for console logging
    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(stream_level)
        stream_format = "{asctime}"
        if stream_milliseconds:
            stream_format += ".{msecs:03.0f}"
        stream_format += " | {color}{levelname:8}{reset}| {name} | {message}"
        stream_formatter = ColoredFormatter(
            stream_format,
            style="{",
            datefmt="%H:%M:%S",
        )
        stream_handler.setFormatter(stream_formatter)
        handlers.append(stream_handler)

    # FileHandler for file logging, added only if file path is provided
    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            "{asctime} | {levelname:8} | {name} | {message}", style="{"
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Create filter for ignoring logs from specified libraries
    def create_filter(ignored_libs):
        def ignore_logs(record):
            return not any(record.name.startswith(lib) for lib in ignored_libs)

        return ignore_logs

    if ignore_libs:
        ignore_filter = create_filter(ignore_libs)
        for handler in handlers:
            handler.addFilter(ignore_filter)

    # Clear any previously added handlers from the root logger
    logging.getLogger().handlers = []

    # Set up the root logger configuration with the specified handlers
    logging.basicConfig(level=min(stream_level, file_level), handlers=handlers)


def close_root_logging():
    """
    Safely closes and removes all handlers associated with the root logger.

    This function can be called when you no longer need logging or before re-configuring
    logging. It is particularly useful for ensuring that FileHandlers release
    their file resources.

    Note that you typically do not need to manually close and remove handlers,
    as Python's logging module will handle the cleanup when the program exits.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)


class Color:
    """A class for terminal color codes using ANSI escape sequences."""

    BLUE = "\033[36m"
    WHITE = "\033[97m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    BOLD_RED = BOLD + RED + UNDERLINE
    END = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Logging Formatter class that adds colors and styles to log messages."""

    COLORS = {
        "DEBUG": Color.BLUE,
        "INFO": Color.GREEN,
        "WARNING": Color.YELLOW,
        "ERROR": Color.RED,
        "CRITICAL": Color.BOLD_RED,
        "RESET": Color.END,
    }

    # Enable ANSI escape sequences on Windows
    if platform.system() == "Windows":
        try:
            from colorama import just_fix_windows_console

            just_fix_windows_console()
        except ImportError:
            print("Colorama module not found, proceeding without colored logging.")
            # No colors, but include 'RESET' for consistency
            COLORS = {"RESET": ""}

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""
        super().__init__(*args, **kwargs)
        self.colors = ColoredFormatter.COLORS

    def format(self, record) -> str:
        """Format the specified record as text."""
        record.color = self.colors.get(record.levelname, "")
        record.reset = self.colors["RESET"]
        return super().format(record)


def main():
    """Example usage of the configure_logging function."""
    configure_logging(stream_level=logging.DEBUG, file_level=10, file_path="debug.log")
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")


if __name__ == "__main__":
    main()
