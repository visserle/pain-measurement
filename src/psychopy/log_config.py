# work in proqress

import logging


def configure_logging(
        stream_level=logging.INFO, stream=True,
        file_level=logging.DEBUG, file_path=None,
        ignore_libs=None):
    """
    Configures the root logger for logging messages to the console and optionally to a file.
    Supports ignoring logs from specified libraries.

    Parameters:
    - stream_level: The logging level for the stream handler.
    - stream: Whether to enable the stream handler for console logging.
    - file_level: The logging level for the file handler.
    - file_path: The path to the debug log file for the file handler, logs are only saved to a file if this is provided.
    - ignore_libs: A list of library names whose logs should be ignored.
    """
    handlers = []

    # StreamHandler for console logging
    if stream:
        stream_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s', datefmt='%H:%M:%S')
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(stream_level)
        stream_handler.setFormatter(stream_formatter)
        handlers.append(stream_handler)

    # FileHandler for file logging, added only if file path is provided
    if file_path:
        file_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s')
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(file_level)
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
    logging.basicConfig(level=min(stream_level, logging.DEBUG), handlers=handlers)


def close_root_logging():
    """
    Safely closes and removes all handlers associated with the root logger.

    This function should be called when you no longer need logging or before re-configuring
    logging. It is particularly useful for ensuring that FileHandlers release
    their file resources.

    Note that you typically do not need to manually close and remove handlers, 
    as Python's logging module will handle the cleanup when the program exits.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
