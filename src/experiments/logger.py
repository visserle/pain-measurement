# work in proqress

# TODO
# - find out exactly where/when and how to use close_logger (not in del, it's tricky)
# - find out what happens when the experiment gets aborted - logs still get written? closed?


import logging


def setup_logger(logger_name, level=logging.INFO, log_file=None, stream_handler=True):
    """
    Configures and returns a logger instance for logging messages to the console
    and optionally to a specified log file.

    This function sets up a logger with custom formatting. If enabled, it adds two handlers:
    1. StreamHandler for console output.
    2. FileHandler for logging to a file.

    Parameters
    ----------
    logger_name : str
        Name of the logger to be configured. This is typically the name of the
        module where the logger is used.
    
    level : int, optional
        The logging level to filter messages. Default is logging.INFO.
        Levels are defined in Python's logging module (e.g., logging.DEBUG,
        logging.WARNING).

    log_file : str, optional
        Path to the log file where messages will be written. If `None`, messages
        are only logged to the console. Default is None.
    
    stream_handler : bool, optional
        Flag to control the addition of a StreamHandler for console output.
        Default is True.

    Returns
    -------
    logging.Logger
        A configured logger instance ready for use.

    Examples
    --------
    >>> logger = setup_logger('my_module', log_file='my_log.log', level=logging.DEBUG)
    >>> logger.debug('This is a debug message')
    
    The above example will log the debug message both to the console and to a file named 'my_log.log'.
    """

    # Initialize logger and set logging level
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = True

    # Add StreamHandler for console logging, if enabled
    if stream_handler:
        stream_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s', datefmt='%H:%M:%S')
        stream_handle = logging.StreamHandler()
        stream_handle.setLevel(level)
        stream_handle.setFormatter(stream_formatter)
        logger.addHandler(stream_handle)

    # Add FileHandler for file logging, if log_file is specified
    if log_file:
        file_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s')
        file_handle = logging.FileHandler(log_file, mode='a')
        file_handle.setLevel(level)
        file_handle.setFormatter(file_formatter)
        logger.addHandler(file_handle)

    return logger

def close_logger(logger):
    """
    Safely closes and removes all handlers associated with a logger instance.

    This function should be called when you no longer need the logger or before re-configuring
    an existing logger. It is particularly useful for ensuring that FileHandlers release
    their file resources.

    Parameters
    ----------
    logger : logging.Logger
        The logger instance to close.

    Examples
    --------
    >>> logger = logging.getLogger('my_module')
    >>> close_logger(logger)
    
    This will close all handlers, releasing any resources used, and remove them from the 'my_module' logger.
    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
