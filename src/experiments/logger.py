# work in proqress

# TODO
# - find out exactly where/when and how to use close_logger (not in del, it's tricky)
# - find out what happens when the experiment gets aborted - logs still get written? closed?
# - add doc


import logging

def setup_logger(logger_name, log_file = None, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Create a stream handler (console handler)
    stream_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    if log_file is None:
        return logger
    
    # Create a file handler if log_file is specified
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file, mode='w') # mode='w' for overwriting, mode='a' for appending
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    return logger
    
def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
