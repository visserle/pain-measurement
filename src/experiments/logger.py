# work in proqress

# TODO
# - add file logger
#   -> find out exactly where/when and how to use close_logger (not in del, it's tricky)
# - add per class logger? answer: no, always initialize at module level
# - add doc
# - logger.propagate = False needed?


import logging

def setup_logger(logger_name, log_file = None, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Create a stream handler (console handler) and set the level and formatter
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(stream_handler)
    if log_file is None:
        return logger
    
    # Create a file handler and set the level and formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
    
def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    pass
