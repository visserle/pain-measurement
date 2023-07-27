# work in proqress

# TODO
# - find out where and how to use close_logger
# - add file logger
# - add per class logger? 
#    -> logger = logging.getLogger(__name__+"."+__class__.__name__)
# - test with thermoino
# - add to all other files
# - add doc


import logging

def setup_logger(logger_name, log_file = None, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # # Create a file handler and set the level and formatter
    # file_handler = logging.FileHandler(log_file, mode='w')
    # file_handler.setLevel(level)
    # file_handler.setFormatter(formatter)

    # Create a stream handler (console handler) and set the level and formatter
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    # Add the handlers to the logger
    logger.addHandler(stream_handler)
    # logger.addHandler(file_handler)

    return logger
    
def close_logger(logger):
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

