"""
Logging module based on python's built-in `logging`
"""

import logging
import logging.config
from pathlib import Path
from lib.utils import read_json


def setup_logging(save_dir, log_config='logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if not log_config.is_absolute():
        log_config = Path(__file__).parent / log_config
    log_config = log_config.resolve()
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)


def get_logger(name, verbosity=2):
    """
    Creates a logger with the given name and verbosity level.

    Please note the following when using loggers:
    a) Do not call .format() on messages you pass to the logger. The logger should
       decide whether the call is required (i.e. a message might get discarded
       due to low verbosity). Instead use them with the following syntax:
       `logger.debug("Message with %s argument", argument)``
    b) If `argument` is expensive to compute and the only purpose for the computation
       is for the logger to potentially output a message then the following is recommended:
       ```
       if logger.isEnabledFor(logging.DEBUG):
           logger.debug("Message with %s argument", expensive_function())
       ```
    More on logging optimization can be found here:
    https://docs.python.org/3/howto/logging.html#optimization
    """
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, log_levels.keys())
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(name)
    logger.setLevel(log_levels[verbosity])
    return logger
