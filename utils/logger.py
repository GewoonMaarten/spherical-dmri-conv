
"""
Logging levels:
CRITICAL    50
ERROR       40
WARNING     30
INFO        20
DEBUG       10
NOTSET      0
"""

import logging
import sys

from tqdm import tqdm

from .env import LOGGING_LEVEL

LOGGER_NAME = 'geometric-dl'


class logging_tqdm(tqdm):
    def __init__(
            self,
            *args,
            logger: logging.Logger = None,
            mininterval: float = 1,
            bar_format: str = '{desc}{percentage:3.0f}%{r_bar}',
            desc: str = 'progress: ',
            **kwargs):
        self._logger = logger
        super().__init__(
            *args,
            mininterval=mininterval,
            bar_format=bar_format,
            desc=desc,
            **kwargs
        )

    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        return logging.getLogger(LOGGER_NAME)

    def display(self, msg=None, pos=None):
        if not self.n:
            # skip progress bar before having processed anything
            return
        if not msg:
            msg = self.__str__()
        self.logger.info('%s', msg)


class ColorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(name, log_level = 30):
    """Create a logger and add a colored formatter if not added already"""

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    if not logger.hasHandlers():
        stream = logging.StreamHandler(stream=sys.stdout)
        stream.setLevel(log_level)
        stream.setFormatter(ColorFormatter())

        logger.addHandler(stream)


init_logger(LOGGER_NAME, LOGGING_LEVEL)
logger = logging.getLogger(LOGGER_NAME)
