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
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from tqdm import tqdm

LOGGER_NAME = "geometric-dl"
FORMATTER = (
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
)


class logging_tqdm(tqdm):
    def __init__(
        self,
        *args,
        logger: logging.Logger = None,
        mininterval: float = 1,
        bar_format: str = "{desc}{percentage:3.0f}%{r_bar}",
        desc: str = "progress: ",
        **kwargs,
    ):
        self._logger = logger
        super().__init__(
            *args, mininterval=mininterval, bar_format=bar_format, desc=desc, **kwargs
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
        self.logger.info("%s", msg)


class BraceString(str):
    def __mod__(self, other):
        return self.format(*other)

    def __str__(self):
        return self


class StyleAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra=None):
        super(StyleAdapter, self).__init__(logger, extra)

    def process(self, msg, kwargs):
        if kwargs.pop("style", "%") == "{":  # optional
            msg = BraceString(msg)
        return msg, kwargs


class ColorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey + FORMATTER + reset,
        logging.INFO: grey + FORMATTER + reset,
        logging.WARNING: yellow + FORMATTER + reset,
        logging.ERROR: red + FORMATTER + reset,
        logging.CRITICAL: bold_red + FORMATTER + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(name, log_level=30):
    """Create a logger with a stream handler and a file handler"""

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    if not logger.hasHandlers():
        path = Path(Path(__file__).parent.parent, "logs", f"{name}.log")
        path.parent.mkdir(parents=True, exist_ok=True)

        streamHandler = logging.StreamHandler(stream=sys.stdout)
        streamHandler.setLevel(log_level)
        streamHandler.setFormatter(ColorFormatter())

        fileHandler = TimedRotatingFileHandler(path, when="h", interval=1)
        fileHandler.setLevel(log_level)
        formatter = logging.Formatter(FORMATTER)
        fileHandler.setFormatter(formatter)

        logger.addHandler(streamHandler)
        logger.addHandler(fileHandler)


logger = StyleAdapter(logging.getLogger(LOGGER_NAME))
