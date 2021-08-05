"""
Custom logging module to create and store logs
"""

import logging
import sys
from logging import Logger
from logging.handlers import TimedRotatingFileHandler


class CustomLogger(Logger):
    """Creates and stores logs.

    Attributes:
        log_file: File used to store logs.
        log_format: Format of storing logs.
    """

    def __init__(self, log_file: str = None,
                 log_format: str = "%(asctime)s - %(name)s - \
                                    %(levelname)s - %(message)s",
                 *args,
                 **kwargs) -> None:

        self.formatter = logging.Formatter(log_format)
        self.log_file = log_file

        Logger.__init__(self, *args, **kwargs)

        self.addHandler(self.get_console_handler())
        if log_file:
            self.addHandler(self.get_file_handler())

        # With this pattern, it's rarely necessary to propagate
        # the| error up to parent
        self.propagate = False

    def get_console_handler(self) -> logging.StreamHandler:
        """
        Format logs on the console.
        """

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)

        return console_handler

    def get_file_handler(self) -> logging.handlers.TimedRotatingFileHandler:
        """
        Format logs on the file.
        """

        file_handler = TimedRotatingFileHandler(self.log_file, when="midnight")
        file_handler.setFormatter(self.formatter)

        return file_handler
