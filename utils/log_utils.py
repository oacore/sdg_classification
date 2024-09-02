"""
Utilities for logging
"""

import os
import json
import logging
from typing import NewType
from logging import handlers
from logging.config import dictConfig

LogLevel = NewType('LogLevel', int)


class MakeFileHandler(handlers.TimedRotatingFileHandler):
    """
    Creates the log directory in case it doesn't exist.
    """

    def __init__(
            self, filename, when='h', interval=1, backupCount=0, encoding=None,
            delay=False, utc=False, atTime=None
    ):
        """
        :param filename:
        :param when:
        :param interval:
        :param backupCount:
        :param encoding:
        :param delay:
        :param utc:
        :param atTime:
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        super(MakeFileHandler, self).__init__(
            filename, when, interval, backupCount, encoding, delay, utc, atTime
        )


class LogUtils(object):

    @staticmethod
    def setup_logging(
            logging_config_path: str = 'logging.json',
            logging_level: LogLevel = logging.DEBUG,
            log_file_path: str = 'repo.log'
    ) -> None:
        """
        Setup logging configuration
        :param logging_config_path: path to the logging configuration file
        :param logging_level: which message levels should be enabled
        :param log_file_path: path to the log file
        :return: None
        """
        # Basic configuration
        logging.basicConfig(filename=log_file_path, level=logging_level)

        # Load logging configuration from file if available
        if os.path.exists(logging_config_path):
            with open(logging_config_path, 'rt') as f:
                logging_config = json.load(f)
            dictConfig(logging_config)