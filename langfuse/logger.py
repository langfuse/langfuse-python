import logging
import sys


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a console handler and set its level to INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Create a formatter and set the format for log messages
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


# Use this logger throughout your library
logger = setup_logger()
