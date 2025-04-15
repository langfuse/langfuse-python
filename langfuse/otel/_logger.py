import logging

logger = logging.getLogger("langfuse")

# handle httpx logging
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)  # Set the desired log level

console_handler = logging.StreamHandler()
httpx_logger.addHandler(console_handler)
