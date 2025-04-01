import logging

# Create a logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler("logs/my_log.log")
file_handler.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Example usage of the logger
# logger.debug("This is a debug message")
# logger.info("This is an info message")
# logger.warning("This is a warning message")
# logger.error("This is an error message")
# logger.critical("This is a critical message")


# Create a logger
logger2 = logging.getLogger("my_logger2")
logger2.setLevel(logging.DEBUG)

# Create a file handler
file_handler2 = logging.FileHandler("logs/my_log_lineaer.log")
file_handler2.setLevel(logging.DEBUG)

# Create a formatter
file_handler2.setFormatter(formatter)

logger2.addHandler(console_handler)
logger2.addHandler(file_handler2)


def get_logger(name):
    if name == "my_logger2":
        return logger2
    else:
        return logger
