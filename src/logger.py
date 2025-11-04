import logging, os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"log_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# # Basic Logging:
# logging.basicConfig(
#     level=logging.INFO,
#     format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(LOG_FILE_PATH), 
#         logging.StreamHandler()
#     ]
# )

"""Final Logging Setup:

- Use only INFO, WARNING, ERROR levels.

- INFO logs: go to log file only (routine progress, debug info).

- WARNING and ERROR logs: go to both terminal and log file (critical, operational, and user-visible warnings/errors).

- Print statements: for transient UI feedback (thinking..., accessing..., generating output...) that aren't important for audit.
"""
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d)  - %(message)s"
))

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
stream_handler.setFormatter(logging.Formatter(
    "%(levelname)-8s | %(message)s"
))

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Suppressing Transformer Logs except ERROR
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_logger(name):
    return logging.getLogger(name)
