from src.logger import get_logger

logger = get_logger(__name__)

def log_and_notify_exception(e, message="ChatBot encountered a problem. Please check logs for details."):
    logger.error(f"{message} | Exception Details: {e}")
