import logging
import logging.handlers
import os

from dotenv import load_dotenv
from pyprojroot import here


def setup_logging() -> logging.Logger:
    """Configures the hf_runner logging"""
    load_dotenv(here() / ".env")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger = logging.getLogger(__name__)
    logging.getLogger("rasterio").setLevel(logging.WARNING)  # turning off rasterio INFO logging

    log_file_path = here() / "logs/hf_builder.log"
    max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))

    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=max_bytes, backupCount=backup_count
    )

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(file_handler)
    return logger
