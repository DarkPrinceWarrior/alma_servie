import sys
from loguru import logger

# Configure default logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# You can add file sinks here if needed, e.g.
# logger.add("pipeline.log", rotation="10 MB")

__all__ = ["logger"]
