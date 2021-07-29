import sys
from loguru import logger
import pendulum


def _set_datetime(record):
    record["extra"]["datetime"] = pendulum.now("Asia/Shanghai").strftime('%Y-%m-%d %H:%M:%S')


def get_logger():
    logger.remove()
    logger.configure(patcher=_set_datetime)
    logger.add(sys.stderr, format="<green>{extra[datetime]}</green> - <level>{message}</level>")
    return logger