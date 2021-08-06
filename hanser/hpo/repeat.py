import time
from typing import Optional, Tuple, Type, Union, Callable
import multiprocessing
from hanser.hpo.common import get_logger
logger = get_logger()

import tensorflow as tf


def get_tpu_errors():
    return (
        tf.errors.UnavailableError,
    )


def run_trial(
    train_fn: Callable,
    catch: Tuple[Type[Exception], ...] = (),
    i: int = 0,
):
    try:
        train_fn(i)
    except Exception as e:
        func_err = e
        logger.warning(
            "Experiment failed because of the following error: {}".format(repr(func_err)))
        if not isinstance(func_err, catch):
            raise e


def repeat(
    train_fn: Callable,
    times: int,
    catch: Tuple[Type[Exception], ...] = get_tpu_errors(),
    timeout: int = None,
):
    i = 0
    while True:
        p = multiprocessing.Process(target=run_trial, args=(train_fn, catch, i))
        p.start()
        logger.info("Experiment {} started".format(i + 1))
        p.join(timeout=timeout)
        if p.is_alive():
            logger.warning("Maybe connection timeout in TPU")
            while p.is_alive():
                p.kill()
                time.sleep(1)
        exitcode = p.exitcode
        p.close()

        if exitcode == 0:
            i += 1
            if times is not None and i >= times:
                break
        else:
            break