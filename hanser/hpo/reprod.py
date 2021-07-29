import time
from typing import Optional, Tuple, Type, Union, Callable
from termcolor import colored
import multiprocessing

import tensorflow as tf

from hhutil.datetime import datetime_now


def get_tpu_errors():
    return (
        tf.errors.UnavailableError,
    )


def _time_now():
    dt = datetime_now()
    dt = dt.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
    return dt


def info(msg):
    print(colored(f"[I {_time_now()}]", "green") + " " + msg)


def warn(msg):
    print(colored(f"[I {_time_now()}]", "red") + " " + msg)


def run_trial(
    train_fn: Callable,
    catch: Tuple[Type[Exception], ...] = (),
):
    try:
        train_fn()
    except Exception as e:
        func_err = e
        warn(
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
        p = multiprocessing.Process(target=run_trial, args=(train_fn, catch))
        p.start()
        print("Experiment {} started".format(i + 1))
        p.join(timeout=timeout)
        if p.is_alive():
            warn("Maybe connection timeout in TPU")
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