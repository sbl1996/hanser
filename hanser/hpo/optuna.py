import tensorflow as tf
from typing import Optional, Tuple, Type
from termcolor import colored
import multiprocessing

import optuna

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
    study_fn,
    objective,
    catch: Tuple[Type[Exception], ...] = (),
):
    study = study_fn()
    trial = study.ask()
    try:
        score = objective(trial)
        study.tell(trial, score)
        info(
            "Trial {} finished with value: {} and parameters: {}. "
            "Best is trial {} with value: {}.".format(
                trial.number,
                score,
                trial.params,
                study.best_trial.number,
                study.best_value,
            )
        )
    except optuna.TrialPruned:
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        info("Trial {} pruned.".format(trial.number))
    except Exception as e:
        func_err = e
        warn(
            "Trial {} failed because of the following error: {}".format(
                trial.number, repr(func_err)))
        if not isinstance(func_err, catch):
            raise e


def optimize_mp(
    study_fn: optuna.Study,
    objective: optuna.study.ObjectiveFuncType,
    n_trials: Optional[int] = None,
    catch: Tuple[Type[Exception], ...] = (),
):
    i_trial = 0
    while True:
        p = multiprocessing.Process(target=run_trial, args=(study_fn, objective, catch))
        p.start()
        p.join()
        exitcode = p.exitcode
        p.close()

        if exitcode == 0:
            i_trial += 1
            if n_trials is not None and i_trial >= n_trials:
                break
        else:
            break
