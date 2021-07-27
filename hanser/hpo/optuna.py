import optuna

from hhutil.datetime import datetime_now
from termcolor import colored

def info(msg):
    dt = datetime_now(format=True)
    print(colored(f"[I {dt}]", "green") + " " + msg)


def run_trial(study_fn, objective):
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