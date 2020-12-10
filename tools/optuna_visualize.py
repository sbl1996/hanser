import optuna

study_name = "no-name-7e431866-43a5-4bd4-958c-c9279a6a3983"
study = optuna.load_study(study_name, f"sqlite:////Users/hrvvi/Downloads/example.db")
optuna.visualization.plot_intermediate_values(study)
