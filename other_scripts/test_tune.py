from ray import train, tune
import matplotlib.pyplot as plt
import os

def objective(config, trial):
    score = config["a"] ** 2 + config["b"]

    # Create a plot
    fig, ax = plt.subplots()
    ax.plot([config["a"], config["b"]], [score, score])
    ax.set_title('Score Plot')
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Score')

    # Save the plot
    fig.savefig(os.path.join(trial.logdir, "score_plot.png"))

    return {"score": score}


search_space = {  # ②
    "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
    "b": tune.choice([1, 2, 3]),
}

tuner = tune.Tuner(objective, param_space=search_space)  # ③

results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)