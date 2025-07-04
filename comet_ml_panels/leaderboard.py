"""Creates the Leaderboard for Comet ML Panels

This script queries from a given Comet ML project a DataFrame of
model metrics at each step for each model in the project
Then displays the top models.

Example:
    This is not intended to be run locally. Please test on Comet-ML.

For Developers:
    For more on adding to this see docs at
    https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/

    Note that updating this file does not update comet-ml. Please
    go into the project to update after pushing to GitHub.

    Do not include Doc string in comet-ml... for some reason this
    is displayed in the comet-ml panel if copied directly
"""
from comet_ml import API, APIExperiment, ui
import pandas as pd
import numpy as np

# Initialize Comet API
api = API()

# Select the experiments and metrics to compare
available_metrics = ["train/valid_cMAP", "train/valid_ROCAUC"]
selected_metric = ui.dropdown("Select a metric:", available_metrics)

experiment_keys = api.get_panel_experiment_keys()
data = api.get_metrics_for_chart(
    experiment_keys, metrics=[selected_metric], parameters=["task"])

# Given all experiments, find all possible tasks to measure!
available_tasks = list(
    set(data[key]["params"]["task"]
        for key in data if "task" in data[key]["params"])
)
available_tasks.append(None)
selected_task = ui.dropdown("Select a Task:", available_tasks)

processed_data = []

for key in data:
    # Note, some of the early runs have no value for the task
    # The following code handles those cases
    TASK = None
    if "task" in data[key]["params"]:
        TASK = data[key]["params"]["task"]

    # Only display the leaderboard for tasks we want
    # This CAN include runs with no task
    if TASK is not selected_task and TASK != selected_task:
        continue

    # Failed runs may not have metrics
    if len(data[key]["metrics"]) == 0:
        continue

    metric_values = data[key]["metrics"][0]["values"]
    max_index = np.argmax(metric_values)

    processed_data.append({
        "experiment_name": data[key]["experimentName"],
        "experiment_key": key,
        selected_metric: max(metric_values),
        "step": data[key]["metrics"][0]["steps"][max_index],
    })

leaderboard_df = pd.DataFrame(processed_data).sort_values(
    selected_metric, ascending=False)

leaderboard_df["users"] = leaderboard_df["experiment_key"].apply(
    lambda key: APIExperiment(previous_experiment=key).get_user()
)

col_order = [
    "experiment_name",
    selected_metric,
    "experiment_key",
    "step",
    "users"
]
ui.display(leaderboard_df[col_order])
