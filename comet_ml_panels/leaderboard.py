"""Creates the Leaderboard for Comet ML Panels

This script queries from a given Comet ML project a DataFrame of
model metrics at each step for each model in the project
Then displays the top models.

Note that updating this file does not update comet-ml. Please
go into the project to update after pushing to GitHub.

Example:
    This is not intended to be run locally. Please test on Comet-ML.

For Developers:
    For more on adding to this see docs at
    https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/

"""

from comet_ml import API, APIExperiment, ui


def get_max_metric(df, metric_col="metric"):
    # Doing a simple groupby max removes extra useful metadata
    # For example
    # We may want to know the exact step we had the best score
    # But a max groupby will only show the last step at the end
    index = df[metric_col].argmax()
    return df.iloc[index]


# Initialize Comet API
api = API()

# Get available metrics and select one
available_metrics = ["train/valid_cMAP", "train/valid_ROCAUC"]
selected_metric = ui.dropdown("Select a metric:", available_metrics)

# Fetch experiment data
experiment_keys = api.get_panel_experiment_keys()
if experiment_keys and selected_metric:
    # Fetch the selected metric data for all experiments
    metrics_df = api.get_metrics_df(experiment_keys, [selected_metric])

    # Create Leaderboard View
    leaderboard_df = metrics_df.groupby("experiment_key").apply(
        lambda df: get_max_metric(df, selected_metric)
    ).sort_values(by=selected_metric, ascending=False).reset_index(drop=True)

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
else:
    ui.display(
        "No data to plot. Make sure your metric data is logged by step."
    )
