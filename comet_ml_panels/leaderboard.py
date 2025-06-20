# Comet Python Panels BETA, full documentation available at:
# https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/
# Code from original python template
# Modified by Sean Perry, 6/202/2025
# TODO: FIGURE OUT HOW TO VERSION CONTROL THIS...


from comet_ml import API, APIExperiment, ui
import pandas as pd
# import plotly.express as px

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

    col_order = ["experiment_name", selected_metric, "experiment_key", "step", "users"]


    
    #api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')
    #print(api_experiment.get_user())
    #
    
    ui.display(leaderboard_df[col_order])
else:
    ui.display("No data to plot. Make sure your metric data is logged by step.")
