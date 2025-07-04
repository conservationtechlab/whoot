""" Metrics for Bioacoustic Mutlilabel Models

Helps us evaluate which models do well

These the metrics with HF Trainer and are called
as part of a callback during training
"""

import comet_ml
import torch
from sklearn.metrics import confusion_matrix

# from torchmetrics.classification import (
#     MultilabelAveragePrecision,
#     MultilabelAUROC,
# )

from pyha_analyzer.metrics.classification_metrics  \
    import AudioClassificationMetrics


class WhootMutliClassMetrics(AudioClassificationMetrics):
    """Gets CMAP, ROCAUC, and confusion matrices and reports them to
    Comet-ML dashboards
    """
    def __init__(self, classes: list):
        self.classes = classes
        self.training = True
        super().__init__([], len(classes), mutlilabel=True)

    def __call__(self, eval_pred) -> dict[str, float]:
        # CMAP / ROCAUC
        initial_metrics = super().__call__(eval_pred=eval_pred)

        # Confusion Matrix
        self.log_comet_ml_only(eval_pred)

        # Return the metrics that can be logged to console AND comet-ml
        return initial_metrics

    def log_comet_ml_only(self, eval_pred):
        """Logs confusion matrix each eval step
        """
        # For metrics that are not loggable to console
        # We can only have comet_ml for these metrics
        experiment = comet_ml.get_running_experiment()
        if experiment is None:
            return
        logits = torch.Tensor(eval_pred.predictions)
        target = torch.Tensor(eval_pred.label_ids).to(torch.long)

        # Confusion Matrix WARNING, ONLY USE IF DATA IS MOSTLY MUTLICLASS
        cm = confusion_matrix(
            torch.argmax(target, dim=1),
            torch.argmax(logits, dim=1)
        )
        experiment.log_confusion_matrix(
            matrix=cm.tolist(), labels=self.classes)
