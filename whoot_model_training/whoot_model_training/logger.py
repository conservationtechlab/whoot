""" Contains useful tools for additional logging

For example, CometMLLoggerSupplement adds additional
logging for data augmentations used compared
to the base logging done by the HF trainer
integration
"""

import comet_ml


# pylint disable-next=R0903
class CometMLLoggerSupplement():
    """Note, that is working with the Trainer!

    The Trainer class implements their own CometML Callback during training
    This handles a lot but NOT ALL of the logging we want

    This class handles the last 10% of the logging we want such as
    - Better dataset hashing
    - git hash saving
    - etc
    """

    def __init__(self, augmentations, name):
        comet_ml.login()
        self.start(augmentations, name)

    def start(self, augmentations, name):
        """Begins a new set of experiments

        Helpful for cases where a new run has begun
        """
        self.experiment = comet_ml.start()

        self.experiment.log_parameter("augmentations", augmentations)
        self.experiment.set_name(name)

    def end(self):
        """Fully ends experiment if still running
        """
        return self.experiment.end()
    
    def log_task(self, task_name):
        """Log what task this model should be listed under
        """
        self.experiment.log_parameter("task", task_name)

