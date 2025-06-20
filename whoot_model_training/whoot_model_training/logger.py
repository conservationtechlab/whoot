import comet_ml

class CometMLLoggerSupplement():
    """Note, that is working with the Trainer!

    The Trainer class implements their own CometML Callback during training
    See https://github.com/huggingface/transformers/blob/2166b6b4ff09f6dd3867ab982f262f66482aa968/src/transformers/integrations/integration_utils.py#L1031
    This handles a lot but NOT ALL of the logging we want

    This class handles the last 10% of the logging we want such as
    - Better dataset hashing
    - git hash saving
    - etc
    """

    def __init__(self, dataset_info, githash, ):
        comet_ml.login()
        self.experiment = comet_ml.start()
        print("experiment key", self.experiment.id)

        #TODO add these logs to comet_ml
        #TODO Check to make sure training doesn't create a new experiment