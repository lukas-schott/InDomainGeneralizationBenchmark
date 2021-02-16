from sklearn.metrics import r2_score
import numpy as np


def r2(ground_truths, predictions):
    """Returns the r2 score"""
    return r2_score(ground_truths, predictions)


def evaluate_model(model_fn, dataset, metrics=None):
    """ Returns the benchmark scores of a given model under a particular dataset
    Args:
        model_fn: a function of the model that has an array of images as input and returns the predicted labels
        dataset: a dataset on which the model shall be evaluated
        metrics (dict): a dict of the metrics to be evaluated
    Returns:
        score_dict (dict): a dict with the score for each metric
    """
    if metrics is None:
        metrics = dict()
        metrics['r2_score'] = r2 #add potential frther scores here

    score_dict = dict()
    ground_truths = None
    predictions = None
    for images, factors in dataset:
        batch_prediction = model_fn(images)

        if ground_truths is None and predictions is None:
            ground_truths = factors
            predictions = batch_prediction
        else:
            np.append(ground_truths, factors)
            np.append(predictions, batch_prediction)

    for metric_key in metrics.keys():
        score_dict[metric_key] = metrics[metric_key](ground_truths, predictions)
    return score_dict
