from sklearn.metrics import r2_score
import numpy as np


def r2(ground_truths, predictions):
    """Returns the r2 score"""
    assert ground_truths.shape == predictions.shape
    r2_per_factor = np.zeros((ground_truths.shape[1]))
    for i in range(ground_truths.shape[1]):
        r2_per_factor[i] = r2_score(ground_truths[:, i], predictions[:, i])
    return r2_per_factor


def evaluate_model(model_fn, dataset_loader, metrics=None):
    """ Returns the benchmark scores of a given model under a particular dataset
    Args:
        model_fn: a function of the model that has an array of images as input and returns the predicted labels
        dataset_loader: a dataset on which the model shall be evaluated
        metrics (dict): a dict of the metrics to be evaluated
    Returns:
        score_dict (dict): a dict with the score for each metric
    """
    if metrics is None:
        metrics = dict()
        metrics['r2_score'] = r2

    score_dict = dict()
    ground_truths = None
    predictions = None
    for batch in dataset_loader:
        images, labels = batch['image'], batch['labels'].numpy()
        batch_prediction = model_fn(images)

        if ground_truths is None and predictions is None:
            ground_truths = labels
            predictions = batch_prediction
        else:
            ground_truths = np.append(ground_truths, labels, axis=0)
            predictions = np.append(predictions, batch_prediction, axis=0)

    for metric_key in metrics.keys():
        score_dict[metric_key] = metrics[metric_key](ground_truths, predictions)
    return score_dict
