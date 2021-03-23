import numpy as np
import sklearn.metrics


class RSquared:
    def __init__(self, normalized_labels: np.ndarray):
        variance_per_factor = (
            (normalized_labels -
             normalized_labels.mean(axis=0, keepdims=True))**2).mean(axis=0)
        self.variance_per_factor = variance_per_factor

    def __call__(self, predictions, targets):
        assert predictions.shape == targets.shape
        mse_loss_per_factor = np.mean(np.power(predictions - targets, 2),
                                      axis=0)
        return 1 - mse_loss_per_factor / self.variance_per_factor


def r2(ground_truths, predictions):
    """Returns the r2 score"""
    assert ground_truths.shape == predictions.shape
    r2_per_factor = np.zeros((ground_truths.shape[1]))
    for i in range(ground_truths.shape[1]):
        r2_per_factor[i] = sklearn.metrics.r2_score(ground_truths[:, i], predictions[:, i])
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
    targets = None
    predictions = None
    for batch in dataset_loader:
        images, labels = batch['image'], batch['labels'].numpy()
        batch_prediction = model_fn(images)

        if targets is None and predictions is None:
            targets = labels
            predictions = batch_prediction
        else:
            targets = np.append(targets, labels, axis=0)
            predictions = np.append(predictions, batch_prediction, axis=0)

    labels_01 = dataset_loader.dataset.get_normalized_labels()
    r_squared = RSquared(labels_01)

    squared_diff = np.power(targets - predictions, 2)
    r_squared_per_factor = r_squared(predictions, targets)

    # book keeping
    log = {
        'rsquared': r_squared_per_factor,
        'mse': np.mean(squared_diff, axis=0)
    }
    factor_names = dataset_loader.dataset._factor_names
    for factor_index in range(r_squared_per_factor.shape[0]):
        score_dict['rsquared_{}'.format(
            factor_names[factor_index])] = log['rsquared'][factor_index]
        score_dict['mse_{}'.format(
            factor_names[factor_index])] = log['mse'][factor_index]
    return score_dict
