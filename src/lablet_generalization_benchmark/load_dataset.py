import torch
from torch.utils.data import Dataset
from typing import List
import numpy as np
import os
import logging
from sklearn.utils.extmath import cartesian

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class IndexManger(object):
    """Index mapping from features to positions of state space atoms."""

    def __init__(self, factor_sizes: List[int]):
        """Index to latent (= features) space and vice versa.
        Args:
          factor_sizes: List of integers with the number of distinct values for each
            of the factors.
        """
        self.factor_sizes = np.array(factor_sizes)
        self.num_total = np.prod(self.factor_sizes)
        self.factor_bases = self.num_total / np.cumprod(self.factor_sizes)

        self.index_to_feat = cartesian(
            [np.array(list(range(i))) for i in self.factor_sizes])

    def features_to_index(self, features):
        """Returns the indices in the input space for given factor configurations.
        Args:
          features: Numpy matrix where each row contains a different factor
            configuration for which the indices in the input space should be
            returned.
        """
        assert np.all((0 <= features) & (features <= self.factor_sizes))
        index = np.array(np.dot(features, self.factor_bases), dtype=np.int64)
        assert np.all((0 <= index) & (index < self.num_total))
        return index

    def index_to_features(self, index: int) -> np.ndarray:
        assert np.all((0 <= index) & (index < self.num_total))
        features = self.index_to_feat[index]
        assert np.all((0 <= features) & (features <= self.factor_sizes))
        return features


class BenchmarkDataset(Dataset):

    def __init__(self, dataset_name, variant, mode, dir=None):
        DATASET_PATH = "dataset_splits"
        if dir is not None:
            DATASET_PATH = os.path.join(dir, DATASET_PATH)
        super().__init__()
        images_filename = "{}_{}_{}_images.npz".format(dataset_name, variant, mode)
        labels_filename = "{}_{}_{}_labels.npz".format(dataset_name, variant, mode)

        self._factor_sizes = None
        self._factor_names = None
        if dataset_name == 'dsprites':
            self._factor_sizes = [3, 6, 40, 32, 32]
            self._factor_names = ['shape', 'scale', 'orientation', 'x-position',
                            'y-position']
        elif dataset_name == 'shapes3d':
            self._factor_sizes = [10, 10, 10, 8, 4, 15]
            self._factor_names = [
            'floor color', 'wall color', 'object color', 'object size',
            'object type', 'azimuth']
        elif dataset_name == 'mpi3d':
            self._factor_sizes = [6, 6, 2, 3, 3, 40, 40]
            self._factor_names = ['color', 'shape', 'size', 'height', 'bg color',
                        'x-axis', 'y-axis']

        self._index_manager = IndexManger(self._factor_sizes)

        images_path = os.path.join(DATASET_PATH, images_filename)
        labels_path = os.path.join(DATASET_PATH, labels_filename)
        if os.path.exists(images_path) and os.path.exists(labels_path):
            self._dataset_images = np.load(images_path,
                                           encoding='latin1',
                                           allow_pickle=True)['arr_0']
            self._dataset_labels = np.load(labels_path,
                                           encoding='latin1',
                                           allow_pickle=True)['arr_0']
        else:
            if not os.path.exists(DATASET_PATH):
                os.makedirs(DATASET_PATH)
            url = None  # TODO Implement
            raise NotImplementedError('Downloading splits not yet implemented')

    def __len__(self):
        return len(self._dataset_labels)

    def get_normalized_labels(self):
        return self._labels / (np.array(self._factor_sizes) - 1)

    @property
    def _labels(self):
        return self._index_manager.index_to_feat

    def __getitem__(self, idx: int):
        image = self._dataset_images[idx]
        labels = self._dataset_labels[idx]

        sample = {'image': image, 'labels': labels}
        return sample


def load_dataset(dataset_name='shapes3d', variant='random', mode='train', dir=None, batch_size=4, num_workers=0):
    """ Returns a torch dataset loader for the requested split
    Args:
        dataset_name (str): the dataset name, can be either 'shapes3d, 'dsprites' or 'mpi3d'
        variant (str): the split variant, can be either 'none', 'random', 'composition',
        'interpolation', 'extrapolation'
        mode (str): mode, can be either 'train' or 'test', default is 'train'
        batch_size (int): batch_size, default is 4
        num_workers (int): num_workers, default = 1
    Returns:
        dataset
    """

    dataset = BenchmarkDataset(dataset_name, variant, mode, dir)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)
    return data_loader
