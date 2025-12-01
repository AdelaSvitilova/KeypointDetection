from abc import ABC, abstractmethod

class BaseDataset(ABC):
    """
    Abstract base class for all dataset classes (framework-agnostic).
    """

    @abstractmethod
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """
        Return a single sample (image, keypoints/heatmaps) as a NumPy array.
        """
        pass

    def transform(self, image, keypoints):
        """
        Optional method to apply transformations or augmentations to a sample.
        Returns the original image and keypoints by default.
        """
        return image, keypoints
