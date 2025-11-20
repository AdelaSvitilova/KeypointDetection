from abc import ABC, abstractmethod

class BaseDataset(ABC):
    """
    Abstraktní třída pro všechny dataset třídy (framework-agnostic).
    """
    @abstractmethod
    def __len__(self):
        """
        Počet vzorků v datasetu.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """
        Vrátí jeden vzorek (obrázek, keypoints) jako NumPy array.
        """
        pass

    def transform(self, image, keypoints):
        """
        Volitelná metoda pro augmentaci nebo transformaci dat.
        """
        return image, keypoints
