from abc import ABC, abstractmethod

class BaseMetric(ABC):
    """
    Abstract base class for all metric classes (framework-agnostic).

    Subclasses must implement the `update` and `compute` methods to track and
    calculate metrics over predictions and targets in a framework-independent way.
    The `reset` method can be optionally overridden to clear internal state.
    """

    @abstractmethod
    def update(self, preds, targets):
        """
        Update the internal state with new predictions and targets.

        Parameters:
            preds: Predicted outputs (any array-like type, framework-agnostic)
            targets: Ground truth outputs (same type as preds)
        """
        pass

    @abstractmethod
    def compute(self):
        """
        Compute the metric based on the accumulated state.

        Returns:
            Metric value (framework-agnostic, e.g., float or array-like)
        """
        pass

    def reset(self):
        """
        Reset the internal state of the metric.

        Can be overridden by subclasses. By default, does nothing.
        """
        pass
