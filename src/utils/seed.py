import os
import random
import numpy as np
import optuna


def set_global_seed(seed_to_set):
    """
    Set global seed for reproducibility across different libraries.

    Args:
        seed_to_set (int): Seed value used for all random generators.
    """
    SEED = seed_to_set

    # Ensure reproducible hashing behavior in Python
    os.environ["PYTHONHASHSEED"] = str(SEED)

    # Python and NumPy seeds
    random.seed(SEED)
    np.random.seed(SEED)

    # Optuna sampler seed (for reproducible hyperparameter search)
    sampler = optuna.samplers.TPESampler(seed=SEED)

    # PyTorch reproducibility settings (if available)
    try:
        import torch
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # TensorFlow reproducibility settings (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
    except ImportError:
        pass


def seed_worker(worker_id):
    """
    Set seed for DataLoader worker processes.

    Args:
        worker_id (int): ID of the worker process.
    """
    worker_seed = SEED + worker_id  # Ensure unique but deterministic seed per worker

    import random
    import numpy as np

    # Re-seed each worker to ensure deterministic data loading
    random.seed(worker_seed)
    np.random.seed(worker_seed)