import os
import random
import numpy as np

def set_global_seed(seed_to_set):
    SEED = seed_to_set
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(SEED)
    np.random.seed(SEED)

    # PyTorch
    try:
        import torch
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
    except ImportError:
        pass

def seed_worker(worker_id):
    worker_seed = SEED + worker_id

    import random
    import numpy as np
    random.seed(worker_seed)
    np.random.seed(worker_seed)