def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch. Called inside main() blocks only (rule C11)."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
