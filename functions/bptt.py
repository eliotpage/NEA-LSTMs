def loss_mse(predicted, target):
    import numpy as np
    return np.mean((predicted-target)**2)