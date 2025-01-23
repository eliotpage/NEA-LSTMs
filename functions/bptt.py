def loss_mse(predicted, target):
    predicted, target = predicted.reshape(-1, 1), target.reshape(-1, 1)
    return (0.5 * (predicted - target)**2)

def find_grad(loss, predicted):
    loss_pred_grad = predicted - loss
    return loss_pred_grad

def sig(x):
    import numpy as np
    return 1/(1 + np.exp(-x))

def tanh(x):
    import numpy as np
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def sig_deriv(x):
    return sig(x) * (1 - sig(x))

def tanh_deriv(x):
    return 1 - tanh(x)**2