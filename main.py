
import numpy as np
from functions.weight_initializer import init as initialize_weights
from gates.forget_gate import forget_gate

#CONSTANTS
INPUT_SIZE = np.array([5])
HIDDEN_SIZE = np.array([12])


if __name__ == '__main__': 
    weights = initialize_weights(INPUT_SIZE, HIDDEN_SIZE)
    print(forget_gate(weights, 0, 0))