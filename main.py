
import numpy as np
from functions.initializer import init as initialize_weights
from functions.gates import *

#CONSTANTS
INPUT_SIZE = 3
HIDDEN_SIZE = 4

#INITIAL VALUES
hidden_state = 0
cell_state = 0
weights = initialize_weights(INPUT_SIZE, HIDDEN_SIZE)

def foward_pass(input, debug=False):
    global hidden_state, cell_state

    if debug:
        print(f'initial cell state: {cell_state}')
        print(f'initial hidden state: {hidden_state}')
        print(f'\nFt: {forget_gate(weights, 1, hidden_state)}')

    cell_state *= forget_gate(weights, 1, hidden_state)
    if debug:
        print(f'cell state after forget gate: {cell_state}')
        print(f'hidden state after forget gate: {hidden_state}')
        print(f'\nIt: {input_gate(weights, 1, hidden_state)}')

    cell_state += input_gate(weights, 1, hidden_state)
    if debug:
        print(f'cell state after input gate: {cell_state}')
        print(f'hidden state after input gate: {hidden_state}')


if __name__ == '__main__': 
    foward_pass(1, debug=True)