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


#RUN ONE FOWARD PASS THROUGH EVERY GATE
def foward_pass(input, debug=False):
    global hidden_state, cell_state

    #'WEIGHTS' CONTAINS (FORGET GATE WEIGHTS AND BIASES, INPUT CANDIDATE, INPUT GATE, OUTPUT GATE) IN THAT ORDER
    #SO FORGET GATE USES FIRST, INPUT USES 2ND AND 3RD AND OUTPUT USES 4TH

    if debug:
        print(f'initial cell state: {cell_state}')
        print(f'initial hidden state: {hidden_state}')
        print(f'input: {input}')


    #UPDATE CELL STATE USING FORGET GATE
    if debug:
        print(f'\nFt: {forget_gate(weights[0], 1, hidden_state)}')

    cell_state *= forget_gate(weights[0], 1, hidden_state)

    if debug:
        print(f'cell state after forget gate: {cell_state}')
        print(f'hidden state after forget gate: {hidden_state}')


    #UPDATE CELL STATE USING INPUT GATE
    if debug:
        print(f'\nIt: {input_gate((weights[1], weights[2]), 1, hidden_state)}')

    cell_state += input_gate((weights[1], weights[2]), 1, hidden_state)

    if debug:
        print(f'cell state after input gate: {cell_state}')
        print(f'hidden state after input gate: {hidden_state}')


    #UPDATE HIDDEN STATE USING OUTPUT GATE
    if debug:
        print(f'\nOt: {output_gate(weights[3], 1, hidden_state, cell_state)}')

    hidden_state = output_gate(weights[3], 1, hidden_state, cell_state)

    if debug:
        print(f'cell state after output gate: {cell_state}')
        print(f'hidden state after output gate: {hidden_state}')

    return hidden_state, cell_state


if __name__ == '__main__': 
    foward_pass(1)
    print(hidden_state, cell_state)
    foward_pass(2)
    print(hidden_state, cell_state)
    