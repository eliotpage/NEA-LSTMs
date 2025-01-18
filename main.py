import numpy as np
from functions.initializer import init as initialize_weights
from functions.gates import *


#CONSTANTS
INPUT_SIZE = 5
HIDDEN_SIZE = 3

#INITIAL VALUES
hidden_state = np.zeros((HIDDEN_SIZE, 1))  # shape (3, 1)
cell_state = np.zeros((HIDDEN_SIZE, 1))
weights = initialize_weights(INPUT_SIZE, HIDDEN_SIZE)


#RUN ONE FOWARD PASS THROUGH EVERY GATE
def foward_pass(input, hidden_state, cell_state, debug=False):

    #'WEIGHTS' CONTAINS (FORGET GATE WEIGHTS AND BIASES, INPUT CANDIDATE, INPUT GATE, OUTPUT GATE) IN THAT ORDER
    #SO FORGET GATE USES FIRST, INPUT USES 2ND AND 3RD AND OUTPUT USES 4TH

    forget_gate_weights = weights['forget_gate_weights']
    input_gate_weights = [weights['input_gate_candidate_weights'], weights['input_gate_weights']]
    output_gate_weights = weights['output_gate_weights']

    input = np.array([input[0], input[1], input[2], input[3], input[4]])


    if debug:
        print(f'initial cell state: {cell_state}')
        print(f'initial hidden state: {hidden_state}')
        print(f'input: {input}')

    #UPDATE CELL STATE USING FORGET GATE
    if debug:
        print(f'\nFt: {forget_gate(forget_gate_weights, input, hidden_state)}')

    cell_state *= forget_gate(forget_gate_weights, input, hidden_state, debug=debug)

    if debug:
        print(f'cell state after forget gate: {cell_state}')
        print(f'hidden state after forget gate: {hidden_state}')


    #UPDATE CELL STATE USING INPUT GATE
    if debug:
        print(f'\nIt: {input_gate((input_gate_weights, weights[2]), input, hidden_state)}')

    cell_state += input_gate(input_gate_weights, input, hidden_state, debug=debug)

    if debug:
        print(f'cell state after input gate: {cell_state}')
        print(f'hidden state after input gate: {hidden_state}')


    #UPDATE HIDDEN STATE USING OUTPUT GATE
    if debug:
        print(f'\nOt: {output_gate(output_gate_weights, input, hidden_state, cell_state)}')

    hidden_state = output_gate(output_gate_weights, input, hidden_state, cell_state, debug=debug)

    if debug:
        print(f'cell state after output gate: {cell_state}')
        print(f'hidden state after output gate: {hidden_state}')

    return hidden_state, cell_state


if __name__ == '__main__': 
    foward_pass(1, hidden_state, cell_state, debug=True)
    print(hidden_state, cell_state)