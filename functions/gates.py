#FORGET GATE DETERMINES HOW MUCH TO FORGET OF THE LONG TERM MEMORY / HOW RELEVANT IT IS USING THE INPUT AND HIDDEN STATE (SHORT TERM MEMORY)
#FORMULA: Ft = sigmoid((Ht-1 * Uf + Xt * Wf) + Bf), where Ht-1 = hidden state, Uf = h-h weight, Wf = i-h weight, and Bf = bias

def forget_gate(weights, input, hidden_state, debug=False):
    import numpy as np
    #UNPACK WEIGHTS AND REFORMAT INPUT AND HIDDEN STATE INTO MATRICES
    Wf, Uf, Bf = weights
    input = np.reshape(input, (-1, 1))
    hidden_state = np.reshape(hidden_state, (-1, 1))
    
    print(hidden_state.shape, Uf.shape, input.shape, Wf.shape, Bf.shape)

    sum = (hidden_state @ Uf) + (input @ Wf) + Bf

    if debug:
        print('FORGET GATE:')
        print(f'weights:{weights}\n\nWf:{Wf}\n\nUf:{Uf}\n\nBf:{Bf}\n\nhidden:{hidden_state}\n\ninput:{input}\n\nformula:{hidden_state} * {Uf} + {input} * {Wf} + {Bf}\n\nsum:{sum}')

    #PLUG VALUE INTO SIGMOID ACTIVATION FUNCTION: f(x) = -1/1+e^-x
    def sig(x):
        return 1/(1 + np.exp(-x))
    Ft = sig(sum)

    return Ft
    

#INPUT GATE CONSISTS OF CANDIDATE CELL, WHICH CALCULATES POTENTIAL NEW LONG TERM MEMORY, AND MAIN INPUT GATE,
#WHICH CALCULATES THE % OF THE CANDIDATE OUTPUT TO ADD TO CELL STATE (LONG TERM MEMORY)
#BOTH USE THE INPUT AND HIDDEN STATE (SHORT TERM MEMORY) TO DO SO
def input_gate_candidate(weights, input, hidden_state, debug=False):
    import numpy as np
    Wf = weights[0][0]
    Uf = weights[0][1]
    Bf = weights[0][2]
    
    sum = hidden_state * Uf + input * Wf + Bf

    if debug:
        print('INPUT CANDIDATE GATE:')
        print(f'weights:{weights}\n\nWf:{Wf}\n\nUf:{Uf}\n\nBf:{Bf}\n\nhidden:{hidden_state}\n\ninput:{input}\n\nformula:{hidden_state} * {Uf} + {input} * {Wf} + {Bf}\n\nsum:{sum}')

    #PLUG VALUE INTO TANH ACTIVATION FUNCTION: f(x) = (e^x-e^-x)/(e^x+e^-x)
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    Ct = tanh(sum)

    return Ct

def input_gate(weights, input, hidden_state, debug=False):
    import numpy as np
    Wf = weights[1][0]
    Uf = weights[1][1]
    Bf = weights[1][2]
    
    sum = hidden_state * Uf + input * Wf + Bf

    if debug:
        print('INPUT GATE:')
        print(f'weights:{weights}\n\nWf:{Wf}\n\nUf:{Uf}\n\nBf:{Bf}\n\nhidden:{hidden_state}\n\ninput:{input}\n\nformula:{hidden_state} * {Uf} + {input} * {Wf} + {Bf}\n\nsum:{sum}')

    #PLUG VALUE INTO SIGMOID ACTIVATION FUNCTION: f(x) = -1/1+e^-x
    def sig(x):
        return 1/(1 + np.exp(-x))
    It = sig(sum)

    Ct = input_gate_candidate(weights, input, hidden_state)

    return It * Ct


#OUTPUT GATE CONSISTS OF CANDIDATE CELL, WHICH CALCULATES POTENTIAL NEW SHORT TERM MEMORY, AND MAIN OUTPUT GATE,
#WHICH CALCULATES THE % OF THE CANDIDATE OUTPUT TO USE AS NEW HIDDEN STATE (SHORT TERM MEMORY)
def output_gate_candidate(cell_state):
    import numpy as np
    #OUTPUT CANDIDATE CELL SIMPLY INPUTS CELL STATE AFTER THE OTHER GATES INTO A TANH ACTIVATION FUNCTION
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    Ct = tanh(cell_state)

    return Ct


def output_gate(weights, input, hidden_state, cell_state, debug=False):
    import numpy as np
    Wf = weights[0]
    Uf = weights[1]
    Bf = weights[2]

    sum = hidden_state * Uf + input * Wf + Bf

    if debug:
        print('OUTPUT GATE:')
        print(f'weights:{weights}\n\nWf:{Wf}\n\nUf:{Uf}\n\nBf:{Bf}\n\nhidden:{hidden_state}\n\ninput:{input}\n\nformula:{hidden_state} * {Uf} + {input} * {Wf} + {Bf}\n\nsum:{sum}')

    #PLUG VALUE INTO SIGMOID ACTIVATION FUNCTION: f(x) = -1/1+e^-x
    def sig(x):
        return 1/(1 + np.exp(-x))
    Ft = sig(sum)

    Ct = output_gate_candidate(cell_state)

    return Ft * Ct