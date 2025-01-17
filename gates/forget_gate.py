#FORGET GATE DETERMINES HOW MUCH TO FORGET OF THE LONG TERM MEMORY / HOW RELEVANT IT IS
#FORMULA: Ft = sigmoid((Ht-1 * Uf + Xt * Wf) + Bf), where Ht-1 = hidden state, Uf = h-h weight, Wf = i-h weight, and Bf = bias

def forget_gate(input, hidden_state, cell_state):
    import numpy as np
    Wf = input[0][0]
    Uf = input[0][1]
    Bf = input[0][2]
    
    sum = np.dot(hidden_state, Uf) + np.dot(input, Wf) + Bf

    #PLUG VALUE INTO SIGMOID ACTIVATION FUNCTION - 1 / 1+e^-x
    def sig(x):
        return 1/(1 + np.exp(-x))
    Ft = sig(sum)

    return Ft * cell_state
    
