#INITIALIZE RANDOM WEIGHTS AND SET BIASES AND HIDDEN & CELL STATES TO 0
def init(input_size, hidden_size):
    import numpy as np

    #Uniform Xavier initialization method, W ~ U( -sqrt( 6 / (n_in + n_out) ),  sqrt( 6 / (n_in + n_out) ) ),
    #where n_in is num of inputs and n_out is num of outputs

    def uniform_xavier(n_in, n_out):
        limit = np.sqrt(6 / (n_in+n_out))

        #Uniform distribution between limits, returning a vector of size n_in by n_out or input size by hidden state size
        return np.random.uniform(limit, -limit, size=(n_in, n_out))
    
    #WEIGHTS AND BIASES FOR EACH GATE, FORGET GATE, INPUT GATE(POTENTIAL AND % GATES) AND OUTPUT GATE(POTENTIAL AND % GATES)
    # `-> OUTPUT POTENTIAL GATE HAS NO WEIGHTS OR BIASES

    #FORGET GATE WEIGHTS
    fg_i_w = uniform_xavier(input_size, hidden_size)
    fg_h_w = uniform_xavier(hidden_size, hidden_size)
    #FORGET GATE BIAS
    fg_b = np.zeros(hidden_size)


    #INPUT CANDIDATE CELL WEIGHTS
    ic_i_w = uniform_xavier(input_size, hidden_size)
    ic_h_w = uniform_xavier(hidden_size, hidden_size)
    #INPUT CANDIDATE CELL BIAS
    ic_b = np.zeros(hidden_size)

    #INPUT GATE WEIGHTS
    ig_i_w = uniform_xavier(input_size, hidden_size)
    ig_h_w = uniform_xavier(hidden_size, hidden_size)
    #INPUT GATE BIAS
    ig_b = np.zeros(hidden_size)


    #OUTPUT GATE WEIGHTS
    og_i_w = uniform_xavier(input_size, hidden_size)
    og_h_w = uniform_xavier(hidden_size, hidden_size)
    #OUPUT GATE BIAS
    og_b = np.zeros(hidden_size)

    return {'forget_gate_weights': [fg_i_w, fg_h_w, fg_b], 
            'input_gate_candidate_weights': [ic_i_w, ic_h_w, ic_b], 
            'input_gate_weights': [ig_i_w, ig_h_w, ig_b], 
            'output_gate_weights': [og_i_w, og_h_w, og_b]}

    
if __name__ == '__main__':
    #test case
    weights = init(3, 4)
    print(weights['forget_gate_weights'])