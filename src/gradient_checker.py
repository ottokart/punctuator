# coding: utf-8

import numpy as np
import copy
import adaptation_models5 as models

VOCAB = {"tere": 0, "maailm": 1, "mis": 2, "teed": 3, "<unk>": 4, "<END>": 5}
PUNCT = {" ": 0, ".PERIOD": 1, ",COMMA": 2}

INPUT_SIZE = 5
BPTT_STEPS = 5
BATCH_SIZE = 100

assert INPUT_SIZE == BPTT_STEPS, "If input is larger than BPTT steps, then the gradients will be approximate"

SHOW_CORRECT = False

def predict(net, inputs, outputs, pauses, with_backpropagation=False):
    net.reset_state()

    for inputs_i, outputs_i, pauses_i in zip(inputs, outputs, pauses):
        neg_log_probs = net.neg_log_prob(inputs_i, outputs_i, pauses_i)
        if with_backpropagation:
            net._backpropagate(outputs_i)

    return np.sum(neg_log_probs) # we calculate derivatives wrt to the error at last time step

def check(net_type):

    print "\nChecking: %s" % net_type.__name__
    print "="*60

    net = net_type()
    net.initialize(hidden_size=3,
                   projection_size=4,
                   in_vocabulary=VOCAB,
                   out_vocabulary=PUNCT,
                   batch_size=BATCH_SIZE,
                   hidden_activation="Tanh",
                   bptt_steps=BPTT_STEPS,
                   use_pauses=True)

    net_copy = copy.deepcopy(net)
    predict(net_copy, inputs, outputs, pauses, with_backpropagation=True)

    errors = 0

    for param in net.params:

        print "Checking: %s" % param

        W = getattr(net, param)
        dE_dW = getattr(net_copy, "dE_d%s" % param)

        if W.ndim == 2:

            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    
                    W[i,j] += tiny
                    E_plus = predict(net, inputs, outputs, pauses)
                    
                    W[i,j] -= 2.*tiny
                    E_minus = predict(net, inputs, outputs, pauses)

                    W[i,j] += tiny # back to original

                    dE_dW_ij_numerical = (E_plus - E_minus)/(2.*tiny)

                    if isinstance(dE_dW, dict):
                        dE_dW_ij = dE_dW.get(i, np.zeros(W.shape[1]))[j]
                    elif isinstance(dE_dW, float):
                        dE_dW_ij = dE_dW
                    else:
                        dE_dW_ij = dE_dW[i,j]

                    if not np.allclose(dE_dW_ij_numerical, dE_dW_ij):
                        print "Gradient for %s[%d,%d] does not match (numerical: %f != real: %f)" \
                              % (param, i, j, dE_dW_ij_numerical, dE_dW_ij)
                        errors += 1
                    elif SHOW_CORRECT:
                        print "Gradient for %s[%d,%d] is OK (%f)" \
                              % (param, i, j, dE_dW_ij)

        elif W.ndim == 1:

            for i in xrange(W.size):
                
                W[i] += tiny
                E_plus = predict(net, inputs, outputs, pauses)
                
                W[i] -= 2.*tiny
                E_minus = predict(net, inputs, outputs, pauses)

                W[i] += tiny # back to original

                dE_dW_i_numerical = (E_plus - E_minus)/(2.*tiny)

                if not np.allclose(dE_dW_i_numerical, dE_dW[i]):
                    print "Gradient for %s[%d] does not match (numerical: %f != real: %f)" \
                          % (param, i, dE_dW_i_numerical, dE_dW[i])
                    errors += 1
                elif SHOW_CORRECT:
                    print "Gradient for %s[%d] is OK (%f)" \
                          % (param, i, dE_dW[i])

    if errors > 0:
        print "\n### TOTAL ERRORS: %s ###\n" % errors
    else:
        print "\n### OK ###\n"


np.random.seed(1)

tiny = 1e-5 # tiny  constant
inputs = np.random.randint(len(VOCAB), size=(INPUT_SIZE, BATCH_SIZE))
outputs = np.random.randint(len(PUNCT), size=(INPUT_SIZE, BATCH_SIZE))
pauses = np.random.uniform(0, 1, size=(INPUT_SIZE, BATCH_SIZE))

#check(models.RNN)
check(models.LSTM)
