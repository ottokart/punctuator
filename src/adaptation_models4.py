# coding: utf-8

import numpy as np
import cPickle
import activation_functions

from itertools import izip
from activation_functions import Softmax, Sigmoid, Tanh
from utils import get_vocabulary_size

FLOATX = np.float64

# CHECKPOINT
#Taneli idee, kus suure korpuse mudeli LSTM otsa on ehitatud LSTM featureid ja pausi kasutav 2-kihiline LSTM mudel. Vana mudel on fikseeritud ja treenitakse ainult uut mudelit.


class Model(object):

    def __init__(self):
        super(Model, self).__init__()
        self.initialized = False
        self.use_sgd = False

    def output_word_probability(self, output_word_index):
        assert self.initialized, "initialize or load before using"
        assert hasattr(self, "y2"), "predict before trying to use output"
        return self.y2[range(len(output_word_index)), output_word_index]

    def train(self, input_word_index, output_word_index, pause=None, learning_rate=0.1):
        assert self.initialized, "initialize or load before using"
        self.predict(input_word_index, pause)
        self.update(input_word_index, output_word_index, learning_rate)
        return -np.log(self.output_word_probability(output_word_index))

    def neg_log_prob(self, input_word_index, output_word_index, pause=None):
        assert self.initialized, "initialize or load before using"
        self.predict(input_word_index, pause)
        return -np.log(self.output_word_probability(output_word_index))

    def predict_punctuation(self, input_word_index, pause=None):
        assert self.initialized, "initialize or load before using"
        self.predict(input_word_index, pause)
        return np.argmax(self.y2, axis=1)

    def load(self, model):        
        for attr in model:
            setattr(self, attr, model[attr])
        self.hidden_activation = getattr(activation_functions, self.hidden_activation_name)
        self.reset_state()
        self.initialized = True
    
    def weights(self, i, o):
        s = 0.005#np.sqrt(6./(i+o))
        return np.random.uniform(low=-s, high=s, size=(i, o)).astype(FLOATX)

class LSTM(Model):

    def __init__(self):
        super(LSTM, self).__init__()
        self.params = ["Wh2", "Wy2", "Wp", "Wip2", "Wfp2", "Wop2", "Wr2" ] # peepholes
        self.initialized = False

    def initialize(self, hidden_size, projection_size, in_vocabulary, out_vocabulary, batch_size, hidden_activation="Tanh", bptt_steps=1, use_pauses=False):

        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.bptt_steps = bptt_steps
        self.batch_size = batch_size
        self.use_pauses = use_pauses

        self.in_vocabulary = in_vocabulary
        self.out_vocabulary = out_vocabulary

        self.hidden_activation_name = hidden_activation
        self.hidden_activation = getattr(activation_functions, hidden_activation)
        
        self.We = self.weights(get_vocabulary_size(self.in_vocabulary), self.projection_size)
        self.W = self.weights(self.projection_size, self.hidden_size*4)
        self.Wip = self.weights(1, self.hidden_size)
        self.Wfp = self.weights(1, self.hidden_size)
        self.Wop = self.weights(1, self.hidden_size)
        self.Wh = self.weights(self.hidden_size, self.hidden_size*4)

        self.hidden_size2 = 10
        self.Wh2 = self.weights(self.hidden_size, self.hidden_size2*4)
        self.Wp = self.weights(1, self.hidden_size2*4)
        self.Wy2 = self.weights(self.hidden_size2, get_vocabulary_size(self.out_vocabulary))
        self.Wip2 = self.weights(1, self.hidden_size2)
        self.Wfp2 = self.weights(1, self.hidden_size2)
        self.Wop2 = self.weights(1, self.hidden_size2)
        self.Wr2 = self.weights(self.hidden_size2, self.hidden_size2*4)

        # AdaGread sum of squares of per feature historical gradients
        for p in self.params:
            setattr(self, p+"_hg", np.zeros_like(getattr(self, p)))

        self.reset_state()

        self.initialized = True

    def reset_state(self):
        self.m = np.zeros(shape=(self.batch_size, self.hidden_size))
        self.h = np.zeros(shape=(self.batch_size, self.hidden_size))
        self.m2 = np.zeros(shape=(self.batch_size, self.hidden_size2))
        self.h2 = np.zeros(shape=(self.batch_size, self.hidden_size2))
        if self.use_pauses:
            self.pause_history = []
        self.m2_tm1_history = []
        self.h2_tm1_history = []
        self.z2_history = []
        self.h_history = []
        self.i2_history = []
        self.ig2_history = []
        self.fg2_history = []
        self.og2_history = []
        self.Wr2_history = []
        self.Wh2_history = []
        self.Wip2_history = []
        self.Wfp2_history = []
        self.Wop2_history = []

    def _remember_state(self, pause):
        if self.use_pauses:
            self.pause_history.append(pause)
        self.m2_tm1_history.append(self.m2_tm1)
        self.h2_tm1_history.append(self.h2_tm1)
        self.z2_history.append(self.z2)
        self.h_history.append(self.h)
        self.i2_history.append(self.i2)
        self.ig2_history.append(self.ig2)
        self.fg2_history.append(self.fg2)
        self.og2_history.append(self.og2)
        self.Wr2_history.append(self.Wr2.copy())
        self.Wh2_history.append(self.Wh2.copy())
        self.Wip2_history.append(self.Wip2.copy())
        self.Wfp2_history.append(self.Wfp2.copy())
        self.Wop2_history.append(self.Wop2.copy())
        if len(self.h_history) > self.bptt_steps:
            if self.use_pauses:
                del self.pause_history[0]
            del self.m2_tm1_history[0]
            del self.h2_tm1_history[0]
            del self.z2_history[0]
            del self.h_history[0]
            del self.i2_history[0]
            del self.ig2_history[0]
            del self.fg2_history[0]
            del self.og2_history[0]
            del self.Wr2_history[0]
            del self.Wh2_history[0]
            del self.Wip2_history[0]
            del self.Wfp2_history[0]
            del self.Wop2_history[0]
            
    def _slice(self, r, n):
        return r[:, n*self.hidden_size:(n+1)*self.hidden_size]

    def _slice2(self, r, n):
        return r[:, n*self.hidden_size2:(n+1)*self.hidden_size2]

    def predict(self, input_word_index, pause_duration=None):
        assert self.initialized, "initialize or load before using"
        # see math in: http://www.cs.toronto.edu/~fritz/absps/RNN13.pdf

        self.m_tm1 = self.m
        self.h_tm1 = self.h
        self.m2_tm1 = self.m2
        self.h2_tm1 = self.h2


        r = np.dot(self.h_tm1, self.Wh)

        z = self.We[input_word_index]
        self.x = self.hidden_activation.y(z)

        z1 = np.dot(self.x, self.W)

        z = self._slice(r, 0) + self._slice(z1, 0)
        self.i  = self.hidden_activation.y(z)

        z = self._slice(r, 1) + self._slice(z1, 1) + self.m_tm1 * self.Wip
        self.ig = Sigmoid.y(z)
        
        z = self._slice(r, 2) + self._slice(z1, 2) + self.m_tm1 * self.Wfp
        self.fg = Sigmoid.y(z)
        
        self.m = self.i * self.ig + self.m_tm1 * self.fg
        
        z = self._slice(r, 3) + self._slice(z1, 3) + self.m * self.Wop
        self.og = Sigmoid.y(z)
        
        self.z = self.hidden_activation.y(self.m)
        self.h = self.z * self.og

        #########

        r = np.dot(self.h2_tm1, self.Wr2)
        z1 = np.dot(pause_duration[:,np.newaxis], self.Wp) + np.dot(self.h, self.Wh2)

        z = self._slice2(r, 0) + self._slice2(z1, 0)
        self.i2  = self.hidden_activation.y(z)

        z = self._slice2(r, 1) + self._slice2(z1, 1) + self.m2_tm1 * self.Wip2
        self.ig2 = Sigmoid.y(z)
        
        z = self._slice2(r, 2) + self._slice2(z1, 2) + self.m2_tm1 * self.Wfp2
        self.fg2 = Sigmoid.y(z)
        
        self.m2 = self.i2 * self.ig2 + self.m2_tm1 * self.fg2
        
        z = self._slice2(r, 3) + self._slice2(z1, 3) + self.m2 * self.Wop2
        self.og2 = Sigmoid.y(z)
        
        self.z2 = self.hidden_activation.y(self.m2)
        self.h2 = self.z2 * self.og2

        z_y = np.dot(self.h2, self.Wy2)
        self.y2 = Softmax.y(z=z_y)

        self._remember_state(pause_duration)

    def _backpropagate(self, output_word_index):

        dE_dz_y = self.y2.copy() # don't remove the copy() part
        dE_dz_y[range(len(output_word_index)), output_word_index] -= 1.
        self.dE_dWy2 = np.dot(self.h2.T, dE_dz_y)

        dE_dh2 = np.dot(dE_dz_y, self.Wy2.T) * self.hidden_activation.dy_dz(y=self.h2)

        self.dE_dWr2 = np.zeros_like(self.Wr2)
        self.dE_dWh2 = np.zeros_like(self.Wh2)
        self.dE_dWip2 = np.zeros_like(self.Wip2)
        self.dE_dWfp2 = np.zeros_like(self.Wfp2)
        self.dE_dWop2 = np.zeros_like(self.Wop2)
        self.dE_dWp = np.zeros_like(self.Wp)

        dE_dm2_tm1 = 0.
        dE_dh2_tm1 = 0.

        m2 = self.m2

        pause_history = self.pause_history if self.use_pauses else [None]*len(self.h_history)

        for pauses, Wr2, Wh2, Wip2, Wfp2, Wop2, h, m2_tm1, h2_tm1, z2, i2, ig2, fg2, og2 in reversed(zip(
                                                pause_history, 
                                                self.Wr2_history, self.Wh2_history,
                                                self.Wip2_history, self.Wfp2_history, self.Wop2_history,
                                                self.h_history, self.m2_tm1_history, self.h2_tm1_history, self.z2_history,
                                                self.i2_history, self.ig2_history,
                                                self.fg2_history, self.og2_history)):

            dE_dh2 = dE_dh2 + dE_dh2_tm1
            dE_dog2 = dE_dh2 * z2 * Sigmoid.dy_dz(y=og2)
            dE_dz2 = dE_dh2 * og2 * self.hidden_activation.dy_dz(y=z2)
            dE_dm2 = dE_dz2 + dE_dm2_tm1 + dE_dog2 * Wop2
            dE_dfg2 = dE_dm2 * m2_tm1 * Sigmoid.dy_dz(y=fg2)
            dE_di2 = dE_dm2 * ig2 * self.hidden_activation.dy_dz(y=i2)
            dE_dig2 = dE_dm2 * i2 * Sigmoid.dy_dz(y=ig2)
            dE_dm2_tm1 = dE_dm2 * fg2 + dE_dig2 * Wip2 + dE_dfg2 * Wfp2

            self.dE_dWip2 += (dE_dig2 * m2_tm1).sum(0)
            self.dE_dWfp2 += (dE_dfg2 * m2_tm1).sum(0)
            self.dE_dWop2 += (dE_dog2 * m2).sum(0)

            d = np.hstack((dE_di2, dE_dig2, dE_dfg2, dE_dog2))
            
            dE_dh2_tm1 = np.dot(d, Wr2.T)
            if self.use_pauses:
                self.dE_dWp += np.dot(pauses.T, d)
            self.dE_dWh2 += np.dot(h.T, d)
            self.dE_dWr2 += np.dot(h2_tm1.T, d)

            dE_dh2 = 0.
            m2 = m2_tm1


    def update(self, _, output_word_index, learning_rate):
        assert self.initialized, "initialize or load before using"

        self._backpropagate(output_word_index)

        self.Wy2_hg += self.dE_dWy2**2
        self.Wy2 -= learning_rate * self.dE_dWy2 / (1e-6 + np.sqrt(self.Wy2_hg))
        
        self.Wh2_hg += self.dE_dWh2**2
        self.Wh2 -= learning_rate * self.dE_dWh2 / (1e-6 + np.sqrt(self.Wh2_hg))

        if self.use_pauses:
            self.Wp_hg += self.dE_dWp**2
            self.Wp -= learning_rate * self.dE_dWp / (1e-6 + np.sqrt(self.Wp_hg))
        
        self.Wr2_hg += self.dE_dWr2**2
        self.Wr2 -= learning_rate * self.dE_dWr2 / (1e-6 + np.sqrt(self.Wr2_hg))

        self.Wip2_hg += self.dE_dWip2**2
        self.Wip2 -= learning_rate * self.dE_dWip2 / (1e-6 + np.sqrt(self.Wip2_hg))

        self.Wfp2_hg += self.dE_dWfp2**2
        self.Wfp2 -= learning_rate * self.dE_dWfp2 / (1e-6 + np.sqrt(self.Wfp2_hg))

        self.Wop2_hg += self.dE_dWop2**2
        self.Wop2 -= learning_rate * self.dE_dWop2 / (1e-6 + np.sqrt(self.Wop2_hg))

    def save(self, file_name, final):
        assert self.initialized, "initialize or load before using"

        model = {
            "type":                         self.__class__.__name__,
            "hidden_size":                  self.hidden_size,
            "hidden_size2":                  self.hidden_size2,
            "projection_size":              self.projection_size,
            "bptt_steps":                   self.bptt_steps,
            "batch_size":                   self.batch_size,
            "use_pauses":                   self.use_pauses,
            "in_vocabulary":                self.in_vocabulary,
            "out_vocabulary":               self.out_vocabulary,
            "hidden_activation_name":       self.hidden_activation_name,
        }
        for p in self.params:
            model[p] = getattr(self, p)  
            if not final:  
                model[p+"_hg"] = getattr(self, p+"_hg")

        for p in ["We", "Wp", "W", "Wh", "Wip", "Wfp", "Wop", "Wy"]:
            model[p] = getattr(self, p)    

        cPickle.dump(model, file(file_name, 'wb'))

    def load(self, model):        
        for attr in model:
            setattr(self, attr, model[attr])
        self.hidden_activation = getattr(activation_functions, self.hidden_activation_name)
        
        self.hidden_size2 = self.hidden_size
        if not hasattr(self, "Wh2"):
            self.Wh2 = self.weights(self.hidden_size, self.hidden_size2*4)
            self.Wp = self.weights(1, self.hidden_size2*4)
            self.Wy2 = self.weights(self.hidden_size2, get_vocabulary_size(self.out_vocabulary))
            self.Wip2 = self.weights(1, self.hidden_size2)
            self.Wfp2 = self.weights(1, self.hidden_size2)
            self.Wop2 = self.weights(1, self.hidden_size2)
            self.Wr2 = self.weights(self.hidden_size2, self.hidden_size2*4)

            # AdaGread sum of squares of per feature historical gradients
            for p in self.params:
                setattr(self, p+"_hg", np.zeros_like(getattr(self, p)))

        self.reset_state()
        self.initialized = True