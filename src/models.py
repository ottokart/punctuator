# coding: utf-8

import numpy as np
import cPickle
import activation_functions

from itertools import izip
from activation_functions import Softmax, Sigmoid, Tanh
from utils import get_vocabulary_size

FLOATX = np.float64

class Model(object):

    def __init__(self):
        super(Model, self).__init__()
        self.initialized = False
        self.use_sgd = False

    def output_word_probability(self, output_word_index):
        assert self.initialized, "initialize or load before using"
        assert hasattr(self, "y"), "predict before trying to use output"
        return self.y[range(len(output_word_index)), output_word_index]

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
        return np.argmax(self.y, axis=1)

    def load(self, model):        
        for attr in model:
            setattr(self, attr, model[attr])
        self.hidden_activation = getattr(activation_functions, self.hidden_activation_name)
        self.reset_state()
        self.initialized = True
    
    def weights(self, i, o):
        s = 0.005#np.sqrt(6./(i+o))
        return np.random.uniform(low=-s, high=s, size=(i, o)).astype(FLOATX)

class PauseModel(Model):

    def __init__(self):
        super(PauseModel, self).__init__()
        self.params = ["Wh", "Wy"]
        self.initialized = False

    def initialize(self, input_size, hidden_size, out_vocabulary, hidden_activation="Tanh"):

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.out_vocabulary = out_vocabulary
        
        self.hidden_activation_name = hidden_activation
        self.hidden_activation = getattr(activation_functions, hidden_activation)

        self.Wh = self.weights(self.input_size, self.hidden_size)
        self.Wy = self.weights(self.hidden_size, get_vocabulary_size(self.out_vocabulary))
        
        self.reset_state()

        self.initialized = True

    def reset_state(self):
        self.Wh_hg = np.zeros_like(self.Wh)
        self.Wy_hg = np.zeros_like(self.Wy)

    def predict(self, inputs, pause_duration=None):
        assert self.initialized, "initialize or load before using"

        z_h = np.dot(inputs, self.Wh)
        self.h = self.hidden_activation.y(z=z_h)

        z_y = np.dot(self.h, self.Wy)
        self.y = Softmax.y(z=z_y)

    def _backpropagate(self, inputs, output_word_index):

        dE_dz_y = self.y.copy() # don't remove the copy() part
        dE_dz_y[range(len(output_word_index)), output_word_index] -= 1.
        self.dE_dWy = np.dot(self.h.T, dE_dz_y)

        dE_dz_h = np.dot(dE_dz_y, self.Wy.T) * self.hidden_activation.dy_dz(y=self.h)
        self.dE_dWh = np.dot(inputs.T, dE_dz_h)

    def update(self, inputs, output_word_index, learning_rate):
        assert self.initialized, "initialize or load before using"

        self._backpropagate(inputs, output_word_index)

        #self.Wy_hg += self.dE_dWy**2
        #self.Wh_hg += self.dE_dWh**2

        self.Wy -= learning_rate * self.dE_dWy# / (1e-6 + np.sqrt(self.Wy_hg))
        self.Wh -= learning_rate * self.dE_dWh# / (1e-6 + np.sqrt(self.Wh_hg))
   
    def save(self, file_name):
        assert self.initialized, "initialize or load before using"

        model = {
            "type":                         "PauseModel",
            "input_size":                   self.input_size,
            "hidden_size":                  self.hidden_size,
            "out_vocabulary":               self.out_vocabulary,
            "hidden_activation_name":       self.hidden_activation_name,
            "Wh":                           self.Wh,
            "Wy":                           self.Wy
        }
        cPickle.dump(model, file(file_name, 'wb'))



class RNN(Model):

    def __init__(self):
        super(RNN, self).__init__()
        self.params = ["We", "Wp", "Wx", "Wh", "Wy"]
        self.initialized = False

    def initialize(self, hidden_size, projection_size, in_vocabulary, out_vocabulary, batch_size, hidden_activation="Tanh", bptt_steps=1, use_pauses=False):

        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.bptt_steps = bptt_steps
        self.batch_size = batch_size
        self.use_pauses = use_pauses
        
        self.in_vocabulary = in_vocabulary
        self.out_vocabulary = out_vocabulary
        
        self.hidden_activation_name = hidden_activation
        self.hidden_activation = getattr(activation_functions, hidden_activation)

        # Word embedding weights
        self.We = self.weights(get_vocabulary_size(self.in_vocabulary), self.projection_size)
        # Pause input weights
        self.Wp = self.weights(1, self.projection_size)
        # input to hidden weights
        self.Wx = self.weights(self.projection_size, self.hidden_size)
        # hidden to hidden recurrency weights
        self.Wh = self.weights(self.hidden_size, self.hidden_size)
        # hidden to output weights
        self.Wy = self.weights(self.hidden_size, get_vocabulary_size(self.out_vocabulary))
        
        # AdaGread sum of squares of per feature historical gradients
        for p in self.params:
            setattr(self, p+"_hg", np.zeros_like(getattr(self, p)))

        self.reset_state()

        self.initialized = True

    def reset_state(self):
        self.h = np.zeros(shape=(self.batch_size, self.hidden_size))
        self.x = np.zeros(shape=(self.batch_size, self.projection_size))
        self.h_tm1_history = []
        self.x_history = []
        self.input_history = []
        self.Wx_history = []
        self.Wh_history = []
        if self.use_pauses:
            self.pause_history = []

    def _remember_state(self, input_word_index, pause=None):
        self.h_tm1_history.append(self.h_tm1)
        self.x_history.append(self.x)
        self.input_history.append(input_word_index)
        self.Wx_history.append(self.Wx.copy())
        self.Wh_history.append(self.Wh.copy())
        if self.use_pauses:
            self.pause_history.append(pause)
        if len(self.h_tm1_history) > self.bptt_steps:
            del self.h_tm1_history[0]
            del self.x_history[0]
            del self.input_history[0]
            del self.Wx_history[0]
            del self.Wh_history[0]
            if self.use_pauses:
                del self.pause_history[0]

    def predict(self, input_word_index, pause_duration=None):
        assert self.initialized, "initialize or load before using"

        self.h_tm1 = self.h
        
        z_x = self.We[input_word_index]
        if self.use_pauses:
            z_x += np.dot(pause_duration[:,np.newaxis], self.Wp)    
        self.x = self.hidden_activation.y(z=z_x)

        z_h = np.dot(self.x, self.Wx) + np.dot(self.h_tm1, self.Wh)
        self.h = self.hidden_activation.y(z=z_h)

        z_y = np.dot(self.h, self.Wy)
        self.y = Softmax.y(z=z_y)

        if self.use_pauses:
            self._remember_state(input_word_index, pause_duration[:,np.newaxis])
        else:
            self._remember_state(input_word_index)

    def _backpropagate(self, output_word_index):

        dE_dz_y = self.y.copy() # don't remove the copy() part
        dE_dz_y[range(len(output_word_index)), output_word_index] -= 1.
        self.dE_dWy = np.dot(self.h.T, dE_dz_y)

        dE_dz_h = np.dot(dE_dz_y, self.Wy.T) * self.hidden_activation.dy_dz(y=self.h)

        self.dE_dWh = np.zeros_like(self.Wh)
        self.dE_dWx = np.zeros_like(self.Wx)
        self.dE_dWe = {}
        if self.use_pauses:            
            self.dE_dWp = np.zeros_like(self.Wp)

        pause_history = self.pause_history if self.use_pauses else [None]*len(self.h_tm1_history)

        # Backpropagation through time
        for Wx, Wh, words, pauses, x, h_tm1 in reversed(zip(
            self.Wx_history, self.Wh_history,
            self.input_history, pause_history,
            self.x_history, self.h_tm1_history)):

            self.dE_dWh += np.dot(h_tm1.T, dE_dz_h)    
            self.dE_dWx += np.dot(x.T, dE_dz_h)

            dE_dz_x = np.dot(dE_dz_h, Wx.T) * self.hidden_activation.dy_dz(y=x)

            for word, dE_dz_x_word in izip(words, dE_dz_x):
                self.dE_dWe[word] = self.dE_dWe.get(word, 0.) + dE_dz_x_word

            if self.use_pauses:
                self.dE_dWp += np.dot(pauses.T, dE_dz_x)

            dE_dz_h = np.dot(dE_dz_h, Wh.T) * self.hidden_activation.dy_dz(y=h_tm1)
        
    def update(self, _, output_word_index, learning_rate):
        assert self.initialized, "initialize or load before using"

        self._backpropagate(output_word_index)

        self.Wy_hg += self.dE_dWy**2
        self.Wh_hg += self.dE_dWh**2
        self.Wx_hg += self.dE_dWx**2
        if self.use_pauses:
            self.Wp_hg += self.dE_dWp**2

        self.Wy -= learning_rate * self.dE_dWy / (1e-6 + np.sqrt(self.Wy_hg))
        self.Wh -= learning_rate * self.dE_dWh / (1e-6 + np.sqrt(self.Wh_hg))
        self.Wx -= learning_rate * self.dE_dWx / (1e-6 + np.sqrt(self.Wx_hg))
        if self.use_pauses:
            self.Wp -= learning_rate * self.dE_dWp / (1e-6 + np.sqrt(self.Wp_hg))

        for i in self.dE_dWe:
            self.We_hg[i] += self.dE_dWe[i]**2

            self.We[i] -= learning_rate * self.dE_dWe[i] / (1e-6 + np.sqrt(self.We_hg[i]))

    def save(self, file_name):
        assert self.initialized, "initialize or load before using"

        model = {
            "type":                         self.__class__.__name__,
            "hidden_size":                  self.hidden_size,
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
            model[p+"_hg"] = getattr(self, p+"_hg")

        cPickle.dump(model, file(file_name, 'wb'))


class LSTM(Model):

    def __init__(self):
        super(LSTM, self).__init__()
        self.params = ["We", "Wp", "W", "Wh", "Wy", # Word embeddings, pauses, inputs-to-LSTM, recurrency, outputs
                       "Wip", "Wfp", "Wop"] # peepholes
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
        self.Wp = self.weights(1, self.projection_size)
        self.W = self.weights(self.projection_size, self.hidden_size*4)
        self.Wip = self.weights(1, self.hidden_size)
        self.Wfp = self.weights(1, self.hidden_size)
        self.Wop = self.weights(1, self.hidden_size)
        self.Wh = self.weights(self.hidden_size, self.hidden_size*4)
        self.Wy = self.weights(self.hidden_size, get_vocabulary_size(self.out_vocabulary))

        # AdaGread sum of squares of per feature historical gradients
        for p in self.params:
            setattr(self, p+"_hg", np.zeros_like(getattr(self, p)))

        self.reset_state()

        self.initialized = True

    def reset_state(self):
        self.m = np.zeros(shape=(self.batch_size, self.hidden_size))
        self.h = np.zeros(shape=(self.batch_size, self.hidden_size))
        self.word_history = []
        if self.use_pauses:
            self.pause_history = []
        self.m_tm1_history = []
        self.h_tm1_history = []
        self.z_history = []
        self.x_history = []
        self.i_history = []
        self.ig_history = []
        self.fg_history = []
        self.og_history = []
        self.W_history = []
        self.Wh_history = []
        self.Wip_history = []
        self.Wfp_history = []
        self.Wop_history = []

    def _remember_state(self, input_word_index, pause=None):
        self.word_history.append(input_word_index)
        if self.use_pauses:
            self.pause_history.append(pause)
        self.m_tm1_history.append(self.m_tm1)
        self.h_tm1_history.append(self.h_tm1)
        self.z_history.append(self.z)
        self.x_history.append(self.x)
        self.i_history.append(self.i)
        self.ig_history.append(self.ig)
        self.fg_history.append(self.fg)
        self.og_history.append(self.og)
        self.W_history.append(self.W.copy())
        self.Wh_history.append(self.Wh.copy())
        self.Wip_history.append(self.Wip.copy())
        self.Wfp_history.append(self.Wfp.copy())
        self.Wop_history.append(self.Wop.copy())
        if len(self.word_history) > self.bptt_steps:
            del self.word_history[0]
            if self.use_pauses:
                del self.pause_history[0]
            del self.m_tm1_history[0]
            del self.h_tm1_history[0]
            del self.z_history[0]
            del self.x_history[0]
            del self.i_history[0]
            del self.ig_history[0]
            del self.fg_history[0]
            del self.og_history[0]
            del self.W_history[0]
            del self.Wh_history[0]
            del self.Wip_history[0]
            del self.Wfp_history[0]
            del self.Wop_history[0]

    def _slice(self, r, n):
        return r[:, n*self.hidden_size:(n+1)*self.hidden_size]

    def predict(self, input_word_index, pause_duration=None):
        assert self.initialized, "initialize or load before using"
        # see math in: http://www.cs.toronto.edu/~fritz/absps/RNN13.pdf

        self.m_tm1 = self.m
        self.h_tm1 = self.h

        r = np.dot(self.h_tm1, self.Wh)

        z = self.We[input_word_index]
        if self.use_pauses:
            z += np.dot(pause_duration[:,np.newaxis], self.Wp) 
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

        z_y = np.dot(self.h, self.Wy)
        self.y = Softmax.y(z=z_y)

        if self.use_pauses:
            self._remember_state(input_word_index, pause_duration[:,np.newaxis])
        else:
            self._remember_state(input_word_index)

    def _backpropagate(self, output_word_index):

        dE_dz_y = self.y.copy() # don't remove the copy() part
        dE_dz_y[range(len(output_word_index)), output_word_index] -= 1.
        self.dE_dWy = np.dot(self.h.T, dE_dz_y)

        dE_dh = np.dot(dE_dz_y, self.Wy.T)

        self.dE_dWe = {}
        self.dE_dW = np.zeros_like(self.W)
        self.dE_dWh = np.zeros_like(self.Wh)
        self.dE_dWip = np.zeros_like(self.Wip)
        self.dE_dWfp = np.zeros_like(self.Wfp)
        self.dE_dWop = np.zeros_like(self.Wop)
        self.dE_dWp = np.zeros_like(self.Wp)

        dE_dm_tm1 = 0.
        dE_dh_tm1 = 0.

        m = self.m

        pause_history = self.pause_history if self.use_pauses else [None]*len(self.word_history)

        for pauses, words, W, Wh, Wip, Wfp, Wop, x, m_tm1, h_tm1, z, i, ig, fg, og in reversed(zip(
                                                pause_history, self.word_history,
                                                self.W_history, self.Wh_history,
                                                self.Wip_history, self.Wfp_history, self.Wop_history,
                                                self.x_history, self.m_tm1_history, self.h_tm1_history, self.z_history,
                                                self.i_history, self.ig_history,
                                                self.fg_history, self.og_history)):

            dE_dh = dE_dh + dE_dh_tm1
            dE_dog = dE_dh * z * Sigmoid.dy_dz(y=og)
            dE_dz = dE_dh * og * self.hidden_activation.dy_dz(y=z)
            dE_dm = dE_dz + dE_dm_tm1 + dE_dog * Wop
            dE_dfg = dE_dm * m_tm1 * Sigmoid.dy_dz(y=fg)
            dE_di = dE_dm * ig * self.hidden_activation.dy_dz(y=i)
            dE_dig = dE_dm * i * Sigmoid.dy_dz(y=ig)
            dE_dm_tm1 = dE_dm * fg + dE_dig * Wip + dE_dfg * Wfp

            self.dE_dWip += (dE_dig * m_tm1).sum(0)
            self.dE_dWfp += (dE_dfg * m_tm1).sum(0)
            self.dE_dWop += (dE_dog * m).sum(0)

            d = np.hstack((dE_di, dE_dig, dE_dfg, dE_dog))
            
            dE_dx = np.dot(d, W.T) * self.hidden_activation.dy_dz(y=x)
            dE_dh_tm1 = np.dot(d, Wh.T)

            self.dE_dW += np.dot(x.T, d)
            self.dE_dWh += np.dot(h_tm1.T, d)

            for word, dE_dx_word in izip(words, dE_dx):
                self.dE_dWe[word] = self.dE_dWe.get(word, 0.) + dE_dx_word

            if self.use_pauses:
                self.dE_dWp += np.dot(pauses.T, dE_dx)

            dE_dh = 0.
            m = m_tm1

    def update(self, _, output_word_index, learning_rate):
        assert self.initialized, "initialize or load before using"

        self._backpropagate(output_word_index)

        self.Wy_hg += self.dE_dWy**2
        self.Wy -= learning_rate * self.dE_dWy / (1e-6 + np.sqrt(self.Wy_hg))
        
        self.Wh_hg += self.dE_dWh**2
        self.Wh -= learning_rate * self.dE_dWh / (1e-6 + np.sqrt(self.Wh_hg))

        self.Wip_hg += self.dE_dWip**2
        self.Wip -= learning_rate * self.dE_dWip / (1e-6 + np.sqrt(self.Wip_hg))

        self.Wfp_hg += self.dE_dWfp**2
        self.Wfp -= learning_rate * self.dE_dWfp / (1e-6 + np.sqrt(self.Wfp_hg))

        self.Wop_hg += self.dE_dWop**2
        self.Wop -= learning_rate * self.dE_dWop / (1e-6 + np.sqrt(self.Wop_hg))

        self.W_hg += self.dE_dW**2
        self.W -= learning_rate * self.dE_dW / (1e-6 + np.sqrt(self.W_hg))

        if self.use_pauses:
            self.Wp_hg += self.dE_dWp**2
            self.Wp -= learning_rate * self.dE_dWp / (1e-6 + np.sqrt(self.Wp_hg))                     

        for i in self.dE_dWe:
            self.We_hg[i] += self.dE_dWe[i]**2 
            self.We[i] -= learning_rate * self.dE_dWe[i] / (1e-6 + np.sqrt(self.We_hg[i]))

    def save(self, file_name):
        assert self.initialized, "initialize or load before using"

        model = {
            "type":                         self.__class__.__name__,
            "hidden_size":                  self.hidden_size,
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
            model[p+"_hg"] = getattr(self, p+"_hg")

        cPickle.dump(model, file(file_name, 'wb'))