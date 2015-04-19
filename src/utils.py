# coding: utf-8

def get_reverse_map(dictionary):
	return {v:k for k,v in dictionary.iteritems()}

def get_vocabulary_size(vocabulary):
    return max(vocabulary.values()) + 1

def input_word_index(vocabulary, input_word):
    return vocabulary.get(input_word, vocabulary["<unk>"])

def punctuation_index(punctuations, punctuation):
    return punctuations[punctuation]

def load_vocabulary(file_path):
    with open(file_path, 'r') as vocab:
        vocabulary = {w.strip(): i for (i, w) in enumerate(vocab)}
    if "<unk>" not in vocabulary:
        vocabulary["<unk>"] = len(vocabulary)
    if "<END>" not in vocabulary:
        vocabulary["<END>"] = len(vocabulary)
    return vocabulary
    
def load_model(file_path):
    import models
    import numpy as np
    
    model = np.load(file_path)
    net = getattr(models, model["type"])()
    
    net.load(model)
    
    return net
