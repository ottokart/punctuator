# coding: utf-8

def shuffle_arrays(*arrays):
    import numpy as np
    rng_state = np.random.get_state()
    for array in arrays:
        np.random.set_state(rng_state)
        np.random.shuffle(array)

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
    #import models
    import adaptation_models4 as models
    import numpy as np
    
    model = np.load(file_path)
    net = getattr(models, model["type"])()
    
    TEMPORARY_FIX = True
    if TEMPORARY_FIX:
        model["use_pauses"] = False

    net.load(model)
    
    return net
