# coding: utf-8

import numpy as np
import cPickle
import utils

DEBUG = False
USE_PAUSES = True
BATCH_SIZE = 100

def convert_files(file_paths, vocabulary, punctuations, batch_size, output_path):
    inputs = []
    outputs = []
    punctuation = " "
    pause = 0.
    
    if USE_PAUSES:
        pauses = []

    for file_path in file_paths:
        with open(file_path, 'r') as corpus:
            for line in corpus:
                for token in line.split():
                    if token in punctuations:
                        punctuation = token
                        continue
                    elif token.startswith("<sil="):
                        pause = float(token.replace("<sil=","").replace(">",""))
                        continue
                    else:
                        if DEBUG:
                            print punctuation, token, pause
                        inputs.append(utils.input_word_index(vocabulary, token))
                        outputs.append(utils.punctuation_index(punctuations, punctuation))
                        if USE_PAUSES:
                            pauses.append(pause)
                        punctuation = " "
                        pause = 0.

    if DEBUG:
        print punctuation, "<END>", pause
    inputs.append(utils.input_word_index(vocabulary, "<END>"))
    outputs.append(utils.punctuation_index(punctuations, punctuation))
    if USE_PAUSES:
        pauses.append(pause)

    assert len(inputs) == len(outputs)
    num_batches = np.floor(len(inputs) / batch_size)

    dtype = np.int32 if len(vocabulary) > 32767 else np.int16

    inputs = np.array(inputs, dtype=dtype)[:batch_size*num_batches].reshape((batch_size, num_batches)).T
    outputs = np.array(outputs, dtype=np.int16)[:batch_size*num_batches].reshape((batch_size, num_batches)).T
    if USE_PAUSES:
        pauses = np.array(pauses, dtype=np.float32)[:batch_size*num_batches].reshape((batch_size, num_batches)).T

    total_size = batch_size*num_batches

    data = {"inputs": inputs, "outputs": outputs,
            "vocabulary": vocabulary, "punctuations": punctuations,
            "batch_size": batch_size, "total_size": total_size}
    
    if USE_PAUSES:
        data["pauses"] = pauses

    with open(output_path, 'wb') as output_file:
        cPickle.dump(data, output_file, protocol=cPickle.HIGHEST_PROTOCOL)

#punctuations = {" ": 0, ".PERIOD": 1, ",COMMA": 2, "?QUESTIONMARK": 3, ";SEMICOLON": 4, "!EXCLAMATIONMARK": 5, ":COLON": 6}
punctuations = {" ": 0, ".PERIOD": 1, ",COMMA": 2, "?QUESTIONMARK": 1, ";SEMICOLON": 1, "!EXCLAMATIONMARK": 1, ":COLON": 1}
vocabulary = utils.load_vocabulary("../data/vocab")

convert_files([#"../../data/web.train.txt",
              #"../../data/social.train.txt",
              #"../../data/riigikogu.train.txt",
              #"../../data/fiction.train.txt",
              #"../../data/ajalehed.train.txt",
              #"../../data/ajakirjad.train.txt",
              #"../../data/vestlused.train.txt",
              #"../../data/konverentsid.train.txt",
              #"../../data/er-uudised.train.txt"
              "../data/pauses.train"
              ],
              vocabulary, punctuations, BATCH_SIZE, "../data/train")

convert_files([#"../data/vestlused.dev.txt",
              #"../data/konverentsid.dev.txt",
              #"../data/er-uudised.dev.txt"
              "../data/pauses.dev"
              ],
              vocabulary, punctuations, BATCH_SIZE, "../data/dev")
