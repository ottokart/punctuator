# coding: utf-8

import numpy as np
import cPickle
import utils

def convert_files(file_paths, vocabulary, punctuations, batch_size, use_pauses, output_path):
    inputs = []
    outputs = []
    punctuation = " "
    pause = 0.
    
    if use_pauses:
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
                        inputs.append(utils.input_word_index(vocabulary, token))
                        outputs.append(utils.punctuation_index(punctuations, punctuation))
                        if use_pauses:
                            pauses.append(pause)
                        punctuation = " "
                        pause = 0.

    inputs.append(utils.input_word_index(vocabulary, "<END>"))
    outputs.append(utils.punctuation_index(punctuations, punctuation))
    if use_pauses:
        pauses.append(pause)

    assert len(inputs) == len(outputs)
    num_batches = np.floor(len(inputs) / batch_size)

    dtype = np.int32 if len(vocabulary) > 32767 else np.int16

    inputs = np.array(inputs, dtype=dtype)[:batch_size*num_batches].reshape((batch_size, num_batches)).T
    outputs = np.array(outputs, dtype=np.int16)[:batch_size*num_batches].reshape((batch_size, num_batches)).T
    if use_pauses:
        pauses = np.array(pauses, dtype=np.float32)[:batch_size*num_batches].reshape((batch_size, num_batches)).T

    total_size = batch_size*num_batches

    data = {"inputs": inputs, "outputs": outputs,
            "vocabulary": vocabulary, "punctuations": punctuations,
            "batch_size": batch_size, "total_size": total_size}
    
    if use_pauses:
        data["pauses"] = pauses

    with open(output_path, 'wb') as output_file:
        cPickle.dump(data, output_file, protocol=cPickle.HIGHEST_PROTOCOL)