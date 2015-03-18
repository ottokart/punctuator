# coding: utf-8

import numpy as np
import sys
import utils
import models

from time import time
from itertools import izip
from pause_model import duration_normalizer

"""
TODO:
*<DONE>! copy weight matrix past for BPTT?      
*<DONE>! parallel sequences?
*GPU in hidde layer
*look further into future before predicting (longer forward context than 1)
*<DONE>! more complex LSTM implementation
*<DONE>! peephole connections for LSTM
*feedforward NNLM
*<DONE>! detailed error analysis (print text with mistakes)
*<DONE>! gradient checking
*<DONE>! projection layer for pauses and words

NB!!!!!!!!! Denormal handling????????
NB! Both, recurrency and peepholes, seem to help with LSTM
"""

SHOW_WPS = True

CONF = {
    "MAX_EPOCHS": 20,
    "LEARNING_RATE": 0.1,
    "MIN_LEARNING_RATE": 1e-6,
    "MIN_IMPROVEMENT": 1.003,
    "PROJECTION_SIZE": 100,
    "HIDDEN_SIZE": 100,
    "HIDDEN_ACTIVATION": "Tanh",
    "BPTT_STEPS": 5,
    "MODEL_FILE_TEMPLATE": "../out/model_%s",
    "LOG_FILE_TEMPLATE": "../out/log_%s",
    "TRAIN_DATA": "../data/train_large",#"../../Punctuator2/data/train",
    "DEV_DATA": "../data/dev_large",#"../../Punctuator2/data/dev",
    "USE_PAUSES": False
}

def process_corpus(net, dataset, mode='train', learning_rate=None):
    assert mode in ('train', 'test')
    counter = 0
    total_neg_log_prob = 0

    t0 = time()
    max_output_length = 0

    net.reset_state()

    if CONF["USE_PAUSES"]:
        pass#dataset["pauses"] = duration_normalizer(dataset["pauses"]) #Paistab, et ilma normaliseerimiseta töötab paremini!

    for inputs, outputs, pauses in izip(dataset["inputs"], dataset["outputs"], dataset.get("pauses", [None] * dataset["total_size"])):

        if mode == 'train':
            neg_log_probs = net.train(inputs, outputs, pauses, learning_rate)
        elif mode == 'test':
            neg_log_probs = net.neg_log_prob(inputs, outputs, pauses)

        total_neg_log_prob += np.sum(neg_log_probs)
        counter += dataset["batch_size"]

        if SHOW_WPS:
            elapsed_time = time() - t0
            if elapsed_time > 0:
                output = "%d%% @ %.1f wps  " % (float(counter) / dataset["total_size"] * 100,
                                                counter / elapsed_time)
                max_output_length = max(max_output_length, len(output))
                sys.stdout.write(output+"\r")
                sys.stdout.flush()
        
    if SHOW_WPS:
        sys.stdout.write(" "*max_output_length+"\r")
        sys.stdout.flush()

    ppl = np.exp(total_neg_log_prob / counter)

    return ppl

if __name__ == "__main__":

    assert len(sys.argv) > 1, "Please give model name"
    assert len(sys.argv) > 2 and sys.argv[2] in ("RNN", "LSTM"), "Please give model type RNN/LSTM"

    model_type = sys.argv[2]
    NeuralNetwork = getattr(models, model_type)

    model_file_name = CONF["MODEL_FILE_TEMPLATE"] % sys.argv[1]
    log_file_name = CONF["LOG_FILE_TEMPLATE"] % sys.argv[1]

    np.random.seed(1)
    
    description = "%s model\n" % model_type
    for k,v in CONF.items():
        description += "{:<40} {:<40}\n".format(k,v)
    description += "-" * 80 + "\n\n"
    with open(log_file_name, 'w') as log_file:
        log_file.write(description)

    net = NeuralNetwork()

    training_data = np.load(CONF["TRAIN_DATA"])
    validation_data = np.load(CONF["DEV_DATA"])

    assert training_data["batch_size"] == validation_data["batch_size"]
    assert training_data["vocabulary"] == validation_data["vocabulary"]
    assert training_data["punctuations"] == validation_data["punctuations"]

    print "Data loaded..."

    print "Vocabulary size is %d" % utils.get_vocabulary_size(validation_data["vocabulary"])
    print "Training set size is %d" % training_data["total_size"]
    print "Validation set size is %d" % validation_data["total_size"]

    net.initialize(hidden_size=CONF["HIDDEN_SIZE"],
                   projection_size=CONF["PROJECTION_SIZE"],
                   in_vocabulary=training_data["vocabulary"],
                   out_vocabulary=training_data["punctuations"],
                   batch_size=training_data["batch_size"],
                   hidden_activation=CONF["HIDDEN_ACTIVATION"],
                   bptt_steps=CONF["BPTT_STEPS"],
                   use_pauses=CONF["USE_PAUSES"])

    best_validation_ppl = np.inf
    learning_rate = CONF["LEARNING_RATE"]
    divide = False

    for epoch in range(1, CONF["MAX_EPOCHS"]+1):
        
        epoch_start = time()
        
        print "\n======= EPOCH %s =======" % epoch
        print "\tLearning rate is %s" % learning_rate

        train_ppl = process_corpus(net, training_data, mode='train', learning_rate=learning_rate) 
        print "\tTrain PPL is %.3f" % train_ppl

        validation_ppl = process_corpus(net, validation_data, mode='test')
        print "\tValidation PPL is %.3f" % validation_ppl

        epoch_duration = time() - epoch_start
        print "\tTime taken: %ds" % epoch_duration
        with open(log_file_name, 'a') as log_file:
            log_file.write("%d. Validation PPL %s; Time %ds\n" % (epoch, validation_ppl, epoch_duration))

        if np.log(validation_ppl) * CONF["MIN_IMPROVEMENT"] > np.log(best_validation_ppl): # Mikolovs recipe
            if not divide:
                divide = True
                print "\tStarting to reduce the learning rate..."
                if validation_ppl > best_validation_ppl:
                    print "\tLoading best model."
                    net = utils.load_model(model_file_name)
            else:
                if validation_ppl < best_validation_ppl:
                    print "\tSaving model."
                    net.save(model_file_name, final=True)
                break
        else:
            print "\tNew best model! Saving..."
            best_validation_ppl = validation_ppl
            final = learning_rate / 2. < CONF["MIN_LEARNING_RATE"]
            net.save(model_file_name, final)

        if divide:
            learning_rate /= 2.
        
        if learning_rate < CONF["MIN_LEARNING_RATE"]:
            break
            
    print "-"*30
    print "Finished training."
    print "Best validation PPL is %.3f" % best_validation_ppl
