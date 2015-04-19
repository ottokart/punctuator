# coding: utf-8

import numpy as np
import sys
import os
import utils
import models
import conf

from time import time
from itertools import izip

def _process_corpus(net, dataset, mode='train', learning_rate=None):
    assert mode in ('train', 'test')
    counter = 0
    total_neg_log_prob = 0

    t0 = time()
    max_output_length = 0

    net.reset_state()

    for inputs, outputs, pauses in izip(dataset["inputs"], dataset["outputs"], dataset.get("pauses", [None] * dataset["total_size"])):

        if mode == 'train':
            neg_log_probs = net.train(inputs, outputs, pauses, learning_rate)
        elif mode == 'test':
            neg_log_probs = net.neg_log_prob(inputs, outputs, pauses)

        total_neg_log_prob += np.sum(neg_log_probs)
        counter += dataset["batch_size"]

        if conf.SHOW_WPS:
            elapsed_time = time() - t0
            if elapsed_time > 0:
                output = "%d%% @ %.1f wps  " % (float(counter) / dataset["total_size"] * 100,
                                                counter / elapsed_time)
                max_output_length = max(max_output_length, len(output))
                sys.stdout.write(output+"\r")
                sys.stdout.flush()
        
    if conf.SHOW_WPS:
        sys.stdout.write(" "*max_output_length+"\r")
        sys.stdout.flush()

    ppl = np.exp(total_neg_log_prob / counter)

    return ppl

def _train(net, training_data, validation_data, model_name, learning_rate, max_epochs, min_improvement):
    min_learning_rate = 1e-6
    best_validation_ppl = np.inf
    divide = False

    for epoch in range(1, max_epochs+1):
        
        epoch_start = time()
        
        print "\n======= EPOCH %s =======" % epoch
        print "\tLearning rate is %s" % learning_rate

        train_ppl = _process_corpus(net, training_data, mode='train', learning_rate=learning_rate) 
        print "\tTrain PPL is %.3f" % train_ppl

        validation_ppl = _process_corpus(net, validation_data, mode='test')
        print "\tValidation PPL is %.3f" % validation_ppl

        print "\tTime taken: %ds" % (time() - epoch_start)

        if np.log(validation_ppl) * min_improvement > np.log(best_validation_ppl): # Mikolovs recipe
            if not divide:
                divide = True
                print "\tStarting to reduce the learning rate..."
                if validation_ppl > best_validation_ppl:
                    print "\tLoading best model."
                    net = utils.load_model("../out/" + model_name)
            else:
                if validation_ppl < best_validation_ppl:
                    print "\tSaving model."
                    net.save("../out/" + model_name, final=True)
                break
        else:
            print "\tNew best model! Saving..."
            best_validation_ppl = validation_ppl
            final = learning_rate / 2. < min_learning_rate or epoch == max_epochs
            net.save("../out/" + model_name, final)

        if divide:
            learning_rate /= 2.
        
        if learning_rate < min_learning_rate:
            break
            
    print "-"*30
    print "Finished training."
    print "Best validation PPL is %.3f\n\n" % best_validation_ppl

def train(model_name, p1_train_data, p1_dev_data, p2_train_data, p2_dev_data):

    ### PHASE 1 ###    

    training_data = np.load(p1_train_data)
    validation_data = np.load(p1_dev_data)

    assert training_data["batch_size"] == validation_data["batch_size"]
    assert training_data["vocabulary"] == validation_data["vocabulary"]
    assert training_data["punctuations"] == validation_data["punctuations"]

    print "1st phase data loaded..."

    print "Vocabulary size is %d" % utils.get_vocabulary_size(validation_data["vocabulary"])
    print "Training set size is %d" % training_data["total_size"]
    print "Validation set size is %d" % validation_data["total_size"]

    net = models.T_LSTM()
    net.initialize(hidden_size=conf.PHASE1["HIDDEN_SIZE"],
                   projection_size=conf.PHASE1["PROJECTION_SIZE"],
                   in_vocabulary=training_data["vocabulary"],
                   out_vocabulary=training_data["punctuations"],
                   batch_size=training_data["batch_size"],
                   hidden_activation=conf.PHASE1["HIDDEN_ACTIVATION"],
                   bptt_steps=conf.PHASE1["BPTT_STEPS"],
                   use_pauses=False)

    _train(net, training_data, validation_data, model_name, conf.PHASE1["LEARNING_RATE"], conf.PHASE1["MAX_EPOCHS"], conf.PHASE1["MIN_IMPROVEMENT"])

    ### PHASE 2 ###

    if not os.path.isfile(p2_train_data) or not os.path.isfile(p2_train_data):
        print "No second phase data."
        return

    training_data = np.load(p2_train_data)
    validation_data = np.load(p2_dev_data)

    assert training_data["batch_size"] == validation_data["batch_size"] == net.batch_size
    assert training_data["vocabulary"] == validation_data["vocabulary"] == net.in_vocabulary
    assert training_data["punctuations"] == validation_data["punctuations"] == net.out_vocabulary

    print "2nd phase data loaded..."

    print "Training set size is %d" % training_data["total_size"]
    print "Validation set size is %d" % validation_data["total_size"]
    print "Trainging %s pause durations." % ("with" if conf.PHASE2["USE_PAUSES"] else "without")

    t_lstm = net

    net = models.TA_LSTM()
    net.initialize(hidden_size=conf.PHASE2["HIDDEN_SIZE"],
                   t_lstm=t_lstm,
                   out_vocabulary=training_data["punctuations"],
                   batch_size=training_data["batch_size"],
                   hidden_activation=conf.PHASE2["HIDDEN_ACTIVATION"],
                   bptt_steps=conf.PHASE2["BPTT_STEPS"],
                   use_pauses=conf.PHASE2["USE_PAUSES"])

    _train(net, training_data, validation_data, model_name, conf.PHASE2["LEARNING_RATE"], conf.PHASE2["MAX_EPOCHS"], conf.PHASE2["MIN_IMPROVEMENT"])