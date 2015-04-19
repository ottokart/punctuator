# coding: utf-8

import sys
import os
import utils
import trainer
import conf
import converter

import numpy as np

from time import time

PHASE1_TRAIN_PATH = "../data/train1"
PHASE1_DEV_PATH = "../data/dev1"

PHASE2_TRAIN_PATH = "../data/train2"
PHASE2_DEV_PATH = "../data/dev2"

if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "model"

    np.random.seed(conf.RANDOM_SEED)

    t0 = time()

    ### convert data ###
    if not os.path.exists("../data"):

        print "Converting data...\n"

        os.makedirs("../data")

        vocabulary = utils.load_vocabulary(conf.VOCABULARY_FILE)

        converter.convert_files(conf.PHASE1["TRAIN_DATA"], vocabulary, conf.PUNCTUATIONS, conf.BATCH_SIZE, False, PHASE1_TRAIN_PATH)
        converter.convert_files(conf.PHASE1["DEV_DATA"], vocabulary, conf.PUNCTUATIONS, conf.BATCH_SIZE, False, PHASE1_DEV_PATH)
        if conf.PHASE2["TRAIN_DATA"] and conf.PHASE2["DEV_DATA"]:
            converter.convert_files(conf.PHASE2["TRAIN_DATA"], vocabulary, conf.PUNCTUATIONS, conf.BATCH_SIZE, conf.PHASE2["USE_PAUSES"], PHASE2_TRAIN_PATH)
            converter.convert_files(conf.PHASE2["DEV_DATA"], vocabulary, conf.PUNCTUATIONS, conf.BATCH_SIZE, conf.PHASE2["USE_PAUSES"], PHASE2_DEV_PATH)

    ### train model ###
    print "Training model...\n"

    if not os.path.exists("../out"):
        os.makedirs("../out")
    
    trainer.train(model_name, PHASE1_TRAIN_PATH, PHASE1_DEV_PATH, PHASE2_TRAIN_PATH, PHASE2_DEV_PATH)

    print "Done in %.2f minutes" % ((time() - t0) / 60.)