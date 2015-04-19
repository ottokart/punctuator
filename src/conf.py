# coding: utf-8

# If you change configuration then you might need to reconvert the dataset (delete ../data directory and run main.py again). 

PUNCTUATIONS = {" ": 0, ".PERIOD": 1, ",COMMA": 2}
VOCABULARY_FILE = "../raw_data/vocab"

RANDOM_SEED = 1
SHOW_WPS = True # Show training progress in % and speed in words per second (should be turned of when output is logged into a files)
BATCH_SIZE = 100

PHASE1 = {
            "MAX_EPOCHS": 20,
            "LEARNING_RATE": 0.1,
            "MIN_IMPROVEMENT": 1.003,
            "PROJECTION_SIZE": 100,
            "HIDDEN_SIZE": 100,
            "HIDDEN_ACTIVATION": "Tanh",
            "BPTT_STEPS": 5,
            "TRAIN_DATA": ["../raw_data/train.txt"],
            "DEV_DATA": ["../raw_data/dev.txt"],
         }

PHASE2 = {
            "MAX_EPOCHS": 20,
            "LEARNING_RATE": 0.1,
            "MIN_IMPROVEMENT": 1.003,
            "HIDDEN_SIZE": 100,
            "HIDDEN_ACTIVATION": "Tanh",
            "BPTT_STEPS": 5,
            "TRAIN_DATA": ["../raw_data/pauses.train.txt"],
            "DEV_DATA": ["../raw_data/pauses.dev.txt"],
            "USE_PAUSES": True
         }