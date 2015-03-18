# coding: utf-8
import sys
import numpy as np
import utils
import adaptation_models4 as models

def is_pause(token):
    return token.startswith("<sil=")

def is_word(token):
    return not token.startswith("<") and not token.endswith(">")

def fix_missing_pauses(inputs):
    fixed_inputs = []

    previous = ""
    current = ""

    for i, token in enumerate(inputs):

        previous = current
        current = token

        if previous:
            fixed_inputs.append(previous)

        if is_word(previous) and not is_pause(current) and i > 0:
            fixed_inputs.append("<sil=0.000>")

    fixed_inputs.append(current)
    if not is_pause(current):
        fixed_inputs.append("<sil=0.000>")

    return fixed_inputs

def write_punctuations(net, punctuation_reverse_map, document):
    inputs = document.split()
    inputs = fix_missing_pauses(inputs) + ["<END>"]

    word = None
    pause = 0.
    tags = []

    first_word = True

    for token in inputs:

        if is_pause(token):
            
            previous_pause = pause
            pause = float(token.replace("<sil=","").replace(">",""))                

            word_index = utils.input_word_index(net.in_vocabulary, word)
            punctuation_index = net.predict_punctuation([word_index], np.array([previous_pause]))[0]
            
            if first_word:     
                punctuation = ""
            else:
                punctuation = punctuation_reverse_map[punctuation_index]

            tagstring = " ".join(tags) + " " if tags else ""
            tags = []

            if punctuation.strip() == "":
                sys.stdout.write("%s%s%s" % (punctuation, tagstring, word))
            else:
                sys.stdout.write("%s %s%s" % (punctuation[:1], tagstring, word))

            first_word = False

        else:
            if is_word(token):
                word = token
            else:
                tags.append(token)

    sys.stdout.write("\n")
    sys.stdout.flush()

if __name__ == "__main__":
    
    model_name = "../out/model_tanel4"
    model = np.load(model_name)
    net = getattr(models, model["type"])()
    net.load(model)
    net.batch_size = 1

    punctuation_reverse_map = utils.get_reverse_map(net.out_vocabulary)

    for line in iter(sys.stdin.readline, ""):
        net.reset_state()
        write_punctuations(net, punctuation_reverse_map, line)
    
