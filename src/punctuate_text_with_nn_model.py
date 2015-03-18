# coding: utf-8

import utils
import sys
import cPickle
import numpy as np

WRITE_PROBS = False


def write_punctuations(net, unpunctuated_text, output_file_path, punctuation_reverse_map, write_readable_text):
    stream = unpunctuated_text.split()# + ["<END>"]
    if WRITE_PROBS:
        probs = []

    word = None
    pause = 0.

    with open(output_file_path, 'w') as output_file:
        for token in stream:

            if token.startswith("<sil="):
                
                previous_pause = pause
                pause = float(token.replace("<sil=","").replace(">",""))                

                word_index = utils.input_word_index(net.in_vocabulary, word)
                punctuation_index = net.predict_punctuation([word_index], np.array([previous_pause]))[0]

                if WRITE_PROBS:
                    probs.append(net.y[0])
                
                punctuation = punctuation_reverse_map[punctuation_index]

                if punctuation == " ":
                    output_file.write("%s%s" % (punctuation, word))
                else:
                    if write_readable_text:
                        output_file.write("%s %s" % (punctuation[:1], word))
                    else:
                        output_file.write(" %s %s" % (punctuation, word))

            else:
                word = token
    
    if WRITE_PROBS:
        cPickle.dump(np.array(probs), file(output_file_path + ".punct_probs", 'wb'))

if __name__ == "__main__":
    
    if len(sys.argv) > 3:
        model_name = sys.argv[1]    
        net = utils.load_model(model_name)
        net.batch_size = 1
        net.reset_state()
        punctuation_reverse_map = utils.get_reverse_map(net.out_vocabulary)
        
        write_readable_text = bool(int(sys.argv[2]))
         
        output_file_path = sys.argv[3]
        if output_file_path == "-":
            output_file_path = sys.stdout

        if len(sys.argv) > 4:
            with open(sys.argv[4], 'r') as unpunctuated_file:
                unpunctuated_text = " ".join(unpunctuated_file.readlines())
        else:
            unpunctuated_text = " ".join(sys.stdin.readlines())
        
        write_punctuations(net, unpunctuated_text, output_file_path, punctuation_reverse_map, write_readable_text)
        
    else:
        print "Usage: python punctuate_text_with_nn_model.py <model path> <1/0 write readable format> <output file or - for stdout> <unpunctuated text>\n" + \
              "   or: cat <unpunctuated text> | python punctuate_text_with_nn_model.py <model path> <1/0 write readable format> <output file or - for stdout>"
