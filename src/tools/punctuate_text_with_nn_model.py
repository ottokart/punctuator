# coding: utf-8

import sys, os
sys.path.append(os.path.pardir)

from src import utils
import numpy as np

def write_punctuations(net, text_has_pause_duration_tags, unpunctuated_text, output_file_path, punctuation_reverse_map, write_readable_text):
    stream = unpunctuated_text.split()# + ["<END>"]

    word = None
    pause = 0.

    with open(output_file_path, 'w') as output_file:

        if text_has_pause_duration_tags:
            for token in stream:

                if token.startswith("<sil="):
                    
                    previous_pause = pause
                    pause = float(token.replace("<sil=","").replace(">",""))                

                    word_index = utils.input_word_index(net.in_vocabulary, word)
                    punctuation_index = net.predict_punctuation([word_index], np.array([previous_pause]))[0]

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
        else:
            for word in stream:

                word_index = utils.input_word_index(net.in_vocabulary, word)
                punctuation_index = net.predict_punctuation([word_index], np.array([0.0]))[0]

                punctuation = punctuation_reverse_map[punctuation_index]

                if punctuation == " ":
                    output_file.write("%s%s" % (punctuation, word))
                else:
                    if write_readable_text:
                        output_file.write("%s %s" % (punctuation[:1], word))
                    else:
                        output_file.write(" %s %s" % (punctuation, word))
    
if __name__ == "__main__":
    
    if len(sys.argv) > 4:
        model_name = sys.argv[1]    
        net = utils.load_model(model_name)
        net.batch_size = 1
        net.reset_state()
        punctuation_reverse_map = utils.get_reverse_map(net.out_vocabulary)
        
        write_readable_text = bool(int(sys.argv[2]))        
        text_has_pause_duration_tags = bool(int(sys.argv[3]))

        output_file_path = sys.argv[4]
        if output_file_path == "-":
            output_file_path = sys.stdout

        if len(sys.argv) > 5:
            with open(sys.argv[5], 'r') as unpunctuated_file:
                unpunctuated_text = " ".join(unpunctuated_file.readlines())
        else:
            unpunctuated_text = " ".join(sys.stdin.readlines())
        
        write_punctuations(net, text_has_pause_duration_tags, unpunctuated_text, output_file_path, punctuation_reverse_map, write_readable_text)
        
    else:
        print "Execute in src directory." + \
              "Usage: python tools/punctuate_text_with_nn_model.py <model path> <1/0 write readable format> <1/0 text has pause duration tags> <output file path or - for stdout> <unpunctuated text path>\n" + \
              "   or: cat <unpunctuated text path> | python tools/punctuate_text_with_nn_model.py <model path> <1/0 write readable format>  <1/0 text has pause duration tags> <output file path or - for stdout>"
