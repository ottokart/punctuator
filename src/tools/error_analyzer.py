# coding: utf-8

from numpy import nan

PUNCTUATION = {" ", ".PERIOD", ",COMMA", "?QUESTIONMARK", ";SEMICOLON", "!EXCLAMATIONMARK", ":COLON"}

def write_detailed_errors_to_html(target_paths, predicted_paths):
    
    for target_path, predicted_path in zip(target_paths, predicted_paths):

        target_punctuation = " "
        predicted_punctuation = " "

        t_i = 0
        p_i = 0

        with open(target_path, 'r') as target, open(predicted_path, 'r') as predicted, open(predicted_path+".mistakes.html", 'w') as mistakes:

            mistakes.write("<h2><span style='color:red'>PREDICTED</span>|<span style='color:lime'>EXPECTED</span></h2>")

            target_stream = target.read().split() + ["<END>"]
            predicted_stream = predicted.read().split() + ["<END>"]
            
            while True:

                if target_stream[t_i] in PUNCTUATION:
                    while target_stream[t_i] in PUNCTUATION: # skip multiple consecutive punctuations
                        target_punctuation = target_stream[t_i]
                        t_i += 1
                else:
                    target_punctuation = " "

                if predicted_stream[p_i] in PUNCTUATION:
                    predicted_punctuation = predicted_stream[p_i]
                    p_i += 1
                else:
                    predicted_punctuation = " "

                # Write mistakes into an html file
                if target_punctuation == " ":
                    punct = " "
                    if target_punctuation != predicted_punctuation:
                        targ = "-"
                        pred = predicted_punctuation
                        punct = " <span style='color:red'>%s</span>|<span style='color:lime'>%s</span> " % (pred, targ)
                    mistakes.write(punct + target_stream[t_i])
                else:
                    punct = target_punctuation
                    if target_punctuation != predicted_punctuation:
                        targ = target_punctuation if target_punctuation != " " else "-"
                        pred = predicted_punctuation if predicted_punctuation != " " else "-"
                        punct = " <span style='color:red'>%s</span>|<span style='color:lime'>%s</span> " % (pred, targ)
                    mistakes.write(" " + punct + " " + target_stream[t_i])

                t_i += 1
                p_i += 1

                if t_i >= len(target_stream)-1 and p_i >= len(predicted_stream)-1:
                    break