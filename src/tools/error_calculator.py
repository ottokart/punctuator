# coding: utf-8

"""
Computes and prints the overall classification error and precision, recall, F-score over punctuations.
"""

from numpy import nan

PUNCTUATION = {" ", ".PERIOD", ",COMMA"}
SHOULD_NOT_BE_IN_FILES = {"?QUESTIONMARK", ";SEMICOLON", "!EXCLAMATIONMARK", ":COLON"}

def compute_error(target_paths, predicted_paths):
    counter = 0
    total_correct = 0

    true_positives = {}
    false_positives = {}
    false_negatives = {}

    for target_path, predicted_path in zip(target_paths, predicted_paths):

        target_punctuation = " "
        predicted_punctuation = " "

        t_i = 0
        p_i = 0

        with open(target_path, 'r') as target, open(predicted_path, 'r') as predicted:

            target_stream = target.read().split() + ["<END>"]
            predicted_stream = predicted.read().split() + ["<END>"]
            
            while True:

                assert target_stream[t_i] not in SHOULD_NOT_BE_IN_FILES and predicted_stream[p_i] not in SHOULD_NOT_BE_IN_FILES

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

                correct = target_punctuation == predicted_punctuation

                counter += 1 
                total_correct += correct

                true_positives[target_punctuation] = true_positives.get(target_punctuation, 0.) + float(correct)
                false_positives[predicted_punctuation] = false_positives.get(predicted_punctuation, 0.) + float(not correct)
                false_negatives[target_punctuation] = false_negatives.get(target_punctuation, 0.) + float(not correct)

                assert target_stream[t_i] == predicted_stream[p_i] or predicted_stream[p_i] == "<unk>", \
                       ("File: %s \n" + \
                       "Error: %s (%s) != %s (%s) \n" + \
                       "Target context: %s \n" + \
                       "Predicted context: %s") % \
                       (target_path,
                        target_stream[t_i], t_i, predicted_stream[p_i], p_i,
                        " ".join(target_stream[t_i-2:t_i+2]),
                        " ".join(predicted_stream[p_i-2:p_i+2]))

                t_i += 1
                p_i += 1

                if t_i >= len(target_stream)-1 and p_i >= len(predicted_stream)-1:
                    break

    print "-"*46
    print "{:<16} {:<9} {:<9} {:<9}".format('PUNCTUATION','PRECISION','RECALL','F-SCORE')
    for p in PUNCTUATION:
        precision = (true_positives.get(p,0.) / (true_positives.get(p,0.) + false_positives[p])) if p in false_positives else nan
        recall = (true_positives.get(p,0.) / (true_positives.get(p,0.) + false_negatives[p])) if p in false_negatives else nan
        f_score = (2. * precision * recall / (precision + recall)) if (precision + recall) > 0 else nan        
        print "{:<16} {:<9} {:<9} {:<9}".format(p, round(precision,2), round(recall,2), round(f_score,2))
    print "-"*46
    print "Accuracy: %.2f%%" % (float(total_correct) / float(counter-1) * 100.0)