# coding: utf-8

from numpy import nan

PUNCTUATION = {" ", ".PERIOD", ",COMMA", "?QUESTIONMARK", ";SEMICOLON", "!EXCLAMATIONMARK", ":COLON"}

TEST = False

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

        with open(target_path, 'r') as target, open(predicted_path, 'r') as predicted, open(predicted_path+".mistakes.html", 'w') as mistakes:

            mistakes.write("<h2><span style='color:red'>PREDICTED</span>|<span style='color:lime'>EXPECTED</span></h2>")

            target_stream = target.read().split() + ["<END>"]
            predicted_stream = predicted.read().split() + ["<END>"]
            
            if TEST:
                print "TARGET: "+" ".join(target_stream)
                print "PREDICTION: "+" ".join(predicted_stream)

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

                if TEST:
                    print "T: ", target_punctuation, target_stream[t_i]
                    print "P: ", predicted_punctuation, predicted_stream[p_i]

                correct = target_punctuation == predicted_punctuation

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

                counter += 1 
                total_correct += correct

                true_positives[target_punctuation] = true_positives.get(target_punctuation, 0.) + float(correct)
                false_positives[predicted_punctuation] = false_positives.get(predicted_punctuation, 0.) + float(not correct)
                false_negatives[target_punctuation] = false_negatives.get(target_punctuation, 0.) + float(not correct)

                assert target_stream[t_i] == predicted_stream[p_i], \
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

if TEST:
    error = compute_error(["test_target.txt"], ["test_prediction.txt"])
    assert error == 50.0, "Expected 50.0, but got %.2f" % error

print "\nN-gram word predictions: "
compute_error(
    ["../data/pauses.dev.punct"],
    ["../data/pauses.dev.ngram_punct"])

print "\nRNN big-corpus word continued with word+pause predictions with smaller lr 0.01: "
compute_error(
    ["../data/pauses.dev.punct"],
    ["../data/pauses.dev.rnn_large_proj_punct2_pauses"])

print "\nLSTM big-corpus word continued with word+pause predictions with smaller lr 0.01: "
compute_error(
    ["../data/pauses.dev.punct"],
    ["../data/pauses.dev.lstm_large_proj_punct2_pauses"])

#C:\Users\Latitude\Desktop\Projektid\Punctuator\src> python .\punctuate_text_with_nn_model.py  ..\out\model_lstm_continue_with_big_corpus_params 0 ..\data\pauses.ASR.dev.lstm_continue_with_big_corpus_params_punct ..\data\pauses.ASR.dev.pause_nopunct
print "\n#ASR# LSTM big-corpus word continued with word+pause predictions with smaller lr 0.01: "
compute_error(
    ["../data/pauses.ASR.dev.punct"],
    ["../data/pauses.ASR.dev.lstm_continue_with_big_corpus_params_punct"])
