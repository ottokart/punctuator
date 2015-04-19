# coding: utf-8

"""
NB! First manually check the files for multiple consecutive punctuation marks, remove punctuation that's not
    going to be predicted by the model and also use fix_broken_pause_file.py to add missing pauses!
    
This script creates the files necessary for evaluation of pause annotated corpus!
1. File without punctuations nor pauses - to compute 'no punctutation' baseline score,
2. File with only pauses but no punctuation - this file will be used to restore punctuation with the model.
3. File with punctuation but no pauses - this will be compared to model output.
"""
def create_files(file_path, punctuations, punctuations_reverse):
    
    with open(file_path, 'r') as source:
        with open(file_path + ".nopause_punct", 'w') as nopause_punct:
            with open(file_path + ".nopause_nopunct", 'w') as nopause_nopunct:
                with open(file_path + ".pause_nopunct", 'w') as pause_nopunct:

                    for line in source:

                        punctuation = 0
                        silence = ""
                        word = ""

                        tokens = line.split()
                        i = 0

                        while i < len(tokens):

                            if tokens[i].startswith("<sil="):
                                silence = tokens[i]

                                i += 1
                                
                                if tokens[i] in punctuations:
                                    punctuation = punctuations[tokens[i]]
                                    i += 1
                                else:
                                    punctuation = punctuations[" "]
                            
                            word = tokens[i] if i < len(tokens) else ""

                            i += 1

                            nopause_punct.write("%s %s" % (punctuations_reverse[punctuation] if punctuation==0 else " " + punctuations_reverse[punctuation], word))
                            nopause_nopunct.write("%s " % word)
                            pause_nopunct.write(" %s %s" % (silence, word))

                        nopause_punct.write("%s" % (punctuations_reverse[punctuation] if punctuation==0 else " " + punctuations_reverse[punctuation]))
                    

punctuations = {" ": 0, ".PERIOD": 1, ",COMMA": 2, "?QUESTIONMARK": 1, ";SEMICOLON": 1, "!EXCLAMATIONMARK": 1, ":COLON": 1}
punctuations_reverse = {0: " ", 1: ".PERIOD", 2: ",COMMA"}

create_files("../data/pauses.test", punctuations, punctuations_reverse)
