# coding: utf-8

"""
NB! First manually check the files and also use fix_broken_pause_file.py to add missing pauses!
    Also after conversion, check whether .punct file contains any punctuation besides COMMAS and PERIODS.
This script creates the files necessary for evaluation of pause annotated corpus!
1. Unpuntuated file,
2. File with only pauses but nopunct.
3. File with punctuation but no pauses.
"""
def punctuate_file(file_path, punctuations, punctuations_reverse):
    
    with open(file_path, 'r') as source:
        with open(file_path + ".punct", 'w') as punct:
            with open(file_path + ".nopunct", 'w') as nopunct:
                with open(file_path + ".pause_nopunct", 'w') as pause_nopunct:

                    for line in source:

                        punctuation = 0
                        silence = ""
                        word = ""

                        tokens = line.split()[1:] # skip first word which is the title of document
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

                            punct.write("%s %s" % (punctuations_reverse[punctuation] if punctuation==0 else " " + punctuations_reverse[punctuation], word))
                            nopunct.write("%s " % word)
                            pause_nopunct.write(" %s %s" % (silence, word))

                        punct.write("%s" % (punctuations_reverse[punctuation] if punctuation==0 else " " + punctuations_reverse[punctuation]))
                    

punctuations = {" ": 0, ".PERIOD": 1, ",COMMA": 2, "?QUESTIONMARK": 1, ";SEMICOLON": 1, "!EXCLAMATIONMARK": 1, ":COLON": 1}
punctuations_reverse = {0: " ", 1: ".PERIOD", 2: ",COMMA"}

punctuate_file("../test_data/pauses.test", punctuations, punctuations_reverse)
