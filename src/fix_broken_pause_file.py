# coding: utf-8

"""
Puts missing pauses between words
"""

def is_word(token):
	return not (token.startswith("<sil=") or token in {".PERIOD", ",COMMA", "?QUESTIONMARK", ";SEMICOLON", "!EXCLAMATIONMARK", ":COLON"})

def is_pause(token):
	return token.startswith("<sil=")

with open("../test_data/pauses.test.orig", 'r') as source:
	with open("../test_data/pauses.test", 'w') as target:
		for line in source:

			previous = ""
			current = ""

			for i, token in enumerate(line.split()):

				previous = current
				current = token

				target.write(previous + " ")

				if is_word(previous) and not is_pause(current) and i > 2:
					target.write("<sil=0.000> ")

			target.write(current + "\n")