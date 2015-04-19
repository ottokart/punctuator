# coding: utf-8

"""
Puts missing pauses between words. Creates a fixed copy of input file with a suffix "_fixed"
"""

import sys
import conf

def is_word(token):
	return not (token.startswith("<sil=") or token in conf.PUNCTUATIONS)

def is_pause(token):
	return token.startswith("<sil=")

if __name__ == "__main__":

	assert len(sys.argv) > 1, "Give the path to the file to fix"

	file_path = sys.argv[1]

	with open(file_path, 'r') as source:
		with open(file_path + "_fixed", 'w') as target:
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