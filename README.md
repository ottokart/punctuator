# Punctuator
An LSTM RNN for restoring missing punctuation in text.

Model is trained in two stages (second stage is optional though):

1. First stage is trained on punctuation annotated text. Here the model learns to restore puncutation based on textual features only.
2. Optional second stage can be trained on punctuation *and* pause annotated text. In this stage the model learns to combine pause durations with textual features and adapts to the target domain. If pauses are omitted then only adaptation is performed. Second stage with pause durations can be used for example for restoring punctuation in automatic speech recognition system output.

# Requirements
* Python (tested with v.2.7.5)
* Numpy (tested with v.1.8.0)

# Requirements for data:

* Vocabulary file to specify the input vocabulary of the model. One word per line.
* Cleaned text files for training and validation of the first phase model. Punctuation marks that are not going to be restored by the model should be removed or replaced with appropriate substitute (e.g. questionmarks can be replaced with periods etc). Each punctuation symbol must be surrounded by spaces.

  Example:
  ```to be ,COMMA or not to be ,COMMA that is the question .PERIOD```
* *(Optional)* Pause annotated text files for training and validation of the second phase model. These should be cleaned in the same way as the first phase data. Pause durations in seconds should be marked after each word with a special tag `<sil=0.200>`. Punctuation mark, if any, must come after the pause tag.

  Example:
  ```to <sil=0.000> be <sil=0.100> ,COMMA or <sil=0.000> not <sil=0.000> to <sil=0.000> be <sil=0.150> ,COMMA that <sil=0.000> is <sil=0.000> the <sil=0.000> question <sil=1.000> .PERIOD```

  Second phase data can also be without pause annotations to do just target domain adaptation (PHASE2['USE_PAUSES'] in **conf.py** should be changed accordingly then).
  
Make sure that first words of sentences don't have capitalized first letters. This would give the model unfair hints about period locations.

# Configuration

Configuration is located in **conf.py**.

Punctuation annotation symbols are specified in the PUNCTUATIONS dictionary. *.PERIOD* and *,COMMA* are the default and the dictionary should also include *space* (no punctuation).

Location of the vocabulary file is specified in VOCABULARY_FILE.

The locations of the data files can be configured in PHASE1['TRAIN_DATA'], PHASE1['DEV_DATA'], PHASE2['TRAIN_DATA'] and PHASE2['DEV_DATA'].

Changing some configuration options (batch size, data files, pause usage, punctuations or vocabulary) may require deleting the *data* directory so the data files will be reconverted during the next run.

# Usage

Run `python main.py <model_name>`.

Model name is optional. Default is 'model'.

A small example dataset is also included. 

**Tools might not work properly yet!**
