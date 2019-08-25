# -*- coding: utf-8 -*-

"""
variables to be used across multiple files of the project
"""

# symbols
PARAGRAPH_DELIM = "§"
SENTENCE_DELIM = "●"

# file paths, relative to project root
RAW_LETTERS_PATH = "data/raw/letters/"

# VW: main dataset of letters
VW_ORIGINAL = "data/interim/vw_original.csv"
VW_PREPROCESSED = "data/interim/vw_preprocessed.json"

# VWP: dataset of paragraphs from the letters
VWP_PREPROCESSED = "data/interim/vwp_preprocessed.json"

# trained models
TRAINED_WORD2VEC = "src/trained_models/vw_word2vec"

# tf-idf stuff
VWP_TFIDF_VOCAB = "src/tfidf/vwp_tfidf_vocab.json"
VWP_TFIDF_MATRIX = "src/tfidf/vwp_tfidf_matrix.npz"

# virginia woolf info
VW_BIRTH = 1882
VW_DEATH = 1941
