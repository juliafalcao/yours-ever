# -*- coding: utf-8 -*-

"""
variables to be used across multiple files of the project
"""

# symbols
PARAGRAPH_SIGN = "§"
SENTENCE_SIGN = "●"

# file paths, relative to project root

# VW: main dataset of letters
VW_RAW = "data/vw/vw_raw.csv"
VW_PREPROCESSED = "data/vw/vw_preprocessed.csv"
VW_TOKENIZED = "data/vw/vw_tokenized.json"

# VWP: dataset of paragraphs from the letters
VWP_RAW = "data/vw/vwp_raw.csv"
VWP_PREPROCESSED = "data/vw/vwp_preprocessed.json"
VWP_TOKENIZED = "data/vw/vwp_tokenized.json"
VWP_SCORED = "data/vw/vwp_scored.json"
VWP_CLUSTERED = "data/vw/vwp_clustered.json"

# VWB: dataset of books
VWB_RAW = "data/vw/vwb_raw.csv"
VWB_TOKENIZED = "data/vw/vwb_tokenized.json"


# trained models
TRAINED_DOC2VEC = "src/trained_models/vw_doc2vec"
TRAINED_WORD2VEC = "src/trained_models/vw_word2vec"

# tf-idf stuff
VWP_TFIDF_VOCAB = "src/tfidf/vwp_tfidf_vocab.json"
VWP_TFIDF_MATRIX = "src/tfidf/vwp_tfidf_matrix.npz"

# virginia woolf data
VW_BIRTH = 1882
VW_DEATH = 1941
