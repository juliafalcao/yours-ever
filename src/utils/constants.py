# -*- coding: utf-8 -*-

"""
variables to be used across multiple files of the project
"""

# symbols
PARAGRAPH_DELIM = "§"
SENTENCE_DELIM = "●"

# file paths, relative to project root
RAW_LETTERS_PATH = "data/raw/letters/"
GRAPHS_PATH = "reports/graphs/"
SILHOUETTE_PLOTS_PATH = "reports/graphs/silhouette/"
LDA_LETTERS_PATH = "reports/lda_assigned_letters/"
PYLDAVIS_PATH = "reports/pyldavis/"
TOPIC_WORDCLOUDS_PATH = "reports/graphs/topic_wordclouds/"
EXEC_LOGS_PATH = "reports/logs/"

# VW: main dataset of letters
VW_ORIGINAL = "data/interim/vw_original.csv"
VW_PREPROCESSED = "data/interim/vw_preprocessed.json"
VW_ASSIGNED = "data/processed/vw_assigned.json" # topic probabilities

# VWP: dataset of paragraphs from the letters
VWP_PREPROCESSED = "data/interim/vwp_preprocessed.json"

# trained models
TRAINED_WORD2VEC = "models/vw_word2vec.model"
TRAINED_LDA = "models/lda/lda"

# tf-idf stuff
VWP_TFIDF_VOCAB = "src/tfidf/vwp_tfidf_vocab.json"
VWP_TFIDF_MATRIX = "src/tfidf/vwp_tfidf_matrix.npz"

# virginia woolf info
VW_BIRTH = 1882
VW_DEATH = 1941
