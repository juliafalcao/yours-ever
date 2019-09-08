# -*- coding: utf-8 -*-

import sys, codecs, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import deepcopy
from gensim import corpora
from gensim.models import LdaMulticore, CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
from sklearn.metrics import silhouette_samples, silhouette_score

sys.path.append("src/utils")
from constants import *
from utils import *

warnings.filterwarnings("ignore",category=DeprecationWarning)

N_TOPICS: int = 5

vw: pd.DataFrame = read_dataframe(VW_PREPROCESSED)
print("Pre-processed VW dataset successfully imported.")

"""
cleanup
"""
vw["unique_words"] = vw["text"].apply(set).apply(len)
vw["text"] = vw["text"].apply(lambda tokens: [t for t in tokens if t != PARAGRAPH_DELIM]) # remove paragraph signs

"""
obtain representations
"""

vw = vw.sort_index(axis=0)
letters = list(vw["text"]) # each line is a tokenized letter
assert list(vw.index) == list(range(len(vw))) # assert that index is sequential

# create dictionary
dictionary = corpora.Dictionary(letters)
# dict_before = deepcopy(dictionary)
# tokens_before = [dict_before[id] for id in dict_before]
dictionary.filter_n_most_frequent(100)
# tokens_after = [dictionary[id] for id in dictionary]
# removed = [token for token in tokens_before if token not in tokens_after] # store removed words

corpus = [dictionary.doc2bow(letter) for letter in letters]
print("Gensim dictionary and corpus initialized.")

# LDA
lda = LdaMulticore(
	corpus,
	num_topics=N_TOPICS,
	id2word=dictionary,
	passes=15,
	random_state=1,
	workers=3 # num cores - 1
	# alpha='auto',
	# per_word_topics=True
)
lda.save(TRAINED_LDA)

def print_model_topics(lda_model):
	for topic, words in lda_model.show_topics(formatted=True, num_topics=N_TOPICS, num_words=20):
			print(str(topic) + ": " + words + "\n")

"""
function that receives a list of documents and a particular document id
and returns its distribution of assigned topics
"""
def get_document_topics(documents: list, id: int) -> list:
	bow = dictionary.doc2bow(documents[id])
	topics = lda.get_document_topics(bow)
	topics.sort(key=(lambda pair: pair[1]), reverse=True) # sort by highest to lowest probabilities
	return topics

# metrics
perplexity = lda.log_perplexity(corpus) # the lower the better
print(f"Perplexity score: {perplexity}")
coherence = CoherenceModel(model=lda, texts=letters, dictionary=dictionary, coherence='c_v').get_coherence()
print(f"Coherence score: {coherence}")


# create new dataframe with letter index, original text, processed text and scores for each topic
vwt = vw.reset_index()[["index", "text"]]
vwo = pd.read_csv(VW_ORIGINAL, index_col="index")
vwt["raw_text"] = vwt["index"].apply(lambda index: vwo.at[index, "text"]) # recover original text
vwt["topics"] = vwt["index"].apply(lambda index: get_document_topics(letters, index))
vwt["dominant_topic"] = vwt["topics"].apply(lambda topics: topics[0][0])
vwt["dominant_topic_score"] = vwt["topics"].apply(lambda topics: topics[0][1])
vwt = vwt.set_index("index", drop=True).rename(columns={"text": "processed_text"})
vwt = vwt.sort_values(by=["dominant_topic", "dominant_topic_score"], ascending=[True, False]) # sort by best representatives per topic

for t in range(N_TOPICS):
	reps = vwt[vwt["dominant_topic"] == t]
	best_rep = list(reps.index)[0]
	print(f"\nHIGHEST SCORED DOCUMENT FOR TOPIC {t}: {vwt.at[i, 'raw_text']}")


# vis = pyLDAvis.gensim.prepare(topic_model=lda, corpus=corpus, dictionary=dictionary, n_jobs=3)
# pyLDAvis.show(vis)