# -*- coding: utf-8 -*-

from const import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from scipy import sparse
import typing
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


MIN_WORDS = 10

"""
yet another analysis
"""
vw: pd.DataFrame = pd.read_json(VW_TOKENIZED, orient="index")
vwp: pd.DataFrame = pd.read_json(VWP_TOKENIZED, orient="index")

vwp["words"] = vwp["text"].apply(len)
vwp["unique"] = vwp["text"].apply(set).apply(len)

# print(vwp.head(10).to_string())
print(f"Average word count: {np.average(vwp['words'])}")
print(f"Average unique word count: {np.average(vwp['unique'])}")

# vwp = vwp[vwp["unique"] >= MIN_WORDS] # KEEP ONLY PARAGRAPHS WITH AT LEAST 10 UNIQUE WORDS
# print("Removed paragraphs with less than 10 unique letters from VWP dataframe.")
# print(f"Average unique word count after cleanup: {np.average(vwp['unique'])}")

"""
TF-IDF for word embeddings
"""

def _dummy(text: str) -> str:
	return text

TFIDF = TfidfVectorizer(analyzer="word", preprocessor=_dummy, tokenizer=_dummy) # receives pre-processed and tokenized paragraphs
result: sparse.csr_matrix = TFIDF.fit_transform(vwp["text"])
""" TF-IDF result() output: csr_matrix -- result[doc_id, term_id] = tfidf score """
vocab: dict = {int(key):term for (term, key) in TFIDF.vocabulary_.items()}

with open(VWP_TFIDF_VOCAB, "w+") as vocab_file:
	json.dump(vocab, vocab_file) # save vocab to file

sparse.save_npz(VWP_TFIDF_MATRIX, result) # save result matrix to file

def get_tfidf_scores(doc_id: int) -> list:
	tokens: list = vwp.at[doc_id, "text"]
	scores_row = result[doc_id] # line of the csr matrix that refers to this document
	scores: list = [(vocab[term_id], scores_row[0, term_id]) for term_id in list(scores_row.indices)]
	scores.sort(key = lambda t: t[1], reverse=True)
	return scores # [(term, score)]

vwp["tfidf"] = vwp.index
vwp["tfidf"] = vwp["tfidf"].apply(get_tfidf_scores)
print("Calculated TF-IDF scores for all paragraphs.")

# vwp = vwp[vwp["unique"] >= MIN_WORDS] # KEEP ONLY PARAGRAPHS WITH AT LEAST 10 UNIQUE WORDS
# print("Removed paragraphs with less than 10 unique letters from VWP dataframe.")
# print(f"Average unique word count after cleanup: {np.average(vwp['unique'])}")

# print(vwp.sample(20).to_string())


"""
Word2Vec
"""
model = Word2Vec(
	vwp["text"],
	size=150, # best test results (tested 150, 200, 300, 500)
	window=15, # best test results (tested 2, 10, 15, 20)
	min_count=2,
	workers=6
	)

print("Training Word2Vec model...")
model.train(vwp["text"], total_examples=len(vwp["text"]), epochs=30)
model.save(TRAINED_WORD2VEC)
print("Saved trained Word2Vec model to file.")

# model = Word2Vec.load(TRAINED_WORD2VEC)
# print("Loaded saved Word2Vec model from file.")

# plot
def tsne_plot(model):
	vectors: list = [model.wv[word] for word in model.wv.vocab]
	labels: list = [word for word in model.wv.vocab]
	print("Constructed lists of vectors and labels.")
	
	tsne_model = TSNE(
		perplexity=40,
		n_components=2,
		init="pca",
		n_iter=2500,
		random_state=23
	)

	new_values = tsne_model.fit_transform(vectors)

	x: list = [value[0] for value in new_values]
	y: list = [value[1] for value in new_values]

	plt.figure()

	for i in range(len(x)):
		plt.scatter(x[i], y[i])
		plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5,2), textcoords="offset points", ha="right", va="bottom")
	plt.show()


# tsne_plot(model)