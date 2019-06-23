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

WORDS_IN_PARAGRAPH_EMBEDDING = 10

def flatten_paragraph(paragraph: list) -> list: # convert list of sentences to list of tokens (opposite of split_into_sentences)
	return [word for sentence in paragraph for word in sentence]

MIN_WORDS = 10

"""
word analysis and TF-IDF calculations
"""
def calculate_tfidf():
	vw: pd.DataFrame = pd.read_json(VW_TOKENIZED, orient="index")
	vwp: pd.DataFrame = pd.read_json(VWP_TOKENIZED, orient="index")

	""" yet another analysis """
	vwp["words"] = vwp["text"].apply(flatten_paragraph).apply(len)
	vwp["unique"] = vwp["text"].apply(flatten_paragraph).apply(set).apply(len)

	# print(vwp.head(10).to_string())
	# print(f"Average word count: {np.average(vwp['words'])}")
	# print(f"Average unique word count: {np.average(vwp['unique'])}")

	# vwp = vwp[vwp["unique"] >= MIN_WORDS] # KEEP ONLY PARAGRAPHS WITH AT LEAST 10 UNIQUE WORDS
	# print("Removed paragraphs with less than 10 unique letters from VWP dataframe.")
	# print(f"Average unique word count after cleanup: {np.average(vwp['unique'])}")

	""" TF-IDF for word embeddings """
	def _dummy(text: str) -> str:
		return text

	TFIDF = TfidfVectorizer(analyzer="word", preprocessor=_dummy, tokenizer=_dummy) # receives pre-processed and tokenized paragraphs
	result: sparse.csr_matrix = TFIDF.fit_transform(vwp["text"].apply(flatten_paragraph))
	""" TF-IDF result() output: csr_matrix -- result[doc_id, term_id] = tfidf score """
	vocab: dict = {int(key):term for (term, key) in TFIDF.vocabulary_.items()}

	with open(VWP_TFIDF_VOCAB, "w+") as vocab_file:
		json.dump(vocab, vocab_file) # save vocab to file

	sparse.save_npz(VWP_TFIDF_MATRIX, result) # save result matrix to file

	def get_tfidf_scores(doc_id: int) -> list:
		scores_row = result[doc_id] # line of the csr matrix that refers to this document
		scores: list = [(vocab[term_id], scores_row[0, term_id]) for term_id in list(scores_row.indices)]
		scores.sort(key = lambda t: t[1], reverse=True)
		return scores # [(term, score), ...]

	vwp["tfidf"] = vwp.index
	vwp["tfidf"] = vwp["tfidf"].apply(get_tfidf_scores)
	print("Calculated TF-IDF scores for all paragraphs.")

	# vwp = vwp[vwp["unique"] >= MIN_WORDS] # KEEP ONLY PARAGRAPHS WITH AT LEAST 10 UNIQUE WORDS
	# print("Removed paragraphs with less than 10 unique letters from VWP dataframe.")
	# print(f"Average unique word count after cleanup: {np.average(vwp['unique'])}")

	# print(vwp.sample(20).to_string())

	vwp.to_json(VWP_SCORED, orient="index")


"""
word embeddings (Word2Vec)
"""
def create_word_embeddings():
	vwp: pd.DataFrame = pd.read_json(VWP_TOKENIZED, orient="index")
	vwb: pd.DataFrame = pd.read_json(VWB_TOKENIZED, orient="index")

	# assemble corpus
	corpus = [] # list books and letters as sentences (which are lists of tokens)

	for _, paragraph in vwp["text"].iteritems():
		for sentence in paragraph:
			corpus.append(sentence)

	for _, book in vwb["text"].iteritems():
		for sentence in book:
			corpus.append(sentence)

	print(f"Corpus size: {len(corpus)} sentences")

	model = Word2Vec(
		corpus,
		size=250, # best test results (tested 150, 200, 300, 500)
		window=15, # best test results (tested 2, 10, 15, 20)
		min_count=1,
		workers=6
		)

	print("Training Word2Vec model with all the sentences from VW's books and letters...")
	model.train(corpus, total_examples=len(corpus), epochs=45)
	model.save(TRAINED_WORD2VEC)

	print("Saved trained Word2Vec model to file.")

"""
paragraph embeddings
"""
def create_paragraph_embeddings():

	vwp = pd.read_json(VWP_SCORED, orient="index")
	model = Word2Vec.load(TRAINED_WORD2VEC)
	vwp = vwp[vwp["tfidf"].apply(len) > WORDS_IN_PARAGRAPH_EMBEDDING]

	def get_paragraph_embedding(tfidf_scores: list) -> list: # [(word, score), ...]
		# paragraph = [model.wv[tfidf_scores[i][0]] for i in range(10)] # shape: (10,250)
		# word_embeddings: list = [model.wv[tfidf_scores[i][0]] for i in range(10)]

		def nearest(array: list, value: float):
			id: int = (np.abs(array - value)).argmin() # index of the element from the array which is nearest to the value passed
			return id
		
		WORD = 0
		SCORE = 1
		
		words: list = [tfidf_scores[i][WORD] for i in range(len(tfidf_scores))]
		scores: list = [tfidf_scores[i][SCORE] for i in range(len(tfidf_scores))]
		avg_score: float = np.average(scores)
		words_to_embed: list = []

		

		for i in range(WORDS_IN_PARAGRAPH_EMBEDDING):
			id_nearest: int = nearest(scores, avg_score) # get value closest to average
			words_to_embed.append(words[id_nearest]) # add respective word to list

			del words[id_nearest]
			del scores[id_nearest] # remove word and value from list

			# next iteration will get the score closest to average from the list with this iteration's word and score removed


		word_embeddings = [model.wv[word] for word in words_to_embed]
		paragraph = np.concatenate(word_embeddings) # shape (2500,)

		return paragraph


	vwp["embedding"] = vwp["tfidf"].apply(get_paragraph_embedding)
	print(vwp[["letter", "offset", "embedding"]].sample(10).to_string())

	vwp.to_json(VWP_SCORED, orient="index")

def main():
	calculate_tfidf()
	create_word_embeddings()
	create_paragraph_embeddings()

if __name__ == '__main__':
	main()