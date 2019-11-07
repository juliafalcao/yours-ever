# -*- coding: utf-8 -*-

import sys, codecs
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import deepcopy
from nltk import pos_tag
from gensim import corpora
from gensim.models import LdaMulticore, CoherenceModel
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim
from sklearn.metrics import silhouette_samples, silhouette_score
from wordcloud import WordCloud
from time import time

sys.path.append("src/utils")
from constants import *
from utils import *
from plotting import *

"""
constants
"""
# N_MOST_FREQUENT_TO_REMOVE: int = 150
IGNORE: int = 1

"""
prints topics as word distributions
1: 0.004*"upon" + 0.003*"clive" + 0.003*"man" + 0.003*"poor"
"""
def print_model_topics(lda_model, n_words: int = 10):
	for topic, words in lda_model.show_topics(formatted=True, num_topics=lda_model.num_topics, num_words=n_words):
			print(str(topic) + ": " + words + "\n")

"""
function that receives a model, a list of documents and a particular document id
and returns its distribution of assigned topics
"""
def get_document_topics(lda_model, tokens: list, letter_id = -1) -> list:
	bow = dictionary.doc2bow(tokens)
	res = lda_model.get_document_topics(bow, per_word_topics=True)
	topics = res[0]
	phi = res[2]
	topics.sort(key=(lambda pair: pair[1]), reverse=True) # sort by highest to lowest probabilities
	# insert per_word_topics into matrix
	for (word_id, topic_phi) in phi:
		occurs = int(np.ceil(np.sum([phi for (topic, phi) in topic_phi], axis=0))) # number of word occurrences in doc
		for (topic, phi) in topic_phi:
			pwt[word_id][topic][letter_id] = phi/occurs
			# dividing each word's phi value by the number of occurrences so that they'll add up to 1 in the end
	return topics


def get_paragraph_topics(lda_model, tokens: list) -> list:
	bow = dictionary.doc2bow(tokens)
	topics = lda_model.get_document_topics(bow)
	topics.sort(key=(lambda pair: pair[1]), reverse=True) # sort by highest to lowest probabilities
	return topics

"""
receives: [(4, 0.79440266), (1, 0.20198905)]
returns: {'Topic 4': 0.79440266, 'Topic 1': 0.20198905, 'Topic 0': 0.0, 'Topic 2': 0.0, 'Topic 3': 0.0}
"""
def transform_topic_distribution(topic_dist: list, num_topics: int) -> dict:
	topic_dict = {f"Topic {num}" : prob for (num, prob) in topic_dist}

	if len(topic_dict) < num_topics:
		missing_topics = [t for t in range(num_topics) if t not in list(dict(topic_dist).keys())]
		topic_dict.update({f"Topic {t}" : 0.0 for t in missing_topics})

	return topic_dict

"""
function that receives a model and returns a dataframe of topic distributions (columns) per letters (rows)
       Topic 0   Topic 1   Topic 2   Topic 3   Topic 4
0     0.000000  0.000000  0.000000  0.000000  0.982988
1     0.000000  0.000000  0.282820  0.286516  0.424436
      ...
"""
def get_topic_dists_dataframe(lda_model) -> pd.DataFrame:
	topic_dists: pd.Series = vw.reset_index()["index"].apply(lambda i: get_document_topics(lda_model, letters[i], i))
	topic_dists.sort_index(axis=0)
	vws = pd.DataFrame({f"Topic {t}":[] for t in range(lda_model.num_topics)})
	vws.sort_index(axis=0)
	lines = topic_dists.apply(lambda t: transform_topic_distribution(t, lda_model.num_topics))
	vws = vws.append(list(lines), ignore_index=True)

	assert len(vws) == len(topic_dists)
	return vws


"""
function that saves most representative letters of each topic to text files for manual validation
# TODO: receives?
"""
def save_representative_letters(vws: pd.DataFrame, n_letters: int = 3, recipient: str = None) -> None:
	assert n_letters > 0
	assert "main" in vws.columns

	vwo = pd.read_csv(VW_ORIGINAL, index_col="index")

	if recipient is not None:
		vws["recipient"] = vw["recipient"]
		vws_copy = vws[vws["recipient"] == recipient].copy()
		assert len(vws_copy) > 0
		acronym = f"_{''.join([name[0] for name in recipient.split(' ')]).lower()}"
	else:
		vws_copy = vws
		acronym = ""
	
	for t in range(n_topics):
		vws_t = vws_copy[vws_copy["main"] == t]
		vws_t = vws_t.sort_values(by=f"Topic {t}", ascending=False)

		if len(vws_t) < n_letters: # in case there aren't n_letters
			n_letters = len(vws_t)

		for i in range(n_letters):
			letter_id = vws_t.index[i]
			vwo_row: pd.Series = vwo.loc[letter_id]
			vwo_row["tokens"] = vw.at[letter_id, "text"] # save also actual tokens used in model
			vwo_row.to_csv(f"{LDA_LETTERS_PATH}lda{n_topics}{acronym}_topic{t}_letter{i}.csv", sep=":", header=True)

	print("Saved most representative letters for LDA model.")


"""
save n random letters assigned to all topics (for validation)
"""
def save_random_letters(vws: pd.DataFrame, n_letters: int = 3, recipient: str = None, year: int = None) -> None:
	assert n_letters > 0
	assert "main" in vws.columns

	vwo = pd.read_csv(VW_ORIGINAL, index_col="index")

	if recipient is not None:
		vwr = vws[vw["recipient"] == recipient].copy()
		assert len(vwr) > 0
		suffix = f"_{''.join([name[0] for name in recipient.split(' ')]).lower()}"
	else:
		vwr = vws
		suffix = ""
	
	if year is not None:
		vwr = vws[vws["year"] == year].copy()
		assert len(vwr) > 0
		suffix = f"_{year}"
	else:
		vwr = vws
		suffix = ""
	
	for t in range(n_topics):
		vwr_t = vwr[vwr["main"] == t]

		if len(vwr_t) < n_letters: # in case there aren't n_letters
			n_letters = len(vwr_t)
		
		chosen_ids = list(vwr_t.sample(n_letters).index) # random sample

		for letter_id in chosen_ids:
			vwo_row: pd.Series = vwo.loc[letter_id]
			vwo_row["tokens"] = vws.at[letter_id, "text"] # save also actual tokens used in model
			vwo_row.to_csv(f"{LDA_LETTERS_PATH}lda{n_topics}_topic{t}{suffix}_random{letter_id}_{suffix}.csv", sep=":", header=True)

	print(f"Saved {n_letters} random letters for each topic of LDA model.")

"""
create dictionary and corpus of bags-of-words
"""
def build_letter_corpus(vw: pd.DataFrame) -> (pd.DataFrame, list, list, corpora.dictionary.Dictionary):
	vw["text"] = vw["text"].apply(lambda tokens: [t for t in tokens if t != PARAGRAPH_DELIM]) # remove paragraph delim

	vw = vw.sort_index(axis=0)
	letters = list(vw["text"]) # each line is a tokenized letter
	dictionary = corpora.Dictionary(letters)

	"""
	remove extremes (most frequent and least frequent) tokens
	"""
	original_dict = deepcopy(dictionary) # for logging purposes
	original_tokens = [original_dict[id] for id in original_dict]
	dictionary.filter_extremes(no_above=0.1, no_below=IGNORE) # remove most frequent
	tokens_without_frequent = [dictionary[id] for id in dictionary]
	removed_frequent = [token for token in original_tokens if token not in tokens_without_frequent]
	log(removed_frequent, "removed_frequent_tokens")
	dictionary.filter_extremes(no_below=4, no_above=IGNORE) # remove least frequent (below 4 documents)
	tokens_without_rare = [dictionary[id] for id in dictionary]
	removed_rare = [token for token in tokens_without_frequent if token not in tokens_without_rare]
	log(removed_rare, "removed_rare_tokens")

	print(F"Gensim dictionary initialized and stripped of extremes: {len(removed_frequent)} frequent tokens and {len(removed_rare)} rare tokens. (Dictionary size: {len(original_dict)} -> {len(dictionary)})")

	dictionary_tokens: list = list(dictionary.token2id.keys())
	# vw["text"] = vw["text"].apply(lambda tokens: [t for t in tokens if t in dictionary_tokens])
	# remove from vw the tokens that have been removed from dictionary
	# [!] takes ages to run

	corpus = [dictionary.doc2bow(letter) for letter in letters]
	print("Gensim corpus of BOWs initialized.")

	return vw, letters, corpus, dictionary


"""
function that builds an LDA model and returns the model and some evaluation metrics
parameters:
	n_topics: number of topics (K) for the LDA model
	beta: the LDA beta ('eta' on gensim) hyperparameter
	saved: whether to load saved LDA model (True) or train from scratch and save (False)
	pyldavis: whether to prepare [long execution time!] and save HTML of pyLDAvis visualization model
	wordclouds: whether to generate new topic wordclouds
	rep_letters: whether to save most representative letters for each topic
	plots: whether to plot and save the graphs of topics per year and per recipient
"""
def model(n_topics, alpha=None, beta=None, saved=False, pyldavis=False, wordclouds=False, rep_letters=False, plots=False) -> dict:
	assert n_topics >= 2

	"""
	aux functions to make sure it's loading the desired model
	"""
	def verify_alpha(lda_model, given):
		actual: list = lda_model.alpha
		if given == "asymmetric":
			return not np.isclose(actual[0], actual[-1])
		elif given == "symmetric":
			return np.isclose(actual[0], actual[-1])
		else:
			return np.isclose(given, actual[0]) and np.isclose(given, actual[-1])
	
	def verify_beta(lda_model, given):
		actual = lda_model.eta
		if type(given) == float:
			return np.isclose(given, actual[0]) and np.isclose(given, actual[-1]) # basic == comparison doesn't work bc floats suck
		else:
			return False

	print(f"Building LDA model for {n_topics} topics.")

	if saved:
		lda = LdaMulticore.load(f"{TRAINED_LDA}{n_topics}")

		if not (verify_alpha(lda, alpha) and verify_beta(lda, beta)):
			print("Loaded model didn't pass parameter verification; train it from scratch or load the correct one.")
			return

		print(f"Trained LDA model with {n_topics} topics loaded successfully.")

	else:
		lda = LdaMulticore(
			corpus,
			num_topics=n_topics,
			id2word=dictionary,
			passes=20,
			alpha=alpha if alpha is not None else "symmetric", # default
			eta=beta,
			random_state=1,
			iterations=100,
			eval_every=5,
			workers=3,
			per_word_topics=True
		)

		lda.save(f"{TRAINED_LDA}{n_topics}")
		print(f"LDA model with {n_topics} topics trained and saved successfully.")

	"""
	save topic assignment info in dataframes
	[!] alters global variables
	"""
	global vw
	global vws
	vws = get_topic_dists_dataframe(lda)
	vw, vws = set_main_topics(vw, vws)

	"""
	coherence and silhouette scores
	"""
	coherence = CoherenceModel(model=lda, texts=letters, dictionary=dictionary, coherence='c_v').get_coherence()
	print(f"Coherence score: {coherence}") # the higher the better

	avg_silhouette = plot_silhouette(vws)
	print(f"Average silhouette coefficient: {avg_silhouette}") # the higher the better

	"""
	other validation methods
	"""
	if pyldavis:
		vis = pyLDAvis.gensim.prepare(topic_model=lda, corpus=corpus, dictionary=dictionary, n_jobs=3)
		pyLDAvis.save_html(vis, f"{PYLDAVIS_PATH}/lda{n_topics}.html")
	
	if rep_letters:
		save_representative_letters(vws, 3)
	
	if wordclouds:
		save_topic_wordclouds(vw)

	if plots:	
		plot_topics_per_year(vw)
		plot_topics_per_recipient(vw)

	return {
		"model": lda,
		"num_topics": n_topics,
		"alpha": alpha,
		"beta": beta,
		"coherence": coherence,
		"silhouette": avg_silhouette
		}


"""
function to compare LDA models built with the given N values
receives: list of candidate n_topics to compare, i.e. [3, 5, 7, 10]
returns: results dataframe
		topics  coherence  silhouette
	0       3   0.242163    0.627356
	1       4   0.198382    0.629180
"""
def compare_models(k_values: list, alpha_values: list, beta_values: list) -> pd.DataFrame:
	timestamp = int(time())
	results: list = []
	df_results = pd.DataFrame({"num_topics": [], "alpha": [], "beta": [], "coherence": [], "silhouette": []})

	for num_topics in k_values:
		global n_topics
		n_topics = num_topics

		for alpha in alpha_values:
			for beta in beta_values:
				print(f"BUILDING: num_topics={num_topics}, alpha={alpha}, beta={beta}")
				result = model(num_topics, alpha=alpha, beta=beta, saved=False)
				results.append(result)
	

	for result in results:
		df_results = df_results.append({
			"num_topics": result["num_topics"],
			"alpha": result["alpha"],
			"beta": result["beta"],
			"coherence": result["coherence"],
			"silhouette": result["silhouette"]
		}, ignore_index=True)

	df_results["num_topics"] = pd.to_numeric(df_results["num_topics"], downcast="integer")
	df_results.to_csv(f"{EXEC_LOGS_PATH}comparison_results_{timestamp}.csv", index_label="index")
		
	return results, df_results

"""
function that includes main topic column in vw dataframe, or -1 if all topics have the same probability
receives: vw (original dataset); vws (topics distribution dataframe)
"""
def set_main_topics(vw: pd.DataFrame, vws: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

	# function that verifies if all elements in an array are equal
	def equal_values(v: np.ndarray) -> bool:
		v = np.sort(v, axis=0)
		return v[0] == v[len(v)-1] # if first and last elements of sorted array are equal, the array only has equal elements

	vws["main"] = np.argmax([vws[f"Topic {t}"] for t in range(n_topics)], axis=0)
	prob_cols: list = [col for col in vws.columns if "Topic" in col]
	vws.loc[vws.apply(lambda row: equal_values([row[col] for col in prob_cols]), axis=1), "main"] = -1

	vw["main"] = vws["main"]
	return vw, vws

# --------------------------------------------------------------------------

vw = read_dataframe(VW_PREPROCESSED)
print("Pre-processed VW dataset successfully imported.")
vw, letters, corpus, dictionary = build_letter_corpus(vw)
vws = pd.DataFrame() # global var

n_topics = 5

W = len(dictionary)
K = n_topics
N = len(corpus)
pwt = np.zeros((W, K, N))

lda5a9 = model(n_topics, alpha="asymmetric", beta=0.9, saved=False)["model"]

# TODO: create this inside model(); pwt needs to be initialized before get_topic_dists_dataframe is called
# # topics x words x documents 3d matrix
# to access a single line: [pwt[w][k][n] for i in range(K)] fix 2 and vary 1, or fix 1 and vary 2

save_topic_wordclouds(pwt)

"""
compare results
"""
# results = pd.read_csv("reports/logs/comparison_results.csv", index_col="index")
# plot_metrics(results, variable="num_topics", fixed_alpha="asymmetric", fixed_beta=0.9)
# plot_results(results)
