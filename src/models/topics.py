# -*- coding: utf-8 -*-

import sys, codecs, warnings
import pandas as pd
import numpy as np
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
def get_document_topics(lda_model, tokens: list) -> list:
	bow = dictionary.doc2bow(tokens)
	topics = lda_model.get_document_topics(bow)
	topics.sort(key=(lambda pair: pair[1]), reverse=True) # sort by highest to lowest probabilities
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
	topic_dists: pd.Series = vw.reset_index()["index"].apply(lambda i: get_document_topics(lda_model, letters[i]))
	topic_dists.sort_index(axis=0)
	vws = pd.DataFrame({f"Topic {t}":[] for t in range(lda_model.num_topics)})
	vws.sort_index(axis=0)
	lines = topic_dists.apply(lambda t: transform_topic_distribution(t, lda_model.num_topics))
	vws = vws.append(list(lines), ignore_index=True)

	assert len(vws) == len(topic_dists)
	return vws


"""
compute average silhouette score
"""
def compute_avg_silhouette(topic_df: pd.DataFrame) -> float:
	points = topic_df.values
	mains = points.argmax(axis=1)

	return silhouette_score(points, mains)

"""
plot and save silhouette scores per letter in corpus
receives: vws dataframe (topic dists)
returns: average silhouette score
"""
def plot_silhouette(vws: pd.DataFrame) -> float:
	n = 0 # ?
	cols = [col for col in vws.columns if "Topic" in col]
	points = vws[cols].values # matrix of shape (n_samples, n_topics)
	mains = points.argmax(axis=1) # array of dominant topics
	fig, ax1 = plt.subplots(1, 1)
	samples_silhouette_values = silhouette_samples(points, mains)
	y_lower = 10 # ?

	for i in range(n_topics):
		ith_topic_silhouette_values = samples_silhouette_values[mains == i]
		ith_topic_silhouette_values.sort()
		size_topic_i = ith_topic_silhouette_values.shape[0]
		y_upper = y_lower + size_topic_i
		color = cm.rainbow(float(i) / n_topics)
		ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_topic_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
		ax1.text(-0.05, y_lower + 0.5 * size_topic_i, str(i))
		y_lower = y_upper + 10

	ax1.set_title("Silhouette plot for " + str(n_topics) + " topics")
	ax1.set_xlabel("Silhouette coefficient")
	ax1.set_ylabel("Topic number")
	silhouette_avg = compute_avg_silhouette(vws)
	ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
	ax1.set_yticks([])
	ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1]) # ?
	plt.savefig(f"{SILHOUETTE_PLOTS_PATH}silhouette_lda{n_topics}.png")
	print("New silhouette plot generated and saved.")
	plt.close("all")

	return silhouette_avg

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
def save_random_letters(vw: pd.DataFrame, n_letters: int = 3, recipient: str = None, year: int = None) -> None:
	assert n_letters > 0
	assert "main" in vw.columns

	vwo = pd.read_csv(VW_ORIGINAL, index_col="index")

	if recipient is not None:
		vwr = vw[vw["recipient"] == recipient].copy()
		assert len(vwr) > 0
		suffix = f"_{''.join([name[0] for name in recipient.split(' ')]).lower()}"
	else:
		vwr = vw
		suffix = ""
	
	if year is not None:
		vwr = vw[vw["year"] == year].copy()
		assert len(vwr) > 0
		suffix = f"_{year}"
	else:
		vwr = vw
		suffix = ""
	
	for t in range(n_topics):
		vwr_t = vwr[vwr["main"] == t]

		if len(vwr_t) < n_letters: # in case there aren't n_letters
			n_letters = len(vwr_t)
		
		chosen_ids = list(vwr_t.sample(n_letters).index) # random sample

		for letter_id in chosen_ids:
			vwo_row: pd.Series = vwo.loc[letter_id]
			vwo_row["tokens"] = vw.at[letter_id, "text"] # save also actual tokens used in model
			vwo_row.to_csv(f"{LDA_LETTERS_PATH}lda{n_topics}_topic{t}{suffix}_random{letter_id}.csv", sep=":", header=True)

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
	dictionary.filter_extremes(no_above=0.07, no_below=IGNORE) # remove most frequent (above 10% of corpus)
	tokens_without_frequent = [dictionary[id] for id in dictionary]
	removed_frequent = [token for token in original_tokens if token not in tokens_without_frequent]
	log(removed_frequent, "removed_frequent_tokens")
	dictionary.filter_extremes(no_below=4, no_above=IGNORE) # remove least frequent (below 4 documents)
	tokens_without_rare = [dictionary[id] for id in dictionary]
	removed_rare = [token for token in tokens_without_frequent if token not in tokens_without_rare]
	log(removed_rare, "removed_rare_tokens")

	print(F"Gensim dictionary initialized and stripped of extremes: {len(removed_frequent)} frequent tokens and {len(removed_rare)} rare tokens. (Dictionary size: {len(original_dict)} -> {len(dictionary)})")

	dictionary_tokens: list = list(dictionary.token2id.keys())
	vw["text"] = vw["text"].apply(lambda tokens: [t for t in tokens if t in dictionary_tokens])
	# remove from vw the tokens that have been removed from dictionary

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

	print(f"Building LDA model for {n_topics} topics.")

	if saved:
		lda = LdaMulticore.load(f"{TRAINED_LDA}{n_topics}")
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
			iterations=80,
			eval_every=5,
			workers=3
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
		for alpha in alpha_values:
			for beta in beta_values:
				result = model(num_topics, alpha=alpha, beta=beta, saved=False, wordclouds=True)
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
	df_results.to_csv(f"{EXEC_LOGS_PATH}comparison_results_{timestamp}.csv")
		
	return results, df_results

"""
plot coherence and silhouette scores for candidate models
receives: results dict (output of compare_models)
"""
def plot_metrics(results: list) -> None:
	fig, (ax1, ax2) = plt.subplots(2, 1)
	ax1.plot(results["topics"], results["coherence"], label="coherence score", marker=".", color="slateblue")
	ax2.plot(results["topics"], results["silhouette"], label = "avg silhouette score", marker=".", color="purple")
	ax1.set_ylabel("Coherence")
	ax2.set_ylabel("Silhouette")
	ax2.set_xlabel("Topics")
	ax1.grid(True)
	ax2.grid(True)
	plt.savefig(f"{GRAPHS_PATH}metrics_per_num_topics.png")

	print("Saved plot of metrics comparison between different numbers of topics.")
	plt.close("all")

"""
generates and saves wordclouds per topic of a given LDA model
receives: vw dataframe with "main" column
"""
def save_topic_wordclouds(vw: pd.DataFrame) -> None:
	assert "main" in vw.columns

	for i in range(n_topics):
		letters: pd.Series = vw[vw["main"] == i]["text"]
		all_letters = concat_all_letters(letters)
		wordcloud = WordCloud(width=600, height=500, background_color="white", max_words=300).generate(all_letters)
		plt.imshow(wordcloud, interpolation="bilinear")
		plt.axis("off")
		plt.savefig(f"{TOPIC_WORDCLOUDS_PATH}wordcloud_lda{n_topics}_topic{i}.png")
		plt.clf()

	print("Saved topic wordclouds for LDA model.")
	plt.close("all")


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

"""
function that makes a bar graph of topic frequencies per years
receives: vw dataframe with "main" topic column
"""
def plot_topics_per_year(vw: pd.DataFrame) -> None:
	assert "main" in vw.columns

	yearly: pd.DataFrame = vw[(vw["main"] != -1) & vw["year"].notnull()][["year", "main"]]
	yearly = yearly.sort_values(by="year")
	yearly["year"] = pd.to_numeric(yearly["year"], downcast="integer")

	for i in range(n_topics):
		yearly[str(i)] = yearly["year"].apply(lambda y: len(yearly[(yearly["year"] == y) & (yearly["main"] == i)]))

	yearly = yearly.drop(["main"], axis=1).drop_duplicates(subset=["year"], keep="first")

	fig, ax = plt.subplots(1, 1, figsize=(14,7))
	years = list(yearly["year"])
	xrange = list(range(len(years)))
	colors = [cm.rainbow(float(i)/n_topics) for i in range(n_topics)]
	labels = [f"Topic {i}" for i in range(n_topics)]

	# make bar graphs
	for i in range(n_topics):
		if i == 0:
			btm = 0

		else: # calculate bottom
			sum: pd.Series = yearly["0"]
			for j in range(1, i):
				sum = sum + yearly[str(j)]

			btm = list(sum)

		ax.bar(x=xrange, height=yearly[str(i)], width=0.8, bottom=btm, color=colors[i], align="center", label=labels[i])

	plt.xticks(ticks=range(len(years)), labels=years, rotation="vertical")
	plt.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)
	plt.legend()
	plt.grid(True, axis="y")
	ax.set_axisbelow(True)
	plt.savefig(f"{GRAPHS_PATH}lda{n_topics}_topic_frequency_per_year.png")
	plt.close("all")

"""
function that makes 12 pie charts for topic frequencies in letters to the top 12 recipients
receives: vw dataframe with "main" topic column
"""
def plot_topics_per_recipient(vw: pd.DataFrame) -> None:
	assert "main" in vw.columns

	recs = vw[(vw["recipient"].notnull()) & (vw["main"] != -1)][["recipient", "main"]]

	for i in range(n_topics):
		recs[str(i)] = recs["recipient"].apply(lambda rec: len(recs[(recs["recipient"] == rec) & (recs["main"] == i)]))

	recs = recs.drop(["main"], axis=1).drop_duplicates(subset=["recipient"], keep="first")

	top_recs = vw[vw["recipient"].notnull()][["recipient"]]
	top_recs = top_recs.reset_index().groupby(["recipient"]).count()[["index"]].rename(columns={"index": "num_letters"})
	top_recs = top_recs.sort_values(by="num_letters", ascending=False)

	recs = recs.merge(top_recs, on="recipient", how="left")
	top_recs_list = list(recs.sort_values(by="num_letters", ascending=False)["recipient"][:12])

	fig, axs = plt.subplots(3, 4, figsize=(14, 9))
	colors = [cm.rainbow(float(i)/n_topics) for i in range(n_topics)]
	columns = [str(i) for i in range(n_topics)]

	it = 0
	for i in range(3):
		for j in range(4):
			rec: str = top_recs_list[it]
			row = recs[recs["recipient"] == rec][columns]
			values = [int(row[col]) for col in row.columns]
			axs[i,j].pie(values, colors=colors, labels=columns, labeldistance=0.5)
			axs[i,j].set_xlabel(rec)
			it += 1

	plt.savefig(f"{GRAPHS_PATH}lda{n_topics}_topic_frequency_per_recipient.png")
	plt.close("all")

# --------------------------------------------------------------------------

vw = read_dataframe(VW_PREPROCESSED)
print("Pre-processed VW dataset successfully imported.")
vw, letters, corpus, dictionary = build_letter_corpus(vw)
vws = pd.DataFrame() # global var

"""
compare models
"""
n_topics = 0 # global var
# results = compare_models([3, 4, 5, 6, 7, 8])
# plot_metrics(results)

"""
evaluate 3-topic model
"""
n_topics = 5
# results, df_results = compare_models([3, 4, 5], ["symmetric", "asymmetric", 0.5], [0.1, 0.3, 0.9])
# df_results = df_results.sort_values(by=["coherence", "silhouette"], ascending=[False, False])

lda3 = model(3, alpha="asymmetric", beta=0.9, saved=False, wordclouds=True, rep_letters=True, plots=True)["model"] # highest silhouette
lda5 = model(5, alpha="asymmetric", beta=0.9, saved=False, wordclouds=True, rep_letters=True, plots=True)["model"] # highest coherence

plt.close("all")
# vws.to_json(VW_ASSIGNED, orient="index")