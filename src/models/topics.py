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
import pyLDAvis
import pyLDAvis.gensim
from sklearn.metrics import silhouette_samples, silhouette_score
sys.path.append("src/utils")
from constants import *
from utils import *

"""
constants
"""
# UNIQUE_WORDS_THRESHOLD: int = 10
N_MOST_FREQUENT_TO_REMOVE: int = 100

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
def get_document_topics(lda_model, documents: list, id: int) -> list:
	bow = dictionary.doc2bow(documents[id])
	topics = lda_model.get_document_topics(bow)
	topics.sort(key=(lambda pair: pair[1]), reverse=True) # sort by highest to lowest probabilities

	return topics

"""
argument: [(4, 0.79440266), (1, 0.20198905)]
return: {'Topic 4': 0.79440266, 'Topic 1': 0.20198905, 'Topic 0': 0.0, 'Topic 2': 0.0, 'Topic 3': 0.0}
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
	topic_dists: pd.Series = vw.reset_index()["index"].apply(lambda i: get_document_topics(lda_model, letters, i))
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
	dominant_topics = points.argmax(axis=1)

	return silhouette_score(points, dominant_topics)

"""
plot silhouette scores per letter in corpus
"""
def plot_silhouette(lda_model, corpus) -> None:
	n = 0
	vws = get_topic_dists_dataframe(lda_model)
	points = vws.values # matrix of shape (n_samples, n_topics)
	dominant_topics = points.argmax(axis=1) # array of dominant topics
	fig, ax1 = plt.subplots(1, 1)
	samples_silhouette_values = silhouette_samples(points, dominant_topics)
	y_lower = 10 # ?

	for i in range(lda_model.num_topics):
		ith_topic_silhouette_values = samples_silhouette_values[dominant_topics == i]
		ith_topic_silhouette_values.sort()
		size_topic_i = ith_topic_silhouette_values.shape[0]
		y_upper = y_lower + size_topic_i
		color = cm.rainbow(float(i) / lda_model.num_topics)
		ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_topic_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
		ax1.text(-0.05, y_lower + 0.5 * size_topic_i, str(i))
		y_lower = y_upper + 10

	ax1.set_title("Silhouette plot for " + str(lda_model.num_topics) + " topics")
	ax1.set_xlabel("Silhouette coefficient")
	ax1.set_ylabel("Topic number")
	silhouette_avg = compute_avg_silhouette(vws)
	print(f"Average silhouette score: {silhouette_avg}")
	ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
	ax1.set_yticks([])
	ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1]) # ?
	plt.savefig(f"{SILHOUETTE_PLOTS_PATH}silhouette_lda{lda_model.num_topics}.png")
	# plt.show()

	return silhouette_avg

"""
function that saves most representative letters of each topic to text files for validation
"""
def save_representative_letters(lda_model, n_letters: int = 3):
	vws = get_topic_dists_dataframe(lda_model)
	vwo = pd.read_csv(VW_ORIGINAL, index_col="index")
	vws["main"] = np.argmax([vws[f"Topic {t}"] for t in range(lda_model.num_topics)], axis=0)

	for t in range(lda_model.num_topics):
		vws_t = vws[vws["main"] == t]
		vws_t = vws_t.sort_values(by=f"Topic {t}", ascending=False)

		for i in range(n_letters):
			letter_id = vws_t.index[i]
			vwo_row: pd.Series = vwo.loc[letter_id]
			vwo_row.to_csv(f"{LDA_LETTERS_PATH}/lda{lda_model.num_topics}_topic{t}_letter{i}.csv", sep=":", header=True)



vw: pd.DataFrame = read_dataframe(VW_PREPROCESSED)
print("Pre-processed VW dataset successfully imported.")

"""
create dictionary and corpus of bags-of-words
"""
vw["text"] = vw["text"].apply(lambda tokens: [t for t in tokens if t != PARAGRAPH_DELIM]) # remove paragraph signs

vw = vw.sort_index(axis=0)
letters = list(vw["text"]) # each line is a tokenized letter

dictionary = corpora.Dictionary(letters)
# dict_before = deepcopy(dictionary)
# tokens_before = [dict_before[id] for id in dict_before]
dictionary.filter_n_most_frequent(N_MOST_FREQUENT_TO_REMOVE)
# tokens_after = [dictionary[id] for id in dictionary]
# removed = [token for token in tokens_before if token not in tokens_after]
print(F"Gensim dictionary initialized and stripped of {N_MOST_FREQUENT_TO_REMOVE} most frequent words.")

corpus = [dictionary.doc2bow(letter) for letter in letters]
print("Gensim corpus of BOWs initialized.")

"""
function that builds an LDA model and returns the model and some evaluation metrics
"""
def model(n_topics, saved=False):
	print(f"Building LDA model for {n_topics} topics.")

	if saved:
		lda = LdaMulticore.load(f"{TRAINED_LDA}{n_topics}")
		print(f"Trained LDA model with {n_topics} loaded successfully.")

	else:
		lda = LdaMulticore(
			corpus,
			num_topics=n_topics,
			id2word=dictionary,
			passes=20,
			random_state=1,
			workers=3
		)

		lda.save(f"{TRAINED_LDA}{n_topics}")
		print(f"LDA model with {n_topics} topics trained and saved successfully.")


	"""
	coherence and silhouette scores
	"""
	coherence = CoherenceModel(model=lda, texts=letters, dictionary=dictionary, coherence='c_v').get_coherence()
	print(f"Coherence score: {coherence}") # the higher the better

	avg_silhouette = plot_silhouette(lda, corpus)
	save_representative_letters(lda, 3)

	"""
	prepare and save pyLDAvis visualization
	"""
	# vis = pyLDAvis.gensim.prepare(topic_model=lda, corpus=corpus, dictionary=dictionary, n_jobs=3)
	# pyLDAvis.save_html(vis, f"{PYLDAVIS_PATH}/lda{N_TOPICS}.html")

	return {
		"model": lda,
		"coherence": coherence,
		"silhouette": avg_silhouette
		}


"""
function to compare LDA models built with the given N values
"""
def compare_models(candidate_n_topics: list):
	results = pd.DataFrame({"topics": [], "coherence": [], "silhouette": []})

	for n_topics in candidate_n_topics:
		result = model(n_topics, saved=True)
		
		results = results.append({
			"topics": n_topics,
			"coherence": result["coherence"],
			"silhouette": result["silhouette"]
		}, ignore_index=True)
	
	results["topics"] = pd.to_numeric(results["topics"], downcast="integer")
	
	return results

def plot_results(results: list):
	fig, (ax1, ax2) = plt.subplots(2, 1)
	ax1.plot(results["topics"], results["coherence"], label="coherence score", marker=".", color="slateblue")
	ax2.plot(results["topics"], results["silhouette"], label = "avg silhouette score", marker=".", color="purple")
	ax1.set_ylabel("Coherence")
	ax2.set_ylabel("Silhouette")
	ax2.set_xlabel("Topics")
	ax1.grid(True)
	ax2.grid(True)
	plt.savefig(f"{GRAPHS_PATH}metrics_per_num_topics.png")

results: pd.DataFrame = compare_models([3, 4, 5, 6, 7, 8, 9])
plot_results(results)