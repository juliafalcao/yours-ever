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

# local constants
N_TOPICS: int = 4
UNIQUE_WORDS_THRESHOLD: int = 10
N_MOST_FREQUENT_TO_REMOVE: int = 100


vw: pd.DataFrame = read_dataframe(VW_PREPROCESSED)
print("Pre-processed VW dataset successfully imported.")

"""
cleanup
"""
def filter_by_pos(tokens: list, to_keep: list = ['NOUN']) -> list:
	pos_tags = pos_tag(tokens=tokens, tagset="universal")
	filtered = [token for (token, pos) in pos_tags if pos in to_keep]
	
	return filtered

vw["text"] = vw["text"].apply(lambda tokens: [t for t in tokens if t != PARAGRAPH_DELIM]) # remove paragraph signs
# vw["text"] = vw["text"].apply(lambda tokens: filter_by_pos(tokens, ['NOUN'])) # keep only nouns
# print("Dataset has been stripped of words not POS-tagged as nouns.")

# vw_len_before = len(vw)
# vw["words"] = vw["text"].apply(set).apply(len)
# vw = vw[vw["words"] >= UNIQUE_WORDS_THRESHOLD]
# removed = vw_len_before - len(vw)
# print(f"Removed {removed} ({removed/len(vw) * 100}%) letters from dataset with less than {UNIQUE_WORDS_THRESHOLD} unique nouns.")

# WARNING: index is no longer sequential, but it needs to remain this way so that the mapping to the original letters won't be lost
vw_index = list(vw.index)
vw_index_map = {vw_index[i] : list(range(len(vw)))[i] for i in range(len(vw))} # vw_index_map[actual_index] = seq index for letters list

def get_letter(letters: list, actual_id: int) -> list:
	seq_id = vw_index_map[actual_id]
	assert 0 <= seq_id <= len(vw)
	return letters[seq_id]


"""
create dictionary and corpus of bags-of-words
"""
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

# LDA
lda = LdaMulticore(
	corpus,
	num_topics=N_TOPICS,
	id2word=dictionary,
	passes=15,
	random_state=1,
	workers=3
	# per_word_topics=True
)
lda.save(f"{TRAINED_LDA}{N_TOPICS}")
print(f"LDA model with {N_TOPICS} topics trained and saved successfully.")

# lda = LdaMulticore.load(f"{TRAINED_LDA}{N_TOPICS}")
# print("Trained LDA model loaded successfully.")

def print_model_topics(lda_model):
	for topic, words in lda_model.show_topics(formatted=True, num_topics=N_TOPICS, num_words=20):
			print(str(topic) + ": " + words + "\n")

"""
function that receives a model, a list of documents and a particular document id
and returns its distribution of assigned topics
"""
def get_document_topics(lda_model, documents: list, id: int) -> list:
	bow = dictionary.doc2bow(get_letter(documents, id))
	topics = lda_model.get_document_topics(bow)
	topics.sort(key=(lambda pair: pair[1]), reverse=True) # sort by highest to lowest probabilities

	return topics

# create new dataframe with letter index, raw text, processed text and all topic scores
# (For visualization mostly)
"""
vwt = deepcopy(vw.reset_index()[["index", "text"]])
vwo = pd.read_csv(VW_ORIGINAL, index_col="index")
vwt["raw_text"] = vwt["index"].apply(lambda index: vwo.at[index, "text"]) # recover original text
vwt["topics"] = vwt["index"].apply(lambda index: get_document_topics(lda, letters, index))
vwt["dominant_topic"] = vwt["topics"].apply(lambda topics: topics[0][0])
vwt["dominant_topic_score"] = vwt["topics"].apply(lambda topics: topics[0][1])
vwt = vwt.set_index("index", drop=True).rename(columns={"text": "processed_text"})
vwt = vwt.sort_values(by=["dominant_topic", "dominant_topic_score"], ascending=[True, False]) # sort by best representatives per topic
"""

"""
argument: [(4, 0.79440266), (1, 0.20198905)]
return: {'Topic 4': 0.79440266, 'Topic 1': 0.20198905, 'Topic 0': 0.0, 'Topic 2': 0.0, 'Topic 3': 0.0}
"""
def transform_topic_distribution(topic_dist: list) -> dict:
	topic_dict = {f"Topic {num}" : prob for (num, prob) in topic_dist}

	if len(topic_dict) < N_TOPICS:
		missing_topics = [t for t in range(N_TOPICS) if t not in list(dict(topic_dist).keys())]
		topic_dict.update({f"Topic {t}" : 0.0 for t in missing_topics})

	return topic_dict

"""
return:
       Topic 0   Topic 1   Topic 2   Topic 3   Topic 4
0     0.000000  0.000000  0.000000  0.000000  0.982988
1     0.000000  0.000000  0.282820  0.286516  0.424436
2     0.000000  0.000000  0.520222  0.000000  0.472672
      ...
"""
def get_topic_dists_dataframe(lda_model, num_topics: int) -> pd.DataFrame:
	topic_dists: pd.Series = vw.reset_index()["index"].apply(lambda i: get_document_topics(lda_model, letters, i))
	topic_dists.sort_index(axis=0)
	vws = pd.DataFrame({f"Topic {t}":[] for t in range(N_TOPICS)})
	vws.sort_index(axis=0)
	lines = topic_dists.apply(transform_topic_distribution)
	vws = vws.append(list(lines), ignore_index=True)

	assert len(vws) == len(topic_dists)
	return vws

vws = get_topic_dists_dataframe(lda, N_TOPICS)

"""
evaluation metrics
"""
perplexity = lda.log_perplexity(corpus) # the lower the better
print(f"Perplexity score: {perplexity}")
coherence = CoherenceModel(model=lda, texts=letters, dictionary=dictionary, coherence='c_v').get_coherence()
print(f"Coherence score: {coherence}") # the higher the better?

def compute_avg_silhouette(lda_model, corpus, num_topics: int, topic_df: pd.DataFrame) -> float:
	points = topic_df.values
	dominant_topics = points.argmax(axis=1)

	return silhouette_score(points, dominant_topics)


def plot_silhouette(lda_model, corpus, num_topics: int) -> None:
	n = 0
	points = vws.values # matrix of shape (n_samples, n_topics)
	dominant_topics = points.argmax(axis=1) # array of dominant topics
	fig, ax1 = plt.subplots(1, 1)
	samples_silhouette_values = silhouette_samples(points, dominant_topics)
	y_lower = 10 # ?
	for i in range(num_topics):
		ith_topic_silhouette_values = samples_silhouette_values[dominant_topics == i]
		ith_topic_silhouette_values.sort()
		size_topic_i = ith_topic_silhouette_values.shape[0]
		y_upper = y_lower + size_topic_i
		color = cm.rainbow(float(i) / N_TOPICS)
		ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_topic_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
		ax1.text(-0.05, y_lower + 0.5 * size_topic_i, str(i))
		y_lower = y_upper + 10
	ax1.set_title("Silhouette plot for " + str(N_TOPICS) + " topics")
	ax1.set_xlabel("Silhouette coefficient")
	ax1.set_ylabel("Topic number")
	silhouette_avg = compute_avg_silhouette(lda_model, corpus, num_topics, vws)
	print(f"Average silhouette score: {silhouette_avg}")
	ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
	ax1.set_yticks([])
	ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1]) # ?
	plt.savefig(f"{SILHOUETTE_PLOTS_PATH}silhouette_{N_TOPICS}topics.png")
	plt.show()

plot_silhouette(lda, corpus, N_TOPICS)

# pyLDAvis visualization
vis = pyLDAvis.gensim.prepare(topic_model=lda, corpus=corpus, dictionary=dictionary, n_jobs=3)
pyLDAvis.save_html(vis, f"{PYLDAVIS_PATH}/lda{N_TOPICS}.html")

# save most representative letters to files for validation
LETTERS_TO_PRINT: int = 3

vwo = pd.read_csv(VW_ORIGINAL, index_col="index")
vws["main"] = np.argmax([vws[f"Topic {t}"] for t in range(N_TOPICS)], axis=0)

for t in range(N_TOPICS):
	vws_t = vws[vws["main"] == t]
	vws_t = vws_t.sort_values(by=f"Topic {t}", ascending=False)
	
	for i in range(LETTERS_TO_PRINT):
		letter_id = vws_t.index[i]
		vwo_row = vwo.ix[letter_id]
		vwo_row.to_csv(f"{LDA_LETTERS_PATH}/lda{N_TOPICS}_topic{t}_letter{i}.csv", sep=":") # TODO: move to utils.py
		