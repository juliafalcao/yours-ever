# -*- coding: utf-8 -*-

from const import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import typing
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from nltk.cluster import KMeansClusterer
from nltk.cluster.util import cosine_distance
from scipy.spatial.distance import cosine
import random


# plot
def tsne_plot(model):
	# vectors: list = [model.wv[word] for word in model.wv.vocab]
	# labels: list = [word for word in model.wv.vocab]

	vectors: list = []
	labels: list = []

	words: list = ["monday", "write", "sick", "england", "sister"]

	for word in words:
		vectors.append(model.wv[word])
		labels.append(word)

		for similar_word, _ in model.wv.most_similar(word, topn=30):
			vectors.append(model.wv[similar_word])
			labels.append(similar_word)
	
	tsne_model = TSNE(
		perplexity=10,
		n_components=2,
		init="pca",
		n_iter=2500,
		random_state=32
	)

	tsne_values = tsne_model.fit_transform(vectors)

	x: list = [value[0] for value in tsne_values]
	y: list = [value[1] for value in tsne_values]

	fig = plt.figure()

	colors = cm.rainbow(np.linspace(0, 1, len(words)))

	for i in range(len(x)):
		if labels[i] in words:
			label = labels[i]
		else:
			label = None

		plt.scatter(x[i], y[i], c=[colors[i%len(words)]], label=label)
		plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5,2), textcoords="offset points", ha="right", va="bottom")

	plt.title("Embeddings")
	plt.grid(True)
	plt.legend(loc=4)
	plt.show()


def find_paragraph(df: pd.DataFrame, embedding: list) -> pd.Series: # doesn't work!
	for i in df.index:
		embedding_to_compare: list = vwp.at[i, "embedding"]
		if embedding_to_compare == list(embedding):
			return vwp.ix[i]
	
	return None

def most_similar_paragraphs(df: pd.DataFrame, vector: list, topn: int = 5) -> list:
	new_df = df.copy(deep=True)
	new_df["similarity"] = df["embedding"].apply(lambda embedding: cosine(np.array(embedding), vector))
	new_df = new_df.sort_values(by="similarity", ascending=False)

	return new_df[["letter", "offset", "text", "similarity"]].head(topn)

def recover_raw_paragraph(raw_df: pd.DataFrame, letter: int, offset: int) -> str:
	return str(raw_df[(raw_df["letter"] == letter) & (raw_df["offset"] == offset)][["text"]].values[0])


vwpr = pd.read_csv(VWP_RAW, index_col="index")
vwp = pd.read_json(VWP_SCORED, orient="index")

model = Word2Vec.load(TRAINED_WORD2VEC)

# save embedded words
vwp["chosen_words"] = vwp["tfidf"].apply(lambda tfidf: list([tfidf[i][0] for i in range(10)]))

N_CLUSTERS = 6
random_generator = random.Random()
random_generator.seed(22) # for reproducibility

embeddings = [np.array(embedding) for _, embedding in vwp["embedding"].iteritems()]
kmeans = KMeansClusterer(num_means=N_CLUSTERS, distance=cosine_distance, rng=random_generator)
clusters = kmeans.cluster(embeddings, assign_clusters=True)
vwp["cluster"] = clusters # assign clusters to paragraphs

means = kmeans.means()

print("CLUSTERS")
for i in range(N_CLUSTERS):
	print(f"\n- CLUSTER {i} -")
	clustered: pd.DataFrame = vwp[vwp["cluster"] == i]
	print("Sample:")
	for index, row in clustered.sample(5, random_state=222).iterrows(): # print 5 sample paragraphs from cluster
		content: str = recover_raw_paragraph(vwpr, row["letter"], row["offset"])
		print(f"[{index}] letter {row['letter']}, paragraph {row['offset']}: {content}, embedded words: {row['chosen_words']}")
	
	print(f"Paragraphs most similar to mean vector of cluster {i}:")
	similars: pd.DataFrame = most_similar_paragraphs(vwp, means[i])
	for index, row in similars.head(3).iterrows():
		content: str = recover_raw_paragraph(vwpr, row["letter"], row["offset"])
		in_cluster: bool = "YES" if int(vwp.at[index, "cluster"]) == i else "NO"
		print(f"[{index}] letter {row['letter']}, paragraph {row['offset']} (similarity: {row['similarity']}) (in cluster {i}? {in_cluster}):\n{content}")

print("\n\n")

for i in range(10): # tests
	test = np.random.choice(vwp.index)
	classification = kmeans.classify(np.array(vwp.at[test, "embedding"]))
	print(f"TEST index: {test}, classification: {classification}")
	print(f"CONTENT: {vwpr.at[test, 'text']}")


# save dataframe after clustering
vwp.to_json(VWP_CLUSTERED, orient="index")