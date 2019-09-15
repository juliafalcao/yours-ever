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

vwp = pd.read_json(VWP_CLUSTERED, orient="index")
N_CLUSTERS = 6

def tsne_plot_clusters():
	vectors: list = []
	labels: list = []

	# plot 10 paragraphs for each cluster
	rng = random.Random()
	rng.seed(22)

	N_SAMPLES = 10 # per cluster
	for i in range(N_CLUSTERS):
		vwp_i: pd.DataFrame = vwp[vwp["cluster"] == i]
		index_samples: list = vwp_i.sample(N_SAMPLES, random_state=22).index
		vectors.extend([vwp.at[n, "embedding"] for n in index_samples])
		labels.extend([f"I{n} L{vwp.at[n, 'letter']} §{vwp.at[n, 'offset']}"] for n in index_samples)
	
	tsne_model = TSNE(
		perplexity=5,
		n_components=2,
		init="pca",
		n_iter=2500,
		random_state=32
	)

	tsne_values = tsne_model.fit_transform(vectors)

	x: list = [value[0] for value in tsne_values]
	y: list = [value[1] for value in tsne_values]

	fig = plt.figure()

	colors = cm.rainbow(np.linspace(0, 1, N_CLUSTERS))
	colors = [[color] * N_SAMPLES for color in colors]
	colors = [elem for sublist in colors for elem in sublist] # flatten

	for i in range(N_CLUSTERS*N_SAMPLES):
		if i%N_CLUSTERS == 0:
			label = f"cluster {i/N_CLUSTERS}"
		else:
			label = None

		plt.scatter(x[i], y[i], c=[colors[i]], label=label)
		plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5,2), textcoords="offset points", ha="right", va="bottom")

	plt.title("Clusters")
	plt.grid(True)
	plt.legend(loc=4)
	plt.show()

tsne_plot_clusters()