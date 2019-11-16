# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyLDAvis
import pyLDAvis.gensim
from sklearn.metrics import silhouette_samples, silhouette_score
from wordcloud import WordCloud
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from constants import *
from utils import *


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

	n_topics = len(set(vws["main"]))

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
plot coherence and silhouette scores for candidate models
receives: results dict (output of compare_models)
"""
def plot_metrics(results: pd.DataFrame, variable="num_topics", fixed_alpha=None, fixed_beta=None) -> None:
	df = results.copy()
	df = df.sort_values(by="num_topics")

	if fixed_alpha is not None:
		df = df[df["alpha"] == fixed_alpha]
	
	if fixed_beta is not None:
		df = df[df["beta"] == fixed_beta]

	color = cm.rainbow(0.3)

	fig, (ax1, ax2) = plt.subplots(2, 1)
	ax1.plot(df["num_topics"], df["coherence"], label="Coerência (média)", color=color, marker="o", markerfacecolor="white", markeredgecolor=color)
	ax2.plot(df["num_topics"], df["silhouette"], label = "Silhueta (média)", color=color, marker="o", markerfacecolor="white", markeredgecolor=color)
	ax1.set_ylabel("Coerência")
	ax2.set_ylabel("Silhueta")
	ax2.set_xlabel("Topicos")
	ax1.grid(True)
	ax2.grid(True)
	# plt.savefig(f"{GRAPHS_PATH}metrics_per_num_topics.png")
	plt.show()

	print("Saved plot of metrics comparison between different numbers of topics.")
	plt.close("all")

"""
generates and saves wordclouds per topic of a given LDA model
receives: vw dataframe with "main" column
"""
def save_topic_wordclouds_by_frequency(vw: pd.DataFrame) -> None:
	assert "main" in vw.columns

	n_topics = len(set(vw["main"]))

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
function that makes a bar graph of topic frequencies per years
receives: vw dataframe with "main" topic column
"""
def plot_topics_per_year(vw: pd.DataFrame) -> None:
	assert "main" in vw.columns

	n_topics = len(set(vw["main"]))

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
	labels = [f"Tópico #{i}" for i in range(n_topics)]

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
	plt.xlabel("Ano")
	plt.ylabel("Número de cartas")
	plt.legend()
	plt.grid(True, axis="y")
	ax.set_axisbelow(True)
	plt.savefig(f"{GRAPHS_PATH}lda{n_topics}_topic_frequency_per_year.png") # UTF-8 doesn't work
	# plt.show()
	plt.close("all")

"""
function that makes 12 pie charts for topic frequencies in letters to the top 12 recipients
receives: vw dataframe with "main" topic column
"""
def plot_topics_per_recipient(vw: pd.DataFrame) -> None:
	assert "main" in vw.columns

	n_topics = len(set(vw["main"]))

	recs = vw[(vw["recipient"].notnull()) & (vw["main"] != -1)][["recipient", "main"]]

	for i in range(n_topics):
		recs[str(i)] = recs["recipient"].apply(lambda rec: len(recs[(recs["recipient"] == rec) & (recs["main"] == i)]))

	recs = recs.drop(["main"], axis=1).drop_duplicates(subset=["recipient"], keep="first")

	top_recs = vw[vw["recipient"].notnull()][["recipient"]]
	top_recs = top_recs.reset_index().groupby(["recipient"]).count()[["index"]].rename(columns={"index": "num_letters"})
	top_recs = top_recs.sort_values(by="num_letters", ascending=False)

	recs = recs.merge(top_recs, on="recipient", how="left")
	top_recs_list = list(recs.sort_values(by="num_letters", ascending=False)["recipient"][:12])

	fig, axs = plt.subplots(3, 4, figsize=(13, 9))
	plt.subplots_adjust(wspace=0.5, hspace=0.5)
	colors = [cm.rainbow(float(i)/n_topics) for i in range(n_topics)]
	columns = [str(i) for i in range(n_topics)]

	it = 0
	for i in range(3):
		for j in range(4):
			rec: str = top_recs_list[it]
			row = recs[recs["recipient"] == rec][columns]
			values = [int(row[col]) for col in row.columns]
			axs[i,j].pie(values, colors=colors, labels=columns, labeldistance=0.7)
			axs[i,j].set_xlabel(rec)
			it += 1

	plt.savefig(f"{GRAPHS_PATH}lda{n_topics}_topic_frequency_per_recipient.png") # UTF-8 doesn't work
	plt.close("all")

"""
2x2 scatter plot comparing alpha and beta for 4 values of k, with legend below
"""
def plot_results(results: pd.DataFrame):
	results = results[results["beta"] != "auto"]

	alpha_markers = {
		"asymmetric": "*", # star
		"symmetric": "o",
		"0.3": "^", # triangle
		"0.5": (5, 0), # pentagon
		"0.7": "d" # thin diamond
	}

	beta_colors = {beta : cm.rainbow(float(beta)) for beta in set(results["beta"])}

	results["num_topics"] = pd.to_numeric(results["num_topics"], downcast="integer")
	results.sort_values(by=["silhouette", "coherence", "beta"], ascending=[False, False, False])
	k_values = [3, 4, 5, 6]
	n_subplots = len(k_values)

	fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(6,9))
	plt.subplots_adjust(hspace=0.35, wspace=0.3)

	gs = axs[2,0].get_gridspec()
	axs[2,0].remove()
	axs[2,1].remove()
	axl = fig.add_subplot(gs[2:, :])
	axl.tick_params(axis="both", left=False, bottom=False, labelbottom=False, labelleft=False)
	axl.axis(False)

	# later
	axs = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]] # flatten

	for i in range(n_subplots):
			n_topics = k_values[i]
			results_t = results[results["num_topics"] == n_topics]
			for index, row in results_t.iterrows():
				ax = axs[i] # easier to change later
				ax.plot(
					row["silhouette"],
					row["coherence"],
					marker=alpha_markers[row["alpha"]],
					color=beta_colors[row["beta"]],
					markersize=10 if row["alpha"] == "asymmetric" else 8,
					alpha=0.9
				)
				ax.set_title(f"K = {n_topics}", loc="center")
				ax.tick_params(axis="both", left=False, bottom=False, labelleft=True, labelbottom=True, labelsize="small")
				ax.grid(axis="both", color="grey", alpha=0.5)

	# fig.text(0.5, 0.04, "Silhouette", ha="center")
	# fig.text(0.04, 0.5, "Coherence", va="center", rotation="vertical")
	axs[0].set_ylabel("Coherence")
	axs[2].set_ylabel("Coherence")
	axs[2].set_xlabel("Silhouette")
	axs[3].set_xlabel("Silhouette")

	legend_elems = []
	legend_elems.append(Line2D([0], [0], marker="s", color="white", markerfacecolor="white", markeredgecolor="white", label="ALPHA")) # first title

	for label, marker in list(alpha_markers.items()): # legend for alpha markers
		legend_elems.append(Line2D([0], [0], marker=marker, color="white", markeredgecolor="grey", markerfacecolor="grey", markersize=10, label=label))

	legend_elems.append(Line2D([0], [0], marker="s", color="white", markerfacecolor="white", markeredgecolor="white", label="BETA")) # second title
	beta_tuples = list(beta_colors.items())
	beta_tuples.sort(key=lambda x: x[0])
	for label, color in beta_tuples: # legend for beta colors (squares in given color)
		legend_elems.append(Line2D([0], [0], marker="s", color="white", markeredgecolor=color, markerfacecolor=color, markersize=10, label=label))

	axl.legend(handles=legend_elems, framealpha=1, loc="upper center", edgecolor="grey", fontsize="small", ncol=4, borderpad=1)

	# hide leading zeroes (makes tick labels unchangeable; must adjust figsize first)
	axs[0].set_yticklabels([str(x)[1:] for x in np.round(ax.get_yticks(), 3)])
	axs[0].set_xticklabels([str(x)[1:] for x in np.round(ax.get_xticks(), 3)])

	plt.show()
	# plt.savefig(f"{GRAPHS_PATH}parameter_comparison.png")