# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from scipy.sparse import csr_matrix

pd.set_option('precision', 0)

"""
from basic preprocessed text
"""
vw = pd.read_csv("data/vw/vw_preprocessed.csv", index_col="index")
# print("DATAFRAME SAMPLE:")
# print(vw.sample(10).to_string())

# amount of letters per year
years = list(vw[vw["year"].notnull()]["year"])
plt.hist(years, bins = len(set(years)), rwidth=0.8, color="lightseagreen", alpha=0.9, align="mid")
plt.title("Amount of letters written by Virginia Woolf")
plt.xlabel("Years")
plt.ylabel("Letters")
# plt.savefig("graphs/letters_per_year.png")
# plt.clf()

years = vw[vw["year"].notnull()][["year"]]
years = years.groupby(["year"]).size()
print("\nAMOUNT OF LETTERS SENT PER YEAR:")
# print(years.to_string())
avg_letters_per_year = np.average(years)
# print(f"AVERAGE LETTERS PER YEAR: {avg_letters_per_year} letters")

# amount of letters per recipient
recs = vw[vw["recipient"].notnull()][["recipient"]]
recs = recs.groupby(["recipient"]).size()
recs = recs.sort_values(ascending=False)
# print("\nTOP RECIPIENTS (20+ letters):")
# print(recs[recs >= 20].to_string())

# average length
avg_length = np.average(vw["length"])
print(f"\nAVERAGE LENGTH OF LETTERS: {int(avg_length)} characters")
# print(vw.sort_values(by="length", ascending=False))


"""
from tokenized text
"""

def concat_all_letters(letters_series): # pass series of letters as lists of tokens
	series = letters_series.apply(lambda letter: " ".join(letter))
	all_letters = series.str.cat(sep=" ")

	return all_letters


vw = pd.read_json("data/vw/vw_tokenized.json")
# print(vw.sample(20).to_string())

# word frequency
letters = vw["text"].to_list()
all_words = [word for letter in letters for word in letter]
word_freq = nltk.FreqDist(all_words)
# plt.ion()
# word_freq.plot(40, cumulative=False, title="Most frequent words (after preprocessing)")
# plt.savefig("graphs/word_frequency_hist.png")
# plt.ioff()

# word clouds per year
years = sorted(set(vw[vw["year"].notnull()]["year"]))
letters = concat_all_letters(vw["text"])
wordcloud = WordCloud(width=500, height=400, background_color="white", max_words=100).generate(letters)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.savefig(f"graphs/word_clouds/wordcloud.png")
# plt.clf()

"""
TF-IDF experiments
too many stop words, virginia
"""
def dummy_function(text):
	return text # so that TfidfVectorizer won't tokenize and preprocess my already tokenized and preprocessed text

yvw = vw[["year"]].drop_duplicates(keep="first").sort_values(by="year")
yvw = yvw[yvw["year"].notnull()]
yvw["letters"] = vw["year"].apply(lambda year: concat_all_letters(vw[vw["year"] == year]["text"]))
yvw = yvw.reset_index()[["year", "letters"]]
yvw.to_csv("data/vw/yearly_vw.csv")

# tfidf_df = pd.DataFrame(result.todense()) # lines: documents, columns: words, values: tfidf scores
# tfidf_df = tfidf_df.rename(columns=vocab)
# print(tfidf_df[[col for col in tfidf_df.columns if col not in tfidf_df.isnull().any()]])

wordcloud = WordCloud(width=500, height=400, background_color="white", max_words=100)

tfidf = TfidfVectorizer(analyzer="word", preprocessor=dummy_function)
result = tfidf.fit_transform(yvw["letters"])
vocab = {int(key):term for (term, key) in tfidf.vocabulary_.items()}
with open("graphs/tfidf/vocab.json", "w+") as vocab_out:
	json.dump(vocab, vocab_out)

for y in range(len(yvw)):
	frequencies = {vocab[term]:result[(y, term)] for term in result[y].indices}
	wordcloud.generate_from_frequencies(frequencies)
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	year = yvw.at[y, "year"]
	plt.savefig(f"graphs/tfidf/tfidf_wordcloud_{int(year)}.png")
	plt.clf()
