import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from typing import List

pd.set_option('precision', 0)

"""
from basic preprocessed text
"""
vw = pd.read_csv("data\\vw\\vw_preprocessed.csv", index_col="index")
print("DATAFRAME SAMPLE:")
print(vw.sample(10).to_string())

# amount of letters per year
years = list(vw[vw["year"].notnull()]["year"])
plt.hist(years, bins = len(set(years)), rwidth=0.8, color="lightseagreen", alpha=0.9, align="mid")
plt.title("Amount of letters written by Virginia Woolf")
plt.xlabel("Years")
plt.ylabel("Letters")
plt.savefig("graphs\\letters-per-year.png")
plt.cla()
plt.clf()

years = vw[vw["year"].notnull()][["year"]]
years = years.groupby(["year"]).size()
print("\nAMOUNT OF LETTERS SENT PER YEAR:")
print(years.to_string())

# amount of letters per recipient
recs = vw[vw["recipient"].notnull()][["recipient"]]
recs = recs.groupby(["recipient"]).size()
recs = recs.sort_values(ascending=False)
print("\nTOP RECIPIENTS (20+ letters):")
print(recs[recs >= 20].to_string())

# average length
avg_length = np.average(vw["length"])
print(f"\nAVERAGE LENGTH OF LETTERS: {int(avg_length)} characters")

print(vw.sort_values(by="length", ascending=False))


"""
from tokenized text
"""
vw = pd.read_json("data\\vw\\vw_tokenized.json")
print(vw.sample(20).to_string())

# word frequency
letters = vw["text"].to_list()
all_words = [word for letter in letters for word in letter]
word_freq = nltk.FreqDist(all_words)
plt.ion()
word_freq.plot(40, cumulative=False, title="Most frequent words (after preprocessing)")
plt.savefig("graphs\\word_frequency.png")
plt.ioff()
