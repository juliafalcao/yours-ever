import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('precision', 0)

vw = pd.read_csv("data\\vw\\vw_preprocessed.csv", index_col="index")
print("DATAFRAME SAMPLE:")
print(vw.sample(10).to_string())

# amount of letters per year
years = list(vw[vw["year"].notnull()]["year"])
plt.hist(years, bins = len(set(years)), rwidth=0.8, color="lightseagreen", alpha=0.9, align="mid")
plt.title("Amount of letters written by Virginia Woolf")
plt.xlabel("Years")
plt.ylabel("Letters")
plt.xticks(list(range(0, 46, 5)))
plt.savefig("graphs\\letters-per-year.png")

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