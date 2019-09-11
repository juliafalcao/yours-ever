# -*- coding: utf-8 -*-

import sys, typing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("src/utils")
from constants import *

vwp = pd.read_json(VWP_PREPROCESSED, orient="index")

vwp["length"] = vwp["text"].apply(len)

# vwp = vwp[vwp["length"] > 10]

print(vwp.head(30).to_string())

paragraphs_per_letter: pd.DataFrame = vwp.groupby(by="letter").count()[["offset"]]
paragraphs_per_letter.index.name = "letter"
paragraphs_per_letter = paragraphs_per_letter.rename(columns={"offset": "paragraphs"})
print(paragraphs_per_letter.head(30))

counts = len(set(paragraphs_per_letter["paragraphs"]))
plt.hist(paragraphs_per_letter["paragraphs"], bins=counts, rwidth=0.8, color="lightseagreen", alpha=0.9, align="mid")
# plt.xticks(range(1, 15))
plt.title("Parágrafos por carta")
plt.xlabel("Parágrafos")
plt.ylabel("Cartas")
# plt.savefig(f"{GRAPHS_PATH}paragraphs_per_letter.png")
# plt.show()

top_recipients = vwp.groupby(by="recipient").agg({"letter": "count"})
top_recipients = top_recipients.rename(columns={"letter": "letters"}).sort_values(by="letters", ascending=False)
print(top_recipients)