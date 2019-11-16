# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import deepcopy

sys.path.append("src/utils")
from constants import *
from utils import *
from plotting import *


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

vw = read_dataframe(VW_PREPROCESSED)
vwp = read_dataframe(VWP_PREPROCESSED)

vw = vw[vw["year"].notnull()]
vw["year"] = pd.to_numeric(vw["year"], downcast="integer")
vw["word_count"] = vw["text"].apply(len)
awc = vw.groupby(["year"])["word_count"].mean().reset_index().set_index("year")
lpy = vw.groupby(["year"]).size().reset_index().set_index("year").rename(columns={"0": "count"})

fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1, sharex=True, figsize=(9, 7))
ax1.set_facecolor("gainsboro")
ax2.set_facecolor("gainsboro")
plt.subplots_adjust(hspace=0.2)

ax1.bar(x=awc.index, height=awc["word_count"], color=cm.rainbow(0.3))
ax1.tick_params(bottom=False, labelbottom=True, left=False)
ax1.tick_params(axis="x", labelrotation=90, pad=0.1)
ax1.set_ylabel("Tamanho medio por ano")

plt.xticks(ticks=awc.index, labels=[str(int(year)) for year in awc.index], rotation="vertical")

ax2.plot(lpy.index, lpy, color=cm.rainbow(0.8), marker="o", markersize=4, label="Cartas por ano")
ax2.set_ylabel("Cartas por ano")
ax2.tick_params(bottom=False, labelbottom=False, left=False)

for ax in [ax1, ax2]: # commands to run in both axes
	ax.spines["top"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.grid(True, axis="x", color="white")
	ax.set_axisbelow(True)

plt.savefig(f"{GRAPHS_PATH}count_and_size_per_year.pdf", format="pdf")