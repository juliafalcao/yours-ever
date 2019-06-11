# -*- coding: utf-8 -*-

from const import *
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt

MIN_WORDS = 10

vw = pd.read_json(VW_TOKENIZED, orient="index")
vwp = pd.read_json(VWP_TOKENIZED, orient="index")

vwp["word_count"] = vwp["text"].apply(len)
vwp["unique_word_count"] = vwp["text"].apply(set).apply(len)

# print(vwp.head(10).to_string())
# print(f"avg word count: {np.average(vwp['word_count'])}")
# print(f"avg unique word count: {np.average(vwp['unique_word_count'])}")

vwp = vwp[vwp["unique_word_count"] > MIN_WORDS]
vwp = vwp[vwp["word_count"] != vwp["unique_word_count"]]
# print(vwp.sort_values(by="word_count").head(10).to_string())

# plt.hist(vwp["word_count"], range=[0,50], bins=100)
# plt.show()
# print(f"avg unique word count after cleanup: {np.average(vwp['unique_word_count'])}")

"""
doc2vec
"""

# training

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(vwp["text"])]

"""
model = Doc2Vec(
	documents,
	vector_size=100,
	window=2,
	min_count=1,
	workers=8
	)

model.train(documents, total_examples=len(vwp["text"]), epochs=5)

model.save("trained_models/vw_doc2vec.model")
"""

model = Doc2Vec.load(TRAINED_DOC2VEC)

test_paragraph = np.random.choice(vwp.index)

test = vwp.at[test_paragraph, "text"]

print("test paragraph: " + " ".join(test))

print("most similar docs:")
top_similar_docs = model.docvecs.most_similar([model.infer_vector(test)], topn=10)

i = 0
for (doc_index, _) in top_similar_docs:
	print(f"{i}: {' '.join(documents[doc_index].words)}")
	i += 1
