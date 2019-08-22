import nltk 
nltk.download('wordnet')

"""
doc2vec experiment
"""

# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(vwp["text"])]

# model = Doc2Vec(
# 	documents,
# 	vector_size=300,
# 	window=2,
# 	min_count=1,
# 	workers=6
# 	)

# model.train(documents, total_examples=len(vwp["text"]), epochs=5)
# model.save(TRAINED_DOC2VEC)

# model = Doc2Vec.load(TRAINED_DOC2VEC)

# test_paragraph = np.random.choice(vwp.index)
# test = vwp.at[test_paragraph, "text"]
# print("test paragraph: " + " ".join(test))
# print("most similar docs:")
# top_similar_docs = model.docvecs.most_similar([model.infer_vector(test)], topn=10)
# i = 0

# for (doc_index, _) in top_similar_docs:
# 	print(f"{i}: {' '.join(documents[doc_index].words)}")
# 	i += 1