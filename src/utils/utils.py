# -*- coding: utf-8 -*-

import pandas as pd 
import codecs
from constants import *
from datetime import datetime
from sklearn.metrics import silhouette_samples, silhouette_score

"""
function that receives a letter index and returns the original letter text a string
"""
def get_letter(index: int) -> str:
	vw = pd.read_csv(VW_ORIGINAL, index_col="index")
	assert 0 <= index < len(vw)

	return vw.at[index, "text"]

"""
function that receives a letter index and paragraph offset and returns the original paragraph from the letter as a string
"""
def get_paragraph(letter_index: int, offset: int) -> str:
	letter: str = get_letter(letter_index)
	paragraphs: list = [p.strip() for p in letter.split(PARAGRAPH_DELIM)]
	if len(paragraphs[0]) < 2:
		paragraphs = paragraphs[1:]
	
	assert 0 <= offset < len(paragraphs)
	return paragraphs[offset]


"""
wrapper function to import dataframes saved as json with the proper encoding settings
"""
def read_dataframe(filename: str) -> pd.DataFrame:
	with codecs.open(filename, mode="r", encoding="utf-8") as file:
		df = pd.read_json(file, orient="index")

	return df

"""
dummy function to override tokenization in vectorizers
"""
def _dummy(text: str) -> str:
		return text

"""
function that receives a series of letters as lists of tokens and returns all of them concatenated (one list of tokens)
"""
def concat_all_letters(letters: pd.Series):
	letters = letters.apply(lambda letter: " ".join(letter))
	all_letters = letters.str.cat(sep=" ")
	return all_letters

"""
save something into a file inside reports/logs/
"""
def log(something, filename: str):
	printable = str(something)

	filepath = f"{EXEC_LOGS_PATH}{filename}.txt"

	with open(filepath, encoding="utf8", mode="w") as file:
		_ = file.write(printable)
		print(f"Logged: {filepath}")
	
	return

"""
function that returns a standard-format lda model name such as lda3-a-9 for lda with k = 3, alpha = asymmetric and beta = 0.9
"""
def get_model_name(lda_model):
	k = lda_model.num_topics

	alpha = lda_model.alpha
	if (alpha[0] != alpha[-1]): # asymmetric alpha
		alpha = "a"
	else:
		alpha = str(alpha[0]).split(".") # symmetric alpha

	beta = str(lda_model.eta[0])[2:] # symmetric beta

	return f"lda{3}-{alpha}-{beta}"

"""
compute average silhouette score
"""
def compute_avg_silhouette(topic_df: pd.DataFrame) -> float:
	points = topic_df.values
	mains = points.argmax(axis=1)

	return silhouette_score(points, mains)