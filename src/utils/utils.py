# -*- coding: utf-8 -*-

import pandas as pd 
import codecs
from constants import *

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