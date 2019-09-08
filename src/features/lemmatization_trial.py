import sys, re, typing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dateutil.parser as dateparser
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

test1 = ["terrific", "terrifically", "terrified", "terrifies", "terrify"]
test2 = ["says", "saying", "say"]
test3 = ["terrifies", "terrify", "terrifying"]
test4 = ["parent", "parents"]


def nltk_lemmatize(tokens: list) -> list:
	wnl = WordNetLemmatizer()
	pos_tags = pos_tag(tokens=tokens, tagset="universal")
	print(f"pos_tags: {pos_tags}")
	mapping = {"NOUN": "n", "VERB": "v", "ADV": "r", "ADJ": "a"} # tags recognized by the lemmatizer, else default to "v"
	pos_tags = dict([(_, mapping[tag] if tag in mapping else "v") for (_, tag) in pos_tags])
	print(f"mapped pos tags: {pos_tags}")
	lemmatized = [wnl.lemmatize(word, pos_tags[word]) for word in tokens]
	print(f"LEMMATIZED BY NLTK: {lemmatized}")
	return lemmatized


spacy_lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
sp = spacy.load('en_core_web_sm')

def spacy_lemmatize(tokens: list) -> list:
	s = sp(tokens)
	lemmatized = [word.lemma_ for word in tokens]
	return lemmatized
