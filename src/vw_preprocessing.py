"""
pre-processing of virginia woolf's letters
(to be generalized at some point)
"""

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import dateutil.parser as dateparser
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def remove_brackets(text: str):
	if pd.notnull(text):
		brackets = r'[\[\]]'
		return re.sub(brackets, "", text)

def remove_non_alphabetic(text: str):
	if pd.notnull(text):
		to_replace = r'[\-—\+\n]' # symbols to replace with spaces
		to_remove = r'[^a-zA-Z ]' # remove all non alphanumeric symbols left
		text = re.sub(to_replace, " ", text)
		text = re.sub(to_remove, "", text)
		text = re.sub(r' +', " ", text) # remove extra whitespace
		return text

def trim_recipient(recipient: str):
	recipient = recipient.replace("\n", " ").replace("\r", " ") # trailing whitespace
	recipient = re.sub(" +", " ", recipient) # extra whitespace

	if pd.notnull(recipient) and len(recipient) > 4 and recipient[:3] == "To ":
		return recipient[3:]
	else:
		return recipient

def extract_year(full_date: str):
	if pd.isnull(full_date):
		return None

	try:
		full_date = full_date.replace("?", "")
		year = dateparser.parse(full_date, fuzzy=True, ignoretz=True).year

	except ValueError:
		date_only_digits = re.sub("[^0-9]", "", full_date)
		year = int(date_only_digits[-4:])

	if str(year)[:2] == "20":
		if int(str(year)[2:]) >= 95:
			year = "18" + str(year)[2:] # 1800s
		elif int(str(year)[2:]) <= 41:
			year = "19" + str(year)[2:] # 1990s

	if int(year) < 1895 or int(year) > 1941:
		return None

	return str(year)

def fill_missing_years(df: pd.DataFrame):
	missing = df[df["year"].isnull()][["id", "year"]]

	for i in list(missing.index):
		if i > 0:
			if df.at[i-1, "year"] == df.at[i+1, "year"]:
				df.at[i, "year"] = df.at[i+1, "year"]

	return df

def remove_stopwords(tokens: list):
	stopwords = {'most', 'over', 'until', "shouldnt", 'only', 'all', 'o', 'herself', 'same', 'ma', 'i', 'will', 'now', 'these', 'needn', 'out', 'yours', 'hadn', 'where', 'during', 'above', 'very', 'aren', 'off', 'when', 'once', 'm', 'don', 'then', 'why', 'its', 'weren', "couldn't", 'him', "mightn't", 't', 'to', 'into', 'been', 'wasn', 'wouldn', 'won', 'are', 'whom', 'y', 'more', 'some', 'nor', 'by', 'being', "shes", 'a', 'about', 'he', 'below', 'my', 'it', "needn't", 'they', 'does', 'this', 'any', "wouldnt", 'and', 'ours', "havent", 'you', 'should', 'in', "isn't", 'each', 's', "won't", "don't", 'them', 'himself', 'if', 'so', 'at', 'mustn', 'themselves', 'further', 'myself', 'ain', 'she', 'an', "werent", 'our', 'what', 'on', 'doing', 'll', 'isn', 'yourself', "its", 'too', "arent", 'couldn', 'just', 'we', "hasnt", 'have', "didnt", 'is', 'the', 'for', 've', 'do', 'few', 'hasn', 'those', 'was', 'such', 'which', 'didn', 'because', 'of', 'hers', 'not', "shouldve", 'me', 'were', 'with', 'itself', 'doesn', 'shan', 'or', 'both', 'can', 'has', 'did', 'while', 'no', "youd", 'be', "hadnt", 'but', 'shouldn', 'his', 'their', 're', 'again', 'how', 'your', 'here', 'before', 'through', 'who', 'up', 'between', 'yourselves', "mustnt", 'having', 'her', 'other', 'theirs', 'am', 'haven', 'against', 'as', 'had', "doesnt", 'mightn', "youre", 'from', 'under', 'than', "youve", 'd', 'down', 'ourselves', 'there', "wasnt", "youll", 'that', 'own', 'after'}
	
	new_tokens = [t for t in tokens if t not in stopwords]
	return new_tokens
		

vw = pd.read_csv("data\\vw\\vw_from_epub.csv", index_col = "index")

vw["place"] = vw["place"].apply(remove_brackets)
vw["recipient"] = vw["recipient"].apply(trim_recipient)
vw["recipient"] = vw["recipient"].apply(lambda r: r.replace("\xa0", " "))
vw["recipient"] = vw["recipient"].replace("V. Sackville-West", "Vita Sackville-West") # ♡
vw["length"] = vw["text"].apply(lambda text: len(text))

vw["date"] = vw["date"].apply(remove_brackets)
vw["year"] = vw["date"].apply(extract_year)
vw = fill_missing_years(vw)

"""
text preprocessing
"""
vw["text"] = vw["text"].apply(remove_non_alphabetic)
vw["text"] = vw["text"].str.lower()

# save preprocessed dataframe before tokenization etc.
vw = vw[["id", "year", "recipient", "text", "length"]]
vw.to_csv("data\\vw\\vw_preprocessed.csv", index_label="index")

# tokenization
vw["text"] = vw["text"].apply(lambda t: word_tokenize(t))
vw["text"] = vw["text"].apply(remove_stopwords)


print(vw.sample(20).to_string())
print(vw.info())
vw.to_csv("data\\vw\\vw_tokenized.csv", index_label="index")