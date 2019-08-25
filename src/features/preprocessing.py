"""
pre-processing of virginia woolf's letters
(to be generalized at some point)
"""
import sys, re, typing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dateutil.parser as dateparser
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

sys.path.append("src/utils")
from constants import *

"""
function to remove brackets from string
"""
def remove_brackets(text: str) -> str:
	if pd.notnull(text):
		brackets = r'[\[\]]'
		return re.sub(brackets, "", text)

"""
function to remove non-alphabetic symbols from string
"""
def remove_non_alphabetic(text: str) -> str:
	if pd.notnull(text):
		to_replace = r'[\-—\+\n]'
		text = re.sub(to_replace, " ", text) # replace these symbols with spaces
		to_remove = r'[^a-zA-Z§ ]'
		text = re.sub(to_remove, "", text) # remove all symbols left except for § (paragraph delimiter)
		text = re.sub(r' +', " ", text) # remove extra whitespace

		return text

"""
function to trim the recipients of the letters, saved as "To Vanessa Bell" in the original dataframe
"""
def trim_recipient(recipient: str):
	recipient = re.sub(r'[\n\r\t]', "", recipient) # weird whitespace
	recipient = re.sub(" +", " ", recipient) # extra whitespace

	if pd.notnull(recipient) and len(recipient) > 4 and recipient[:3] == "To ":
		return recipient[3:]
	else:
		return recipient


"""
function to extract year from dates written in multiple different ways
"""
def extract_year(full_date: str) -> str:

	def decade(year4digits: int) -> int: # decade(1941) -> 41
		return int(str(year4digits)[-2:])
	
	def century(year4digits: int) -> int: # century(1941) -> 19
		return int(str(year4digits)[:-2])

	if pd.isnull(full_date):
		return None

	try:
		full_date: str = full_date.replace("?", "")
		year: int = dateparser.parse(full_date, fuzzy=True, ignoretz=True).year

	except ValueError:
		date_only_digits = re.sub("[^0-9]", "", full_date)
		year = int(date_only_digits[-4:])

	# correct years written in two digits that were mistakenly parsed as 20th century years
	if century(year) == "20":
		if decade(year) >= decade(VW_BIRTH):
			year = "18" + str(decade(year)) # 1800s
		elif decade(year) <= decade(VW_DEATH):
			year = "19" + str(decade(year)) # 1990s

	if int(year) < VW_BIRTH or int(year) > VW_DEATH: # invalid
		return None

	return str(year)


"""
function to fill missing years from dataframe using year of previous and following letters if both are the same year
(letters are assumed to be in chronological order)
"""
def fill_missing_years(df: pd.DataFrame) -> pd.DataFrame:
	missing = df[df["year"].isnull()][["id", "year"]]

	for i in list(missing.index):
		if i > 0:
			if df.at[i-1, "year"] == df.at[i+1, "year"]:
				df.at[i, "year"] = df.at[i+1, "year"]

	return df

"""
function to remove stop words from text passed as list of tokens
"""
def remove_stopwords(tokens: list) -> list:
	if len(tokens) < 1:
		return tokens
	
	# based on nltk english stopwords list but modified
	stopwords = {'most', 'over', 'until', "shouldnt", 'only', 'all', 'o', 'herself', 'same', 'ma', 'i', 'will', 'now', 'these', 'needn', 'out', 'yours', 'hadn', 'where', 'during', 'above', 'very', 'aren', 'off', 'when', 'once', 'm', 'don', 'then', 'why', 'its', 'weren', "couldn't", 'him', "mightn't", 't', 'to', 'into', 'been', 'wasn', 'wouldn', 'won', 'are', 'whom', 'y', 'more', 'some', 'nor', 'by', 'being', "shes", 'a', 'about', 'he', 'below', 'my', 'it', "needn't", 'they', 'does', 'this', 'any', "wouldnt", 'and', 'ours', "havent", 'you', 'should', 'in', "isn't", 'each', 's', "won't", "don't", 'them', 'himself', 'if', 'so', 'at', 'mustn', 'themselves', 'further', 'myself', 'ain', 'she', 'an', "werent", 'our', 'what', 'on', 'doing', 'll', 'isn', 'yourself', "its", 'too', "arent", 'couldn', 'just', 'we', "hasnt", 'have', "didnt", 'is', 'the', 'for', 've', 'do', 'few', 'hasn', 'those', 'was', 'such', 'which', 'didn', 'because', 'of', 'hers', 'not', "shouldve", 'me', 'were', 'with', 'itself', 'doesn', 'shan', 'or', 'both', 'can', 'has', 'did', 'while', 'no', "youd", 'be', "hadnt", 'but', 'shouldn', 'his', 'their', 're', 'again', 'how', 'your', 'here', 'before', 'through', 'who', 'up', 'between', 'yourselves', "mustnt", 'having', 'her', 'other', 'theirs', 'am', 'haven', 'against', 'as', 'had', "doesnt", 'mightn', "youre", 'from', 'under', 'than', "youve", 'd', 'down', 'ourselves', 'there', "wasnt", "youll", 'that', 'own', 'after', "one", "dont", "im", "ive"}
	"""
	words added to list after word frequency analysis: one, dont, im, ive
	"""

	new_tokens = [t for t in tokens if t not in stopwords]
	return new_tokens

"""
function to lemmatize text passed as list of tokens, using WordNetLemmatizer
"""
def lemmatize(tokens: list) -> list:
	wnl = WordNetLemmatizer()

	pos_tags = nltk.pos_tag(tokens=tokens, tagset="universal")
	mapping = {"NOUN": "n", "VERB": "v", "ADV": "r", "ADJ": "a"} # tags recognized by the lemmatizer, else default to "v"
	pos_tags = dict([(_, mapping[tag] if tag in mapping else "v") for (_, tag) in pos_tags])

	lemmatized = [wnl.lemmatize(word, pos_tags[word]) for word in tokens]
	return lemmatized

"""
function to split list into list of lists separated by a given delimiter
"""
def split_list(tokens: list, delim: str) -> list:
	flattened = " ".join(tokens) # convert to string
	parts = [part.strip() for part in flattened.split(delim)] # split by delim
	parts = [part for part in parts if part != ""] # remove empty strings
	tokens = [part.replace(delim, "").split(" ") for part in parts] # re-tokenize
	return tokens


vw = pd.read_csv(VW_ORIGINAL, index_col="index")
print("Original letters dataframe imported.")

"""
adjust the metadata of the letters
"""
vw["place"] = vw["place"].apply(remove_brackets)
vw["recipient"] = vw["recipient"].apply(trim_recipient)
vw["recipient"] = vw["recipient"].apply(lambda r: r.replace("\xa0", " "))
vw["recipient"] = vw["recipient"].replace("V. Sackville-West", "Vita Sackville-West") # ♡
vw["date"] = vw["date"].apply(remove_brackets).apply(extract_year) # keep only years
vw = vw.rename(columns={"date": "year"})
vw = fill_missing_years(vw)
vw["year"] = pd.to_numeric(vw["year"], downcast="integer")

"""
text preprocessing
1. removal of non-alphabetic symbols and conversion to lowercase
2. tokenization
3. stop words removal
4. lemmatization
"""

vw["text"] = vw["text"].apply(remove_non_alphabetic).str.lower()
vw["text"] = vw["text"].apply(word_tokenize)
vw["text"] = vw["text"].apply(remove_stopwords)
vw["text"] = vw["text"].apply(lemmatize)


with open(VW_PREPROCESSED, "w", encoding="utf-8") as json_file:
	vw.to_json(json_file, force_ascii=False, orient="index")

print(f"Dataset has been successfully preprocessed, tokenized, stripped of stop words and lemmatized, and written to '{VW_PREPROCESSED}'.")

"""
create dataframe of paragraphs from the preprocessed letters
new columns:
- 'letter': corresponds to index value in vw dataframe
- 'offset': sequential id of paragraph inside letter, where 0 is the offset of the first paragraph
- 'text': raw content of the paragraph
"""

vwp = pd.DataFrame({"letter": [], "offset": [], "year": [], "recipient": [], "text": []})

for (index, row) in vw.iterrows():
	paragraphs = split_list(row["text"], PARAGRAPH_DELIM)
	# paragraphs = [re.sub(r' +', " ", re.sub(r'[§\n\r\t]', "", paragraph)).strip() for paragraph in paragraphs] # cleanup
	# while "" in paragraphs:
		# paragraphs.remove("")
	
	for i in range(len(paragraphs)):
		if len(paragraphs[i]) > 0: # add paragraphs to VWP dataframe and set offset
			vwp = vwp.append({
				"letter": index, # same index as VW dataframe
				"offset": i,
				"year": row["year"],
				"recipient": row["recipient"],
				"text": paragraphs[i]
			}, ignore_index=True)
	

	vwp["letter"] = pd.to_numeric(vwp["letter"], downcast="integer")
	vwp["offset"] = pd.to_numeric(vwp["offset"], downcast="integer")

vwp.to_json(VWP_PREPROCESSED, orient="index")
print(f"Paragraphs dataframe created and written to '{VWP_PREPROCESSED}'.")