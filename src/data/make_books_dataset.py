# -*- coding: utf-8 -*-

"""
fake scraping to obtain content of vw's books to pre-train word2vec
"""

from const import *
import codecs, os, re
from bs4 import BeautifulSoup as bs
from bs4 import NavigableString
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vw_preprocessing import remove_non_alphabetic, remove_stopwords, lemmatize, split_into_sentences
from const import *

def scraping():
	export_path = "data/vw/output"
	books_path = "data/vw/books"

	vwb = pd.DataFrame({"book": [], "text": []})

	for filename in os.listdir(books_path): # loop through html files in dir
		if ".htm" not in filename:
			continue # skip subdirectories

		print(f"Reading '{filename}'...")

		book = re.sub(r'[0-9]', "", filename.replace(".htm", "")) # leave only shortened book title ("Voyage")

		with codecs.open(f"{books_path}/{filename}", mode="r", encoding="utf-8") as file:
			file_content: str = file.read()

			try:
				start: int = file_content.index("sigil_not_in_toc")
				start = file_content.index("</h", start)

			except ValueError: # in case of no "sigil_not_in_toc"
				start = file_content.index('<p class="source">')

			body: str = file_content[start:]
			
			body = re.sub(r'<[^>]+>', "", body)
			vwb = vwb.append({"book": book, "text": body}, ignore_index=True)
		
		print(f"Finished reading '{filename}'.")

	def clean_book_text(text: str) -> str:
		text = text.replace("&nbsp;", "")
		text = re.sub(r'·[0-9]+·', "", text) # remove page numbers like ·78·
		text = re.sub(r'[\r\t]', " ", text) # remove fancy whitespace
		text = re.sub(r' +', " ", text) # remove extra whitespace
		text = text.strip() # trailing whitespace

		return text

	vwb["text"] = vwb["text"].apply(clean_book_text)

	print("DATAFRAME SAMPLE:")
	print(vwb.head(25).to_string())
	vwb.to_csv(VWB_RAW, index_label="index")
	print(f"Raw books dataframe successfully written to '{VWB_RAW}'.")


def preprocessing():
	vwb = pd.read_csv(VWB_RAW, index_col="index")
	
	vwb["text"] = vwb["text"].apply(remove_non_alphabetic).str.lower()
	vwb["text"] = vwb["text"].apply(split_into_sentences)
	print("VWB prepared.")
	vwb["text"] = vwb["text"].apply(lambda text: [list(word_tokenize(sentence)) for sentence in text])
	vwb["text"] = vwb["text"].apply(lambda text: [list(remove_stopwords(sentence)) for sentence in text])
	print("VWB tokenized, split into sentences and stripped of stop words.")

	wnl = WordNetLemmatizer()
	vwb["text"] = vwb["text"].apply(lambda text: [list(lemmatize(sentence, wnl)) for sentence in text])
	print("VWB lemmatized.")

	with open(VWB_TOKENIZED, "w", encoding="utf-8") as json_file:
		vwb.to_json(json_file, force_ascii=False, orient="index")
		print(f"Tokenized and lemmatized VWB dataframe written to '{VWB_TOKENIZED}'.")

scraping()
preprocessing()