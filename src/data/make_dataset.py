# -*- coding: utf-8 -*-

"""
scraping to obtain the content of the letters from the html files that make up the epub book
"""

import sys
import codecs, os, re
from bs4 import BeautifulSoup as bs, NavigableString
import pandas as pd

sys.path.append("src/utils")
from constants import *

vw = pd.DataFrame({"id": [], "date": [], "recipient": [], "place": [], "text": []})

for filename in os.listdir(RAW_LETTERS_PATH): # loop through html files in dir
	assert ".htm" in filename

	print(f"Reading '{filename}'...")

	with codecs.open(f"{RAW_LETTERS_PATH}{filename}", mode="r", encoding="utf-8") as file:
		file_content = file.read()

	soup = bs(file_content, "html.parser")
	html = soup.find("html")
	body = html.find("body")
	soup = body

	delimiter = "link"
	to_ignore = ["date", "card", "sigil_not_in_toc", "imgbrd", "margl"]
	count_per_file = 0

	for h3item in soup.find_all("h3"): # start from h3 header
		if isinstance(h3item, NavigableString):
			continue

		if h3item.get("class") == "sigil_not_in_toc" or h3item.has_attr("id") or h3item.has_attr("title"): # only actual letter headers
			header = h3item.get_text().strip()
			id = header[:header.find(":")].strip()
			recipient = header[header.find(":")+1:].strip()
			time, place = None, None # yet undefined
			letter_body = ""
			
			for item in h3item.next_siblings: # verify next <p ...> elements
				if isinstance(item, NavigableString):
						continue

				elif item.has_attr("class"): # <p class="...">
					item_class = item.get("class")[0]

					if item_class == "time": # letter date
						time = item.get_text()

					elif item_class == "place": # sender address
						place = item.get_text()

					elif item_class in to_ignore:
						continue

					elif item_class == delimiter: # end of current letter
						break # out of <p> loop
					
					else: # parts of letter body (<p class="name"> etc.)
						letter_body += f"§ {str(item.get_text())}\n"

				else: # <p> parts of letter body
					letter_body += f"§ {str(item.get_text())}\n"


			# clean and assemble letter info
			letter_bdody = re.sub(r'<[^>]+>', "", letter_body) # remove html tags if any
			letter_body = letter_body.strip() # remove trailing whitespace

			assert len(letter_body.strip()) > 0

			if place is None or place == "[n.d.]":
				place = "N/A"
			if time is None or time == "[n.d.]":
				time = "N/A"
			# N/A will be interpreted as None by pandas anyway
			
			# add new row to dataframe
			vw = vw.append({"id": id, "date": time, "recipient": recipient, "place": place, "text": letter_body}, ignore_index=True)
			count_per_file += 1

	print(f"Finished reading '{filename}'.")
	print(f"Added {count_per_file} letters to dataframe.")
	count_per_file = 0

print("Dataframe sample:")
print(vw.sample(30).to_string())
print("Dataframe info:")
print(vw.info())
vw.to_csv(VW_ORIGINAL, index_label = "index")
print(f"Original letters dataframe successfully written to '{VW_ORIGINAL}'.")