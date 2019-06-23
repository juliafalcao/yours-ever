# -*- coding: utf-8 -*-

"""
html scraping to obtain content and info of virginia woolf's letters from epub book
"""

from const import *
import codecs, os, re
from bs4 import BeautifulSoup as bs
from bs4 import NavigableString
import pandas as pd

export_path = "data\\vw\\output"
letters_path = "data\\vw\\letters"

vw = pd.DataFrame({"id": [], "date": [], "recipient": [], "place": [], "text": []})

for letters_file in os.listdir(letters_path): # loop through html files in dir
	if ".htm" not in letters_file:
		continue # skip subdirectories

	print(f"Reading '{letters_file}'...")

	with codecs.open(f"{letters_path}\\{letters_file}", mode="r", encoding="utf-8") as file:
		file_content = file.read()

	soup = bs(file_content, "html.parser")
	html = soup.find("html")
	body = html.find("body")
	soup = body

	# reading letters from html file
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
						continue # ignore
					
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
			html_tag_regex = re.compile(r'<[^>]+>')
			letter_body = html_tag_regex.sub("", letter_body) # remove html tags if any
			letter_body = letter_body.strip() # remove trailing whitespace

			if len(letter_body.strip()) < 0:
				print("ERROR: empty letter body :O")
				exit()

			if place is None or place == "[n.d.]":
				place = "N/A"
			if time is None or time == "[n.d.]":
				time = "N/A"
			# N/A will be interpreted as None by pandas anyway
			
			# add new row to dataframe
			vw = vw.append({"id": id, "date": time, "recipient": recipient, "place": place, "text": letter_body}, ignore_index=True)
			count_per_file += 1


	# export letter body (for validation purposes)
	"""
	out_all_files = f"{export_path}\\letters.txt"
	file = codecs.open(out_all_files, mode="a+", encoding="utf-8")
	file.write(letter_body)
	file.write("\n--------------\n")
	file.close()
	"""

	print(f"Finished reading '{letters_file}'.")
	print(f"Added {count_per_file} letters to dataframe.")
	count_per_file = 0

print("DATAFRAME SAMPLE:")
print(vw.sample(30).to_string())
print("INFO:")
print(vw.info())
vw.to_csv(VW_RAW, index_label = "index")
print(f"Raw letters dataframe successfully written to {VW_RAW}.")