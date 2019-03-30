import codecs, os, re
from bs4 import BeautifulSoup as bs
from bs4 import NavigableString
import pandas as pd

export_path = "data\\vw\\letters\\output"
letters_path = "data\\vw\\letters"


for letters_file in os.listdir(letters_path): # loop through files in dir
    if ".htm" not in letters_file:
        continue # skip subdirectories

    with codecs.open(f"{letters_path}\\{letters_file}", mode="r", encoding="utf-8") as file:
        file_content = file.read()

    soup = bs(file_content, "html.parser")
    html = list(soup.children)[0]
    body = list(html.children)[1]
    headers = [item.get_text() for item in soup.find_all("h3")]
    ids = [header[:header.find(":")] for header in headers]
    receivers = [header[header.find(":")+1:].lstrip() for header in headers]
    dates = [item.get_text().replace("[", "").replace("]", "")
            for item in soup.find_all("p", {"class": "time"})]
    places = [item.get_text() for item in soup.find_all("p", {"class": "place"})]
    contents = []

    # reconstruir conteúdo de cada carta -- tudo que estiver entre as linhas de p class "place" e "date"
    letter_body = ""
    to_ignore = ["date", "card", "place", "time", "sigil_not_in_toc", "imgbrd"]
    delimiter = "link"

    for tag in soup.find("p", {"class": "place"}).next_siblings:
        if (not isinstance(tag, NavigableString)) and tag.get("class") is not None:
            if tag.get("class")[0] in to_ignore:
                continue

            if tag.get("class")[0] == delimiter:
                contents.append(letter_body)
                letter_body = ""
            else:
                letter_body += str(tag)
        else:
            letter_body += str(tag)


    # remove html tags
    html_tag_regex = re.compile(r'<[^>]+>')
    for i in range(len(contents)):
        contents[i] = contents[i].lstrip().strip() # remove trailing whitespace
        # contents[i] = contents[i].replace("</p>", "\n").replace("</br>", "\n") # add line breaks
        contents[i] = html_tag_regex.sub("", contents[i]) # remove html tags

    # export all letters (for debugging)
    for letter in contents:
        export_file = codecs.open(f"{export_path}\\{letters_file.replace('.htm', '')}.txt", "a+", "utf-8")
        export_file.write(letter)
        export_file.write("\n--------------\n")


# CONSTRUIR DATAFRAME
vw = pd.DataFrame({"id": ids, "date": dates, "receiver": receivers, "place": places, "text": contents})
print(vw.head(30).to_string())