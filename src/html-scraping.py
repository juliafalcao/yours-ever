import codecs, os, re
from bs4 import BeautifulSoup as bs
from bs4 import NavigableString
import pandas as pd


export_path = "data\\vw\\letters\\output"
letters_path = "data\\vw\\letters"

vw = pd.DataFrame({"id": [], "date": [], "receiver": [], "place": [], "text": []})

for letters_file in os.listdir(letters_path): # loop through files in dir
    if ".htm" not in letters_file:
        continue # skip subdirectories

    print(f"Reading '{letters_file}'...")

    with codecs.open(f"{letters_path}\\{letters_file}", mode="r", encoding="utf-8") as file:
        file_content = file.read()

    soup = bs(file_content, "html.parser")
    html = soup.find("html")
    body = html.find("body")
    soup = body

    # headers separated in ids and receivers
    headers = []
    for h3item in soup.find_all("h3"):
        if h3item.has_attr("id") or h3item.has_attr("title"): # only get actual letter headers
            headers.append(h3item.get_text())

    ids = [header[:header.find(":")] for header in headers]
    receivers = [header[header.find(":")+1:].lstrip() for header in headers]
    print(f"Found {len(headers)} headers ({len(ids)} IDs and {len(receivers)} receivers).")

    # dates
    dates = [item.get_text().replace("[", "").replace("]", "") for item in soup.find_all("p", {"class": "time"})]
    print(f"Found {len(dates)} dates.")

    # places
    places = []
    for item in soup.find_all("p", {"class": "place"}):
        previous = str(item.previous_sibling.previous_sibling.encode("utf-8"))
        if "time" in previous or "card" in previous: # only get class="place" if it comes after class="time" or class="card"
            places.append(item.get_text())
    
    print(f"Found {len(places)} places.")

    # letter bodies -- everything in between class="place" and class="link" except classes in to_ignore
    contents = []
    letter_body = ""
    to_ignore = ["date", "card", "place", "time", "sigil_not_in_toc", "imgbrd", "margl"]
    delimiter = "link"

    for tag in soup.find("p", {"class": "place"}).next_siblings:
        if (not isinstance(tag, NavigableString)) and tag.get("class") is not None:
            if tag.get("class")[0] in to_ignore:
                continue

            if tag.get("class")[0] == delimiter:
                if len(letter_body.strip().lstrip()) > 0: # skip empty bodies - probably scraping mistakes
                    contents.append(letter_body)
                letter_body = ""
            else:
                letter_body += str(tag)
        else:
            letter_body += str(tag)
    
    print(f"Found {len(contents)} letter bodies.")


    # remove html tags
    html_tag_regex = re.compile(r'<[^>]+>')
    for i in range(len(contents)):
        contents[i] = contents[i].lstrip().strip() # remove trailing whitespace
        contents[i] = html_tag_regex.sub("", contents[i]) # remove all html tags

    # export all letters (for validation)
    out_all_files = f"{export_path}\\letters.txt"
    out_current_file = f"{export_path}\\{letters_file.replace('.htm', '.txt')}"

    for letter in contents:
        file = codecs.open(out_current_file, mode="a+", encoding="utf-8")
        file.write(letter)
        file.write("\n--------------\n")
        file.close()

    # add to complete dataframe    
    vw_part = pd.DataFrame({"id": ids, "date": dates, "receiver": receivers, "place": places, "text": contents})
    vw = vw.append(vw_part)
    print(f"finished reading '{letters_file}'.")

print("DATAFRAME HEAD:")
print(vw.head(40).to_string())
print(vw.describe())