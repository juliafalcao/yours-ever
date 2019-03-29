import codecs
from bs4 import BeautifulSoup as bs
import pandas as pd

with codecs.open("data\\vw\\letters\\35Lettera1.htm", mode = "r", encoding = "utf-8") as file:
    file_content = file.read()

soup = bs(file_content, "html.parser")
html = list(soup.children)[0]
body = list(html.children)[1]
headers = [item.get_text() for item in soup.find_all("h3")]
ids = [header[:header.find(":")] for header in headers]
receivers = [header[header.find(":")+1:].lstrip() for header in headers]
dates = [item.get_text().replace("[", "").replace("]", "") for item in soup.find_all("p", {"class": "time"})]
places = [item.get_text() for item in soup.find_all("p", {"class": "place"})]
contents = []

# reconstruir conteúdo de cada carta
letter_body = ""

for item in soup.find_all("p"):
    if "class" in item.attrs:
        if "date" in item.attrs["class"]:
            # achou o separador de cartas -- salvar body e zerar para começar a próxima
            contents.append("".join(letter_body))
            letter_body = ""
        
        elif "name" in item.attrs["class"] or "noind" in item.attrs["class"]: # para os <p> com classe "name" (assinaturas)
            letter_body += item.get_text() + "\n"

        elif "place" in item.attrs["class"] or "time" in item.attrs["class"] or "imgbrd" in item.attrs["class"]:
            continue # campos indesejados (já salvos)

    else: # para os <p> simples
        letter_body += item.get_text() + "\n"


# PRINTAR CARTAS INTEIRAS
i = 0
print(f"CARTA {i}:")
print(f"id: {ids[i]}")
print(f"date: {dates[i]}")
print(f"receiver: {receivers[i]}")
print(f"place: {places[i]}")
print(f"body: {contents[i]}")

i = len(ids)-1
print(f"CARTA {i}:")
print(f"id: {ids[i]}")
print(f"date: {dates[i]}")
print(f"receiver: {receivers[i]}")
print(f"place: {places[i]}")
print(f"body: {contents[i]}")



# CONSTRUIR DATAFRAME
# vw = pd.DataFrame({"id": ids, "date": dates, "receiver": receivers, "place": places, "text": contents})

"""
problema: falta alguma carta do meio na contents (não a primeira nem a última)
"""