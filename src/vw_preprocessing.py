"""
pre-processing of virginia woolf's letters
(to be generalized at some point)
"""

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import dateutil.parser as dateparser

def remove_brackets(text: str):
    if pd.notnull(text):
        brackets = r'[\[\]]'
        return re.sub(brackets, "", text)

def remove_dashes(text: str):
    if pd.notnull(text):
        dashes = r'[-—]'
        return re.sub(dashes, "", text)


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
        year = dateparser.parse(full_date, fuzzy=True).year

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

vw = pd.read_csv("data\\vw\\vw_from_epub.csv", index_col = "index")

vw["place"] = vw["place"].apply(remove_brackets)
vw["recipient"] = vw["recipient"].apply(trim_recipient)
vw["recipient"] = vw["recipient"].apply(lambda r: r.replace("\xa0", " "))
vw["recipient"] = vw["recipient"].replace("V. Sackville-West", "Vita Sackville-West") # ♡
vw["length"] = vw["text"].apply(lambda text: len(text))

vw["date"] = vw["date"].apply(remove_brackets)
vw["year"] = vw["date"].apply(extract_year)
vw = fill_missing_years(vw)

# (...)

# print(vw.sample(20).to_string())
print(vw.info())

vw = vw[["id", "year", "recipient", "text", "length"]]
vw.to_csv("data\\vw\\vw_preprocessed.csv", index_label="index")