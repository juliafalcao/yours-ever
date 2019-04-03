import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import dateutil.parser as dateparser

def remove_brackets(text: str):
    if pd.notnull(text):
        brackets = r'[\[\]]'
        return re.sub(brackets, "", text)

def trim_receiver(receiver: str):
    receiver = receiver.replace("\n", " ").replace("\r", " ")
    receiver = re.sub(" +", " ", receiver)

    if pd.notnull(receiver) and len(receiver) > 4 and receiver[:3] == "To ":
        return receiver[3:]
    else:
        return receiver

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


vw = pd.read_csv("data\\vw\\vw_dataset.csv", index_col = "Unnamed: 0")

vw["date"] = vw["date"].apply(remove_brackets)
vw["place"] = vw["place"].apply(remove_brackets)
vw["receiver"] = vw["receiver"].apply(trim_receiver)
vw["receiver"] = vw["receiver"].replace("V. Sackville-West", "Vita Sackville-West")
vw["length"] = vw["text"].apply(lambda text: len(text))

vw["year"] = vw["date"].apply(extract_year)
vw = fill_missing_years(vw)

years = list(vw[vw["year"].notnull()]["year"])

plt.hist(years, bins = len(set(years)), rwidth=0.8, color="lightseagreen", alpha=0.9, align="mid")
plt.title("Amount of letters written by Virginia Woolf")
plt.xlabel("Years")
plt.ylabel("Letters")
plt.xticks(list(range(0, 46, 5)))
plt.savefig("graphs\\letters-per-year.png")
plt.show()