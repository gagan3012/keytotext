from datasets import load_dataset
import pandas as pd
from keybert import KeyBERT


def clean(keywords):
    source_text = " ".join(map(str, keywords))
    return source_text

def clean_keywords(keywords):
  return clean(list(map(list, zip(*keywords)))[0])

def make_keywords(df):
  kw_model = KeyBERT()
  for i in range(len(df)):
    keyword = kw_model.extract_keywords(df['text'][i])
    clean = clean_keywords(keyword)
    df["keywords"][i] = clean
    print(i)
  return df

def dataset(dataset="common_gen", split="train"):
  try:
    if dataset == "common_gen":
        dataset = load_dataset(dataset, split=split)
        df = pd.DataFrame()
        df["keywords"] = dataset["concepts"]
        df["text"] = dataset["target"]
        df["keywords"] = df["keywords"].apply(clean)
        return df
    else:
        dataset = load_dataset(dataset, split=split)
        df = pd.DataFrame()

    return None
  except:
    return ValueError("Dataset not found")
