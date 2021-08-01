from datasets import load_dataset
import pandas as pd


def clean(keywords):
    source_text = " ".join(map(str, keywords))
    return source_text


def make_dataset(dataset="common_gen", split="train"):
    if dataset == "common_gen":
        dataset = load_dataset(dataset, split=split)
        df = pd.DataFrame()
        df["keywords"] = dataset["concepts"]
        df["text"] = dataset["target"]
        df["keywords"] = df["keywords"].apply(lambda x: clean(x))
        return df
    else:
        return None
