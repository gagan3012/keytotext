from datasets import load_dataset
import pandas as pd


def clean(keywords):
    source_text = " ".join(map(str, keywords))
    return source_text


def make_dataset(dataset="common_gen", split="train"):
        dataset = load_dataset(dataset, split=split)
        df = pd.DataFrame()
        return df
    else:
        return None
