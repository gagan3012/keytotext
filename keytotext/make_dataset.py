from datasets import load_dataset
import pandas as pd

def clean(keywords):
    source_text = ' '.join(map(str, keywords))

