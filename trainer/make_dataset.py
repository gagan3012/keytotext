from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def clean(keywords):
    source_text = ' '.join(map(str, keywords))
    return source_text


def create_df():
    dataset = load_dataset('common_gen', split='train')
    df = pd.DataFrame()
    df['keywords'] = dataset['concepts']
