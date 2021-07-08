from keytotext import trainer
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = load_dataset('common_gen', split='train')


def clean(keywords):
    source_text = ' '.join(map(str, keywords))
    return source_text


def create_df(dataset):


model = trainer()
model.from_pretrained(model_name="t5-small")
model.train(train_df = train_df,test_df=test_df,batch_size=4, max_epochs=3, use_gpu=True)
model.upload("gagan3012","k2t-test3")