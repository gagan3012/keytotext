from keytotext import trainer

    source_text = ' '.join(map(str, keywords))
    return source_text


def create_df(dataset):
    df = pd.DataFrame()
    df['keywords'] = dataset['concepts']
    df['text'] = dataset['target']
    df['keywords'] = df['keywords'].apply(lambda x: clean(x))
    train_df, test_df = train_test_split(df, test_size=0.01, random_state=42)
    return train_df, test_df

model = trainer()
model.from_pretrained(model_name="t5-small")
model.train(train_df=train_df, test_df=test_df, batch_size=4, max_epochs=3, use_gpu=True)
model.upload(hf_username="gagan3012",model_name="k2t-test3")