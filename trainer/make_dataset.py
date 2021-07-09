    source_text = ' '.join(map(str, keywords))
    return source_text


def create_df():
    dataset = load_dataset('common_gen', split='train')
    df = pd.DataFrame()
    df['keywords'] = dataset['concepts']
    df['text'] = dataset['target']
    df['keywords'] = df['keywords'].apply(lambda x: clean(x))
    train_df, test_df = train_test_split(df, test_size=0.01, random_state=42)
    return train_df, test_df
