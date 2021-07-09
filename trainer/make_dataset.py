    df['keywords'] = df['keywords'].apply(lambda x: clean(x))
    train_df, test_df = train_test_split(df, test_size=0.01, random_state=42)
    return train_df, test_df
