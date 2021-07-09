from keytotext import trainer, make_dataset

test_df = make_dataset(split='test')

model = trainer()
model.from_pretrained(model_name="t5-small")
model.train(train_df=train_df, test_df=test_df, batch_size=4, max_epochs=3, use_gpu=True)
model.upload(hf_username="gagan3012",model_name="k2t-test3")