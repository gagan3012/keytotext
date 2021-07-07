from keytotext import trainer

model = trainer()
model.from_pretrained(model_name="t5-small")
model.train(train_df = train_df[:2000],test_df=test_df,batch_size=4, max_epochs=3, use_gpu=True)
