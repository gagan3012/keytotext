from keytotext import trainer, make_dataset

train_df = make_dataset('common_gen', split='train')
eval_df = make_dataset('common_gen', split='val')
test_df = make_dataset('common_gen',split='test')

model = trainer()
model.from_pretrained(model_name="t5-small")
model.train(train_df=train_df, test_df=test_df, batch_size=4, max_epochs=3, use_gpu=True)
