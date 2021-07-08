from keytotext import trainer
from datasets import load_dataset

model = trainer()
model.from_pretrained(model_name="t5-small")
model.train(train_df = train_df,test_df=test_df,batch_size=4, max_epochs=3, use_gpu=True)
model.upload("gagan3012","k2t-test3")