from keytotext import trainer

model = trainer()
model.from_pretrained(model_name="t5-small")
model.upload("gagan3012","k2t-test3")