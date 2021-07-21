from keytotext import trainer

model = trainer()


model.load_model()

keywords = ['New delhi','fire','house']

text = model.predict(keywords)

print(text)