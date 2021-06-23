    if task == "k2t-base":
        return task_class(model=model, tokenizer=tokenizer, use_cuda=use_cuda)


def eval():
    test = pd.read_csv("data/TestNLG.csv")
    keywords_test = test["input_text"]

    nlp = eval_pipeline("k2t")
    prediction = []
    for key in keywords_test:
        prediction.append(nlp(keywords=key))

    test["predctions"] = prediction

    return test
