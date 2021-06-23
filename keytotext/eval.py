            )
    if isinstance(tokenizer, (str, tuple)):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # Instantiate model if needed
    if isinstance(model, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model)

    if task == "k2t":
        return task_class(model=model, tokenizer=tokenizer, use_cuda=use_cuda)
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
