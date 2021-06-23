    nlp = eval_pipeline("k2t")
    prediction = []
    for key in keywords_test:
        prediction.append(nlp(keywords=key))

    test["predctions"] = prediction

    return test
