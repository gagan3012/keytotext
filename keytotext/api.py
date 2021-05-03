from fastapi import FastAPI
from typing import List
from keytotext.pipeline import pipeline

app = FastAPI()


def generate(keywords, model="k2t"):
    nlp = pipeline(model)
    return nlp(keywords)

@app.post("/")
def k2tapi(data: List[str]):
    return generate(data)

