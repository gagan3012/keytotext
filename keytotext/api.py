from fastapi import FastAPI
from typing import List
from keytotext.pipeline import pipeline

app = FastAPI()


def generate(keywords, model="k2t"):
    nlp = pipeline(model)
    return nlp(keywords)


@app.post("/", response_model=List[str])
def k2tpost(data: List[str]):
    return {
        "keywords": data,
        "text": generate(data)
    }
