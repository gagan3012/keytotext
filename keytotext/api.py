from fastapi import FastAPI
from typing import List
from keytotext.pipeline import pipeline

app = FastAPI()


def generate(keywords, model="k2t"):
    nlp = pipeline(model)
    return nlp(keywords)


@app.post("/")
def k2tpost(data: List[str]):
    return {
        "keywords": data,
        "text": generate(data)
    }


@app.get("/")
def k2tget(data: List[str]):
    return {"text": generate(data)}