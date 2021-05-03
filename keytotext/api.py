from fastapi import FastAPI
from typing import List
from keytotext.pipeline import pipeline

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


def generate(keywords, model="k2t"):
    nlp = pipeline(model)
    return nlp(keywords)


@app.post("/")
def k2tpost(data: List[str]):
    return {
        "keywords": data,
        "text": generate(data)
    }
