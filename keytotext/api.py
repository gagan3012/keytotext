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


@app.post("/api")
def k2t_post(data: List[str]):
    return {
        "keywords": data,
        "text": generate(data)
    }
