from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List
from starlette.testclient import TestClient

app = FastAPI()


@app.post("/")
def k2tapi(data: List[str]):
    print(data)

