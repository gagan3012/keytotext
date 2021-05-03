from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()

@app.post("/")
def k2tapi(data: List[str]):
    print(data)