from random import randint

from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/count")
def count() -> dict[str, int]:
    return {"count": randint(5, 50)}
