from random import randint

from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/count")
def count() -> dict[str, int]:
    return {"count": randint(5, 50)}


@app.post("/start")
def start() -> dict[str, str]:
    return {"status": "started"}


@app.post("/stop")
def stop() -> dict[str, str]:
    return {"status": "stopped"}

@app.post("/rounds/start")
def start_round():
    return {"status": "started"}

@app.post("/rounds/stop")
def stop_round():
    return {"status": "stopped"}