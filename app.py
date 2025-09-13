from fastapi import FastAPI

app = FastAPI()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Return a simple health indicator."""
    return {"status": "ok"}
