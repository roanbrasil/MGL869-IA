from typing import Annotated

from fastapi import FastAPI, File

from model.classification import classify

app = FastAPI()


@app.post("/classify_images")
def classify_images(images: Annotated[list[bytes], File()]):
    return {"categories": classify(images)}
