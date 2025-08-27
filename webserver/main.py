import base64
import io
import json
import os
import time
import uuid

import numpy as np
from PIL import Image
import redis
from fastapi import FastAPI, File, UploadFile, HTTPException

# Init FastAPI
app = FastAPI()

# Connect to Redis (hostname comes from docker-compose service name)
redis_host = os.environ.get("REDIS_HOST", "redis")
db = redis.Redis(host=redis_host, decode_responses=False)

# Config from env
IMAGE_QUEUE = os.environ.get("IMAGE_QUEUE", "image_queue")
CLIENT_SLEEP = float(os.environ.get("CLIENT_SLEEP", 0.25))
CLIENT_MAX_TRIES = int(os.environ.get("CLIENT_MAX_TRIES", 100))


def prepare_image(image, target_size):
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize and convert to NumPy
    image = image.resize(target_size)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)  # add batch dim
    return image


@app.get("/")
def index():
    return {"message": "Hello World from FastAPI Webserver"}


@app.post("/predict")
async def predict(img_file: UploadFile = File(...)):
    # Load image
    image_bytes = await img_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = prepare_image(
        image,
        (
            int(os.environ.get("IMAGE_WIDTH", 224)),
            int(os.environ.get("IMAGE_HEIGHT", 224)),
        ),
    )

    # Serialize image
    image = image.copy(order="C")  # ensure contiguous
    image_b64 = base64.b64encode(image).decode("utf-8")

    # Unique ID
    job_id = str(uuid.uuid4())
    job_data = {"id": job_id, "image": image_b64}

    # Push job into Redis
    db.rpush(IMAGE_QUEUE, json.dumps(job_data))

    # Poll for result
    tries = 0
    while tries < CLIENT_MAX_TRIES:
        tries += 1
        output = db.get(job_id)

        if output is not None:
            predictions = json.loads(output.decode("utf-8"))
            db.delete(job_id)
            return {"success": True, "predictions": predictions}

        time.sleep(CLIENT_SLEEP)

    raise HTTPException(status_code=408, detail="Model server did not respond in time")
