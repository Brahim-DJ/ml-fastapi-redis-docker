import base64
import json
import os
import time
import sys

import numpy as np
import redis
from tensorflow.keras.applications import ResNet50, imagenet_utils

# Redis connection (blocking/sync)
redis_host = os.environ.get("REDIS_HOST", "redis")
db = redis.StrictRedis(host=redis_host, decode_responses=False)

# Model & config
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
IMAGE_QUEUE = os.environ.get("IMAGE_QUEUE", "image_queue")
SERVER_SLEEP = float(os.environ.get("SERVER_SLEEP", 0.25))

# Load model once per process
print("Loading model...")
model = ResNet50(weights="imagenet")
print("Model loaded.")

def base64_decode_image(b64_string, dtype=np.uint8, shape=None):
    # Python3: b64_string is text => encode to bytes
    if isinstance(b64_string, str):
        b64_string = b64_string.encode("utf-8")
    decoded = base64.b64decode(b64_string)
    arr = np.frombuffer(decoded, dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr

def classify_process():
    while True:
        # Atomically pop up to BATCH_SIZE items using pipeline (lrange + ltrim)
        with db.pipeline() as pipe:
            pipe.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
            pipe.ltrim(IMAGE_QUEUE, BATCH_SIZE, -1)
            queue, _ = pipe.execute()

        imageIDs = []
        batch = None
        shapes = None
        for q in queue:
            q = json.loads(q.decode("utf-8"))
            image_id = q["id"]
            image_b64 = q["image"]

            # We assume sender sent a flattened C-contiguous uint8 array and knows shape.
            # For simplicity we expect shape env vars on modelserver.
            h = int(os.environ.get("IMAGE_HEIGHT", 224))
            w = int(os.environ.get("IMAGE_WIDTH", 224))
            ch = int(os.environ.get("IMAGE_CHANS", 3))
            img = base64_decode_image(image_b64, dtype=np.uint8, shape=(1, h, w, ch))

            if batch is None:
                batch = img
            else:
                batch = np.vstack([batch, img])

            imageIDs.append(image_id)

        if len(imageIDs) > 0:
            print(f"* Processing batch of size {batch.shape[0]}")
            preds = model.predict(batch)
            results = imagenet_utils.decode_predictions(preds, top=5)

            for (imageID, resultSet) in zip(imageIDs, results):
                output = []
                for (_, label, prob) in resultSet:
                    output.append({"label": label, "probability": float(prob)})
                # Store result with a TTL so Redis doesn't fill up permanently
                db.set(imageID, json.dumps(output))
                db.expire(imageID, int(os.environ.get("RESULT_TTL", 300)))  # default 5 min

        time.sleep(SERVER_SLEEP)

if __name__ == "__main__":
    try:
        classify_process()
    except KeyboardInterrupt:
        print("Shutting down.")
        sys.exit(0)
