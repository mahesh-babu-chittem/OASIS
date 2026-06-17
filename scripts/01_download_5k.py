import os
import requests
from tqdm import tqdm

os.makedirs("data/raw", exist_ok=True)

BASE_URL = "https://source.unsplash.com/random/640x480"
NUM_IMAGES = 5000

headers = {
    "User-Agent": "Mozilla/5.0"
}

for i in tqdm(range(NUM_IMAGES)):
    try:
        response = requests.get(BASE_URL, headers=headers, timeout=5)

        content_type = response.headers.get("Content-Type", "")

        # ✅ Only save if it's actually an image
        if "image" in content_type:
            with open(f"data/raw/img_{i}.jpg", "wb") as f:
                f.write(response.content)
        else:
            print(f"Skipped {i} (not an image)")

    except Exception as e:
        print(f"Error at {i}: {e}")