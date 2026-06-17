import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import random

input_dir = "data/processed/original"

output_dirs = {
    "crop": "data/processed/crop",
    "compression": "data/processed/compression",
    "noise": "data/processed/noise",
    "text_overlay": "data/processed/text_overlay",
    "screenshot": "data/processed/screenshot"
}

# Create folders
for d in output_dirs.values():
    os.makedirs(d, exist_ok=True)

files = os.listdir(input_dir)

for file in tqdm(files):
    try:
        path = os.path.join(input_dir, file)

        pil_img = Image.open(path)
        cv_img = cv2.imread(path)

        # ---------------- CROP ----------------
        w, h = pil_img.size
        crop_w = int(w * random.uniform(0.6, 0.9))
        crop_h = int(h * random.uniform(0.6, 0.9))
        cropped = pil_img.crop((0, 0, crop_w, crop_h)).resize((224, 224))
        cropped.save(os.path.join(output_dirs["crop"], file))

        # ---------------- COMPRESSION ----------------
        pil_img.save(os.path.join(output_dirs["compression"], file), quality=30)

        # ---------------- NOISE ----------------
        noise = np.random.normal(0, 25, cv_img.shape)
        noisy = np.clip(cv_img + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dirs["noise"], file), noisy)

        # ---------------- TEXT OVERLAY ----------------
        text_img = cv_img.copy()
        cv2.putText(text_img, "Sample Text", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.imwrite(os.path.join(output_dirs["text_overlay"], file), text_img)

        # ---------------- SCREENSHOT ----------------
        small = pil_img.resize((180, 180))
        bordered = ImageOps.expand(small, border=20, fill="black")
        final = bordered.resize((224, 224))
        final.save(os.path.join(output_dirs["screenshot"], file))

    except Exception as e:
        print("Error:", file, e)

print("✅ All transformations generated")