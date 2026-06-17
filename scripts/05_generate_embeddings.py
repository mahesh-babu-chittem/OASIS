import os
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------- DEVICE SETUP ----------------
if torch.backends.mps.is_available():
    device = "mps"   # Apple GPU
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

# ---------------- LOAD MODEL ----------------
model, preprocess = clip.load("ViT-B/32", device=device)

# ---------------- DATA ----------------
base_path = "data/processed"
folders = ["original", "crop", "compression", "noise", "text_overlay", "screenshot"]

image_paths = []
embeddings = []

# Collect all image paths first
all_images = []

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    for file in os.listdir(folder_path):
        all_images.append((folder, file))

print("Total images:", len(all_images))

# ---------------- BATCH SETTINGS ----------------
BATCH_SIZE = 32   # adjust if memory issue

# ---------------- PROCESS ----------------
for i in tqdm(range(0, len(all_images), BATCH_SIZE), desc="Generating Embeddings"):
    batch = all_images[i:i+BATCH_SIZE]

    images = []
    valid_paths = []

    for folder, file in batch:
        try:
            path = os.path.join(base_path, folder, file)
            img = preprocess(Image.open(path).convert("RGB"))
            images.append(img)
            valid_paths.append(f"{folder}/{file}")
        except:
            continue

    if len(images) == 0:
        continue

    images = torch.stack(images).to(device)

    with torch.no_grad():
        emb = model.encode_image(images)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    emb = emb.cpu().numpy()

    embeddings.extend(emb)
    image_paths.extend(valid_paths)

# ---------------- SAVE ----------------
embeddings = np.array(embeddings)

os.makedirs("outputs", exist_ok=True)

np.save("outputs/embeddings.npy", embeddings)
np.save("outputs/image_paths.npy", image_paths)

print("✅ Embeddings generated:", embeddings.shape)