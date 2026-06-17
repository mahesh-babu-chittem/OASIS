import os
import pandas as pd
import random

base_dir = "data/processed"

folders = ["original", "crop", "compression", "noise", "text_overlay", "screenshot"]

# collect all files
files = os.listdir(os.path.join(base_dir, "original"))

data = []

# ---------------- POSITIVE PAIRS ----------------
for f in files:
    for folder in folders:
        if folder == "original":
            continue
        
        img1 = f"original/{f}"
        img2 = f"{folder}/{f}"
        
        data.append([img1, img2, 1])

# ---------------- NEGATIVE PAIRS ----------------
for _ in range(len(files)):
    f1, f2 = random.sample(files, 2)
    
    img1 = f"original/{f1}"
    img2 = f"original/{f2}"
    
    data.append([img1, img2, 0])

# ---------------- SAVE ----------------
df = pd.DataFrame(data, columns=["img1", "img2", "label"])

os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/labels.csv", index=False)

print("✅ labels.csv created:", len(df))