import os
from PIL import Image
from tqdm import tqdm

input_dir = "data/raw"
output_dir = "data/processed/original"

os.makedirs(output_dir, exist_ok=True)

for file in tqdm(os.listdir(input_dir)):
    try:
        path = os.path.join(input_dir, file)
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224))
        img.save(os.path.join(output_dir, file))
    except:
        print("Error:", file)

print("✅ Preprocessing complete")