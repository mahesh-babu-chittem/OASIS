import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_features():
    base_path = "data/processed"
    folders = ["original", "crop", "compression", "noise", "text_overlay", "screenshot"]

    features = []
    paths = []

    for folder in folders:
        folder_path = os.path.join(base_path, folder)

        for file in tqdm(os.listdir(folder_path), desc=folder):
            try:
                img = Image.open(os.path.join(folder_path, file)).convert("RGB")
                img = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    feat = model(img).squeeze().cpu().numpy()

                feat = feat / np.linalg.norm(feat)

                features.append(feat)
                paths.append(f"{folder}/{file}")

            except:
                continue

    np.save("outputs/dino_features.npy", np.array(features))
    np.save("outputs/dino_paths.npy", paths)

    print("✅ DINO features saved")

if __name__ == "__main__":
    extract_features()