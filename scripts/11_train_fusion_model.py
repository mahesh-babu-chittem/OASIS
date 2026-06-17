import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# ---------------- LOAD ----------------
embeddings = np.load("outputs/embeddings.npy")
image_paths = np.load("outputs/image_paths.npy", allow_pickle=True)

labels_df = pd.read_csv("outputs/labels.csv")

# ---------------- HELPERS ----------------
def get_index(path):
    return list(image_paths).index(path)

def load_image(path):
    return np.array(Image.open(os.path.join("data/processed", path)).convert("RGB"))

def semantic_sim(e1, e2):
    return np.dot(e1, e2)

def structural_sim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return ssim(gray1, gray2)

# ---------------- BUILD FEATURES ----------------
X = []
y = []

for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
    try:
        idx1 = get_index(row['img1'])
        idx2 = get_index(row['img2'])

        e1 = embeddings[idx1]
        e2 = embeddings[idx2]

        img1 = load_image(row['img1'])
        img2 = load_image(row['img2'])

        s_sem = semantic_sim(e1, e2)
        s_str = structural_sim(img1, img2)

        X.append([s_sem, s_str])
        y.append(row['label'])

    except:
        continue

X = np.array(X)
y = np.array(y)

# ---------------- TRAIN ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

# ---------------- SAVE ----------------
import joblib
joblib.dump(model, "outputs/fusion_model.pkl")

print("✅ Model trained & saved")