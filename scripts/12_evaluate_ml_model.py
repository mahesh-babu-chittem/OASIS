import numpy as np
import os
import faiss
import joblib
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as ssim

# ---------------- LOAD ----------------
embeddings = np.load("outputs/embeddings.npy").astype('float32')
image_paths = np.load("outputs/image_paths.npy", allow_pickle=True)

model = joblib.load("outputs/fusion_model.pkl")
index = faiss.read_index("outputs/faiss.index")

# ---------------- HELPERS ----------------
def load_image(path):
    return np.array(Image.open(os.path.join("data/processed", path)).convert("RGB"))

def get_id(path):
    return path.split("/")[-1]

def semantic_sim(e1, e2):
    return np.dot(e1, e2)

def structural_sim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return ssim(gray1, gray2)

# ---------------- COCS ----------------
def cocs_score(similarities):
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    return mean_sim * (1 - std_sim)

# ---------------- GROUPS ----------------
groups = defaultdict(list)

for i, path in enumerate(image_paths):
    groups[get_id(path)].append(i)

# ---------------- EVALUATION ----------------
k = 5
correct = 0
total = 0

tp = 0
fp = 0
fn = 0

total_cocs = 0

for i in tqdm(range(len(embeddings))):
    query_emb = embeddings[i]
    query_img = load_image(image_paths[i])

    # retrieve candidates
    D, I = index.search(query_emb.reshape(1, -1), 10)

    results = []
    similarities_list = []

    for idx in I[0][1:]:  # skip self
        emb = embeddings[idx]
        img = load_image(image_paths[idx])

        s_sem = semantic_sim(query_emb, emb)
        s_str = structural_sim(query_img, img)

        score = model.predict_proba([[s_sem, s_str]])[0][1]

        results.append((idx, score))
        similarities_list.append(score)

    # sort top-k
    results = sorted(results, key=lambda x: x[1], reverse=True)[:k]

    # compute COCS
    cocs = cocs_score(similarities_list[:k])
    total_cocs += cocs

    # ground truth
    query_id = get_id(image_paths[i])
    gt = set(groups[query_id])

    hit = False

    for idx, _ in results:
        if idx in gt:
            hit = True
            tp += 1
        else:
            fp += 1

    if hit:
        correct += 1

    fn += len(gt) - 1
    total += 1

# ---------------- FINAL METRICS ----------------
accuracy = correct / total
precision = tp / (tp + fp)
recall = tp / (tp + fn)
avg_cocs = total_cocs / total

print("\n📊 ML Model Results")
print("---------------------")
print(f"Top-{k} Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"COCS Score: {avg_cocs:.4f}")