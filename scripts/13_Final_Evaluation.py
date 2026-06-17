import numpy as np
import os
import faiss
import joblib
import pandas as pd
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as ssim

# ---------------- LOAD ----------------
clip_embeddings = np.load("outputs/embeddings.npy").astype('float32')
image_paths = np.load("outputs/image_paths.npy", allow_pickle=True)
index = faiss.read_index("outputs/faiss.index")

resnet_features = np.load("outputs/resnet_features.npy")
dino_features = np.load("outputs/dino_features.npy")

model = joblib.load("outputs/fusion_model.pkl")
from hybrid.hybrid_model import hybrid_score

# ---------------- HELPERS ----------------
def load_image(path):
    return np.array(Image.open(os.path.join("data/processed", path)).convert("RGB"))

def get_id(path):
    return path.split("/")[-1]

def get_type(path):
    return path.split("/")[0]

def semantic_sim(e1, e2):
    return np.dot(e1, e2)

def structural_sim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return ssim(gray1, gray2)

def cocs_score(similarities):
    return np.mean(similarities) * (1 - np.std(similarities))

def build_index(features):
    idx = faiss.IndexFlatIP(features.shape[1])
    idx.add(features.astype('float32'))
    return idx

# ---------------- EXTRA INDEXES ----------------
resnet_index = build_index(resnet_features)
dino_index = build_index(dino_features)

# ---------------- GROUPS ----------------
groups = defaultdict(list)
for i, path in enumerate(image_paths):
    groups[get_id(path)].append(i)

# ---------------- MODELS ----------------
models = ["clip", "hybrid", "ml", "resnet", "dino", "ssim"]

# ---------------- EVALUATION ----------------
def evaluate(mode):
    topk_vals = [1, 3, 5, 10]
    topk_correct = {k: 0 for k in topk_vals}

    tp, fp, fn = 0, 0, 0
    total = 0
    ap_list = []
    total_cocs = 0

    transform_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for i in tqdm(range(len(image_paths)), desc=mode):

        query_path = image_paths[i]
        query_img = load_image(query_path)
        query_type = get_type(query_path)
        gt = set(groups[get_id(query_path)])

        # -------- SELECT FEATURE --------
        if mode == "clip":
            features = clip_embeddings
            idx = index

        elif mode == "resnet":
            features = resnet_features
            idx = resnet_index

        elif mode == "dino":
            features = dino_features
            idx = dino_index

        else:
            features = clip_embeddings
            idx = index

        query_emb = features[i].reshape(1, -1)
        D, I = idx.search(query_emb, 20)

        results = []
        similarities_list = []

        for j, base_score in zip(I[0][1:], D[0][1:]):

            if mode in ["clip", "resnet", "dino"]:
                score = base_score

            elif mode == "hybrid":
                emb = clip_embeddings[j]
                img = load_image(image_paths[j])
                score = hybrid_score(clip_embeddings[i], emb, query_img, img)

            elif mode == "ml":
                emb = clip_embeddings[j]
                img = load_image(image_paths[j])

                s_sem = semantic_sim(clip_embeddings[i], emb)
                s_str = structural_sim(query_img, img)

                score = model.predict_proba([[s_sem, s_str]])[0][1]

            elif mode == "ssim":
                img = load_image(image_paths[j])
                score = structural_sim(query_img, img)

            results.append((j, score))
            similarities_list.append(score)

        results = sorted(results, key=lambda x: x[1], reverse=True)

        # -------- METRICS --------
        for k in topk_vals:
            if any(idx in gt for idx, _ in results[:k]):
                topk_correct[k] += 1

        hit = False
        for idx2, _ in results[:5]:
            if idx2 in gt:
                tp += 1
                hit = True
            else:
                fp += 1

        if hit:
            transform_stats[query_type]["correct"] += 1

        transform_stats[query_type]["total"] += 1

        fn += len(gt) - 1
        total += 1

        # mAP
        hits = 0
        precision_sum = 0

        for rank, (idx2, _) in enumerate(results[:10], start=1):
            if idx2 in gt:
                hits += 1
                precision_sum += hits / rank

        ap_list.append(precision_sum / hits if hits > 0 else 0)

        total_cocs += cocs_score(similarities_list[:5])

    # -------- FINAL --------
    transform_acc = {}
    all_types = ["original", "crop", "compression", "noise", "text_overlay", "screenshot"]

    for t in all_types:
        if t in transform_stats:
            transform_acc[t] = transform_stats[t]["correct"] / transform_stats[t]["total"]
        else:
            transform_acc[t] = 0

    return {
        "accuracy": topk_correct[5] / total,
        "top1": topk_correct[1] / total,
        "top3": topk_correct[3] / total,
        "top5": topk_correct[5] / total,
        "top10": topk_correct[10] / total,
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "mAP": np.mean(ap_list),
        "COCS": total_cocs / total,
        **transform_acc
    }

# ---------------- RUN ----------------
rows = []

for m in models:
    res = evaluate(m)
    res["model"] = m
    rows.append(res)

df = pd.DataFrame(rows)

# ---------------- SAVE CSV ----------------
os.makedirs("results", exist_ok=True)
df.to_csv("results/final_comparison.csv", index=False)

print("\n✅ Results saved to: results/final_comparison.csv")
print(df)