import numpy as np
import os
import faiss
import joblib
import pandas as pd
from collections import defaultdict
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# ---------------- LOAD ----------------
image_paths = np.load("outputs/image_paths.npy", allow_pickle=True)

dino_features = np.load("outputs/dino_features.npy").astype('float32')
clip_embeddings = np.load("outputs/embeddings.npy").astype('float32')

model = joblib.load("outputs/fusion_model.pkl")
from hybrid.hybrid_model import hybrid_score

# ---------------- FAISS ----------------
index = faiss.IndexFlatIP(dino_features.shape[1])
index.add(dino_features)

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

# ---------------- GROUPS ----------------
groups = defaultdict(list)
for i, p in enumerate(image_paths):
    groups[get_id(p)].append(i)

# ---------------- METRICS ----------------
topk_vals = [1, 3, 5, 10]
topk_correct = {k: 0 for k in topk_vals}

tp, fp, fn = 0, 0, 0
total = 0
ap_list = []
total_cocs = 0

transform_stats = defaultdict(lambda: {"correct": 0, "total": 0})

# ---------------- PIPELINE EVALUATION ----------------
for i in tqdm(range(len(image_paths)), desc="pipeline"):

    query_dino = dino_features[i].reshape(1, -1)
    query_clip = clip_embeddings[i]
    query_img = load_image(image_paths[i])
    query_type = get_type(image_paths[i])

    gt = set(groups[get_id(image_paths[i])])

    # -------- DINO Retrieval --------
    D, I = index.search(query_dino, 20)
    candidates = I[0][1:]

    results = []
    similarities = []

    for idx in candidates:
        img = load_image(image_paths[idx])

        # hybrid
        h_score = hybrid_score(query_clip, clip_embeddings[idx], query_img, img)

        # ML
        s_sem = semantic_sim(query_clip, clip_embeddings[idx])
        s_str = structural_sim(query_img, img)

        ml_score = model.predict_proba([[s_sem, s_str]])[0][1]

        # final score
        score = 0.7 * h_score + 0.3 * ml_score

        results.append((idx, score))
        similarities.append(score)

    results = sorted(results, key=lambda x: x[1], reverse=True)

    # ---------------- TOP-K ----------------
    for k in topk_vals:
        if any(idx in gt for idx, _ in results[:k]):
            topk_correct[k] += 1

    # ---------------- Precision / Recall ----------------
    hit = False
    for idx, _ in results[:5]:
        if idx in gt:
            tp += 1
            hit = True
        else:
            fp += 1

    if hit:
        transform_stats[query_type]["correct"] += 1

    transform_stats[query_type]["total"] += 1

    fn += len(gt) - 1
    total += 1

    # ---------------- mAP ----------------
    hits = 0
    precision_sum = 0

    for rank, (idx, _) in enumerate(results[:10], start=1):
        if idx in gt:
            hits += 1
            precision_sum += hits / rank

    ap_list.append(precision_sum / hits if hits > 0 else 0)

    # ---------------- COCS ----------------
    total_cocs += cocs_score(similarities[:5])

# ---------------- FINAL RESULTS ----------------
transform_acc = {}
all_types = ["original", "crop", "compression", "noise", "text_overlay", "screenshot"]

for t in all_types:
    if t in transform_stats:
        transform_acc[t] = transform_stats[t]["correct"] / transform_stats[t]["total"]
    else:
        transform_acc[t] = 0

result = {
    "accuracy": topk_correct[5] / total,
    "top1": topk_correct[1] / total,
    "top3": topk_correct[3] / total,
    "top5": topk_correct[5] / total,
    "top10": topk_correct[10] / total,
    "precision": tp / (tp + fp),
    "recall": tp / (tp + fn),
    "mAP": np.mean(ap_list),
    "COCS": total_cocs / total,
    **transform_acc,
    "model": "pipeline"
}

# ---------------- SAVE CSV ----------------
df = pd.DataFrame([result])

os.makedirs("results", exist_ok=True)
df.to_csv("results/pipeline_results.csv", index=False)

print("\n✅ Pipeline results saved to results/pipeline_results.csv")
print(df)

# ---------------- SAMPLE OUTPUTS + CSV ----------------
import random
import pandas as pd

print("\n\n================ SAMPLE QUALITATIVE RESULTS ================\n")

sample_ids = random.sample(range(len(image_paths)), 10)

rows = []

for query_id in sample_ids:

    print("\n" + "="*60)
    query_name = image_paths[query_id]
    print("🔍 Query:", query_name)

    query_dino = dino_features[query_id].reshape(1, -1)
    query_clip = clip_embeddings[query_id]
    query_img = load_image(query_name)

    # Retrieval
    D, I = index.search(query_dino, 20)
    candidates = I[0][1:]

    results = []
    similarities = []

    for idx in candidates:
        img = load_image(image_paths[idx])

        h_score = hybrid_score(query_clip, clip_embeddings[idx], query_img, img)

        s_sem = semantic_sim(query_clip, clip_embeddings[idx])
        s_str = structural_sim(query_img, img)

        ml_score = model.predict_proba([[s_sem, s_str]])[0][1]

        score = 0.7 * h_score + 0.3 * ml_score

        results.append((idx, score))
        similarities.append(score)

    results = sorted(results, key=lambda x: x[1], reverse=True)[:5]

    cocs = cocs_score(similarities[:5])

    print("\n🏆 Top Matches:\n")

    match_names = []
    match_scores = []

    for rank, (idx, score) in enumerate(results):
        match_name = image_paths[idx]
        print(f"{rank+1}. {match_name} | Score: {score:.4f}")

        match_names.append(match_name)
        match_scores.append(score)

    decision = "CONFIRMED" if cocs > 0.7 else "LOW_CONFIDENCE"

    print("\n📊 Confidence (COCS):", round(cocs, 4))
    print("Result:", "✅ Ownership Match CONFIRMED" if decision=="CONFIRMED" else "⚠️ Low Confidence Match")

    # -------- SAVE ROW --------
    row = {
        "query": query_name,
        "top1": match_names[0],
        "top1_score": match_scores[0],
        "top2": match_names[1],
        "top2_score": match_scores[1],
        "top3": match_names[2],
        "top3_score": match_scores[2],
        "top4": match_names[3],
        "top4_score": match_scores[3],
        "top5": match_names[4],
        "top5_score": match_scores[4],
        "COCS": cocs,
        "decision": decision
    }

    rows.append(row)

print("\n===========================================================")

# ---------------- SAVE CSV ----------------
df = pd.DataFrame(rows)

os.makedirs("results", exist_ok=True)
df.to_csv("results/qualitative_results.csv", index=False)

print("\n✅ Qualitative results saved to results/qualitative_results.csv")