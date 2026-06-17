import numpy as np
import faiss
from collections import defaultdict
from tqdm import tqdm

# ---------------- LOAD ----------------
embeddings = np.load("outputs/embeddings.npy").astype('float32')
image_paths = np.load("outputs/image_paths.npy", allow_pickle=True)

index = faiss.read_index("outputs/faiss.index")

# ---------------- HELPERS ----------------
def get_id(path):
    return path.split("/")[-1]

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
    query_emb = embeddings[i].reshape(1, -1)

    D, I = index.search(query_emb, 10)

    results = []
    similarities_list = []

    for idx, score in zip(I[0][1:], D[0][1:]):  # skip self
        results.append((idx, score))
        similarities_list.append(score)

    results = sorted(results, key=lambda x: x[1], reverse=True)[:k]

    # COCS
    cocs = cocs_score(similarities_list[:k])
    total_cocs += cocs

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

accuracy = correct / total
precision = tp / (tp + fp)
recall = tp / (tp + fn)
avg_cocs = total_cocs / total

print("\n📊 CLIP Results")
print("---------------------")
print(f"Top-{k} Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"COCS Score: {avg_cocs:.4f}")