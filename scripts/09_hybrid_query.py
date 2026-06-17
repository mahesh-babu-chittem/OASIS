import numpy as np
import os
import faiss
from PIL import Image

from hybrid.hybrid_model import hybrid_score

# load
embeddings = np.load("outputs/embeddings.npy").astype('float32')
image_paths = np.load("outputs/image_paths.npy", allow_pickle=True)

index = faiss.read_index("outputs/faiss.index")

def load_image(path):
    return np.array(Image.open(path).convert("RGB"))

query_id = 0
query_emb = embeddings[query_id]
query_img = load_image(os.path.join("data/processed", image_paths[query_id]))

D, I = index.search(query_emb.reshape(1, -1), 10)

results = []

for idx in I[0][1:]:
    emb = embeddings[idx]
    img = load_image(os.path.join("data/processed", image_paths[idx]))

    score = hybrid_score(query_emb, emb, query_img, img)
    results.append((image_paths[idx], score))

results = sorted(results, key=lambda x: x[1], reverse=True)

print("\n🔬 Hybrid Results:\n")
for r in results[:5]:
    print(r)