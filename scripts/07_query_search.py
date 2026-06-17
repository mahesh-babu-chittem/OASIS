import numpy as np
import faiss
import random

# Load data
embeddings = np.load("outputs/embeddings.npy")
image_paths = np.load("outputs/image_paths.npy", allow_pickle=True)

# Load FAISS index
index = faiss.read_index("outputs/faiss.index")

# Pick random query
query_id = random.randint(0, len(embeddings)-1)

query_embedding = embeddings[query_id].reshape(1, -1)

# Search
k = 6
D, I = index.search(query_embedding, k)

# remove query itself
I = I[:, 1:]
D = D[:, 1:]

print("🔍 Query Image:", image_paths[query_id])
print("\nTop Matches:\n")

for rank, idx in enumerate(I[0]):
    print(f"{rank+1}. {image_paths[idx]}  | Similarity: {D[0][rank]:.4f}")