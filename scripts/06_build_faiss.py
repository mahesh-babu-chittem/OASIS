import numpy as np
import faiss
import os

# Load embeddings
embeddings = np.load("outputs/embeddings.npy")

print("Embeddings shape:", embeddings.shape)

# Normalize again (safety)
embeddings = embeddings.astype('float32')

# Dimension
d = embeddings.shape[1]

# Create FAISS index (cosine similarity)
index = faiss.IndexFlatIP(d)

# Add embeddings
index.add(embeddings)

print("Total vectors indexed:", index.ntotal)

# Save index
os.makedirs("outputs", exist_ok=True)
faiss.write_index(index, "outputs/faiss.index")

print("✅ FAISS index saved")