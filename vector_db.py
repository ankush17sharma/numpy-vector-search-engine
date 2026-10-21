import numpy as np
import json
import pickle 
from typing import List, Dict, Any, Tuple

class VectorEngine:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: np.ndarray = np.empty((0, dimension), dtype = np.float32)
        self.doc_ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """L2 Normalization: Scales vector so its magnitude/length equals 1.0"""
        norms = np.linalg.norm(v, axis = 1, keepdims = True)
        #Avoid division by zero 
        norms = np.where(norms == 0, 1.0, norms)
        return v / norms

    def add(self, doc_id: str, vector: List[float], meta: Dict[str, Any] = None):
        """Adds a single vector and its metadata into the database."""
        vec_np = np.array(vector, dtype = np.float32)
        if vec_np.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {vec_np.shape[0]}")

        #L2 Normalize before storing 
        normalized_vec = self._normalize(vec_np)

        #Append to matrix memory
        self.vectors = np.vstack([self.vectors, normalized_vec])
        self.doc_ids.append(doc_id)
        self.metadata.append(meta or {})

    def AddBatch(self, doc_ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]] = None):
        """Batch inserts vectors using matrix operations for speed."""
        vecs_np = np.array(vectors, dtype = np.float32)
        if vecs_np.shape[1] != self.dimesion:
            raise ValueError(f"Vector Dimension Mismatch")

        normalized_vecs = self._normalize(vecs_np)

        if self.vectors.size == 0:
            self.vectors = normalized_vecs
        else:
            self.vectors = np.vstack([self.vectors, normalized_vecs])

        self.doc_ids.extend(doc_ids)
        if self.metadata:
            self.metadata.extend(self.metadatas)
        else:
            self.metadata.extend([{} for _ in doc_ids])

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarity of the query vector against ALL stored vectors
        in a single matrix-vector dot product operation: (N, D) @ (D, 1) -> (N, 1).
        """
        if self.vectors.shape[0] == 0:
            return []

        q_np = np.array(query_vector, dtype = np.float32)
        q_norm = self._normalize(q_np)

        #Matrix dot product calculates cosine similarity across ALL vectora simultaneously
        similarities = np.dot(self.vectors, q_norm)

        #Get indices of the top-k highest similarity scores
        # argsort sorts ascending, so slice from the end [-top_k:] and reverse [::1]
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_k_indices:
            results.append({
                "id": self.doc_ids[idx],
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx]
            })

        return results

    def save(self, filepath: str):
        """Persists index to disk"""
        data = {
            "dimension": self.dimension,
            "vectors": self.vectors,
            "doc_ids": self.doc_ids,
            "metadata": self.metadata
        }
        with open (filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[VectorEngine] Database saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Loads index state from disk"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        engine = cls(dimension = data["dimension"])
        engine.vectors = data["vectors"]
        engine.doc_ids = data['doc_ids']
        engine.metadata = data["metadata"]
        print(f"[VectorEngine] Loaded {len(engine.doc_ids)} vectors from {filepath}")
        return engine

# -------------------------------------------------------------
# DEMO / TEST DRIVE
# -------------------------------------------------------------
if __name__ == "__main__":
    #Create engine for 4-dimensional vectors
    db= VectorEngine(dimension = 4)

    #Dummy vector representations (e.g., embeddings for sports, tech, food)
    #[sports, tech, food, science]
    db.AddBatch(
        doc_ids = ["doc_1", "doc_2", "doc_3", "doc_4"],
        vectors = [
            [0.9, 0.1, 0.0, 0.2],   # Football / Basketball article
            [0.8, 0.2, 0.1, 0.1],   # Tennis tournament
            [0.1, 0.9, 0.2, 0.8],   # Python Programming & AI
            [0.0, 0.2, 0.9, 0.1],   # Italian Pasta Recipe
        ],
        metadatas = [
            {"category": "sports", "title": "Champions League Finals"},
            {"category": "sports", "title": "Wimbledon Highlights"},
            {"category": "technology", "title": "Building Neural Networks"},
            {"category": "food", "title": "Best Spaghetti Recipe"} 
        ]
    )

    #Query: Looking for tech/coding articles [0.1, 0.8, 0.1, 0.7]
    query = [0.1, 0.8, 0.1, 0.7]
    print('--- Searching for nearest vectors ---')
    results = db.search(query_vector= query, top_k = 2)

    for r in results:
        print(f"ID: {r['id']} | Score: {r['score']:.4f} | Title: {r['metadata']['title']}")