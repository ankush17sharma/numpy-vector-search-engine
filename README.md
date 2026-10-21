# NumPy Vector Search Engine 

A lightweight vector search engine built with Python and NumPy. It stores numerical vectors, calculates cosine similarity, and retrieves the most similar results using Top-K search.

## ✨ Features

- Vector storage and retrieval
- L2 vector normalization
- Cosine similarity search
- Top-K nearest-vector retrieval
- Batch vector insertion
- Document metadata support
- Save and load index using Pickle

## Tech Stack

- Python
- NumPy
- Pickle

## Installation

Clone the repository:

```bash
git clone https://github.com/ankush17sharma/numpy-vector-search-engine.git
cd numpy-vector-search-engine
```

Install the dependency:

```bash
pip install numpy
```

## Usage

```python
from vector_engine import VectorEngine

# Initialize the engine
db = VectorEngine(dimension=4)

# Add a document vector
db.add(
    doc_id="doc_1",
    vector=[0.9, 0.1, 0.0, 0.2],
    meta={
        "category": "sports",
        "title": "Champions League Finals"
    }
)

# Search for similar vectors
query = [0.1, 0.8, 0.1, 0.7]
results = db.search(query_vector=query, top_k=1)

for result in results:
    print(result)
```

*Note: This example assumes the implementation's vector normalization and search methods have been corrected and tested.*

## How It Works

1. Store document vectors in a NumPy matrix.
2. Normalize vectors using L2 normalization.
3. Compare a query against stored vectors using a dot product.
4. Rank results by similarity score.
5. Return the Top-K matching documents with their metadata.

## Use Cases

- Semantic search prototypes
- AI document retrieval
- Recommendation systems
- Learning vector databases
- Experiments with Retrieval-Augmented Generation (RAG)

## Limitations

- Uses brute-force search across stored vectors.
- Stores the index in memory.
- Does not generate embeddings from text.
- Requires additional validation for production use.

## Future Improvements

- [ ] Fix and test input validation
- [ ] Add document deletion and updates
- [ ] Add metadata filtering
- [ ] Integrate a text embedding model
- [ ] Add unit tests and benchmarks
- [ ] Explore approximate nearest-neighbor search


**Built with Python 🐍 and NumPy 🔢**