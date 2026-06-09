# A Minimal RAG System

## 1 System Design

- LLM Integration: Preferentially use GitHub API (GitHub Models, reading token from `gh auth login` login state) to call chat interface; fallback to Copilot token when unavailable
- Embedding: Vectorize text files under `data/doc`
- VectorDB: Use `faiss`, index and metadata saved in `data/db_file/`, index defaults to IVF-PQ, distance metric uses METRIC_INNER_PRODUCT

## 2 Environment Dependencies

Use conda to manage virtual environment, can be skipped (recommended):
```bash
conda create -n micro_rag
conda activate micro_rag
```

Install Python dependencies:
```bash
pip install \
  numpy==1.26.4 \
  sentence-transformers==3.4.1 \
  faiss-cpu==1.10.0 \
  requests==2.32.3
```

Install GitHub CLI and login (for Copilot token):

```bash
sudo apt install gh
gh auth login
gh auth status --show-token -h github.com
```

For users in China, prefer using Hugging Face mirror to download embedding models:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Or pass directly in command: `--hf-endpoint https://hf-mirror.com`.

If model is already downloaded locally, use local mode: `--local-files-only --model-name <local model directory>`.

## 3 Data Preparation

- Place all text data in `data/doc` (supports `.txt/.md/.markdown`)

## 4 Running Guide

### 4.1 Build Database

```bash
python rag_chat.py build
```

### 4.2 Query

```bash
python rag_chat.py query "Your question here" --top-k 3
```

The top-k parameter controls how many retrieved results are used to enhance the answer.

### 4.3 Raw LLM Q&A (No RAG, No Retrieval)

```bash
python no_doc_chat.py "Your question here"
```

Used for direct comparison with RAG results from `python rag_chat.py query ...`.

### 4.4 Send All Documents to LLM Q&A (No RAG, No Retrieval)

```bash
python full_doc_chat.py "Your question here"
```

This script demonstrates the typical problem of "full document direct transfer": when documents are too long, the interface may error due to context length limits.

The script prints:

- Request scale (chars/bytes/estimated_tokens)
- Specific error messages from GitHub Models/Copilot (if any)

### 4.5 Overall Example

```bash
export HF_ENDPOINT=https://hf-mirror.com
python rag_chat.py build
python full_doc_chat.py "What are the coffee shop business hours?"
python rag_chat.py query "What are the coffee shop business hours?" --top-k 3
```

This set of commands can be directly used as a teaching demonstration:

1. First run `full_doc_chat.py`, observe the failure information when passing long documents directly.
2. Then run `rag_chat.py query`, observe the stable answer of "retrieve then generate".

Additional notes:

- If the current machine can obtain token via GitHub login state, it will preferentially call GitHub Models to generate the final answer
- If GitHub Models is unavailable, it will try Copilot token
- If no token is available, it automatically degrades to "retrieval result summary" mode (still can verify RAG retrieval pipeline)


## 5 Other Features

### 5.1 Quantization Algorithms

The system supports IVF-Flat and IVF-PQ indexes, where IVF-PQ uses PQ compressed vectors. To facilitate comparison of differences between these two indexes, the system provides a comparison script:

```bash
python compare_index.py
```

This script will build IVF-Flat and IVF-PQ indexes based on documents in doc and the chunk embedding module, and output the disk space of these two indexes.

The current version simultaneously compares two metrics on the same query set:

- Database layer recall: Using top-k from `IndexFlatIP` exact retrieval as baseline, calculate `avg_recall_at_k` for `IVF-Flat` and `IVF-PQ`
- Index storage space: Output index file sizes (bytes) for both indexes, compression ratio (`ivfpq_ratio`) and saved space (`saved_bytes`)

Note: `pq_m` as `null` means using default strategy (automatically set to `dim // 8` based on vector dimension, with divisibility correction). `pq_nbits` defaults to 8 bits storage per subspace.

Under default experimental configuration, test results are:
```json
{
  "query_count": 7,
  "top_k": 3,
  "nlist": 50,
  "nprobe": 30,
  "pq_m": null,
  "pq_nbits": 8,
  "storage": {
    "ivfflat_index_bytes": 1357099,
    "ivfpq_index_bytes": 671188,
    "ivfpq_ratio": 0.494576,
    "saved_bytes": 685911
  },
  "summary": {
    "ivfflat": {
      "avg_recall_at_k": 1.0
    },
    "ivfpq": {
      "avg_recall_at_k": 0.8571
    }
  }
}
```

This means:

- Evaluated 7 query problems in total (`query_count=7`), comparing top-3 results each time (`top_k=3`).
- Inverted index parameters are `nlist=50`, `nprobe=30`; `pq_m=null` means automatically setting subspace count based on `dim // 8`, `pq_nbits=8` means 8-bit encoding per subspace.
- `ivfflat_index_bytes=1357099` and `ivfpq_index_bytes=671188` represent index file sizes for both types; `ivfpq_ratio=0.494576` means IVF-PQ size is approximately 49.46% of IVF-Flat.
- `saved_bytes=685911` means IVF-PQ saves approximately 686 KB storage space compared to IVF-Flat.
- `avg_recall_at_k` represents database layer average recall: IVF-Flat is `1.0` (consistent with exact retrieval top-k), IVF-PQ is `0.8571` (can hit approximately 85.71% of exact retrieval top-k on average).

This shows that under current data scale and parameters, IVF-PQ significantly reduces index volume, but brings certain recall loss; can further trade off by increasing `nprobe`, reducing compression intensity (e.g., decreasing `pq_nbits` or adjusting `pq_m`).