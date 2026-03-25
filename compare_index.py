from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from embedding import Embedder
from vectordb import VectorDB

try:
    import faiss
except Exception as exc:
    raise ImportError(
        "无法导入 faiss（由 faiss-cpu 提供）。请先安装 faiss-cpu==1.10.0。"
    ) from exc


def _extract_queries_from_markdown(file_path: Path) -> list[str]:
    content = file_path.read_text(encoding="utf-8")
    pattern = re.compile(r'rag_chat\.py\s+query\s+"([^"]+)"')
    queries = [match.group(1).strip() for match in pattern.finditer(content)]
    seen: set[str] = set()
    ordered: list[str] = []
    for query in queries:
        if query not in seen:
            seen.add(query)
            ordered.append(query)
    return ordered


def _embed_query(embedder: Embedder, query: str) -> np.ndarray:
    vector = embedder.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)


def _recall_at_k(gold_contexts: list[dict], pred_contexts: list[dict]) -> float:
    gold_ids = {item["chunk_id"] for item in gold_contexts}
    if not gold_ids:
        return 0.0
    pred_ids = {item["chunk_id"] for item in pred_contexts}
    hit = len(gold_ids & pred_ids)
    return hit / len(gold_ids)


def _collect_contexts_from_search(
    metadata: list[dict],
    scores: np.ndarray,
    indices: np.ndarray,
) -> list[dict]:
    contexts: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = dict(metadata[idx])
        item["score"] = float(score)
        contexts.append(item)
    return contexts


def compare_recall_only(
    data_dir: str | Path = "data",
    question_file: str | Path = "提问题库.md",
    db_root_dir: str | Path = "data",
    model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
    hf_endpoint: str | None = None,
    local_files_only: bool = False,
    top_k: int = 3,
    nlist: int = 50,
    nprobe: int = 30,
    pq_m: int | None = None,
    pq_nbits: int = 8,
) -> dict:
    data_dir = Path(data_dir)
    question_file = Path(question_file)
    db_root_dir = Path(db_root_dir)

    queries = _extract_queries_from_markdown(question_file)
    if not queries:
        raise ValueError(f"未在 {question_file} 里解析到 rag_chat.py query 问题")

    embedder = Embedder(
        model_name=model_name,
        hf_endpoint=hf_endpoint,
        local_files_only=local_files_only,
    )
    chunks = embedder.load_chunks(data_dir=data_dir)
    embeddings = embedder.embed_chunks(chunks)

    metadata = [chunk.to_dict() for chunk in chunks]
    dim = embeddings.shape[1]
    exact_index = faiss.IndexFlatIP(dim)
    exact_index.add(embeddings)

    ivfflat_db = VectorDB(
        db_dir=db_root_dir / "db_file_ivfflat",
        index_type="ivfflat",
        nlist=nlist,
        nprobe=nprobe,
    )
    ivfflat_db.build(embeddings=embeddings, chunks=chunks)
    ivfflat_db.save()

    ivfpq_db = VectorDB(
        db_dir=db_root_dir / "db_file_ivfpq",
        index_type="ivfpq",
        nlist=nlist,
        nprobe=nprobe,
        pq_m=pq_m,
        pq_nbits=pq_nbits,
    )
    ivfpq_db.build(embeddings=embeddings, chunks=chunks)
    ivfpq_db.save()

    ivfflat_bytes = ivfflat_db.index_path.stat().st_size
    ivfpq_bytes = ivfpq_db.index_path.stat().st_size

    db_map = {
        "ivfflat": ivfflat_db,
        "ivfpq": ivfpq_db,
    }

    per_query: list[dict] = []
    avg_recall = {"ivfflat": [], "ivfpq": []}

    for query in queries:
        query_vector = _embed_query(embedder, query)
        exact_scores, exact_indices = exact_index.search(query_vector, top_k)
        exact_contexts = _collect_contexts_from_search(metadata, exact_scores, exact_indices)

        query_result = {
            "query": query,
            "exact_topk_chunk_ids": [item["chunk_id"] for item in exact_contexts],
            "index_results": {},
        }

        for index_name, db in db_map.items():
            contexts = db.search(query_vector=query_vector, top_k=top_k)
            recall_k = _recall_at_k(exact_contexts, contexts)

            avg_recall[index_name].append(recall_k)

            query_result["index_results"][index_name] = {
                "recall_at_k": round(recall_k, 4),
                "topk_chunk_ids": [item["chunk_id"] for item in contexts],
            }

        per_query.append(query_result)

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "query_count": len(queries),
        "top_k": top_k,
        "nlist": nlist,
        "nprobe": nprobe,
        "pq_m": pq_m,
        "pq_nbits": pq_nbits,
        "storage": {
            "ivfflat_index_bytes": int(ivfflat_bytes),
            "ivfpq_index_bytes": int(ivfpq_bytes),
            "ivfpq_ratio": round(float(ivfpq_bytes / ivfflat_bytes), 6) if ivfflat_bytes else 0.0,
            "saved_bytes": int(ivfflat_bytes - ivfpq_bytes),
        },
        "summary": {
            "ivfflat": {
                "avg_recall_at_k": round(_mean(avg_recall["ivfflat"]), 4),
            },
            "ivfpq": {
                "avg_recall_at_k": round(_mean(avg_recall["ivfpq"]), 4),
            },
        },
        "per_query": per_query,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="同一查询集下比较 IVFFlat vs IVFPQ 的数据库召回")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--question-file", default="提问题库.md")
    parser.add_argument("--db-root-dir", default="data")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        help="Embedding 模型名称或本地模型路径",
    )
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--nlist", type=int, default=50)
    parser.add_argument("--nprobe", type=int, default=30)
    parser.add_argument("--pq-m", type=int, default=None)
    parser.add_argument("--pq-nbits", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = compare_recall_only(
        data_dir=args.data_dir,
        question_file=args.question_file,
        db_root_dir=args.db_root_dir,
        model_name=args.model_name,
        hf_endpoint=args.hf_endpoint,
        local_files_only=args.local_files_only,
        top_k=args.top_k,
        nlist=args.nlist,
        nprobe=args.nprobe,
        pq_m=args.pq_m,
        pq_nbits=args.pq_nbits,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()