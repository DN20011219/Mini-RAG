from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

try:
	import faiss
except Exception as exc:
	raise ImportError(
		"无法导入 faiss（由 faiss-cpu 提供）。请在 micro_rag 环境安装 faiss-cpu==1.10.0，"
		"并使用 `conda run -n micro_rag python chat.py ...` 运行。"
	) from exc

from embedding import Chunk, Embedder


class VectorDB:
	def __init__(
		self,
		db_dir: str | Path = "data/db_file",
		index_name: str = "index.faiss",
		metadata_name: str = "metadata.json",
		index_type: Literal["ivfflat", "ivfpq"] = "ivfflat",
		nlist: int = 50,
		nprobe: int = 30,
		pq_m: int | None = None,
		pq_nbits: int = 8,
	):
		self.db_dir = Path(db_dir)
		self.db_dir.mkdir(parents=True, exist_ok=True)

		self.index_path = self.db_dir / index_name
		self.metadata_path = self.db_dir / metadata_name
		if index_type not in {"ivfflat", "ivfpq"}:
			raise ValueError("index_type 仅支持 'ivfflat' 或 'ivfpq'")
		self.index_type = index_type
		self.nlist = nlist
		self.nprobe = nprobe
		self.pq_m = pq_m
		self.pq_nbits = pq_nbits

		self.index: faiss.Index | None = None
		self.metadata: list[dict] = []

	@staticmethod
	def _pick_pq_m(dim: int, preferred_m: int) -> int:
		candidate = max(1, min(preferred_m, dim))
		while candidate > 1 and dim % candidate != 0:
			candidate -= 1
		return candidate

	def build(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
		if embeddings.size == 0 or not chunks:
			raise ValueError("没有可建立索引的数据，请检查 data/ 是否有文本文件")

		embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
		dim = embeddings.shape[1]
		num_vectors = embeddings.shape[0]

		quantizer = faiss.IndexFlatIP(dim)
		nlist = max(1, min(self.nlist, num_vectors))

		if self.index_type == "ivfflat":
			index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
		else:
			preferred_pq_m = self.pq_m if self.pq_m is not None else max(1, dim // 8)
			pq_m = self._pick_pq_m(dim, preferred_pq_m)
			index = faiss.IndexIVFPQ(
				quantizer,
				dim,
				nlist,
				pq_m,
				self.pq_nbits,
				faiss.METRIC_INNER_PRODUCT,
			)

		index.train(embeddings)
		index.add(embeddings)
		index.nprobe = max(1, min(self.nprobe, nlist))

		self.index = index
		self.metadata = [chunk.to_dict() for chunk in chunks]

	def save(self) -> None:
		if self.index is None:
			raise RuntimeError("索引不存在，请先 build")

		faiss.write_index(self.index, str(self.index_path))
		self.metadata_path.write_text(
			json.dumps(self.metadata, ensure_ascii=False, indent=2),
			encoding="utf-8",
		)

	def load(self) -> None:
		if not self.index_path.exists() or not self.metadata_path.exists():
			raise FileNotFoundError("未找到索引文件，请先执行 build")

		self.index = faiss.read_index(str(self.index_path))
		self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

	def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[dict]:
		if self.index is None:
			raise RuntimeError("索引未加载")

		if hasattr(self.index, "nprobe"):
			self.index.nprobe = max(1, min(self.nprobe, self.index.nlist))

		top_k = min(top_k, len(self.metadata))
		scores, indices = self.index.search(query_vector, top_k)

		results: list[dict] = []
		for score, idx in zip(scores[0], indices[0]):
			if idx < 0:
				continue
			item = dict(self.metadata[idx])
			item["score"] = float(score)
			results.append(item)
		return results


def build_db(
	data_dir: str | Path = "data",
	db_dir: str | Path = "data/db_file",
	model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
	hf_endpoint: str | None = None,
	local_files_only: bool = False,
	index_type: Literal["ivfflat", "ivfpq"] = "ivfpq",
	nlist: int = 50,
	nprobe: int = 30,
	pq_m: int | None = None,
	pq_nbits: int = 8,
) -> tuple[int, int]:
	embedder = Embedder(
		model_name=model_name,
		hf_endpoint=hf_endpoint,
		local_files_only=local_files_only,
	)
	chunks = embedder.load_chunks(data_dir=data_dir)
	embeddings = embedder.embed_chunks(chunks)

	vectordb = VectorDB(
		db_dir=db_dir,
		index_type=index_type,
		nlist=nlist,
		nprobe=nprobe,
		pq_m=pq_m,
		pq_nbits=pq_nbits,
	)
	vectordb.build(embeddings=embeddings, chunks=chunks)
	vectordb.save()
	return len(chunks), embeddings.shape[1]


def compare_ivf_index_sizes(
	data_dir: str | Path = "data",
	db_root_dir: str | Path = "data",
	model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
	hf_endpoint: str | None = None,
	local_files_only: bool = False,
	nlist: int = 50,
	nprobe: int = 30,
	pq_m: int | None = None,
	pq_nbits: int = 8,
) -> dict[str, float | int]:
	embedder = Embedder(
		model_name=model_name,
		hf_endpoint=hf_endpoint,
		local_files_only=local_files_only,
	)
	chunks = embedder.load_chunks(data_dir=data_dir)
	embeddings = embedder.embed_chunks(chunks)

	db_root = Path(db_root_dir)
	flat_db = VectorDB(
		db_dir=db_root / "db_file_ivfflat",
		index_type="ivfflat",
		nlist=nlist,
		nprobe=nprobe,
	)
	pq_db = VectorDB(
		db_dir=db_root / "db_file_ivfpq",
		index_type="ivfpq",
		nlist=nlist,
		nprobe=nprobe,
		pq_m=pq_m,
		pq_nbits=pq_nbits,
	)

	flat_db.build(embeddings=embeddings, chunks=chunks)
	flat_db.save()
	pq_db.build(embeddings=embeddings, chunks=chunks)
	pq_db.save()

	flat_bytes = flat_db.index_path.stat().st_size
	pq_bytes = pq_db.index_path.stat().st_size

	return {
		"num_vectors": int(embeddings.shape[0]),
		"dimension": int(embeddings.shape[1]),
		"ivfflat_bytes": int(flat_bytes),
		"ivfpq_bytes": int(pq_bytes),
		"ivfpq_ratio": float(pq_bytes / flat_bytes if flat_bytes else 0.0),
		"saved_bytes": int(flat_bytes - pq_bytes),
	}
