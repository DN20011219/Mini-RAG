from __future__ import annotations

import json
from pathlib import Path

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
	):
		self.db_dir = Path(db_dir)
		self.db_dir.mkdir(parents=True, exist_ok=True)

		self.index_path = self.db_dir / index_name
		self.metadata_path = self.db_dir / metadata_name

		self.index: faiss.Index | None = None
		self.metadata: list[dict] = []

	def build(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
		if embeddings.size == 0 or not chunks:
			raise ValueError("没有可建立索引的数据，请检查 data/ 是否有文本文件")

		dim = embeddings.shape[1]
		index = faiss.IndexFlatIP(dim)
		index.add(embeddings)

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
) -> tuple[int, int]:
	embedder = Embedder(
		model_name=model_name,
		hf_endpoint=hf_endpoint,
		local_files_only=local_files_only,
	)
	chunks = embedder.load_chunks(data_dir=data_dir)
	embeddings = embedder.embed_chunks(chunks)

	vectordb = VectorDB(db_dir=db_dir)
	vectordb.build(embeddings=embeddings, chunks=chunks)
	vectordb.save()
	return len(chunks), embeddings.shape[1]
