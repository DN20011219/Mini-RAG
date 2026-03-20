from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


warnings.filterwarnings(
	"ignore",
	message=r"`resume_download` is deprecated and will be removed in version 1\.0\.0.*",
	category=FutureWarning,
	module=r"huggingface_hub\.file_download",
)


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}


@dataclass
class Chunk:
	chunk_id: str
	source: str
	modality: str
	content: str

	def to_dict(self) -> dict:
		return asdict(self)


class Embedder:
	def __init__(
		self,
		model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
		chunk_size: int = 500,
		chunk_overlap: int = 100,
		hf_endpoint: str | None = None,
		local_files_only: bool = False,
	):
		if hf_endpoint:
			os.environ["HF_ENDPOINT"] = hf_endpoint

		try:
			self.model = SentenceTransformer(model_name, local_files_only=local_files_only)
		except TypeError:
			self.model = SentenceTransformer(model_name)
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap

	def _chunk_text(self, text: str) -> list[str]:
		text = text.replace("\r\n", "\n").replace("\r", "\n")
		paragraphs = [" ".join(part.split()) for part in text.split("\n\n") if part.strip()]
		if not paragraphs:
			return []

		chunks: list[str] = []
		step = max(1, self.chunk_size - self.chunk_overlap)
		for paragraph in paragraphs:
			if len(paragraph) <= self.chunk_size:
				chunks.append(paragraph)
				continue

			start = 0
			while start < len(paragraph):
				end = start + self.chunk_size
				chunks.append(paragraph[start:end])
				start += step
		return chunks

	def _iter_text_files(self, data_dir: Path) -> Iterable[Path]:
		if not data_dir.exists():
			return []
		return (
			path
			for path in sorted(data_dir.rglob("*"))
			if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS
		)

	def load_chunks(self, data_dir: str | Path) -> list[Chunk]:
		data_dir = Path(data_dir)
		chunks: list[Chunk] = []

		for file_path in self._iter_text_files(data_dir):
			raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
			text_chunks = self._chunk_text(raw_text)
			for idx, chunk_text in enumerate(text_chunks):
				chunks.append(
					Chunk(
						chunk_id=f"text::{file_path.relative_to(data_dir)}::{idx}",
						source=str(file_path),
						modality="text",
						content=chunk_text,
					)
				)

		return chunks

	def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
		if not chunks:
			return np.empty((0, 0), dtype=np.float32)

		text_inputs = [chunk.content for chunk in chunks]
		text_embeddings = self.model.encode(text_inputs, convert_to_numpy=True, normalize_embeddings=True)
		return np.asarray(text_embeddings, dtype=np.float32)

	def embed_query(self, query: str) -> np.ndarray:
		vector = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
		return np.asarray(vector, dtype=np.float32)
