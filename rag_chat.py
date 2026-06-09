from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import requests


def get_copilot_token() -> str | None:
	try:
		result = subprocess.run(
			["gh", "api", "-H", "Accept: application/json", "/copilot_internal/v2/token"],
			capture_output=True,
			text=True,
			check=True,
		)
		payload = json.loads(result.stdout)
		return payload.get("token")
	except Exception:
		return None


def get_github_token_from_gh() -> str | None:
	env_token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
	if env_token:
		return env_token.strip()

	try:
		result = subprocess.run(
			["gh", "auth", "status", "--show-token", "-h", "github.com"],
			capture_output=True,
			text=True,
			check=True,
		)
		combined = "\n".join([result.stdout or "", result.stderr or ""])
		for line in combined.splitlines():
			if "Token:" in line:
				return line.split("Token:", 1)[1].strip()
		return None
	except Exception:
		return None


def generate_with_github_models(question: str, contexts: list[dict], model: str = "openai/gpt-4.1-mini") -> str | None:
	token = get_github_token_from_gh()
	if not token:
		return None

	context_text = "\n\n".join(
		[
			f"[{idx + 1}] source={item['source']} score={item['score']:.4f}\n{item.get('content', '')}"
			for idx, item in enumerate(contexts)
		]
	)

	messages = [
		{
			"role": "system",
			"content": "You are a RAG Q&A assistant. Answer only based on the given context, and clearly state if the context is insufficient.",
		},
		{
			"role": "user",
			"content": f"Question: {question}\n\nRetrieved Context:\n{context_text}",
		},
	]

	response = requests.post(
		"https://models.github.ai/inference/chat/completions",
		headers={
			"Authorization": f"Bearer {token}",
			"Content-Type": "application/json",
		},
		json={
			"model": model,
			"messages": messages,
			"temperature": 0.2,
		},
		timeout=30,
	)
	if response.status_code >= 400:
		return None

	data = response.json()
	choices = data.get("choices", [])
	if not choices:
		return None

	choice = choices[0]
	if isinstance(choice.get("message"), dict):
		return choice.get("message", {}).get("content")
	return choice.get("content")


def generate_with_copilot(question: str, contexts: list[dict], model: str = "gpt-4o-mini") -> str | None:
	token = get_copilot_token()
	if not token:
		return None

	context_text = "\n\n".join(
		[
			f"[{idx + 1}] source={item['source']} score={item['score']:.4f}\n{item.get('content', '')}"
			for idx, item in enumerate(contexts)
		]
	)

	messages = [
		{
			"role": "system",
			"content": "You are a RAG Q&A assistant. Answer only based on the given context, and clearly explain why if the context is insufficient.",
		},
		{
			"role": "user",
			"content": f"Question: {question}\n\nRetrieved Context:\n{context_text}",
		},
	]

	response = requests.post(
		"https://api.githubcopilot.com/chat/completions",
		headers={
			"Authorization": f"Bearer {token}",
			"Content-Type": "application/json",
		},
		json={
			"model": model,
			"messages": messages,
			"temperature": 0.2,
		},
		timeout=30,
	)
	if response.status_code >= 400:
		return None

	data = response.json()
	choices = data.get("choices", [])
	if not choices:
		return None
	return choices[0].get("message", {}).get("content")


def fallback_answer(question: str, contexts: list[dict]) -> str:
	lines = [f"Question: {question}", "", "Here is the most relevant content:"]
	for idx, item in enumerate(contexts, start=1):
		preview = (item.get("content") or "<image content>")[:180]
		lines.append(f"{idx}. {item['source']} (score={item['score']:.4f})")
		lines.append(f"   {preview}")
	lines.append("")
	lines.append("No available GitHub Models/Copilot token detected, returning retrieval result summary.")
	return "\n".join(lines)


def print_retrieval_results(contexts: list[dict]) -> None:
	print("Retrieval Results:")
	for idx, item in enumerate(contexts, start=1):
		preview = (item.get("content") or "<image content>")[:180]
		print(f"{idx}. {item['source']} (score={item['score']:.4f})")
		print(f"   {preview}")
	print("")


def _extract_query_ngrams(text: str) -> set[str]:
	normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
	grams: set[str] = set()
	for n in (2, 3, 4):
		if len(normalized) < n:
			continue
		for idx in range(len(normalized) - n + 1):
			grams.add(normalized[idx : idx + n])
	return grams


def rerank_contexts(question: str, candidates: list[dict], top_k: int) -> list[dict]:
	grams = _extract_query_ngrams(question)
	if not grams:
		return candidates[:top_k]

	reranked: list[dict] = []
	for item in candidates:
		content = str(item.get("content", ""))
		lexical_hits = sum(1 for gram in grams if gram in content)
		semantic_score = float(item.get("score", 0.0))
		mixed_score = semantic_score + lexical_hits * 0.03
		new_item = dict(item)
		new_item["semantic_score"] = semantic_score
		new_item["lexical_hits"] = lexical_hits
		new_item["score"] = mixed_score
		reranked.append(new_item)

	reranked.sort(key=lambda x: (x.get("lexical_hits", 0), x.get("score", 0.0)), reverse=True)
	return reranked[:top_k]


def cmd_build(args: argparse.Namespace) -> None:
	from vectordb import build_db

	total_chunks, dim = build_db(
		data_dir=args.data_dir,
		db_dir=args.db_dir,
		model_name=args.model_name,
		hf_endpoint=args.hf_endpoint,
		local_files_only=args.local_files_only,
	)
	print(f"Build completed: chunks={total_chunks}, embedding_dim={dim}, db={Path(args.db_dir).resolve()}")


def cmd_query(args: argparse.Namespace) -> None:
	from embedding import Embedder
	from vectordb import VectorDB
	
	print("Loading database and embedding model...")
	vectordb = VectorDB(db_dir=args.db_dir)
	vectordb.load()
	embedder = Embedder(
		model_name=args.model_name,
		hf_endpoint=args.hf_endpoint,
		local_files_only=args.local_files_only,
	)
	print("Loading completed...")
	
	print("Embedding query...")
	query_vector = embedder.embed_query(args.question)

	print("Retrieving from vector database...")
	candidate_k = max(args.top_k * 8, 20)
	candidates = vectordb.search(query_vector=query_vector, top_k=candidate_k)
	contexts = rerank_contexts(args.question, candidates, top_k=args.top_k)
	if not contexts:
		print("No relevant content retrieved")
		return
	print_retrieval_results(contexts)

	print("Generating enhanced answer, please wait...")
	answer = generate_with_github_models(args.question, contexts, model=args.model)
	if not answer:
		answer = generate_with_copilot(args.question, contexts, model="gpt-4o-mini")
	if not answer:
		answer = fallback_answer(args.question, contexts)

	print(answer)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Minimal RAG System")
	subparsers = parser.add_subparsers(dest="command", required=True)

	build_parser = subparsers.add_parser("build", help="Build vector database")
	build_parser.add_argument("--data-dir", default="data", help="Data directory (text only)")
	build_parser.add_argument("--db-dir", default="data/db_file", help="Vector database storage directory")
	build_parser.add_argument(
		"--model-name",
		default="sentence-transformers/clip-ViT-B-32-multilingual-v1",
		help="Embedding model name or local model path",
	)
	build_parser.add_argument(
		"--hf-endpoint",
		default=os.getenv("HF_ENDPOINT"),
		help="Hugging Face endpoint (e.g., https://hf-mirror.com)",
	)
	build_parser.add_argument(
		"--local-files-only",
		action="store_true",
		help="Only load model from local cache/path, no network access",
	)
	build_parser.set_defaults(func=cmd_build)

	query_parser = subparsers.add_parser("query", help="Retrieve and answer")
	query_parser.add_argument("question", help="User question")
	query_parser.add_argument("--db-dir", default="data/db_file", help="Vector database storage directory")
	query_parser.add_argument("--top-k", type=int, default=3, help="Number of results to retrieve")
	query_parser.add_argument("--model", default="openai/gpt-4.1-mini", help="GitHub Models chat model")
	query_parser.add_argument(
		"--model-name",
		default="sentence-transformers/clip-ViT-B-32-multilingual-v1",
		help="Embedding model name or local model path",
	)
	query_parser.add_argument(
		"--hf-endpoint",
		default=os.getenv("HF_ENDPOINT"),
		help="Hugging Face endpoint (e.g., https://hf-mirror.com)",
	)
	query_parser.add_argument(
		"--local-files-only",
		action="store_true",
		help="Only load model from local cache/path, no network access",
	)
	query_parser.set_defaults(func=cmd_query)

	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
