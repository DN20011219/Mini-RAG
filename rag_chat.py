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
			"content": "你是一个RAG问答助手。仅基于给定上下文回答，若上下文不足请明确说明。",
		},
		{
			"role": "user",
			"content": f"问题：{question}\n\n检索上下文：\n{context_text}",
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
			"content": "你是一个RAG问答助手。仅基于给定上下文回答，若上下文不足请明确说明为什么。",
		},
		{
			"role": "user",
			"content": f"问题：{question}\n\n检索上下文：\n{context_text}",
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
	lines = [f"问题：{question}", "", "以下是最相关内容："]
	for idx, item in enumerate(contexts, start=1):
		preview = (item.get("content") or "<图片内容>")[:180]
		lines.append(f"{idx}. {item['source']} (score={item['score']:.4f})")
		lines.append(f"   {preview}")
	lines.append("")
	lines.append("未检测到可用 GitHub Models/Copilot token，已返回检索结果摘要。")
	return "\n".join(lines)


def print_retrieval_results(contexts: list[dict]) -> None:
	print("检索结果：")
	for idx, item in enumerate(contexts, start=1):
		preview = (item.get("content") or "<图片内容>")[:180]
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
	print(f"建库完成: chunks={total_chunks}, embedding_dim={dim}, db={Path(args.db_dir).resolve()}")


def cmd_query(args: argparse.Namespace) -> None:
	from embedding import Embedder
	from vectordb import VectorDB
	
	print("数据库与嵌入模型加载中...")
	vectordb = VectorDB(db_dir=args.db_dir)
	vectordb.load()
	embedder = Embedder(
		model_name=args.model_name,
		hf_endpoint=args.hf_endpoint,
		local_files_only=args.local_files_only,
	)
	print("加载完成...")
	
	print("嵌入query中...")
	query_vector = embedder.embed_query(args.question)

	print("向量数据库检索中...")
	candidate_k = max(args.top_k * 8, 20)
	candidates = vectordb.search(query_vector=query_vector, top_k=candidate_k)
	contexts = rerank_contexts(args.question, candidates, top_k=args.top_k)
	if not contexts:
		print("未检索到相关内容")
		return
	print_retrieval_results(contexts)

	print("增强回答中，请稍候...")
	answer = generate_with_github_models(args.question, contexts, model=args.model)
	if not answer:
		answer = generate_with_copilot(args.question, contexts, model="gpt-4o-mini")
	if not answer:
		answer = fallback_answer(args.question, contexts)

	print(answer)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="最简 RAG 系统")
	subparsers = parser.add_subparsers(dest="command", required=True)

	build_parser = subparsers.add_parser("build", help="构建向量库")
	build_parser.add_argument("--data-dir", default="data", help="数据目录（仅文本）")
	build_parser.add_argument("--db-dir", default="data/db_file", help="向量库存储目录")
	build_parser.add_argument(
		"--model-name",
		default="sentence-transformers/clip-ViT-B-32-multilingual-v1",
		help="Embedding 模型名称或本地模型路径",
	)
	build_parser.add_argument(
		"--hf-endpoint",
		default=os.getenv("HF_ENDPOINT"),
		help="Hugging Face 访问地址（如 https://hf-mirror.com）",
	)
	build_parser.add_argument(
		"--local-files-only",
		action="store_true",
		help="仅从本地缓存/路径加载模型，不访问网络",
	)
	build_parser.set_defaults(func=cmd_build)

	query_parser = subparsers.add_parser("query", help="检索并回答")
	query_parser.add_argument("question", help="用户问题")
	query_parser.add_argument("--db-dir", default="data/db_file", help="向量库存储目录")
	query_parser.add_argument("--top-k", type=int, default=3, help="召回条数")
	query_parser.add_argument("--model", default="openai/gpt-4.1-mini", help="GitHub Models 聊天模型")
	query_parser.add_argument(
		"--model-name",
		default="sentence-transformers/clip-ViT-B-32-multilingual-v1",
		help="Embedding 模型名称或本地模型路径",
	)
	query_parser.add_argument(
		"--hf-endpoint",
		default=os.getenv("HF_ENDPOINT"),
		help="Hugging Face 访问地址（如 https://hf-mirror.com）",
	)
	query_parser.add_argument(
		"--local-files-only",
		action="store_true",
		help="仅从本地缓存/路径加载模型，不访问网络",
	)
	query_parser.set_defaults(func=cmd_query)

	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
