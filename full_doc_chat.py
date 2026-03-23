from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import requests


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}


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


def load_all_documents(doc_dir: str | Path) -> str:
	doc_dir = Path(doc_dir)
	if not doc_dir.exists():
		return ""

	parts: list[str] = []
	for file_path in sorted(doc_dir.rglob("*")):
		if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
			continue
		content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
		if not content:
			continue
		rel = file_path.relative_to(doc_dir)
		parts.append(f"### 文件: {rel}\n{content}")

	return "\n\n".join(parts)


def build_user_content(question: str, docs_content: str) -> str:
	if not docs_content:
		return f"问题：{question}\n\n文档内容为空，请直接基于常识回答并明确说明未提供文档。"

	return (
		f"请仅基于下面给出的完整文档内容回答问题。\n"
		f"如果文档里没有答案，请明确说“文档中未找到”。\n\n"
		f"问题：{question}\n\n"
		f"完整文档内容：\n{docs_content}"
	)


def estimate_prompt_size(user_content: str) -> tuple[int, int, int]:
	char_count = len(user_content)
	byte_count = len(user_content.encode("utf-8"))
	estimated_tokens = max(1, byte_count // 4)
	return char_count, byte_count, estimated_tokens


def _extract_error_message(response: requests.Response) -> str:
	try:
		payload: Any = response.json()
		error = payload.get("error") if isinstance(payload, dict) else None
		if isinstance(error, dict):
			message = error.get("message") or error.get("detail")
			if message:
				return str(message)
		if isinstance(error, str):
			return error
		return json.dumps(payload, ensure_ascii=False)[:500]
	except Exception:
		return (response.text or "").strip()[:500]


def chat_with_github_models(user_content: str, system_prompt: str, model: str) -> tuple[str | None, str | None]:
	token = get_github_token_from_gh()
	if not token:
		return None, "未检测到 GitHub token"

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_content},
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
		detail = _extract_error_message(response)
		return None, f"GitHub Models HTTP {response.status_code}: {detail}"

	try:
		data = response.json()
	except Exception:
		return None, "GitHub Models 返回了非 JSON 响应"
	choices = data.get("choices", [])
	if not choices:
		return None, "GitHub Models 响应中没有 choices"

	choice = choices[0]
	if isinstance(choice.get("message"), dict):
		return choice.get("message", {}).get("content"), None
	return choice.get("content"), None


def chat_with_copilot(user_content: str, system_prompt: str, model: str = "gpt-4o-mini") -> tuple[str | None, str | None]:
	token = get_copilot_token()
	if not token:
		return None, "未检测到 Copilot token"

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_content},
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
		detail = _extract_error_message(response)
		return None, f"Copilot HTTP {response.status_code}: {detail}"

	try:
		data = response.json()
	except Exception:
		return None, "Copilot 返回了非 JSON 响应"
	choices = data.get("choices", [])
	if not choices:
		return None, "Copilot 响应中没有 choices"
	return choices[0].get("message", {}).get("content"), None


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="裸大模型问答（无RAG、无检索）")
	parser.add_argument("question", help="用户问题")
	parser.add_argument("--model", default="openai/gpt-4.1-mini", help="GitHub Models 聊天模型")
	parser.add_argument(
		"--system",
		default="你是一个文档问答助手，请严格根据给定文档回答。",
		help="系统提示词",
	)
	parser.add_argument("--doc-dir", default="data/doc", help="文档目录（将全量拼接发送给模型）")
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	docs_content = load_all_documents(args.doc_dir)
	user_content = build_user_content(args.question, docs_content)
	char_count, byte_count, estimated_tokens = estimate_prompt_size(user_content)

	answer, gh_error = chat_with_github_models(user_content, system_prompt=args.system, model=args.model)
	copilot_error = None
	if not answer:
		answer, copilot_error = chat_with_copilot(user_content, system_prompt=args.system, model="gpt-4o-mini")

	if answer:
		print(answer)
		return

	print("全量文档直传模式调用失败。")
	print(f"请求规模：chars={char_count}, bytes={byte_count}, estimated_tokens≈{estimated_tokens}")
	if gh_error:
		print(f"- GitHub Models: {gh_error}")
	if copilot_error:
		print(f"- Copilot: {copilot_error}")


if __name__ == "__main__":
	main()