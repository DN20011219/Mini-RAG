from __future__ import annotations

import argparse
import json
import os
import subprocess

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


def chat_with_github_models(question: str, system_prompt: str, model: str) -> str | None:
	token = get_github_token_from_gh()
	if not token:
		return None

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": question},
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


def chat_with_copilot(question: str, system_prompt: str, model: str = "gpt-4o-mini") -> str | None:
	token = get_copilot_token()
	if not token:
		return None

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": question},
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


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="裸大模型问答（无RAG、无检索）")
	parser.add_argument("question", help="用户问题")
	parser.add_argument("--model", default="openai/gpt-4.1-mini", help="GitHub Models 聊天模型")
	parser.add_argument(
		"--system",
		default="你是一个有帮助的助手，请直接回答用户问题。",
		help="系统提示词",
	)
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	answer = chat_with_github_models(args.question, system_prompt=args.system, model=args.model)
	if not answer:
		answer = chat_with_copilot(args.question, system_prompt=args.system, model="gpt-4o-mini")

	if answer:
		print(answer)
		return

	print("未检测到可用 GitHub Models/Copilot token，无法进行裸大模型问答。")


if __name__ == "__main__":
	main()