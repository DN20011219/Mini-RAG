# 一个最简单的 RAG 系统

## 系统设计

- LLM 接入：优先使用 GitHub API（GitHub Models，读取 `gh auth login` 登录态 token）调用对话接口；不可用时回退 Copilot token
- Embedding：对 `data/doc` 下文本做向量化
- VectorDB：使用 `faiss`，索引与元数据保存在 `data/db_file/`

## 环境依赖

使用 conda 管理虚拟环境，可跳过（推荐）：
```bash
conda create -n micro_rag
conda activate micro_rag
```

安装 python 依赖：
```bash
pip install \
  numpy==1.26.4 \
  sentence-transformers==3.4.1 \
  faiss-cpu==1.10.0 \
  requests==2.32.3
```

安装 GitHub CLI 并登录（用于 Copilot token）：

```bash
sudo apt install gh
gh auth login
gh auth status --show-token -h github.com
```

国内网络可优先使用 Hugging Face 镜像下载嵌入模型：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

也可直接在命令中传入：`--hf-endpoint https://hf-mirror.com`。

若模型已提前下载到本地，可使用本地模式：`--local-files-only --model-name <本地模型目录>`。

## 数据准备

- 把所有文本数据放到 `data/doc`（支持 `.txt/.md/.markdown`）

## 运行指南

### 1) 建库

```bash
python rag_chat.py build
```

### 2) 提问

```bash
python rag_chat.py query "你的问题" --top-k 3
```

top-k 参数用于控制使用前多少条检索结果增强回答。

### 3) 裸大模型问答（无RAG、无检索）

```bash
python no_doc_chat.py "你的问题"
```

用于和 `python rag_chat.py query ...` 的 RAG 结果做直接对比。

### 4) 将全部文档发送给大模型问答（无RAG、无检索）

```bash
python full_doc_chat.py "你的问题"
```

该脚本用于演示“全量文档直传”的典型问题：当文档过长时，接口容易因上下文超限而报错。

脚本会打印：

- 请求规模（chars/bytes/estimated_tokens）
- GitHub Models/Copilot 的具体错误信息（若有）

### 总体示例：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python rag_chat.py build
python full_doc_chat.py "咖啡店营业时间是什么？"
python rag_chat.py query "咖啡店营业时间是什么？" --top-k 3
```

上面这组命令可直接作为教学演示：

1. 先运行 `full_doc_chat.py`，观察长文档直传时的失败信息。
2. 再运行 `rag_chat.py query`，观察“检索后再生成”的稳定回答。

额外说明：

- 如果当前机器可通过 GitHub 登录态拿到 token，则会优先调用 GitHub Models 生成最终答案
- 如果 GitHub Models 不可用，会尝试 Copilot token
- 如果拿不到 token，则自动退化为“检索结果摘要”模式（仍可验证 RAG 检索链路）

