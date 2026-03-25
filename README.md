# 一个最简单的 RAG 系统

## 1 系统设计

- LLM 接入：优先使用 GitHub API（GitHub Models，读取 `gh auth login` 登录态 token）调用对话接口；不可用时回退 Copilot token
- Embedding：对 `data/doc` 下文本做向量化
- VectorDB：使用 `faiss`，索引与元数据保存在 `data/db_file/`，索引默认使用 IVF-PQ，距离度量采用 METRIC_INNER_PRODUCT

## 2 环境依赖

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

## 3 数据准备

- 把所有文本数据放到 `data/doc`（支持 `.txt/.md/.markdown`）

## 4 运行指南

### 4.1 建库

```bash
python rag_chat.py build
```

### 4.2 提问

```bash
python rag_chat.py query "你的问题" --top-k 3
```

top-k 参数用于控制使用前多少条检索结果增强回答。

### 4.3 裸大模型问答（无RAG、无检索）

```bash
python no_doc_chat.py "你的问题"
```

用于和 `python rag_chat.py query ...` 的 RAG 结果做直接对比。

### 4.4 将全部文档发送给大模型问答（无RAG、无检索）

```bash
python full_doc_chat.py "你的问题"
```

该脚本用于演示“全量文档直传”的典型问题：当文档过长时，接口容易因上下文超限而报错。

脚本会打印：

- 请求规模（chars/bytes/estimated_tokens）
- GitHub Models/Copilot 的具体错误信息（若有）

### 4.5 总体示例

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


## 5 其它功能

### 5.1 量化算法

系统支持了 IVF-Flat 与 IVF-PQ 两种索引，其中 IVF-PQ 使用了 PQ 压缩向量。为了便于对比两类索引的差异，系统提供了对比脚本：

```bash
python compare_index.py
```

该脚本将根据 doc 内部文档和切分嵌入模块，构建 IVF-Flat 和 IVF-PQ 两类索引，并输出这两类索引的磁盘空间。

当前版本会在同一查询集上同时对比两项指标：

- 数据库层召回：以 `IndexFlatIP` 精确检索的 top-k 作为基准，计算 `IVF-Flat` 和 `IVF-PQ` 的 `avg_recall_at_k`
- 索引存储空间：输出两类索引文件大小（bytes）、压缩比例（`ivfpq_ratio`）与节省空间（`saved_bytes`）

说明：`pq_m` 为 `null` 表示使用默认策略（按向量维度自动设置为 `dim // 8`，再做可整除修正）。`pq_nbits` 默认每个子空间使用 8 bit 存储。

如默认实验配置下，测试结果为：
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

其含义为：

- 一共评测了 7 个查询问题（`query_count=7`），每次比较 top-3 结果（`top_k=3`）。
- 倒排参数为 `nlist=50`、`nprobe=30`；`pq_m=null` 表示自动按 `dim // 8` 设定子空间数，`pq_nbits=8` 表示每个子空间 8 bit 编码。
- `ivfflat_index_bytes=1357099` 与 `ivfpq_index_bytes=671188` 表示两类索引文件大小；`ivfpq_ratio=0.494576` 表示 IVF-PQ 大小约为 IVF-Flat 的 49.46%。
- `saved_bytes=685911` 表示 IVF-PQ 相比 IVF-Flat 节省约 686 KB 存储空间。
- `avg_recall_at_k` 表示数据库层平均召回：IVF-Flat 为 `1.0`（与精确检索 top-k 一致），IVF-PQ 为 `0.8571`（平均能命中约 85.71% 的精确检索 top-k）。

这说明在当前数据规模与参数下，IVF-PQ 显著降低了索引体积，但会带来一定召回损失；可通过调大 `nprobe`、降低压缩强度（如减小 `pq_nbits` 或调整 `pq_m`）进一步权衡。