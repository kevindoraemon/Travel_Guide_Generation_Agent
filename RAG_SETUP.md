# 本地 Qdrant 混合 RAG

当前知识库使用 Qdrant 本地持久化模式，数据默认保存在 `data/qdrant/`，无需启动 MongoDB、Docker 或独立 Qdrant 服务。

## 检索链路

每次检索严格经过以下阶段：

1. 查询改写：把对话式或较长的出行简报改写成独立检索查询。
2. 元数据条件提取：提取 `city`、`country`、`topic`、`language`、`source_type`、`search_engine` 精确条件。
3. Dense Top 20：`BAAI/bge-small-zh-v1.5` 本地语义向量召回。
4. Sparse Top 20：中文二元组与英文词的稀疏向量召回，Qdrant 使用 IDF 修正。
5. RRF 融合：默认 `rrf_k=60`，合并两路最多 40 个候选。
6. Reranker Top 8：使用本地 RAG-Retrieval Cross-Encoder 重排。
7. 去重：先按文档去重，再做文本精确和近重复过滤。
8. Evidence Top 5：只把最终五条证据交给 LangGraph/LLM。

若查询改写模型或 Reranker 暂时不可用，系统分别回退到确定性改写/条件规则和 RRF 排序，不会让主图失败。CLI 输出的 `trace` 会明确标记各阶段数量和 Reranker 是否回退。

## 安装

```powershell
python -m pip install -r requirements.txt
```

首次入库会下载约 90 MB 的中文 Dense ONNX 模型到 `data/models/`。此后可离线运行 Dense/Sparse 检索；查询改写如需完全离线，可设置：

```powershell
$env:RAG_QUERY_REWRITE_ENABLED = "false"
$env:RAG_METADATA_EXTRACTION_ENABLED = "false"
```

## 入库

JSONL 每行一份文档，推荐字段：

```json
{"title":"北京三日亲子路书","content":"...","source_url":"https://...","search_engine":"baidu","country":"中国","city":"北京","topic":"family","metadata":{"author":"..."}}
```

```powershell
python scripts/ingest_travel_knowledge.py data/travel_itineraries_50.jsonl
```

支持 JSONL、Markdown、TXT、HTML 文件或目录。重复入库使用稳定文档 ID 覆盖旧分块。

## 查询与链路验证

```powershell
python scripts/query_travel_knowledge.py "北京亲子三日游，偏历史文化，地铁出行" --city 北京
```

输出包含 `evidence` 和 `trace`。主图在生成初稿前调用同一检索器，Scout Agent 也可调用 `search_travel_knowledge`。

## Recall@5 / MRR 评测

只有带人工相关性标签的数据集才能产生可写入简历的检索指标。评测集使用 JSONL，每行格式如下：

```json
{"case_id":"beijing-family-001","query":"北京亲子五日游","relevant_document_ids":["Qdrant 中的 document_id"],"filters":{"city":"北京"}}
```

运行并保存带数据集哈希、逐题排名和检索 Trace 的报告：

```powershell
python scripts/evaluate_rag.py data/eval/travel_rag_eval.jsonl --k 5 --output reports/rag_eval.json
```

在没有足够规模、人工标注且固定版本的评测集与报告前，不应声明 Recall@5 或 MRR 数值。

完整的自建旅游问答评测集、分级相关性、nDCG@5、双人盲评与仲裁流程见
[`data/eval/README.md`](data/eval/README.md)。当前 `travel_qa_eval_pilot.jsonl`
仅用于跑通工具链，标签未经真人复核，不能作为正式指标来源。

## 关键配置

配置位于 `config.yml -> stages.prod.rag`：

- `qdrant.path`：本地数据库目录。
- `embeddings.dense_model` / `dense_dimension`：Dense 模型与维度。
- `retrieval.dense_top_k` / `sparse_top_k`：两路召回数量，均为 20。
- `retrieval.reranker_top_k`：重排后保留 8。
- `retrieval.final_k`：最终证据 5。
- `deduplication`：按文档和近似度去重参数。

实现参考 Qdrant 官方的本地客户端、命名 Dense/Sparse 向量和 RRF 混合检索设计：<https://qdrant.tech/documentation/search/text-search/hybrid-search/>。
