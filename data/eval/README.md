# 自建旅游问答评测集

本目录包含一份可以立即验证工具链的北京 Pilot 集，以及把真实旅游语料构造成正式人工评测集的完整流程。

## 当前能报告什么

`travel_qa_eval_pilot.jsonl` 有 10 个基于当前 Qdrant 内容编写的问题，覆盖完整行程、单日路线、景点、美食、出行提示、人群适配和多方面规划。它的标签状态全部是 `draft_needs_review`，作用是：

- 检查数据结构和评测脚本能否运行；
- 供两位标注员复核并练习评分规范；
- 暴露当前知识库覆盖不足的问题。

它不是正式人工评测集。当前 Qdrant 只有 1 份北京文档、4 个文本块，不能据此声明具有普遍意义的 Recall@5、MRR 或问答正确率。

## 正式集建议规模

先完成 50 份路书的清洗与入库，再构建 100 个问题：

| 类型 | 数量 | 目的 |
|---|---:|---|
| 单事实/单景点 | 20 | 基础精确召回 |
| 单日或多日路线 | 20 | 行程顺序和路线信息 |
| 预算/时间/人群多约束 | 20 | 条件保留和元数据过滤 |
| 美食、住宿、交通、避坑建议 | 15 | 主题检索与实用性 |
| 跨文档综合 | 15 | 多证据召回和综合回答 |
| 语料中无法回答 | 10 | 拒答与幻觉检测，仅计入问答评测 |

建议按 `20% pilot / 20% dev / 60% frozen test` 划分。Test 集在系统调参前冻结，不能根据检索结果反向修改问题。

## 构建步骤

### 1. 冻结语料

记录：文档数、文本块数、抓取时间范围、Qdrant collection 名、源 JSONL 的 SHA-256。语料或分块参数发生变化，就创建新的 snapshot 版本，不能继续沿用旧指标。

### 2. 导出标注材料

```powershell
python scripts/export_eval_annotation_catalog.py `
  --output-dir data/eval/annotation/corpus_v1 `
  --questions-per-document 2
```

输出：

- `chunk_catalog.jsonl`：标注员查阅的真实文本块和稳定 ID；
- `question_authoring.csv`：人工出题与相关性标注表。

### 3. 人工出题

问题不能直接复制标题，应像真实用户提问，并满足：

- 答案确实存在于冻结语料；
- `expected_answer_points_json` 是证据可支持的最小事实点；
- `hard_constraints_json` 只记录不可违反的时间、预算、人群和路线要求；
- 两位标注员分别填写 `relevance_label_1_json`、`relevance_label_2_json`；
- 仲裁员填写 `adjudicated_relevance_json`，使用 `chunk:<id>` 或 `document:<id>` 作为键。

对于语料中确实无法回答的问题，将 `answerable` 改为 `false`，两位标注员与仲裁结果均应为全 0；这类题只进入问答拒答/幻觉评测，检索 Recall/MRR/nDCG 脚本会自动排除。

### 4. 双人相关性标注与仲裁

两位标注员独立给每个候选证据打分：

- `0`：无关；
- `1`：主题相关，但不能直接回答；
- `2`：能支持部分答案；
- `3`：直接、完整支持核心答案。

分歧由第三人仲裁。只有填入两个不同 `labeler` 和 `adjudicator_id` 后，状态才会成为 `adjudicated`。

### 5. 生成并校验 JSONL

```powershell
python scripts/build_travel_qa_eval.py `
  data/eval/annotation/corpus_v1/question_authoring.csv `
  --snapshot "travel-corpus-v1:<sha256>" `
  --output data/eval/travel_qa_eval_v1.jsonl

python scripts/validate_travel_qa_eval.py `
  data/eval/travel_qa_eval_v1.jsonl `
  --check-qdrant `
  --require-adjudicated
```

`--require-adjudicated` 未通过时，不应对外报告指标。

### 6. 跑检索评测

```powershell
python scripts/evaluate_rag.py `
  data/eval/travel_qa_eval_v1.jsonl `
  --k 5 `
  --output reports/rag_eval_v1.json
```

输出包含数据集哈希、Recall@5、MRR、nDCG@5、HitRate@5、平均延迟和逐题 Trace。

### 7. 生成待盲评回答

每套系统保存一个 JSONL，每行至少包含：

```json
{"case_id":"case-001","answer":"系统最终答案……"}
```

然后生成双人盲评表：

```powershell
python scripts/prepare_qa_human_review.py `
  data/eval/travel_qa_eval_v1.jsonl `
  --answers baseline=reports/baseline_answers.jsonl `
  --answers hybrid_rag=reports/hybrid_rag_answers.jsonl `
  --annotators annotator_a,annotator_b `
  --output-dir data/eval/annotation/run_v1
```

不要把 `blind_key.json` 发给标注员。

### 8. 回收并聚合人工评分

标注员填写六个 `1–5` 分维度和严重错误字段。回收后运行：

```powershell
python scripts/aggregate_qa_human_review.py `
  data/eval/annotation/run_v1/answer_review.csv `
  data/eval/annotation/run_v1/blind_key.json `
  --output reports/qa_human_eval_v1.json
```

报告给出各系统综合分、严重错误率和每个维度的二次加权 Cohen's Kappa。建议 Kappa 低于 `0.60` 时重新培训标注员并复标，不要直接发布均分。

## 可写入简历的最低证据

至少保留：冻结数据集、SHA-256、标注规范、匿名标注员编号、冲突仲裁记录、逐题评测结果和运行配置。指标应写为“在 N 条双人标注并仲裁的固定测试集上”，不要只写一个没有样本量和版本的百分比。
