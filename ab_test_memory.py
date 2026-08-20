# -*- coding: utf-8 -*-
"""
记忆系统 Token 消耗 A/B 测试

A 组（RAG 记忆方案）：用 RAG 检索知识块作为记忆，每次调用注入检索到的
    final_k=5 个 chunk（chunk_size=800），完整 RetrievedChunk JSON 注入
    所有消费者节点（无角色裁剪）。

B 组（分层记忆系统）：当前 MemoryPolicy 方案，按消费者角色裁剪 5 层记忆
    （session/profile/summary/sliding_window/skills）。

测量指标：每次 LLM 调用中「记忆上下文」消耗的 token 数
"""
import json
import sys
from pathlib import Path

# ── 加载项目记忆模块 ──
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from travel_planner.memory import MemoryPolicy

# ── Token 计数器 ──
_TOKENIZER = None

def get_tokenizer():
    """优先用已下载的 Qwen3 tokenizer；不可用时回退到字符估算。"""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    try:
        from transformers import AutoTokenizer
        qwen_path = ROOT.parent / "项目源码" / "data" / "rag" / "models" / "hub"
        snapshot = list(qwen_path.glob("models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/tokenizer.json"))
        if snapshot:
            _TOKENIZER = AutoTokenizer.from_pretrained(snapshot[0].parent, use_fast=True)
            print(f"[tokenizer] 已加载 Qwen3 tokenizer")
            return _TOKENIZER
    except Exception as e:
        print(f"[tokenizer] Qwen3 加载失败 ({e})，回退到字符估算")
    _TOKENIZER = "fallback"
    return _TOKENIZER

def count_tokens(text: str) -> int:
    tk = get_tokenizer()
    if tk == "fallback":
        cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - cn_chars
        return int(cn_chars / 1.5 + other_chars / 4)
    return len(tk.encode(text, add_special_tokens=False))


# ════════════════════════════════════════════════════
#  RAG 记忆方案：模拟检索到的 RetrievedChunk（config.yml: final_k=5, chunk_size=800）
# ════════════════════════════════════════════════════

def make_rag_chunks(final_k=5):
    """构造 final_k 个真实的 RetrievedChunk JSON（与 schemas.py 字段一致）。

    chunk_size=800（config.yml），content 长度贴近真实分块。
    """
    sample_contents = [
        # chunk 1: 景点攻略 ~800 chars
        "清水寺是京都最古老的寺院，始建于778年，被列为世界文化遗产。门票400日元，"
        "开放时间6:00-18:00（夜间参拜18:00-21:00，秋季有特别夜间拜观，门票600日元）。"
        "从祇园步行约15分钟可达，建议清晨前往避开人流。寺内音羽瀑布三道水流分别代表"
        "学业、恋爱、长寿，游客可用长柄勺接水饮用。清水舞台是最佳观景点，可俯瞰京都全景。"
        "周边二年坂、三年坂是京都最具风情的石板路，两侧聚集抹茶店、和果子铺、手工艺品店。"
        "推荐在此享用午餐：荞麦面、汤豆腐、抹茶甜品。带孩子建议预留2-3小时游览。",
        # chunk 2: 交通方案 ~800 chars
        "关西国际机场到京都站：JR Haruka 特急约75分钟，单程2980日元（ICOCA&HARUKA套票4060日元含ICOCA卡1500日元余额）。"
        "京都到大阪：JR京都线新快速约29分钟570日元，或京阪本线特急约45分钟410日元。"
        "京都市内交通推荐巴士一日券（700日元/天，覆盖大部分景点），或地铁+巴士组合。"
        "大阪市内推荐大阪地铁一日券（600日元平日/800日元周末），覆盖御堂筋线、中央线等所有线路。"
        "环球影城(USJ)：从大阪站乘JR樱岛线约10分钟，180日元。建议购买Express Pass 4（7800日元）减少排队。"
        "关西广域周游券5日券9700日元，可覆盖京都-大阪-奈良-神户-和歌山区间特急列车。",
        # chunk 3: 住宿推荐 ~800 chars
        "京都祇园地区住宿推荐：1) 花萤之榻（日式旅馆，4星级，双人间8000-12000日元/晚，含早晚餐怀石料理，"
        "距清水寺1.2km）；2) 祇园畑中（精品民宿，6人间带私汤，15000-20000日元/晚，距八坂神社300m）；"
        "3) Cross Hotel Kyoto（商务四星，双人间6000-8000日元/晚，距祇园4条站步行3分钟）。"
        "大阪难波地区住宿推荐：1) 大阪万豪都酒店（五星级，双人间10000-15000日元/晚，直通难波站）；"
        "2) Hotel Hankyu RESPIRE OSAKA（四星级，双人间7000-9000日元/晚，距大阪站2分钟）；"
        "3) 富士屋别馆（日式民宿，8000日元/晚，含早，距道顿堀步行5分钟）。亲子家庭建议选择带连通房的酒店。",
        # chunk 4: 美食指南 ~800 chars
        "京都怀石料理推荐：1) 菊乃井 本店（米其林三星，人均20000-30000日元，需提前1个月预约，"
        "地址：东山区祇园町南侧；2) 南禅寺 顺正（米其林一星，人均8000-12000日元，含汤豆腐套餐）。"
        "拉面推荐：一兰拉面（自助式，980日元/碗，24小时营业，道顿堀店人气最高）；"
        "四代目 鸟居处（鸡白汤拉面，850日元，京都拉面小路）。章鱼烧推荐：本家 大阪烧（道顿堀，850日元/8个）。"
        "亲子餐厅推荐：叙叙苑 烧肉（难波店，午间套餐1580日元起，含儿童椅和儿童餐具）；"
        "美浓吉 京都站店（便当套餐2500日元，便携式便当适合景点午餐）。预算建议：正餐人均1500-3000日元。",
        # chunk 5: 预算明细 ~800 chars
        "6日关西亲子游预算明细（2大1小，单位日元）：交通：关西广域券5日9700×2=19400，机场快线4060×2=8120，"
        "市内交通一日券700×5天×2人=7000，合计34520。住宿：京都3晚×9000=27000，大阪2晚×8500=17000，合计44000。"
        "门票：清水寺400×3=1200，伏见稻荷免费，岚山小火车620×3=1860，大阪城600×3=1800，"
        "环球影城8600×3=25800（含Express Pass），合计30660。餐饮：早餐800×6=4800，"
        "午餐1500×3人×6天=27000，晚餐3000×3人×5晚=45000，合计76800。购物/杂费：20000。"
        "总计：205980日元≈10299元人民币（1日元≈0.05元）。人均约3433元（不含购物），符合1万元/人预算。",
    ]

    metadata_fields = {
        "chunk_id": "chunk_8a3f2e1b",
        "document_id": "doc_kyoto_osaka_6day",
        "title": "京都大阪6日亲子路书",
        "source_url": "https://www.example-travel.com/kyoto-osaka-family-guide",
        "search_engine": "tavily",
        "language": "zh",
        "country": "Japan",
        "city": "Kyoto",
        "topic": "family_travel",
        "dense_score": 0.8234,
        "sparse_score": 0.7156,
        "dense_rank": 1,
        "sparse_rank": 2,
        "rrf_score": 0.03125,
        "rerank_score": 0.9512,
        "metadata": {"author": "travel_writer", "fetched_at": "2026-04-20T10:00:00Z"},
    }

    chunks = []
    for i, content in enumerate(sample_contents[:final_k]):
        chunk = dict(metadata_fields)
        chunk["chunk_id"] = f"chunk_{i+1:04x}a2"
        chunk["content"] = content
        chunks.append(chunk)
    return chunks


def rag_memory_render(chunks, consumer=None):
    """A 组：RAG 检索结果作为记忆上下文注入所有消费者（无角色裁剪）。"""
    return json.dumps(chunks, ensure_ascii=False, separators=(",", ":"))


# ════════════════════════════════════════════════════
#  分层记忆方案：构造真实 5 层记忆
# ════════════════════════════════════════════════════

def make_profile():
    return {
        "pace": "relaxed",
        "budget_level": "mid-range",
        "travel_style": "文化体验+自然风光",
        "dietary": "不吃辣，偏好清淡",
        "accommodation": "四星酒店或精品民宿",
        "group_size": "2大1小",
        "transport_pref": "高铁优先，市内打车",
        "language": "zh",
        "interests": ["历史古迹", "当地美食", "摄影"],
    }

def make_summary():
    return (
        "用户计划 2026 年 5 月带家人（2大1小）去日本京都和大阪 6 日游，"
        "预算 15000 元/人（不含购物），偏好轻松节奏，注重文化体验和亲子友好。"
        "要求：京都住 3 晚（祇园附近），大阪住 2 晚（难波附近）；"
        "必去清水寺、伏见稻荷大社、岚山竹林、大阪城、环球影城；"
        "交通用 JR Pass + 关西周游券；餐饮以怀石料理、拉面、章鱼烧为主；"
        "需含 Day1-Day6 详细安排，含交通时刻表和预算明细。"
        "注意：5 月可能有雨，需备室内备选方案；环球影城需提前购票。"
    )

def make_sliding_window(n=6):
    msgs = [
        ("user", "我想 5 月份去日本关西玩 6 天，2 大人 1 个小孩，预算 1 万 5 每人，想轻松一点"),
        ("assistant", "好的！关西 6 日亲子游，我来帮您规划。请问您对住宿有什么偏好？比如温泉旅馆还是城市酒店？"),
        ("user", "京都想住祇园附近的精品民宿，大阪住难波附近的四星酒店。不吃辣，孩子比较挑食"),
        ("assistant", "了解！京都祇园精品民宿 + 大阪难波四星酒店，清淡饮食。您对交通方式有偏好吗？"),
        ("user", "高铁优先吧，市内打车也行。对了孩子想环球影城，清水寺和伏见稻荷也要去"),
        ("assistant", "收到！我会安排：京都 3 晚 + 大阪 2 晚，含清水寺、伏见稻荷大社、岚山、大阪城、环球影城。"),
    ]
    return [f"{role}: {content}" for role, content in msgs[:n]]

def make_session():
    return {
        "session_id": "sess_20260514_a1b2c3d4",
        "channel": "web",
        "temp_flags": {"urgent": False, "vip": True, "locale": "zh-CN"},
        "requested_skills": ["kyoto_family_may_2026", "osaka_usj_guide"],
    }

def make_skills():
    return {
        "kyoto_family_may_2026": {
            "path": "outputs/kyoto_family_may_2026/itinerary.md",
            "title": "京都亲子 5 日路书",
            "created_at": "2026-04-20T10:30:00Z",
        },
        "osaka_usj_guide": {
            "path": "outputs/osaka_usj_guide/guide.md",
            "title": "大阪环球影城攻略",
            "created_at": "2026-04-15T14:00:00Z",
        },
    }

def build_layers():
    return {
        "session": make_session(),
        "profile": make_profile(),
        "recent_summary": make_summary(),
        "sliding_window": make_sliding_window(),
        "skills": make_skills(),
    }

def layered_memory_render(layers, consumer):
    """B 组：按 MemoryPolicy 裁剪后注入。"""
    return MemoryPolicy.render(layers, consumer)


# ════════════════════════════════════════════════════
#  A/B 测试核心
# ════════════════════════════════════════════════════

CONSUMERS = ["briefing", "writer", "tool"]

def run_ab_test():
    print("=" * 72)
    print("  记忆系统 Token 消耗 A/B 测试")
    print("  A组 = RAG 检索记忆 (final_k=5 chunks, chunk_size=800)")
    print("  B组 = 分层记忆系统 (MemoryPolicy 角色裁剪)")
    print("=" * 72)

    get_tokenizer()

    chunks = make_rag_chunks(final_k=5)
    layers = build_layers()

    # ── RAG chunk 结构分析 ──
    rag_json = rag_memory_render(chunks)
    rag_tokens = count_tokens(rag_json)
    chunk_content_tokens = sum(count_tokens(c["content"]) for c in chunks)
    chunk_metadata_tokens = rag_tokens - chunk_content_tokens

    print(f"\n[A组-RAG检索记忆]")
    print(f"  final_k = {len(chunks)} chunks, 总计 {len(rag_json)} 字符 / {rag_tokens} tokens")
    print(f"  ├─ chunk 内容: {chunk_content_tokens:>5} tokens ({chunk_content_tokens/rag_tokens*100:.1f}%)")
    print(f"  └─ 元数据开销: {chunk_metadata_tokens:>5} tokens ({chunk_metadata_tokens/rag_tokens*100:.1f}%)")
    for i, c in enumerate(chunks):
        print(f"     chunk[{i+1}] {len(c['content'])}字符 / {count_tokens(c['content'])} tokens")

    # ── 分层记忆结构分析 ──
    full_layer_json = json.dumps(layers, ensure_ascii=False, separators=(",", ":"))
    full_layer_tokens = count_tokens(full_layer_json)
    print(f"\n[B组-分层记忆系统]")
    print(f"  5 层完整记忆: {len(full_layer_json)} 字符 / {full_layer_tokens} tokens")
    for name, data in layers.items():
        tok = count_tokens(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        print(f"  ├─ {name:<16} {tok:>4} tokens")

    # ── 单次调用对比 ──
    print(f"\n{'─' * 72}")
    print(f"  单次调用 Token 消耗对比")
    print(f"{'─' * 72}")
    print(f"{'消费者':<14} {'A组(RAG)':>12} {'B组(分层)':>12} {'节省':>10} {'节省%':>8}")
    print(f"{'─' * 72}")

    results = {}
    for consumer in CONSUMERS:
        a_text = rag_memory_render(chunks, consumer)
        b_text = layered_memory_render(layers, consumer)
        a_tok = count_tokens(a_text)
        b_tok = count_tokens(b_text)
        saved = a_tok - b_tok
        pct = saved / a_tok * 100 if a_tok else 0
        results[consumer] = {"a": a_tok, "b": b_tok, "saved": saved, "pct": pct}
        print(f"  {consumer:<12} {a_tok:>10} tok {b_tok:>10} tok {saved:>8} tok {pct:>6.1f}%")

    # ── 完整规划周期对比 ──
    n_scouts = 3
    n_iterations = 5

    print(f"\n{'═' * 72}")
    print(f"  完整路书规划周期 Token 消耗 (briefing×1 + writer×1 + tool×{n_scouts}×{n_iterations})")
    print(f"{'═' * 72}")

    total_a = results["briefing"]["a"] + results["writer"]["a"] + n_scouts * n_iterations * results["tool"]["a"]
    total_b = results["briefing"]["b"] + results["writer"]["b"] + n_scouts * n_iterations * results["tool"]["b"]
    total_saved = total_a - total_b
    total_pct = total_saved / total_a * 100

    print(f"  {'':>4} {'调用次数':<8} {'A组(RAG)':>12} {'B组(分层)':>12}")
    print(f"  {'briefing':>4} {'1':<8} {results['briefing']['a']:>10} tok {results['briefing']['b']:>10} tok")
    print(f"  {'writer':>4} {'1':<8} {results['writer']['a']:>10} tok {results['writer']['b']:>10} tok")
    print(f"  {'tool':>4} {n_scouts*n_iterations:<8} {n_scouts*n_iterations*results['tool']['a']:>10} tok {n_scouts*n_iterations*results['tool']['b']:>10} tok")
    print(f"  {'─' * 56}")
    print(f"  {'合计':>4} {1+1+n_scouts*n_iterations:<8} {total_a:>10} tok {total_b:>10} tok")
    print(f"  {'节省':>4} {'':<8} {'':>12} {total_saved:>10} tok  ({total_pct:.1f}%)")

    # ── 不同迭代轮次下的节省曲线 ──
    print(f"\n{'═' * 72}")
    print(f"  不同迭代轮次下的 Token 消耗对比")
    print(f"{'═' * 72}")
    print(f"{'迭代':<6} {'tool调用':<8} {'A组(RAG)':>12} {'B组(分层)':>12} {'节省':>10} {'节省%':>8}")
    print(f"{'─' * 72}")
    for iters in [1, 2, 3, 5, 8, 10, 15]:
        n_tools = iters * n_scouts
        ta = results["briefing"]["a"] + results["writer"]["a"] + n_tools * results["tool"]["a"]
        tb = results["briefing"]["b"] + results["writer"]["b"] + n_tools * results["tool"]["b"]
        sv = ta - tb
        print(f"  {iters:<4} {n_tools:<8} {ta:>10} tok {tb:>10} tok {sv:>8} tok {sv/ta*100:>6.1f}%")

    # ── 节省来源分析 ──
    print(f"\n{'═' * 72}")
    print(f"  Token 节省来源分析")
    print(f"{'═' * 72}")
    print(f"  1. 结构化 vs 非结构化:")
    print(f"     RAG chunk 含大量元数据开销 ({chunk_metadata_tokens} tok/chunk, 占 {chunk_metadata_tokens/rag_tokens*100:.0f}%)")
    print(f"     分层记忆用精简 key-value 结构，无冗余元数据")
    print(f"  2. 角色裁剪:")
    print(f"     RAG 对所有消费者注入相同全量检索结果")
    print(f"     MemoryPolicy 按角色裁剪:")
    for consumer in CONSUMERS:
        print(f"       {consumer:<12} 注入 {results[consumer]['b']:>4} tok (RAG 注入 {results[consumer]['a']:>4} tok)")

    # ── MemoryPolicy 可见层明细 ──
    print(f"\n{'─' * 72}")
    print(f"  MemoryPolicy 各消费者可见层")
    print(f"{'─' * 72}")
    policy_map = {
        "briefing": ["session.temp_flags", "profile", "recent_summary", "sliding_window"],
        "writer":   ["profile", "recent_summary", "skills"],
        "tool":     ["session", "skills"],
    }
    for consumer, visible in policy_map.items():
        hidden = [l for l in layers.keys() if l not in [v.split(".")[0] for v in visible]]
        print(f"  {consumer:<12} 可见: {', '.join(visible)}")
        print(f"  {' ':>12} 裁剪: {', '.join(hidden) if hidden else '无'}")

    # ── 结论 ──
    print(f"\n{'═' * 72}")
    print(f"  结论")
    print(f"{'═' * 72}")
    print(f"  单次调用: 分层记忆比 RAG 记忆节省 {results['briefing']['pct']:.0f}%-{results['tool']['pct']:.0f}% token")
    print(f"  完整规划周期({1+1+n_scouts*n_iterations}次调用): 节省 {total_saved} tokens ({total_pct:.1f}%)")
    print(f"  节省归因: ① 角色裁剪(避免无关层注入) ② 结构化精简(无检索元数据开销)")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    run_ab_test()
