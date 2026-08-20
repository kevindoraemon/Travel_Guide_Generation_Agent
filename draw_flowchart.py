# -*- coding: utf-8 -*-
"""
MemoryPolicy 角色裁剪流程图 → SVG（零依赖，面试展示用）
输出: memory_policy_flowchart.svg
"""

SVG_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 780" font-family="'Microsoft YaHei','SimHei',sans-serif">
<defs>
  <marker id="arr-blue" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#4A90D9"/></marker>
  <marker id="arr-gray" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#BDC3C7"/></marker>
  <marker id="arr-orange" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#F5A623"/></marker>
  <marker id="arr-green" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#7ED321"/></marker>
  <marker id="arr-red" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#D0021B"/></marker>
  <filter id="shadow" x="-5%" y="-5%" width="115%" height="115%"><feDropShadow dx="1" dy="2" stdDeviation="2" flood-opacity="0.15"/></filter>
  <style>
    .title{font-size:24px;font-weight:bold;fill:#2C3E50}
    .subtitle{font-size:13px;fill:#7F8C8D}
    .section{font-size:14px;font-weight:bold}
    .box-label{font-size:12px;font-weight:bold;fill:#fff}
    .box-sub{font-size:10px;fill:#fff;opacity:0.9}
    .desc{font-size:10px;fill:#5D6D7E}
    .tok{font-size:11px;font-weight:bold}
    .cut{font-size:9px;fill:#BDC3C7}
    .compare-title{font-size:12px;font-weight:bold}
    .compare-text{font-size:10px}
    .save-num{font-size:20px;font-weight:bold;fill:#27AE60}
  </style>
</defs>
"""

def rbox(x, y, w, h, r=8, fill="#4A90D9", opacity=1.0, stroke="none"):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" opacity="{opacity}" stroke="{stroke}" filter="url(#shadow)"/>'

def text(x, y, content, cls="box-label", anchor="middle"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}" xml:space="preserve">{content}</text>'

def arrow(x1, y1, x2, y2, marker="arr-gray", color="#BDC3C7", dash=None, width=1.5):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" marker-end="url(#{marker})"{d}/>'

def path_arrow(d_str, marker="arr-gray", color="#BDC3C7", width=1.5):
    return f'<path d="{d_str}" fill="none" stroke="{color}" stroke-width="{width}" marker-end="url(#{marker})"/>'

svg = [SVG_HEADER]

# ═══════ 标题 ═══════
svg.append(text(600, 38, "MemoryPolicy 角色裁剪流程", "title"))
svg.append(text(600, 60, "5 层记忆 → 按消费者角色裁剪 → 精准注入 Prompt", "subtitle"))

# ═══════ 1. 输入（左侧） ═══════
svg.append(text(110, 100, "输入", "section", "#4A90D9"))
svg.append(rbox(30, 115, 160, 32, fill="#4A90D9"))
svg.append(text(110, 136, "user_id + session", "box-label"))
svg.append(rbox(30, 160, 160, 32, fill="#4A90D9"))
svg.append(text(110, 181, "messages[] 对话历史", "box-label"))
svg.append(rbox(30, 205, 160, 32, fill="#4A90D9"))
svg.append(text(110, 226, "profile_updates", "box-label"))

# ═══════ 2. 5层记忆（中间偏左） ═══════
svg.append(text(400, 100, "5 层记忆 (load_layers)", "section", "#1A5276"))

layer_data = [
    (120, "Session",      "会话ID · 渠道 · 临时标记",      "44 tok"),
    (168, "Profile",      "画像：节奏 · 预算 · 偏好",      "73 tok"),
    (216, "Summary",      "对话摘要 (≤1600字)",            "120 tok"),
    (264, "SlidingWindow","滑动窗口 (最近6条消息)",        "159 tok"),
    (312, "Skills",       "可复用技能 (历史路书路径)",      "69 tok"),
]

for y, name, desc, tok in layer_data:
    svg.append(rbox(270, y, 120, 38, r=6, fill="#67B7DC"))
    svg.append(text(330, y+16, name, "box-label"))
    svg.append(text(330, y+30, tok, "box-sub"))
    svg.append(text(400, y+22, desc, "desc", "start"))

svg.append(rbox(270, 360, 120, 28, r=6, fill="#1A5276"))
svg.append(text(330, 379, "合计: 483 tok", "box-label"))

# 输入 -> 5层
svg.append(arrow(190, 131, 270, 139, "arr-blue", "#4A90D9", width=2))
svg.append(path_arrow("M190 176 C 220 176, 240 187, 270 187", "arr-blue", "#4A90D9", 1.5))
svg.append(path_arrow("M190 221 C 220 221, 240 233, 270 233", "arr-blue", "#4A90D9", 1.5))

# ═══════ 3. MemoryPolicy 中央 ═══════
svg.append(rbox(470, 170, 160, 110, r=10, fill="#34495E", opacity=0.08, stroke="#34495E"))
svg.append('<rect x="470" y="170" width="160" height="110" rx="10" fill="none" stroke="#34495E" stroke-width="1.5" stroke-dasharray="4,3"/>')
svg.append(text(550, 200, "MemoryPolicy", "section", "#2C3E50"))
svg.append(text(550, 222, ".render(layers,", "desc"))
svg.append(text(550, 238, "    consumer)", "desc"))
svg.append(text(550, 258, "按角色裁剪", "desc"))
svg.append(text(550, 272, "仅返回所需层", "desc"))

# 5层 -> Policy 箭头
for y, *_ in layer_data:
    svg.append(arrow(390, y+19, 470, 225, "arr-gray", "#BDC3C7", dash="3,2", width=1))

# ═══════ 4. 三个消费者（右侧） ═══════
svg.append(text(900, 100, "消费者节点 (Agent)", "section", "#8E44AD"))

# --- Briefing ---
svg.append(rbox(760, 120, 280, 36, r=8, fill="#F5A623"))
svg.append(text(900, 143, "Briefing Agent", "box-label"))
svg.append(rbox(760, 162, 280, 30, r=6, fill="#F5A623", opacity=0.2))
svg.append(text(900, 182, "✓ flags + profile + summary + window", "desc"))
svg.append(text(900, 196, "380 tok", "tok", "#F5A623"))
svg.append(text(1042, 196, "✗ skills", "cut", "start"))
svg.append(path_arrow("M630 215 C 680 180, 720 138, 760 138", "arr-orange", "#F5A623", 2.5))

# --- Writer ---
svg.append(rbox(760, 260, 280, 36, r=8, fill="#7ED321"))
svg.append(text(900, 283, "Writer Agent", "box-label"))
svg.append(rbox(760, 302, 280, 30, r=6, fill="#7ED321", opacity=0.2))
svg.append(text(900, 322, "✓ profile + summary + skills", "desc"))
svg.append(text(900, 336, "275 tok", "tok", "#7ED321"))
svg.append(text(1042, 336, "✗ session + window", "cut", "start"))
svg.append(path_arrow("M630 225 C 680 250, 720 278, 760 278", "arr-green", "#7ED321", 2.5))

# --- Tool ---
svg.append(rbox(760, 400, 280, 36, r=8, fill="#D0021B"))
svg.append(text(900, 423, "Tool Agent (Scout)", "box-label"))
svg.append(rbox(760, 442, 280, 30, r=6, fill="#D0021B", opacity=0.2))
svg.append(text(900, 462, "✓ session + skills", "desc"))
svg.append(text(900, 476, "121 tok", "tok", "#D0021B"))
svg.append(text(1042, 476, "✗ profile+summary+window", "cut", "start"))
svg.append(path_arrow("M630 240 C 680 320, 720 418, 760 418", "arr-red", "#D0021B", 2.5))

# ═══════ 5. 底部对比 ═══════
svg.append('<line x1="30" y1="540" x2="1170" y2="540" stroke="#ECF0F1" stroke-width="2"/>')

# RAG 方案
svg.append(rbox(60, 565, 280, 100, r=8, fill="#FDEDEC", opacity=0.6, stroke="#E74C3C"))
svg.append(text(200, 590, "RAG 检索记忆 (对比)", "compare-title", "#C0392B"))
svg.append(text(200, 612, "final_k=5 chunks", "compare-text", "#E74C3C"))
svg.append(text(200, 628, "每次注入 1320 tokens", "compare-text", "#E74C3C"))
svg.append(text(200, 644, "所有消费者无差别全量注入", "compare-text", "#999"))
svg.append(text(200, 660, "含 45% 元数据开销", "compare-text", "#999"))

# 箭头 RAG -> 节省
svg.append(arrow(340, 615, 430, 615, "arr-red", "#E74C3C", width=2))

# 节省结果
svg.append(rbox(430, 565, 300, 100, r=8, fill="#EAFAF1", opacity=0.7, stroke="#27AE60"))
svg.append(text(580, 590, "Token 节省", "compare-title", "#27AE60"))
svg.append(text(580, 620, "单次调用: 71% - 91%", "save-num"))
svg.append(text(580, 645, "完整规划周期 (17次调用):", "compare-text", "#27AE60"))
svg.append(text(580, 660, "节省 89% token (19970/22440)", "compare-text", "#27AE60"))

# 节省归因
svg.append(rbox(770, 565, 380, 100, r=8, fill="#F8F9FA", opacity=0.8, stroke="#BDC3C7"))
svg.append(text(960, 590, "节省归因", "compare-title", "#2C3E50"))
svg.append(text(960, 610, "① 角色裁剪：避免无关记忆层注入", "compare-text", "#5D6D7E"))
svg.append(text(960, 628, "② 结构化精简：key-value 替代 chunk JSON", "compare-text", "#5D6D7E"))
svg.append(text(960, 646, "③ 去元数据：无 scores/ranks/url 开销", "compare-text", "#5D6D7E"))
svg.append(text(960, 664, "④ Tool 节点高频调用 (15次)，节省放大", "compare-text", "#5D6D7E"))

svg.append("</svg>")

out = "d:/Liu Minghao/西电/简历/源码/project-Deep_Research/memory_policy_flowchart.svg"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(svg))
print(f"已保存: {out}")
