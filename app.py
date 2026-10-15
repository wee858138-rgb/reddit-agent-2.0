# -*- coding: utf-8 -*-
"""Reddit 内容重构与本地化控制台

把中文素材经过「人格化英文初稿 -> 对抗性去 AI 味重写（可选）」两阶段流水线，
重构为可直接发到目标 Subreddit 的英文成稿。

v2.0 第 ①~⑥ 步：契约 / LLM / 缓存 / 对抗评价 / Reddit 抓取清洗 / RAG 俚语词库。
"""

import json
import re
import time
from string import Template

import streamlit as st

from src.adversarial_humanizer import QUALITY_SCORE_LABELS_ZH
from src.cache_manager import get_cache_manager
from src.data_cleaner import extract_living_context
from src.llm_client import call_llm
from src.rag_engine import get_rag_engine
from src.reddit_client import RedditClient, RedditClientError
from src.schemas import DeAiIntensity, GenerationConfig, OutputMode
from src.structure_builder import (
    PRESET_BLUEPRINTS,
    build_paragraph_blueprint_block,
    run_structure_pipeline,
)

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

SUBREDDIT_NOTES = {
    "r/technology": (
        "Technology-focused general audience. Readers like concrete reporting and "
        "sharp takes, are deeply skeptical of corporate hype and PR wording, and "
        "run a strong, argumentative comments culture."
    ),
    "r/programming": (
        "Working software developers. They value technical accuracy, specifics and "
        "honest trade-off discussion, and are allergic to marketing fluff, listicle "
        "clickbait and low-effort beginner tutorials."
    ),
    "r/CasualConversation": (
        "A casual, friendly sub for personal chat. First-person anecdote style, warm "
        "and low-key, no essaying and no lecturing; the post should invite replies."
    ),
    "r/tifu": (
        "Today I Fucked Up — confessional, self-deprecating storytelling. Readers expect "
        "a vivid build-up, an ironic twist and a hard-earned lesson; they reward punchy "
        "first-person narration and punish humblebrags and obviously fake sagas."
    ),
    "r/AmItheAsshole": (
        "Moral judgment court. Posts lay out a conflict and ask the crowd to rule who is "
        "the asshole. Readers crave clear stakes, both sides presented fairly, and a "
        "satisfying 'so, AITA?' closer; they punish one-sided or manipulative framing."
    ),
    "r/AskReddit": (
        "Open-ended prompts that mine the crowd for stories and hot takes. The best posts "
        "are broad but specific enough to spark wildly different answers; comments are the "
        "real content, so the post must be a strong conversation starter, not an essay."
    ),
    "r/wallstreetbets": (
        "High-energy, irreverent finance/trading meme culture. Readers love loss porn, "
        "degenerate risk jokes, ticker talk and 'diamond hands / YOLO / to the moon' energy; "
        "they despise corporate disclaimers, sober risk warnings and anything that reads "
        "like professional financial advice."
    ),
    "r/SaaS": (
        "Indie founders and SaaS builders comparing MRR, churn, CAC and product-led-growth "
        "war stories. Readers value concrete numbers, honest failure postmortems and "
        "actionable tactics over hype; they are skeptical of 'I hit $100k MRR overnight' fluff."
    ),
}

PERSONA_INSTRUCTIONS = {
    "严谨毒舌的技术老哥": (
        "You write like a grizzled senior engineer: technically precise, deeply "
        "skeptical of hype, and dryly sarcastic. Favor short declarative sentences. "
        "Jargon is fine for technical audiences. Quietly respect the reader's "
        "intelligence and never talk down."
    ),
    "轻松幽默的 GenZ 网友": (
        "You write like a GenZ Redditor: ultra-casual, playful, slightly hyperbolic "
        "and meme-literate, dropping slang only where it sounds natural (never "
        "forced). Low formality, funny asides; it reads like a group-chat message "
        "that happens to be public."
    ),
    "客观中立的行业观察者": (
        "You write like a neutral industry analyst: measured, evidence-first and "
        "fair to multiple sides. No sarcasm and no hype. Plain, professional prose "
        "that still sounds like a human wrote it, not a press release."
    ),
    "TIFU 戏剧化故事大王": (
        "You write like a born storyteller recounting a personal disaster: vivid, "
        "self-deprecating and slightly theatrical, building suspense before landing an "
        "ironic twist. First-person, conversational, concrete sensory details and a "
        "punchy payoff line; never lectures, always entertains."
    ),
    "硅谷独立开发者/创业老兵": (
        "You write like a bootstrapped indie founder who has shipped real products and "
        "survived: blunt about churn, CAC, technical debt and feature creep, allergic to "
        "VC hype-speak, practical and a little jaded but generous with hard-won lessons. "
        "Concrete numbers beat adjectives."
    ),
    "贴吧/小红书神评嘴替": (
        "You write like the sharp-tongued voice of the internet comment section: punchy, "
        "meme-aware, mercilessly funny, mixing street-smart sarcasm with spot-on pop-culture "
        "references. Short bursts, high energy, zero politeness theater; it lands like the "
        "top-voted roast in a thread."
    ),
}

SENTIMENT_OPTIONS = ["极度负面", "略微批评", "客观中立", "略微积极", "极度热情"]
AUDIENCE_OPTIONS = ["GenZ", "千禧一代", "资深开发者", "摸鱼上班族", "创业者"]

DE_AI_INTENSITY_LABELS: dict[str, DeAiIntensity] = {
    "关闭 (off)": "off",
    "弱 (weak) — 句式起伏 + 本地 scrub，1 次 LLM": "weak",
    "中 (medium) — Stage2 对抗重写 + 四维评分": "medium",
    "强 (strong) — 不达标自动回炉 1 次": "strong",
}

OUTPUT_MODE_LABELS: dict[str, OutputMode] = {
    "🅱️ Mode B — 英文洗稿 + 逐段中文对照（旗舰）": "B_en_rewrite_zh_gloss",
    "🅰️ Mode A — 跨文化汉化（英文原帖 → 中文爆款）": "A_zh_social",
}

# 侧边栏「目标 Subreddit」下拉中的自定义版块哨兵选项
CUSTOM_SUBREDDIT_LABEL = "➕ 自定义版块…"

STAGE1_SYSTEM_TEMPLATE = Template(
    """You are a native English Reddit writer. You receive Chinese-language source
material and turn it into an authentic English Reddit post. NEVER mention that the
material was translated, summarized from Chinese, or AI-generated.

TARGET SUBREDDIT: $subreddit
Audience notes: $sub_notes

WRITER PERSONA / TONE: $persona
Voice instructions: $persona_instructions

$viewpoint_block

$control_block

$rag_block

$blueprint_block

HARD RULES:
1. Output the post body ONLY. No title, no preface, no translator notes, no markdown
   code fences.
2. Write idiomatic English that reads like a real person typing on Reddit, not a
   translation.
3. The core viewpoint above must drive the post's angle, but never quote it like a
   slogan or state it mechanically.
4. Length: approximately $paragraph_count paragraph(s).
5. Match the target subreddit's culture and the persona's voice throughout.
6. Strictly obey STANCE, TARGET AUDIENCE and BLACKLIST from the control block.
7. If RAG_LEXICON_CONTEXT is present, weave fitting slang naturally — never force it."""
)

STAGE2_SYSTEM_TEMPLATE = Template(
    """You are an adversarial "de-AI" rewrite engine. You receive an English Reddit
draft and rewrite it so absolutely nobody would guess a machine wrote it.

TARGET SUBREDDIT: $subreddit
PERSONA / TONE: $persona (stay fully in this voice)

$viewpoint_block

$control_block

$rag_block

$blueprint_block

REWRITE RULES:
1. Hunt down and kill every AI tell: "in conclusion", "it is worth noting",
   "delve", "tapestry", "landscape", "navigating", "moreover", "furthermore",
   perfectly balanced clauses, rule-of-three listicles, neat summary wrap-ups,
   em-dash overload, and hollow rhetorical questions.
2. Maximize burstiness: mix very short punchy sentences (occasional fragments
   allowed) with longer, messier, conversational ones. Never let two consecutive
   sentences share the same rhythm or opening.
3. Maximize perplexity: prefer the surprising-but-right word, concrete details and
   real Reddit-native colloquialisms/slang. No corporate smoothness.
4. Keep every factual claim from the draft intact and preserve the core viewpoint;
   do not invent new claims.
5. Approximately $paragraph_count paragraph(s). No title, no preface, no code
   fences.
6. Strictly obey STANCE, TARGET AUDIENCE and BLACKLIST from the control block;
   purge any blacklisted terms that survived the draft.

OUTPUT FORMAT — emit exactly these three markers, with nothing before or after them:
===FINAL_POST===
[the rewritten post goes here]
===REVIEW_NOTES===
[5-8 bullet points written in Simplified Chinese. Each bullet names ONE concrete
AI-tell you removed or ONE humanizing change you made. Be specific, for example:
- 删除了 "It is worth noting" 这类 AI 高频套话，换成口语化的 "here's the thing though"。]
===QUALITY_SCORES===
{"semantic_fidelity": 0-100, "viewpoint_blend": 0-100, "slang_authenticity": 0-100, "human_likeness": 0-100, "diagnosis": "简体中文挑刺诊断，2-4句"}"""
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def build_viewpoint_block(viewpoint, stage_label):
    """核心观点注入：有观点时强制要求融入，没有时让模型从素材自行提炼角度。"""
    if viewpoint.strip():
        return (
            "CORE VIEWPOINT (the post MUST embody this clearly and naturally):\n"
            + viewpoint.strip()
        )
    return (
        "CORE VIEWPOINT: none provided. Derive the most natural angle from the "
        + stage_label
        + " itself."
    )


def build_stage1_prompt(config: GenerationConfig, rag_block: str = "") -> str:
    """根据 GenerationConfig 组装阶段一 system prompt（含 stance/受众/黑名单/RAG/蓝图）。"""
    return STAGE1_SYSTEM_TEMPLATE.substitute(
        subreddit=config.subreddit,
        sub_notes=SUBREDDIT_NOTES.get(
            config.subreddit, "General Reddit audience."
        ),
        persona=config.tone,
        persona_instructions=PERSONA_INSTRUCTIONS.get(config.tone, ""),
        viewpoint_block=build_viewpoint_block(
            config.injected_viewpoint, "source material"
        ),
        control_block=config.prompt_control_block(),
        rag_block=(rag_block or "").strip() or "(no RAG lexicon injected)",
        blueprint_block=build_paragraph_blueprint_block(config.paragraph_blueprint),
        paragraph_count=config.paragraph_count,
    )


def build_stage2_prompt(config: GenerationConfig, rag_block: str = "") -> str:
    """根据 GenerationConfig 组装阶段二 system prompt（含 stance/受众/黑名单/RAG/蓝图）。"""
    return STAGE2_SYSTEM_TEMPLATE.substitute(
        subreddit=config.subreddit,
        persona=config.tone,
        viewpoint_block=build_viewpoint_block(config.injected_viewpoint, "draft"),
        control_block=config.prompt_control_block(),
        rag_block=(rag_block or "").strip() or "(no RAG lexicon injected)",
        blueprint_block=build_paragraph_blueprint_block(config.paragraph_blueprint),
        paragraph_count=config.paragraph_count,
    )


def render_copy_button(text, key):
    """渲染一个不依赖 Streamlit 自带 code 复制图标的显式一键复制按钮。"""
    safe_payload = (
        json.dumps(text, ensure_ascii=False)
        .replace("\\", "\\\\")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    html = (
        """
<div style="font-family:sans-serif;line-height:1.4">
  <textarea id="ta_""" + key + """" style="position:absolute;left:-9999px;top:0"
            aria-hidden="true" tabindex="-1"></textarea>
  <button id="btn_""" + key + """"
    style="padding:6px 16px;border-radius:8px;border:1px solid rgba(128,128,128,0.45);
           background:#ffffff;color:#111;cursor:pointer;font-size:14px">
    📋 一键复制成稿
  </button>
  <span id="msg_""" + key + """" style="margin-left:10px;color:#1a7f37;font-size:13px"></span>
</div>
<script>
  const textPayload_""" + key + """ = """ + safe_payload + """;
  const btn_""" + key + """ = document.getElementById('btn_""" + key + """');
  btn_""" + key + """.addEventListener('click', async () => {
    const ta = document.getElementById('ta_""" + key + """');
    let ok = false;
    try {
      await navigator.clipboard.writeText(textPayload_""" + key + """);
      ok = true;
    } catch (err) {
      ta.value = textPayload_""" + key + """;
      ta.focus();
      ta.select();
      try { ok = document.execCommand('copy'); } catch (e2) { ok = false; }
    }
    const msg = document.getElementById('msg_""" + key + """');
    msg.textContent = ok ? '✅ 已复制到剪贴板' : '❌ 复制失败，请手动选择文本';
    setTimeout(() => { msg.textContent = ''; }, 2500);
  });
</script>
"""
    )
    st.components.v1.html(html, height=52)


def render_config_chips(chips):
    """把一组 (标签, 值) 渲染为仪表盘风格的配置芯片。"""
    spans = "".join(
        '<span class="cfg-chip"><span class="cfg-k">'
        + str(k)
        + "</span> "
        + str(v)
        + "</span>"
        for k, v in chips
    )
    st.markdown('<div class="cfg-chips">' + spans + "</div>", unsafe_allow_html=True)


def build_export_markdown(result, metrics, meta):
    """把本次生成结果组装为排版精美的 Markdown 双语审稿档案。"""
    mode = result.get("mode")
    is_b = mode == "B_en_rewrite_zh_gloss"

    lines = [
        "# 🎯 Reddit 跨文化搬运 · 双语审稿档案",
        "",
        f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 一、配置元数据",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 目标版块 | {meta.get('subreddit', '—')} |",
        f"| 数字人格 | {meta.get('persona', '—')} |",
        f"| 温度 | {meta.get('temperature', '—')} |",
        f"| 输出模式 | {meta.get('mode_label', '—')} |",
        f"| 段落蓝图 | {meta.get('blueprint_name', '—')} |",
        f"| 段落数 | {meta.get('paragraph_count', '—')} |",
        f"| 情感立场 | {meta.get('stance', '—')} |",
        f"| 目标受众 | {meta.get('audience', '—')} |",
        f"| 去痕迹强度 | {meta.get('de_ai', '—')} |",
        f"| RAG 注入 | {meta.get('rag', '—')} |",
        f"| 核心观点 | {meta.get('viewpoint', '未指定')} |",
        f"| 禁用词 | {meta.get('blacklist', '（无）')} |",
        "",
    ]

    if is_b:
        lines += [
            "## 二、英文洗稿成稿",
            "",
            result.get("final", ""),
            "",
            "## 三、逐段中文对照",
            "",
        ]
        zh_gloss = result.get("zh_gloss") or []
        if zh_gloss:
            for g in zh_gloss:
                mission = g.get("mission", "")
                header = f"### 第 {g.get('paragraph', '?')} 段"
                if mission:
                    header += f"｜{mission}"
                lines += [header, "", g.get("zh", ""), ""]
        else:
            lines += ["（暂无逐段对照）", ""]
        analysis = result.get("analysis", "")
        if analysis:
            lines += ["## 四、观点植入与俚语运用解析", "", analysis, ""]
    else:
        lines += [
            "## 二、中文爆款汉化成稿",
            "",
            result.get("final", ""),
            "",
        ]

    # 四维评分诊断（两种模式均可用）
    quality = metrics.get("quality") or result.get("quality") or {}
    lines += ["## 五、四维评分诊断", ""]
    for key, label in QUALITY_SCORE_LABELS_ZH.items():
        val = quality.get(key)
        lines.append(f"- {label}：{round(float(val), 1) if val is not None else '—'}")
    avg = quality.get("average")
    if avg is None and quality:
        nums = [float(quality.get(k, 0) or 0) for k in QUALITY_SCORE_LABELS_ZH]
        avg = round(sum(nums) / 4, 1) if nums else 0
    lines.append(f"- 四维均分：**{avg}**")
    diagnosis = (quality.get("diagnosis") or "").strip()
    if diagnosis:
        lines += ["", "**挑刺诊断**：" + diagnosis]

    lines += ["", "### 客观指标", ""]
    lines.append(f"- AI 检测概率：{metrics.get('ai_risk', '—')}%")
    lines.append(f"- 可读性 (Flesch/中文折算)：{metrics.get('flesch', '—')}")
    lines.append(f"- 突发度 (句长 CV 缩放)：{metrics.get('burstiness', '—')}")
    hits = metrics.get("cliche_hits") or []
    lines.append(f"- 套话/禁用词命中：{'、'.join(hits) if hits else '无'}")

    notes = result.get("notes")
    if notes:
        lines += ["", "## 六、反派审查 / 改写细节", "", notes, ""]

    return "\n".join(lines).strip() + "\n"


LIVING_LLM_SYSTEM = (
    "You are a Reddit community corpus analyst. Extract structured JSON only, no extra text."
)


def _living_llm_fn():
    """构建 Living Context 提纯用的 LLM 调用（无 Key 则返回 None → 本地兜底）。"""
    if not api_key.strip():
        return None
    key = api_key.strip()
    url = base_url.strip().rstrip("/")
    model = model_name.strip()

    def _fn(prompt: str) -> str:
        return call_llm(key, url, model, 0.3, LIVING_LLM_SYSTEM, prompt)

    return _fn


def build_living_corpus(subreddit, client_id, client_secret, proxy, *, llm_fn=None):
    """抓取并提纯目标版块今日热帖，写入本地 24h 缓存池。返回 (corpus_dict, message)。"""
    cache = get_cache_manager()
    cached = cache.get_subreddit_corpus(subreddit)
    if cached and cached.get("fresh"):
        return cached, "已命中本地 24h 缓存池，秒级读取。"

    client = RedditClient(
        cache=cache, client_id=client_id, client_secret=client_secret, proxy=proxy
    )
    posts = client.fetch_subreddit_hot(
        subreddit, limit=15, time_filter="day", include_comments=True
    )
    raw_posts = [p.to_dict() for p in posts]
    cleaned_corpus = [
        {"title": p.title, "body": p.body, "score": p.score, "comments": list(p.top_comments)}
        for p in posts
    ]
    style = extract_living_context(cleaned_corpus, subreddit=subreddit, llm_fn=llm_fn)
    cache.put_subreddit_corpus(subreddit, raw_posts, cleaned_corpus, style)
    corpus = cache.get_subreddit_corpus(subreddit)
    return corpus, f"已抓取 {len(posts)} 个热帖并提纯写入缓存。"


DASHBOARD_CSS = """
<style>
/* ---------- 顶部品牌横幅 ---------- */
.hero-banner {
    background: linear-gradient(120deg, #101935 0%, #1d3f8f 55%, #2d6cdf 100%);
    border-radius: 18px;
    padding: 26px 32px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(16, 25, 53, 0.25);
}
.hero-banner h1 { color: #ffffff; font-size: 30px; margin: 0 0 8px 0; }
.hero-banner p { color: #c8d6ff; font-size: 14px; margin: 0; }

/* ---------- Tab 选项卡 ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: none; }
.stTabs [data-baseweb="tab"] {
    border-radius: 12px 12px 0 0;
    padding: 9px 20px;
    background: rgba(127, 127, 127, 0.08);
    border: 1px solid rgba(127, 127, 127, 0.15);
    border-bottom: none;
}
.stTabs [aria-selected="true"] {
    background: rgba(45, 108, 223, 0.12);
}
.stTabs [data-baseweb="tab"] p { font-size: 16px; font-weight: 700; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #2d6cdf; height: 3px; }

/* ---------- 指标卡片 ---------- */
[data-testid="stMetric"] {
    background: linear-gradient(160deg, #ffffff 0%, #f4f7ff 100%);
    border: 1px solid rgba(45, 108, 223, 0.20);
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 8px 22px rgba(31, 52, 98, 0.08);
}
[data-testid="stMetricValue"] { color: #1d3f8f; font-size: 32px !important; }
[data-testid="stMetricLabel"] p { font-weight: 600; color: #4a5568 !important; }

/* ---------- 配置芯片 ---------- */
.cfg-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 4px; }
.cfg-chip {
    display: inline-block;
    background: #eef3ff;
    border: 1px solid rgba(45, 108, 223, 0.25);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 12.5px;
    color: #1d3f8f;
}
.cfg-chip .cfg-k { color: #6b7aa5; font-weight: 600; }

/* ---------- 深色侧边栏 ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101935 0%, #1b2a52 100%);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff; }
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: #d7def5; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.18); }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea { color: #111111; }
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #9fb0e0 !important; }

/* ---------- 区块标题 ---------- */
.section-title {
    font-size: 18px; font-weight: 700; color: #1d3f8f;
    margin: 4px 0 2px;
}
</style>
"""


# ---------------------------------------------------------------------------
# 页面布局
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Reddit 内容重构与本地化控制台",
    page_icon="🎯",
    layout="wide",
)

st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="hero-banner">
  <h1>🎯 Reddit 内容重构与本地化控制台</h1>
  <p>中文素材 → 人格化英文初稿 →（可选）对抗性去 AI 味重写，输出可直接发帖的英文成稿 · Enterprise Pipeline Console</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- 侧边栏：API / 版块 / 人格 / 参数 ----------------

with st.sidebar:
    st.header("⚙️ 控制台设置")

    st.subheader("API 配置")
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-...（DeepSeek / OpenAI / 中转 Key）",
        help="Key 仅保存在当前浏览器会话中，不会被上传到任何第三方。",
    )
    base_url = st.text_input(
        "Base URL",
        value=DEFAULT_BASE_URL,
        help=(
            "DeepSeek：https://api.deepseek.com\n"
            "OpenAI：https://api.openai.com/v1\n"
            "第三方中转：填写中转方提供的 Base URL"
        ),
    )
    model_name = st.text_input(
        "模型名称 (Model)",
        value=DEFAULT_MODEL,
        help=(
            "DeepSeek 常用 deepseek-chat；OpenAI 常用 gpt-4o-mini / gpt-4o；"
            "第三方中转请按其文档填写。"
        ),
    )

    st.divider()

    st.subheader("Reddit 抓取（可选增强）")
    reddit_client_id = st.text_input(
        "Reddit Client ID（可选）",
        type="password",
        placeholder="OAuth 通道 A 需要，留空则走免 Key 通道",
        help="配置后可优先走官方 OAuth，突破 403 更稳。",
    )
    reddit_client_secret = st.text_input(
        "Reddit Client Secret（可选）",
        type="password",
        placeholder="与 Client ID 配对使用",
    )
    proxy_url = st.text_input(
        "🌐 本地代理地址 (Proxy URL，可选)",
        placeholder="例如 http://127.0.0.1:7890（留空则直连）",
        help="填写后所有 Reddit 请求走该代理。",
    )

    st.divider()

    st.subheader("发帖目标")
    _sub_presets = list(SUBREDDIT_NOTES.keys())
    _sub_options = _sub_presets + [CUSTOM_SUBREDDIT_LABEL]
    sub_choice = st.selectbox(
        "目标 Subreddit",
        options=_sub_options,
        key="sub_choice",
        help="下拉套用预设版块，或选择「自定义版块」手动输入（如 r/gamedev）。",
    )
    if sub_choice == CUSTOM_SUBREDDIT_LABEL:
        custom_sub = st.text_input(
            "自定义版块名称",
            placeholder="例如 r/gamedev（可省略 r/ 前缀）",
            key="custom_subreddit",
        )
        subreddit = custom_sub.strip() or "r/technology"
        if not subreddit.lower().startswith("r/"):
            subreddit = "r/" + subreddit
    else:
        subreddit = sub_choice
    persona = st.selectbox(
        "数字人格预设 (Persona)",
        options=list(PERSONA_INSTRUCTIONS.keys()),
    )

    st.divider()

    st.subheader("生成参数")
    temperature = st.slider(
        "温度 (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="越低越稳定保守，越高越发散、越有突发感。",
    )
    paragraph_count = st.slider(
        "输出段落数",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )

    st.divider()
    st.caption("API 状态：" + ("🟢 已配置" if api_key.strip() else "🔴 未配置"))
    st.caption("Runtime · streamlit × openai")

# Tab 顺序：先实例化高阶控件与 RAG，再跑工作台逻辑
tab_pipeline, tab_advanced, tab_rag, tab_metrics = st.tabs(
    [
        "🚀 核心工作台 (Pipeline)",
        "⚙️ 高阶控制阀 (Advanced)",
        "📚 RAG 俚语词库",
        "📊 评估看板 (Metrics)",
    ]
)

# ===========================================================================
# Tab 2：高阶控制阀
# ===========================================================================

with tab_advanced:
    st.subheader("⚙️ 高阶控制阀")
    st.caption("精细化调控生成策略。核心观点与对抗性节点直接作用于两阶段流水线；新增控制项已纳入配置面板。")

    adv_col1, adv_col2 = st.columns(2, gap="large")

    with adv_col1:
        st.markdown('<p class="section-title">观点与词汇约束</p>', unsafe_allow_html=True)
        viewpoint = st.text_input(
            "核心观点注入",
            placeholder="强制成稿融入的观点，例如：「开源模型在端侧场景已经追平闭源旗舰」",
            help="填写后，两阶段都会被强制要求把该观点作为帖子的核心立场。",
        )
        blacklist_raw = st.text_input(
            "禁用词黑名单 (Blacklist Words)",
            placeholder="多个词用逗号分隔，如：delve, moreover, tapestry",
            help="希望在成稿中避免出现的词汇，逗号 / 空格分隔。",
        )

    with adv_col2:
        st.markdown('<p class="section-title">风格与受众</p>', unsafe_allow_html=True)
        de_ai_label = st.selectbox(
            "去痕迹强度 (de_ai_intensity)",
            options=list(DE_AI_INTENSITY_LABELS.keys()),
            index=0,
            help=(
                "off：仅 Stage1 + 本地公式评分；"
                "weak：Stage1 强句式起伏 + 本地套话 scrub（1 次 LLM）；"
                "medium：Stage2 对抗重写并返回四维评分；"
                "strong：不达标（拟真/均分<75 或命中黑名单）最多回炉 1 次。"
            ),
        )
        de_ai_intensity: DeAiIntensity = DE_AI_INTENSITY_LABELS[de_ai_label]
        enable_de_ai = de_ai_intensity != "off"
        sentiment = st.select_slider(
            "情感倾向 (Sentiment / Stance)",
            options=SENTIMENT_OPTIONS,
            value="客观中立",
            help="控制成稿的情感光谱位置（已注入 Prompt）。",
        )
        audiences = st.multiselect(
            "目标受众群体",
            options=AUDIENCE_OPTIONS,
            default=["GenZ"],
            help="可多选，用于锁定帖子的对话对象（已注入 Prompt）。",
        )
        enable_rag = st.checkbox(
            "启用 RAG 圈子黑话注入 (enable_rag)",
            value=False,
            help="开启后，生成前从本地词库混合检索俚语/黑话，注入 Stage1/Stage2 Prompt。",
        )

    blacklist_words = [
        w.strip() for w in re.split(r"[,，;；\s]+", blacklist_raw) if w.strip()
    ]

    st.divider()
    st.markdown("#### 🧾 当前高阶配置快照")
    render_config_chips(
        [
            ("核心观点", "已注入" if viewpoint.strip() else "未指定"),
            ("去痕迹强度", de_ai_intensity),
            ("立场 Stance", sentiment),
            ("目标受众", "、".join(audiences) if audiences else "未选择"),
            ("禁用词数量", str(len(blacklist_words))),
            ("RAG 注入", "ON" if enable_rag else "OFF"),
        ]
    )
    if blacklist_words:
        st.caption("黑名单（已注入 Prompt）：" + " / ".join(blacklist_words))
    st.caption(
        "提示：立场 / 目标受众 / 禁用词已写入 Prompt；去痕迹强度与 RAG 开关经 GenerationConfig 调度。"
    )

# ===========================================================================
# Tab：RAG 俚语词库
# ===========================================================================

with tab_rag:
    st.subheader("📚 RAG 俚语词库")
    st.caption(
        "轻量 JSON 混合检索（关键词命中 + 目标 Subreddit 关联）。"
        "词库文件：`data/rag_lexicon.json`，支持一键新增并持久化。"
    )
    rag_engine = get_rag_engine()
    rag_entries = rag_engine.list_entries()

    r1, r2, r3 = st.columns(3)
    r1.metric("词条总数", len(rag_entries))
    r2.metric("词库版本", rag_engine.version)
    r3.metric("最近更新", (rag_engine.updated_at or "—")[:19])

    st.markdown("#### 🔍 检索试跑")
    test_col1, test_col2 = st.columns([3, 2])
    with test_col1:
        rag_test_text = st.text_area(
            "测试原文（用于关键词命中）",
            height=100,
            placeholder="粘贴一段英文/中文素材，看看会召回哪些黑话…",
            key="rag_test_text",
        )
    with test_col2:
        rag_test_sub = st.selectbox(
            "测试目标版块",
            options=list(SUBREDDIT_NOTES.keys()),
            key="rag_test_sub",
        )
        rag_top_k = st.slider("Top-K", 1, 10, 5, key="rag_top_k")
        if st.button("执行混合检索", key="rag_search_btn"):
            hits = rag_engine.retrieve(
                rag_test_text, rag_test_sub, top_k=rag_top_k
            )
            st.session_state["rag_search_hits"] = [
                {
                    "term": h.entry.term,
                    "score": h.score,
                    "reasons": h.reasons,
                    "zh": h.entry.zh_equivalent,
                    "example": h.entry.example,
                }
                for h in hits
            ]
            st.session_state["rag_context_preview"] = rag_engine.retrieve_context(
                rag_test_text, rag_test_sub, top_k=rag_top_k
            )

    if st.session_state.get("rag_search_hits") is not None:
        st.markdown("##### 命中结果")
        st.dataframe(st.session_state["rag_search_hits"], use_container_width=True)
        with st.expander("查看将注入 Prompt 的 RAG_LEXICON_CONTEXT"):
            st.code(st.session_state.get("rag_context_preview") or "", language="text")

    st.divider()
    st.markdown("#### 🔥 动态热帖语料池 (Living Context RAG)")
    st.caption("抓取指定版块当天最热帖子，提纯「今日热议/鲜活黑话/句法钩子」，24h 本地缓存防限流。")

    _living_sub_options = list(SUBREDDIT_NOTES.keys()) + [CUSTOM_SUBREDDIT_LABEL]
    living_sub_choice = st.selectbox(
        "🎯 选择要抓取今日热帖的版块",
        options=_living_sub_options,
        index=0,
        key="living_sub_choice",
    )
    if living_sub_choice == CUSTOM_SUBREDDIT_LABEL:
        _living_custom = st.text_input(
            "自定义版块名称",
            placeholder="例如 r/DeepSeek（可省略 r/ 前缀）",
            key="living_custom_sub",
        )
        living_target_sub = (_living_custom or "").strip() or "r/technology"
        if not living_target_sub.lower().startswith("r/"):
            living_target_sub = "r/" + living_target_sub
    else:
        living_target_sub = living_sub_choice

    living_corpus = get_cache_manager().get_subreddit_corpus(living_target_sub)
    lc1, lc2 = st.columns([1, 1])
    with lc1:
        if st.button(
            "📡 抓取并提纯该版块今日 Top 热帖",
            use_container_width=True,
            key="living_fetch_btn",
        ):
            with st.spinner(f"正在抓取 r/{living_target_sub.lstrip('r/')} 今日热帖并提纯…"):
                try:
                    living_corpus, _msg = build_living_corpus(
                        living_target_sub, reddit_client_id, reddit_client_secret, proxy_url,
                        llm_fn=_living_llm_fn(),
                    )
                    st.success(_msg)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"语料抓取失败：{exc}")
    with lc2:
        if living_corpus:
            if living_corpus.get("fresh"):
                age = living_corpus.get("age_seconds", 0)
                remain = max(0, 24 * 3600 - age)
                st.success(
                    f"🟢 缓存新鲜 · {str(living_corpus.get('fetched_at', ''))[:16]} · "
                    f"{remain // 3600}h{(remain % 3600) // 60}m 后过期"
                )
            else:
                st.warning(f"🟡 缓存已过期（{living_corpus.get('age_seconds', 0) // 3600}h 前）")
        else:
            st.info("尚未抓取过该版块语料。")
    if living_corpus and living_corpus.get("extracted_style"):
        style = living_corpus["extracted_style"]
        topics = style.get("trending_topics") or []
        slang = style.get("living_slang") or []
        patterns = style.get("syntax_patterns") or []
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.markdown("**今日热议话题**")
                if topics:
                    st.markdown("\n".join(f"- {t}" for t in topics))
                else:
                    st.caption("（无）")
        with c2:
            with st.container(border=True):
                st.markdown("**鲜活黑话/行话**")
                if slang:
                    for s in slang:
                        st.markdown(f"- **{s.get('term', '')}**：{s.get('context', '')}")
                else:
                    st.caption("（无）")
        with c3:
            with st.container(border=True):
                st.markdown("**句法/钩子范例**")
                if patterns:
                    st.markdown("\n".join(f"- {p}" for p in patterns))
                else:
                    st.caption("（无）")

    st.divider()
    st.markdown("#### ➕ 一键新增词条")
    with st.form("rag_add_form", clear_on_submit=True):
        f1, f2 = st.columns(2)
        with f1:
            new_term = st.text_input("term（词条）*", placeholder="例如：cope")
            new_subs = st.text_input(
                "subreddits（逗号分隔）",
                value="all",
                placeholder="all, r/programming",
            )
            new_zh = st.text_input(
                "zh_equivalent（中文网感）",
                placeholder="例如：自我安慰续命",
            )
        with f2:
            new_en = st.text_area(
                "en_meaning（英文释义）",
                height=80,
                placeholder="English definition / usage note",
            )
            new_ex = st.text_area(
                "example（英文造句）",
                height=80,
                placeholder="Native-sounding example sentence",
            )
        submitted = st.form_submit_button("💾 保存到词库", type="primary")
        if submitted:
            if not new_term.strip():
                st.error("term 不能为空")
            else:
                subs = [
                    s.strip()
                    for s in re.split(r"[,，;；\s]+", new_subs)
                    if s.strip()
                ] or ["all"]
                try:
                    rag_engine.add_entry(
                        new_term,
                        subreddits=subs,
                        en_meaning=new_en,
                        zh_equivalent=new_zh,
                        example=new_ex,
                    )
                    st.success(f"已写入词条「{new_term.strip()}」→ data/rag_lexicon.json")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"保存失败：{exc}")

    st.divider()
    st.markdown("#### 📖 现有词库一览")
    st.dataframe(
        [
            {
                "term": e.term,
                "subreddits": ", ".join(e.subreddits),
                "zh_equivalent": e.zh_equivalent,
                "en_meaning": e.en_meaning[:80]
                + ("…" if len(e.en_meaning) > 80 else ""),
            }
            for e in rag_entries
        ],
        use_container_width=True,
        height=360,
    )

# ===========================================================================
# Tab 3：评估看板
# ===========================================================================

with tab_metrics:
    st.subheader("📊 成稿质量评估看板")

    metrics = st.session_state.get("pipeline_metrics")
    metric_result = st.session_state.get("pipeline_result")

    if metrics and metric_result:
        m_col1, m_col2, m_col3 = st.columns(3)
        risk = float(metrics.get("ai_risk", 0))
        m_col1.metric(
            "AI 检测概率 (规则估算)",
            f"{risk}%",
            delta="较低" if risk < 15 else ("偏高" if risk >= 30 else "中等"),
            delta_color="off",
        )
        m_col2.metric(
            "可读性 (Flesch / 中文折算)",
            f"{metrics.get('flesch', '—')}",
            delta="真实公式计算",
            delta_color="off",
        )
        m_col3.metric(
            "突发度 (句长 CV 缩放)",
            f"{metrics.get('burstiness', '—')}",
            delta="真实统计",
            delta_color="off",
        )

        hits = metrics.get("cliche_hits") or []
        if hits:
            st.warning("检测到套话/禁用词命中：" + " / ".join(hits[:12]))
        else:
            st.success("未命中内置高危 AI 套话与用户禁用词。")

        quality = metrics.get("quality") or metric_result.get("quality") or {}
        st.divider()
        st.markdown("#### 🎯 四维评分卡")
        q_cols = st.columns(4)
        for col, key in zip(
            q_cols,
            [
                "semantic_fidelity",
                "viewpoint_blend",
                "slang_authenticity",
                "human_likeness",
            ],
        ):
            label = QUALITY_SCORE_LABELS_ZH.get(key, key)
            val = float(quality.get(key, 0) or 0)
            col.metric(label, f"{val:.0f}")
            col.progress(min(max(val / 100.0, 0.0), 1.0))

        avg = quality.get("average")
        if avg is None and quality:
            nums = [
                float(quality.get(k, 0) or 0)
                for k in (
                    "semantic_fidelity",
                    "viewpoint_blend",
                    "slang_authenticity",
                    "human_likeness",
                )
            ]
            avg = round(sum(nums) / 4, 1) if nums else 0
        src = quality.get("source", "—")
        st.caption(
            f"四维均分：**{avg}**　｜　评分来源：`{src}`"
            + ("　｜　已强档回炉" if metric_result.get("retried") else "")
        )
        diagnosis = (quality.get("diagnosis") or "").strip()
        if diagnosis:
            st.markdown("#### 🩺 挑刺诊断")
            st.info(diagnosis)

        st.divider()
        st.markdown("#### 🗂️ 本次运行档案")
        p1, p2, p3, p4, p5 = st.columns(5)
        intensity = metric_result.get("intensity") or metrics.get("intensity") or "—"
        p1.metric("去痕迹强度", intensity)
        p2.metric("执行模式", "两阶段" if metric_result.get("two_phase") else "单阶段")
        p3.metric("目标版块", subreddit.replace("r/", ""))
        p4.metric("数字人格", persona)
        p5.metric("输出段落", f"{paragraph_count} 段")

        st.caption(
            "客观三指标由本地真实公式计算（突发度 CV、Flesch/中文折算、套话命中风险）；"
            "四维评分在 medium/strong 下由 LLM 结构化返回，off/weak 下为规则启发式。"
        )
    else:
        st.info(
            "📭 暂无评估数据。请前往「🚀 核心工作台」粘贴素材并点击「一键流水线重构」，"
            "成稿生成成功后，客观指标与四维评分卡将出现在这里。"
        )

# ===========================================================================
# Tab 1：核心工作台
# ===========================================================================

with tab_pipeline:
    if st.session_state.pop("pipeline_done", False):
        if st.session_state.get("cache_hit"):
            st.success(
                "⚡ 已命中本地缓存 — 0 Token 秒级返回。可切换到「📊 评估看板」查看指标。"
            )
        else:
            st.success(
                "流水线执行完成 ✅ 成稿已就绪，可切换到「📊 评估看板」查看质量指标。"
            )

    st.subheader("🚀 核心工作台")
    st.caption(
        "左侧支持「粘贴素材」或「Reddit URL 抓取清洗」；右侧展示成稿与审查细节。"
    )

    # ---- 输出模式单选 ----
    mode_label = st.radio(
        "输出模式 (Output Mode)",
        options=list(OUTPUT_MODE_LABELS.keys()),
        index=0,
        horizontal=True,
        help=(
            "Mode B（旗舰）：英文洗稿成稿 + 逐段中文对照与梗解析；"
            "Mode A（跨文化汉化）：英文原帖 → 地道中文社媒爆款帖。"
        ),
    )
    output_mode: OutputMode = OUTPUT_MODE_LABELS[mode_label]

    work_col_in, work_col_out = st.columns([5, 6], gap="large")

    # ---- 左栏：素材输入 + 编排 ----
    with work_col_in:
        fetched_content = st.session_state.get("fetched_content", "")

        input_tab_manual, input_tab_reddit = st.tabs(
            ["✍️ 纯手动粘贴素材", "🔗 粘贴 Reddit 链接自动抓取"]
        )

        with input_tab_manual:
            manual_material = st.text_area(
                "✍️ 手动素材（中文/英文均可）",
                height=260,
                key="manual_material",
                placeholder="在这里直接粘贴中文/英文素材，或粘贴原帖内容。",
            )

        with input_tab_reddit:
            with st.expander("🔗 通过 Reddit 链接自动抓取", expanded=True):
                reddit_url = st.text_input(
                    "Reddit 帖子 URL",
                    placeholder=(
                        "支持 www/old.reddit.com、redd.it 短链、带 query 的分享链接…"
                    ),
                    key="reddit_url_input",
                )
                fetch_c1, fetch_c2 = st.columns(2)
                with fetch_c1:
                    include_hot_comments = st.checkbox(
                        "同时融入原帖高赞热评",
                        value=True,
                        key="include_hot_comments",
                    )
                with fetch_c2:
                    bypass_reddit_cache = st.checkbox(
                        "强制刷新 Reddit 缓存",
                        value=False,
                        key="bypass_reddit_cache",
                    )
                if st.button("抓取并清洗", use_container_width=True, key="fetch_reddit_btn"):
                    if not reddit_url.strip():
                        st.error("请先粘贴 Reddit 帖子链接。")
                    else:
                        try:
                            client = RedditClient(
                                cache=get_cache_manager(),
                                client_id=reddit_client_id,
                                client_secret=reddit_client_secret,
                                proxy=proxy_url,
                            )
                            fetch_res = client.fetch_post(
                                reddit_url.strip(),
                                bypass_cache=bypass_reddit_cache,
                                include_top_comments=5 if include_hot_comments else 0,
                            )
                            cleaned = fetch_res.cleaned
                            formatted = cleaned.format_for_llm(
                                include_comments=include_hot_comments
                            )
                            st.session_state["fetched_content"] = formatted
                            st.session_state["fetched_post_meta"] = {
                                "post_id": cleaned.post_id,
                                "subreddit": cleaned.subreddit,
                                "title": cleaned.title,
                                "score": cleaned.score,
                                "upvote_ratio": cleaned.upvote_ratio,
                                "is_nsfw": cleaned.is_nsfw,
                                "author": cleaned.author,
                                "from_cache": fetch_res.from_cache,
                                "channel": fetch_res.source_channel,
                                "url": cleaned.url,
                                "top_comments_n": len(cleaned.top_comments),
                            }
                            st.rerun()
                        except RedditClientError as exc:
                            st.error(f"抓取失败：{exc}")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"抓取异常：{exc}")

            if fetched_content.strip():
                st.text_area(
                    "📥 抓取到的结构化内容（只读，自动参与下方生成）",
                    value=fetched_content,
                    height=180,
                    disabled=True,
                )

        meta = st.session_state.get("fetched_post_meta")
        if meta:
            _channel = str(meta.get("channel", ""))
            _score = int(meta.get("score", 0) or 0)
            _sub = str(meta.get("subreddit", "")).strip()
            cache_tag = "⚡ 缓存命中" if meta.get("from_cache") else "🌐 实时抓取"
            st.markdown(
                f"**原帖元数据** · `{cache_tag}` · channel=`{_channel}`"
            )
            if _score == 0:
                # RSS 兜底通道（无点赞数据）：仅展示热评数，避免误导
                st.metric("💬 热评数", meta.get("top_comments_n", 0))
            else:
                _upvote = float(meta.get("upvote_ratio") or 0)
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("版块", f"r/{_sub.lstrip('r/')}" if _sub else "—")
                mc2.metric("点赞", _score)
                mc3.metric("Upvote%", f"{_upvote * 100:.0f}%")
                mc4.metric("热评", meta.get("top_comments_n", 0))
            if meta.get("is_nsfw"):
                st.warning("该帖标记为 NSFW，已保留正文但仍建议人工复核。")
            st.caption(
                f"u/{meta.get('author', '?')} · id=`{meta.get('post_id')}` · "
                f"{meta.get('title', '')[:80]}"
            )
            _fetched_sub = str(meta.get("subreddit", "")).strip()
            if _fetched_sub:
                sync_c1, sync_c2 = st.columns([1, 2])
                with sync_c1:
                    if st.button("🔗 同步到目标版块", key="sync_sub_btn"):
                        matched = next(
                            (
                                k
                                for k in SUBREDDIT_NOTES
                                if k.lstrip("r/").lower() == _fetched_sub.lstrip("r/").lower()
                            ),
                            None,
                        )
                        if matched:
                            st.session_state["sub_choice"] = matched
                            st.session_state["custom_subreddit"] = ""
                        else:
                            st.session_state["sub_choice"] = CUSTOM_SUBREDDIT_LABEL
                            st.session_state["custom_subreddit"] = (
                                "r/" + _fetched_sub.lstrip("r/")
                            )
                        st.rerun()
                with sync_c2:
                    st.caption(
                        f"识别到原帖版块 `r/{_fetched_sub.lstrip('r/')}`，"
                        "可一键同步为发帖目标。"
                    )

        _source_parts = [
            p.strip() for p in (fetched_content, manual_material) if p and p.strip()
        ]
        source_material = "\n\n".join(_source_parts)

        # ---- 段落结构编排器 ----
        with st.expander("🧩 段落结构编排器 (Paragraph Blueprint)", expanded=True):
            bp_col1, bp_col2 = st.columns([1, 2])
            with bp_col1:
                blueprint_choice = st.selectbox(
                    "预设段落蓝图",
                    options=[
                        "default_free",
                        "controversy_bait",
                        "tifu_story",
                        "geek_deepdive",
                        "custom",
                    ],
                    format_func=lambda k: (
                        PRESET_BLUEPRINTS[k].name
                        if k in PRESET_BLUEPRINTS
                        else "自定义编排 (Custom)"
                    ),
                    index=0,
                    key="blueprint_choice",
                    help="一键套用爆款段落结构，也可展开微调每一段的任务描述。",
                )
            with bp_col2:
                if blueprint_choice in PRESET_BLUEPRINTS:
                    st.caption(PRESET_BLUEPRINTS[blueprint_choice].description)
                else:
                    st.caption("为第 1~N 段分别指定主题任务，组装为高权重段落蓝图约束。")

            resolved_tasks: list[str] = []
            resolved_blueprint_name = "默认自由结构"
            if blueprint_choice == "default_free":
                st.caption("自由结构：按「输出段落数」滑块自由生成，不限定每段具体任务。")
            elif blueprint_choice == "custom":
                with st.expander("填写每一段任务", expanded=True):
                    custom_blueprint = []
                    for i in range(paragraph_count):
                        v = st.text_input(
                            f"第 {i + 1} 段任务",
                            key=f"bp_custom_{i}",
                            placeholder="例如：反常识暴论/痛点钩子",
                        )
                        custom_blueprint.append(v)
                resolved_tasks = [t.strip() for t in custom_blueprint if t.strip()]
                resolved_blueprint_name = "自定义编排" if resolved_tasks else "默认自由结构"
            else:
                preset = PRESET_BLUEPRINTS[blueprint_choice]
                with st.expander("查看 / 微调每一段任务", expanded=False):
                    edited = []
                    for i, default_task in enumerate(preset.paragraphs):
                        v = st.text_input(
                            f"第 {i + 1} 段任务",
                            value=default_task,
                            key=f"bp_preset_{blueprint_choice}_{i}",
                        )
                        edited.append(v)
                resolved_tasks = [
                    t.strip() for t in edited if t.strip()
                ] or list(preset.paragraphs)
                resolved_blueprint_name = preset.name

            effective_paragraph_count = len(resolved_tasks) if resolved_tasks else paragraph_count
            st.caption(
                f"当前蓝图：**{resolved_blueprint_name}**　｜　实际段落数：**{effective_paragraph_count}**"
            )

        render_config_chips(
            [
                ("版块", subreddit),
                ("人格", persona),
                ("温度", f"{temperature:.2f}"),
                ("段落", f"{effective_paragraph_count}"),
                ("模式", "B 双轨" if output_mode == "B_en_rewrite_zh_gloss" else "A 汉化"),
                ("蓝图", resolved_blueprint_name),
                ("情感", sentiment),
                ("受众", "、".join(audiences) if audiences else "未选择"),
                ("黑名单", f"{len(blacklist_words)} 词"),
                ("去痕迹", de_ai_intensity),
                ("RAG", "ON" if enable_rag else "OFF"),
            ]
        )

        bypass_cache = st.checkbox(
            "🔄 忽略缓存强制重生成 (Bypass Cache)",
            value=False,
            help="勾选后跳过 SQLite 生成缓存，强制重新调用 LLM 并覆盖。",
        )
        auto_sync_living = False
        if enable_rag:
            auto_sync_living = st.checkbox(
                "🔥 生成前自动同步目标版块今日热帖 (Living Context)",
                value=False,
                help="若该版块今日热帖语料尚未缓存，将在生成前自动抓取并提纯后注入 Prompt。",
            )
        run_clicked = st.button(
            "⚡ 一键流水线重构",
            type="primary",
            use_container_width=True,
        )

    # ---- 右栏：成稿输出 ----
    with work_col_out:
        result = st.session_state.get("pipeline_result")
        metrics = st.session_state.get("pipeline_metrics") or {}
        mode = result.get("mode") if result else None
        export_meta = {
            "subreddit": subreddit,
            "persona": persona,
            "temperature": f"{temperature:.2f}",
            "mode_label": (
                "Mode B 双轨洗稿（英文 + 逐段中文对照）"
                if output_mode == "B_en_rewrite_zh_gloss"
                else "Mode A 跨文化汉化（中文社媒爆款）"
            ),
            "blueprint_name": resolved_blueprint_name,
            "paragraph_count": effective_paragraph_count,
            "stance": sentiment,
            "audience": "、".join(audiences) if audiences else "未选择",
            "de_ai": de_ai_intensity,
            "rag": "ON" if enable_rag else "OFF",
            "viewpoint": (viewpoint or "").strip() or "未指定",
            "blacklist": " / ".join(blacklist_words) if blacklist_words else "（无）",
        }
        if result and mode == "B_en_rewrite_zh_gloss":
            st.markdown('<p class="section-title">② 双轨洗稿成稿（左：英文 / 右：中文对照+解析）</p>', unsafe_allow_html=True)
            if st.session_state.get("cache_hit"):
                st.markdown(
                    '<div style="display:inline-block;background:#fff7e6;border:1px solid #fa8c16;'
                    'color:#ad4e00;border-radius:999px;padding:4px 14px;font-weight:700;'
                    'font-size:13px;margin-bottom:8px">⚡ 已命中本地缓存</div>',
                    unsafe_allow_html=True,
                )
            st.caption(
                f"目标版块：{subreddit}　｜　人格：{persona}　｜　蓝图：{result.get('blueprint_name', '—')}"
                f"　｜　去痕迹：{result.get('intensity', de_ai_intensity)}"
            )

            en_col, gloss_col = st.columns([6, 5], gap="large")
            with en_col:
                st.markdown("**🇬🇧 英文洗稿成稿**")
                with st.container(border=True):
                    st.markdown(result["final"])
                _copy_col, _dl_col = st.columns([1, 1], gap="small")
                with _copy_col:
                    render_copy_button(result["final"], key="final_post")
                with _dl_col:
                    st.download_button(
                        "📥 导出双语审稿档案 (.md)",
                        data=build_export_markdown(result, metrics, export_meta).encode("utf-8"),
                        file_name="reddit_bilingual_review.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

            with gloss_col:
                st.markdown("**🇨🇳 逐段中文对照 & 梗解析**")
                zh_gloss = result.get("zh_gloss") or []
                if zh_gloss:
                    for g in zh_gloss:
                        mission = g.get("mission", "")
                        header = f"第 {g.get('paragraph', '?')} 段"
                        if mission:
                            header += f"｜{mission}"
                        with st.container(border=True):
                            st.markdown(f"**{header}**")
                            st.markdown(g.get("zh", ""))
                else:
                    st.caption("（暂无逐段对照）")
                analysis = result.get("analysis", "")
                if analysis:
                    with st.container(border=True):
                        st.markdown(analysis)

            if result.get("notes"):
                st.markdown("#### 😈 反派审查 / 改写细节")
                st.markdown(result["notes"])

            if result.get("stage1") and result.get("two_phase"):
                with st.expander("🔍 查看阶段一初始英文草稿（对比用）"):
                    st.markdown(result["stage1"])
            elif result.get("scrub_notes"):
                with st.expander("🧼 本地 scrub 记录"):
                    st.write(result["scrub_notes"])

        elif result and mode == "A_zh_social":
            st.markdown('<p class="section-title">② 中文爆款汉化成稿</p>', unsafe_allow_html=True)
            if st.session_state.get("cache_hit"):
                st.markdown(
                    '<div style="display:inline-block;background:#fff7e6;border:1px solid #fa8c16;'
                    'color:#ad4e00;border-radius:999px;padding:4px 14px;font-weight:700;'
                    'font-size:13px;margin-bottom:8px">⚡ 已命中本地缓存</div>',
                    unsafe_allow_html=True,
                )
            st.caption(
                f"目标版块：{subreddit}　｜　人格：{persona}　｜　蓝图：{result.get('blueprint_name', '—')}"
            )
            with st.container(border=True):
                st.markdown(result["final"])
            _copy_col, _dl_col = st.columns([1, 1], gap="small")
            with _copy_col:
                render_copy_button(result["final"], key="final_post")
            with _dl_col:
                st.download_button(
                    "📥 导出双语审稿档案 (.md)",
                    data=build_export_markdown(result, metrics, export_meta).encode("utf-8"),
                    file_name="reddit_bilingual_review.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        else:
            st.markdown('<p class="section-title">② 成稿输出</p>', unsafe_allow_html=True)
            st.info(
                "🗒️ 尚未生成成稿。\n\n"
                "1. 确认左侧侧边栏已填写 API Key（缓存命中时可省略）；\n"
                "2. 在「高阶控制阀」中调整观点 / 立场 / 受众 / 去痕迹强度；\n"
                "3. 选择输出模式与段落蓝图，粘贴素材并点击「⚡ 一键流水线重构」。"
            )

    # ---- 核心执行逻辑：缓存查询 → 段落/双轨流水线 → 写回缓存 ----
    if run_clicked:
        if not source_material.strip():
            st.error("请先输入原始素材（中/英均可）。")
            st.stop()

        gen_config = GenerationConfig(
            temperature=temperature,
            tone=persona,
            injected_viewpoint=viewpoint,
            stance=sentiment,
            target_audience=list(audiences),
            subreddit=subreddit,
            paragraph_count=effective_paragraph_count,
            enable_de_ai=enable_de_ai,
            blacklist_words=list(blacklist_words),
            de_ai_intensity=de_ai_intensity,
            enable_rag=enable_rag,
            output_mode=output_mode,
            paragraph_blueprint=list(resolved_tasks),
        )

        cache = get_cache_manager()
        param_hash, source_hash, cached_output = cache.lookup_or_none(
            source_material,
            gen_config,
            bypass=bypass_cache,
        )

        if cached_output is not None:
            st.session_state["pipeline_result"] = cached_output.get(
                "pipeline_result", cached_output
            )
            st.session_state["pipeline_metrics"] = cached_output.get(
                "pipeline_metrics", {}
            )
            st.session_state["cache_hit"] = True
            st.session_state["pipeline_done"] = True
            st.session_state["last_param_hash"] = param_hash
            st.rerun()

        if not api_key.strip():
            st.error("未命中缓存，请先在左侧侧边栏填写 API Key。")
            st.stop()

        rag_block = ""
        if gen_config.enable_rag:
            if auto_sync_living:
                _corpus = get_cache_manager().get_subreddit_corpus(gen_config.subreddit)
                if not (_corpus or {}).get("fresh"):
                    with st.spinner("正在自动同步目标版块今日热帖语料…"):
                        try:
                            build_living_corpus(
                                gen_config.subreddit,
                                reddit_client_id,
                                reddit_client_secret,
                                proxy_url,
                                llm_fn=_living_llm_fn(),
                            )
                        except Exception:  # noqa: BLE001
                            pass
            rag_block = get_rag_engine().retrieve_context(
                source_material, gen_config.subreddit, top_k=5
            )

        def _stage1(cfg: GenerationConfig) -> str:
            return build_stage1_prompt(cfg, rag_block=rag_block)

        def _stage2(cfg: GenerationConfig) -> str:
            return build_stage2_prompt(cfg, rag_block=rag_block)

        try:
            with st.status("系统运行状态监控...", expanded=True) as status:
                st.write(
                    f"🔹 `[INFO] 缓存未命中（hash={param_hash[:12]}…），开始生成…`"
                )
                if gen_config.enable_rag:
                    st.write("📚 `[INFO] RAG 圈子黑话上下文已注入 Prompt。`")
                time.sleep(0.3)

                structure_result = run_structure_pipeline(
                    source=source_material,
                    config=gen_config,
                    paragraph_tasks=list(resolved_tasks),
                    blueprint_name=resolved_blueprint_name,
                    rag_block=rag_block,
                    call_llm=call_llm,
                    api_key=api_key.strip(),
                    base_url=base_url.strip().rstrip("/"),
                    model=model_name.strip(),
                    build_stage1_prompt=_stage1,
                    build_stage2_prompt=_stage2,
                    log=st.write,
                )

                result = structure_result.to_pipeline_dict()
                metrics = structure_result.metrics_for_ui()

                st.write("✅ `[SUCCESS] 文本流重组完毕，正在写入本地缓存…`")
                cache.put_generation(
                    param_hash=param_hash,
                    source_hash=source_hash,
                    config=gen_config,
                    output={
                        "pipeline_result": result,
                        "pipeline_metrics": metrics,
                    },
                )
                time.sleep(0.2)
                status.update(
                    label="系统运行状态监控 — 全部节点执行完毕",
                    state="complete",
                    expanded=False,
                )

            st.session_state["pipeline_result"] = result
            st.session_state["pipeline_metrics"] = metrics
            st.session_state["cache_hit"] = False
            st.session_state["pipeline_done"] = True
            st.session_state["last_param_hash"] = param_hash
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            try:
                status.update(
                    label="系统运行状态监控 — 节点执行失败",
                    state="error",
                    expanded=True,
                )
            except Exception:  # noqa: BLE001
                pass
            st.error(
                "LLM 调用失败，请检查 API Key、Base URL、模型名称与网络后重试。\n\n"
                f"错误信息：`{exc}`"
            )
