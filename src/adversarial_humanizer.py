# -*- coding: utf-8 -*-
"""对抗性去 AI 痕迹引擎 + 真实数学指标 + 四维质量评价。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from src.schemas import DeAiIntensity, GenerationConfig

# ---------------------------------------------------------------------------
# 内置中英双语 AI 高频套话黑名单
# ---------------------------------------------------------------------------

BUILTIN_AI_CLICHES_EN: tuple[str, ...] = (
    "delve",
    "delve into",
    "tapestry",
    "testament",
    "moreover",
    "furthermore",
    "in conclusion",
    "it is worth noting",
    "worth noting",
    "in today's world",
    "in today's digital age",
    "landscape",
    "navigating",
    "aforementioned",
    "multifaceted",
    "leverage",
    "utilize",
    "robust",
    "seamless",
    "cutting-edge",
    "game-changer",
    "unlock the potential",
    "at the end of the day",
    "moving forward",
    "dive deep",
    "deep dive",
)

BUILTIN_AI_CLICHES_ZH: tuple[str, ...] = (
    "总而言之",
    "值得注意的是",
    "不可否认",
    "综上所述",
    "宛如",
    "在当今",
    "在当今社会",
    "不言而喻",
    "众所周知",
    "归根结底",
    "毋庸置疑",
    "与此同时",
    "由此可见",
    "其重要性不言而喻",
    "赋能",
    "助力",
    "深度剖析",
    "全方位",
)

# 弱强度本地 scrub 替换表（仅处理高危英文套话）
_LOCAL_REPLACEMENTS: dict[str, str] = {
    "delve into": "look at",
    "delve": "look into",
    "moreover": "also",
    "furthermore": "plus",
    "in conclusion": "so yeah",
    "it is worth noting that": "one thing though —",
    "it is worth noting": "one thing though —",
    "worth noting": "worth a look",
    "tapestry": "mix",
    "testament": "sign",
    "aforementioned": "that",
    "multifaceted": "messy",
    "utilize": "use",
    "leverage": "use",
    "seamless": "smooth",
    "cutting-edge": "new",
    "game-changer": "big deal",
    "deep dive": "closer look",
    "dive deep": "dig in",
    "总而言之": "说白了",
    "值得注意的是": "有一点得提",
    "不可否认": "确实",
    "综上所述": "综上",
    "在当今社会": "眼下",
    "在当今": "眼下",
    "众所周知": "大家都知道",
    "毋庸置疑": "这没什么好争的",
}

QUALITY_SCORE_KEYS = (
    "semantic_fidelity",
    "viewpoint_blend",
    "slang_authenticity",
    "human_likeness",
)

QUALITY_SCORE_LABELS_ZH = {
    "semantic_fidelity": "语义还原度",
    "viewpoint_blend": "观点融合自然度",
    "slang_authenticity": "文化梗地道度",
    "human_likeness": "去 AI 拟真度",
}

STRONG_SCORE_THRESHOLD = 75
STRONG_MAX_RETRY = 1

CallLlmFn = Callable[..., str]


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class QualityScores:
    semantic_fidelity: float = 0.0
    viewpoint_blend: float = 0.0
    slang_authenticity: float = 0.0
    human_likeness: float = 0.0
    diagnosis: str = ""
    source: str = "heuristic"  # heuristic | llm

    def average(self) -> float:
        vals = [
            self.semantic_fidelity,
            self.viewpoint_blend,
            self.slang_authenticity,
            self.human_likeness,
        ]
        return sum(vals) / len(vals) if vals else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["average"] = round(self.average(), 1)
        return d


@dataclass
class ObjectiveMetrics:
    ai_risk: float
    flesch: float
    burstiness: float
    cliche_hits: list[str] = field(default_factory=list)
    sentence_count: int = 0
    word_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ai_risk": self.ai_risk,
            "flesch": self.flesch,
            "burstiness": self.burstiness,
            "cliche_hits": list(self.cliche_hits),
            "sentence_count": self.sentence_count,
            "word_count": self.word_count,
        }


@dataclass
class HumanizerResult:
    final: str
    stage1: str
    notes: str
    two_phase: bool
    intensity: DeAiIntensity
    metrics: ObjectiveMetrics
    quality: QualityScores
    retried: bool = False
    scrub_notes: list[str] = field(default_factory=list)

    def to_pipeline_dict(self) -> dict[str, Any]:
        return {
            "final": self.final,
            "stage1": self.stage1,
            "notes": self.notes,
            "two_phase": self.two_phase,
            "intensity": self.intensity,
            "retried": self.retried,
            "scrub_notes": list(self.scrub_notes),
            "quality": self.quality.to_dict(),
        }

    def metrics_for_ui(self) -> dict[str, Any]:
        out = self.metrics.to_dict()
        out["quality"] = self.quality.to_dict()
        out["intensity"] = self.intensity
        out["retried"] = self.retried
        return out


# ---------------------------------------------------------------------------
# 文本分析：分句 / 音节 / 指标
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[。！？!?\.…])\s*|(?<=\n)\s*",
)


def split_sentences(text: str) -> list[str]:
    """兼容中英文的分句。"""
    raw = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text or "") if s and s.strip()]
    # 过滤纯标点碎片
    return [s for s in raw if re.search(r"[\w\u4e00-\u9fff]", s)]


def _is_mostly_cjk(text: str) -> bool:
    if not text:
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return cjk >= latin


def _count_syllables_word(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def count_words_en(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def compute_burstiness(text: str) -> tuple[float, int]:
    """句长变异系数（CV = std/mean），再缩放到约 0.5~3.5 展示区间。"""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return 0.5, len(sentences)

    if _is_mostly_cjk(text):
        lengths = [len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", s)) for s in sentences]
    else:
        lengths = [max(len(count_words_en(s)), 1) for s in sentences]

    mean = sum(lengths) / len(lengths)
    if mean <= 0:
        return 0.5, len(sentences)
    var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    std = math.sqrt(var)
    cv = std / mean
    # 人类口语 CV 常见约 0.3~1.2；映射到前端友好区间
    scaled = round(min(3.5, max(0.5, 0.5 + cv * 2.2)), 2)
    return scaled, len(sentences)


def compute_flesch(text: str) -> float:
    """英文 Flesch Reading Ease；中文按均句字数与标点密度折算到 0~100。"""
    sentences = split_sentences(text)
    if not sentences:
        return 50.0

    if _is_mostly_cjk(text):
        chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        punct = len(re.findall(r"[，。！？；、,.!?;:]", text))
        avg_len = chars / max(len(sentences), 1)
        dens = punct / max(chars, 1)
        # 短句 + 适度标点 → 更高分
        score = 100.0 - (avg_len - 12.0) * 2.5 - max(0.0, 0.08 - dens) * 200
        return round(min(100.0, max(0.0, score)), 1)

    words = count_words_en(text)
    if not words:
        return 50.0
    syllables = sum(_count_syllables_word(w) for w in words)
    n_sent = max(len(sentences), 1)
    n_words = len(words)
    # Flesch Reading Ease
    score = (
        206.835
        - 1.015 * (n_words / n_sent)
        - 84.6 * (syllables / n_words)
    )
    return round(min(100.0, max(0.0, score)), 1)


def merged_blacklist(user_words: list[str] | None = None) -> list[str]:
    pool = list(BUILTIN_AI_CLICHES_EN) + list(BUILTIN_AI_CLICHES_ZH)
    if user_words:
        pool.extend(w.strip() for w in user_words if w and w.strip())
    # 长词优先匹配，去重保序
    seen: set[str] = set()
    ordered: list[str] = []
    for w in sorted(pool, key=len, reverse=True):
        key = w.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(w)
    return ordered


def find_cliche_hits(text: str, user_words: list[str] | None = None) -> list[str]:
    hits: list[str] = []
    lower = text.lower()
    for term in merged_blacklist(user_words):
        if any("\u4e00" <= ch <= "\u9fff" for ch in term):
            if term in text:
                hits.append(term)
        else:
            # 词边界友好：短语直接子串；单词用边界
            if " " in term or "-" in term:
                if term.lower() in lower:
                    hits.append(term)
            else:
                if re.search(rf"\b{re.escape(term.lower())}\b", lower):
                    hits.append(term)
    # 去重保序
    out: list[str] = []
    seen: set[str] = set()
    for h in hits:
        k = h.lower()
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out


def compute_ai_risk(text: str, user_words: list[str] | None = None) -> tuple[float, list[str]]:
    """基于套话命中密度估算 AI 检测风险（%）。"""
    hits = find_cliche_hits(text, user_words)
    words = count_words_en(text)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    token_n = max(len(words) + cjk // 2, 1)
    density = len(hits) / max(token_n / 40.0, 1.0)  # 每 ~40 token 一次命中为基准
    # 基准分 + 命中加成，限制在 1~95
    risk = 2.0 + len(hits) * 6.5 + density * 8.0
    # 突发度偏低时略抬高风险
    burst, _ = compute_burstiness(text)
    if burst < 1.0:
        risk += 4.0
    return round(min(95.0, max(1.0, risk)), 2), hits


def compute_objective_metrics(
    text: str,
    user_blacklist: list[str] | None = None,
) -> ObjectiveMetrics:
    burst, n_sent = compute_burstiness(text)
    flesch = compute_flesch(text)
    ai_risk, hits = compute_ai_risk(text, user_blacklist)
    words = count_words_en(text)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    return ObjectiveMetrics(
        ai_risk=ai_risk,
        flesch=flesch,
        burstiness=burst,
        cliche_hits=hits,
        sentence_count=n_sent,
        word_count=len(words) + cjk,
    )


# ---------------------------------------------------------------------------
# 本地 scrub / 启发式四维分
# ---------------------------------------------------------------------------

def apply_local_cliche_scrub(
    text: str,
    user_blacklist: list[str] | None = None,
) -> tuple[str, list[str]]:
    """弱强度：本地替换高危套话（不调用 LLM）。"""
    notes: list[str] = []
    out = text
    # 先处理内置替换（长键优先）
    for src, dst in sorted(_LOCAL_REPLACEMENTS.items(), key=lambda kv: len(kv[0]), reverse=True):
        if any("\u4e00" <= ch <= "\u9fff" for ch in src):
            if src in out:
                out = out.replace(src, dst)
                notes.append(f'本地替换「{src}」→「{dst}」')
        else:
            pattern = re.compile(re.escape(src), re.IGNORECASE)
            if pattern.search(out):
                out = pattern.sub(dst, out)
                notes.append(f'本地替换 "{src}" → "{dst}"')

    # 用户黑名单：无法安全意译则标记（不强行删词破坏语法）
    for w in user_blacklist or []:
        w = w.strip()
        if not w:
            continue
        if any("\u4e00" <= ch <= "\u9fff" for ch in w):
            if w in out and w not in _LOCAL_REPLACEMENTS:
                notes.append(f"检测到用户禁用词「{w}」，建议人工改写")
        else:
            if re.search(rf"\b{re.escape(w)}\b", out, flags=re.IGNORECASE):
                if w.lower() not in {k.lower() for k in _LOCAL_REPLACEMENTS}:
                    notes.append(f'检测到用户禁用词 "{w}"，建议人工改写')
    return out, notes


def heuristic_quality_scores(
    text: str,
    config: GenerationConfig,
    metrics: ObjectiveMetrics,
) -> QualityScores:
    """off / weak 模式：用规则估算四维分，保证看板始终有数据。"""
    hit_penalty = min(40.0, len(metrics.cliche_hits) * 8.0)
    human = max(35.0, min(95.0, 88.0 - hit_penalty + (metrics.burstiness - 1.0) * 6.0))
    # 可读性适中加分
    flesch_bonus = 0.0
    if 55 <= metrics.flesch <= 80:
        flesch_bonus = 5.0
    human = min(98.0, human + flesch_bonus)

    semantic = 78.0 if config.injected_viewpoint.strip() else 82.0
    viewpoint = 70.0
    if config.injected_viewpoint.strip():
        # 粗检：观点关键词是否有回声（中英混合粗糙匹配）
        keys = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{4,}", config.injected_viewpoint)
        echoed = sum(1 for k in keys[:8] if k.lower() in text.lower())
        viewpoint = min(92.0, 62.0 + echoed * 6.0)
    else:
        viewpoint = 80.0

    slang = 68.0 + min(15.0, metrics.burstiness * 4.0)
    if config.tone and "GenZ" in config.tone:
        slang += 4.0
    slang = min(92.0, slang)

    diagnosis_parts = []
    if metrics.cliche_hits:
        diagnosis_parts.append(
            "命中套话/禁用词：" + "、".join(metrics.cliche_hits[:8])
        )
    else:
        diagnosis_parts.append("未命中内置高危 AI 套话。")
    diagnosis_parts.append(
        f"突发度 {metrics.burstiness}，可读性 {metrics.flesch}，"
        f"估算 AI 风险 {metrics.ai_risk}%。"
    )
    if config.injected_viewpoint.strip() and viewpoint < 75:
        diagnosis_parts.append("观点关键词回声偏弱，建议加强自然融入。")

    return QualityScores(
        semantic_fidelity=round(semantic, 1),
        viewpoint_blend=round(viewpoint, 1),
        slang_authenticity=round(slang, 1),
        human_likeness=round(human, 1),
        diagnosis=" ".join(diagnosis_parts),
        source="heuristic",
    )


def weak_burstiness_addon() -> str:
    """weak 模式追加到 Stage1 的句式起伏硬约束。"""
    return (
        "\n\nBURSTINESS / HUMAN RHYTHM (MANDATORY for this run):\n"
        "- Alternate very short punchy sentences with longer, messier ones.\n"
        "- Never start two consecutive sentences the same way.\n"
        "- Prefer concrete nouns and Reddit-native asides over abstract summaries.\n"
        "- Ban corporate smoothness; fragments are allowed sparingly."
    )


# ---------------------------------------------------------------------------
# Stage2 结构化解析（含四维评分）
# ---------------------------------------------------------------------------

STAGE2_SCORE_FORMAT_HINT = """
OUTPUT FORMAT — emit exactly these three markers, with nothing before or after them:
===FINAL_POST===
[the rewritten post goes here]
===REVIEW_NOTES===
[5-8 bullet points written in Simplified Chinese. Each bullet names ONE concrete
AI-tell you removed or ONE humanizing change you made.]
===QUALITY_SCORES===
{"semantic_fidelity": 0-100, "viewpoint_blend": 0-100, "slang_authenticity": 0-100, "human_likeness": 0-100, "diagnosis": "简体中文挑刺诊断，2-4句"}
""".strip()


def parse_stage2_structured(raw: str) -> tuple[str, str, Optional[QualityScores]]:
    """解析 FINAL_POST / REVIEW_NOTES / QUALITY_SCORES。"""
    text = (raw or "").strip()
    if "===FINAL_POST===" not in text:
        return text, "", None

    after_final = text.split("===FINAL_POST===", 1)[1]
    notes = ""
    scores: Optional[QualityScores] = None

    if "===REVIEW_NOTES===" in after_final:
        final_part, rest = after_final.split("===REVIEW_NOTES===", 1)
        final_text = final_part.strip()
        if "===QUALITY_SCORES===" in rest:
            notes_part, score_part = rest.split("===QUALITY_SCORES===", 1)
            notes = notes_part.strip()
            scores = _parse_scores_json(score_part.strip())
        else:
            notes = rest.strip()
    elif "===QUALITY_SCORES===" in after_final:
        final_part, score_part = after_final.split("===QUALITY_SCORES===", 1)
        final_text = final_part.strip()
        scores = _parse_scores_json(score_part.strip())
    else:
        final_text = after_final.strip()

    return final_text, notes, scores


def _parse_scores_json(blob: str) -> Optional[QualityScores]:
    blob = blob.strip()
    # 允许被 markdown code fence 包裹
    fence = re.search(r"\{[\s\S]*\}", blob)
    if not fence:
        return None
    try:
        data = json.loads(fence.group(0))
    except json.JSONDecodeError:
        return None

    def _num(key: str, default: float = 70.0) -> float:
        try:
            v = float(data.get(key, default))
        except (TypeError, ValueError):
            v = default
        return round(min(100.0, max(0.0, v)), 1)

    return QualityScores(
        semantic_fidelity=_num("semantic_fidelity"),
        viewpoint_blend=_num("viewpoint_blend"),
        slang_authenticity=_num("slang_authenticity"),
        human_likeness=_num("human_likeness"),
        diagnosis=str(data.get("diagnosis", "")).strip(),
        source="llm",
    )


def needs_strong_retry(
    text: str,
    quality: QualityScores,
    user_blacklist: list[str] | None = None,
) -> tuple[bool, str]:
    """strong 模式：黑名单命中或拟真/均分 < 75 → 打回。"""
    hits = find_cliche_hits(text, user_blacklist)
    reasons: list[str] = []
    if hits:
        reasons.append("仍命中套话/禁用词：" + "、".join(hits[:10]))
    avg = quality.average()
    if quality.human_likeness < STRONG_SCORE_THRESHOLD:
        reasons.append(
            f"去 AI 拟真度 {quality.human_likeness} < {STRONG_SCORE_THRESHOLD}"
        )
    if avg < STRONG_SCORE_THRESHOLD:
        reasons.append(f"四维均分 {avg:.1f} < {STRONG_SCORE_THRESHOLD}")
    if not reasons:
        return False, ""
    return True, "；".join(reasons)


# ---------------------------------------------------------------------------
# 流水线调度
# ---------------------------------------------------------------------------

def run_humanizer_pipeline(
    *,
    source: str,
    config: GenerationConfig,
    call_llm: CallLlmFn,
    api_key: str,
    base_url: str,
    model: str,
    build_stage1_prompt: Callable[[GenerationConfig], str],
    build_stage2_prompt: Callable[[GenerationConfig], str],
    log: Optional[Callable[[str], None]] = None,
) -> HumanizerResult:
    """按 ``de_ai_intensity`` 调度 off / weak / medium / strong。"""

    def _log(msg: str) -> None:
        if log:
            log(msg)

    intensity = config.resolved_de_ai_intensity()
    user_bl = list(config.blacklist_words)

    stage1_system = build_stage1_prompt(config)
    if intensity == "weak":
        stage1_system = stage1_system + weak_burstiness_addon()

    _log(f"🔹 `[INFO] Stage1 生成中（intensity={intensity}）...`")
    stage1_output = call_llm(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=config.temperature,
        system_prompt=stage1_system,
        user_prompt=(
            "以下是中文原始素材，请据此输出英文 Reddit 帖子正文：\n\n"
            + source.strip()
        ),
    )

    scrub_notes: list[str] = []
    notes = ""
    retried = False
    two_phase = intensity in ("medium", "strong")

    if intensity in ("off", "weak"):
        final_text = stage1_output
        if intensity == "weak":
            _log("🧼 `[INFO] weak：本地套话 scrub...`")
            final_text, scrub_notes = apply_local_cliche_scrub(final_text, user_bl)
            if scrub_notes:
                notes = "\n".join(f"- {n}" for n in scrub_notes)
        else:
            _log("⚪ `[INFO] off：跳过对抗重写，仅做本地客观评分。`")

        metrics = compute_objective_metrics(final_text, user_bl)
        quality = heuristic_quality_scores(final_text, config, metrics)
        return HumanizerResult(
            final=final_text,
            stage1=stage1_output,
            notes=notes,
            two_phase=False,
            intensity=intensity,
            metrics=metrics,
            quality=quality,
            retried=False,
            scrub_notes=scrub_notes,
        )

    # medium / strong：Stage2
    _log("⚔️ `[INFO] Stage2 对抗性去 AI 味重写 + 四维评分...`")
    stage2_system = _augment_stage2_prompt(build_stage2_prompt(config))
    stage2_raw = call_llm(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=min(1.0, config.temperature + 0.1),
        system_prompt=stage2_system,
        user_prompt="以下是需要去 AI 味重写的英文草稿：\n\n" + stage1_output,
    )
    final_text, notes, llm_quality = parse_stage2_structured(stage2_raw)
    metrics = compute_objective_metrics(final_text, user_bl)
    quality = llm_quality or heuristic_quality_scores(final_text, config, metrics)
    if llm_quality is None:
        _log("⚠️ `[WARN] Stage2 未返回 QUALITY_SCORES，已回退启发式四维分。`")

    if intensity == "strong":
        should, reason = needs_strong_retry(final_text, quality, user_bl)
        if should:
            _log(f"🔁 `[INFO] strong 打回重写（Max Retry=1）：{reason}`")
            retry_user = (
                "上一版成稿未达标，请按批评意见靶向重写，"
                "仍输出 FINAL_POST / REVIEW_NOTES / QUALITY_SCORES。\n\n"
                f"批评意见：{reason}\n"
                f"诊断：{quality.diagnosis}\n\n"
                f"待重写成稿：\n{final_text}"
            )
            retry_raw = call_llm(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=min(1.0, config.temperature + 0.15),
                system_prompt=stage2_system,
                user_prompt=retry_user,
            )
            final_text, notes2, llm_quality2 = parse_stage2_structured(retry_raw)
            if notes2:
                notes = (notes + "\n\n【强档回炉】\n" + notes2).strip() if notes else notes2
            metrics = compute_objective_metrics(final_text, user_bl)
            quality = llm_quality2 or heuristic_quality_scores(
                final_text, config, metrics
            )
            retried = True

    return HumanizerResult(
        final=final_text,
        stage1=stage1_output,
        notes=notes,
        two_phase=two_phase,
        intensity=intensity,
        metrics=metrics,
        quality=quality,
        retried=retried,
        scrub_notes=scrub_notes,
    )


def _augment_stage2_prompt(system_prompt: str) -> str:
    """确保 Stage2 prompt 要求输出 QUALITY_SCORES 块。"""
    if "===QUALITY_SCORES===" in system_prompt:
        return system_prompt
    # 替换旧双标记说明，或直接追加
    if "===FINAL_POST===" in system_prompt:
        # 截断旧 OUTPUT FORMAT 段后追加完整三标记说明
        parts = re.split(r"OUTPUT FORMAT[\s\S]*$", system_prompt)
        base = parts[0].rstrip()
        return base + "\n\n" + STAGE2_SCORE_FORMAT_HINT
    return system_prompt.rstrip() + "\n\n" + STAGE2_SCORE_FORMAT_HINT
