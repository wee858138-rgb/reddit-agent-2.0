# -*- coding: utf-8 -*-
"""段落主题结构编排器 + 中英双轨洗稿引擎。

收官模块：把「预设/自定义段落蓝图」与「双轨输出模式」统一编排进生成流水线。

- Mode B（默认旗舰）：英文洗稿成稿 + 逐段中文对照回译 + 观点/俚语解析。
- Mode A（跨文化汉化）：英文原帖 -> 地道中文社媒爆款帖。

设计约定：
- 段落蓝图统一通过 ``build_paragraph_blueprint_block`` 组装为高权重
  ``PARAGRAPH_BLUEPRINT_BLOCK``，注入 Stage1/Stage2 与 Mode A Prompt。
- Mode B 复用 ``adversarial_humanizer.run_humanizer_pipeline`` 完成去 AI 味英文
  成稿（保留四维评分与客观指标），再追加一次 LLM 调用产出逐段中文对照与解析。
- Mode A 直接产出中文，沿用 ``compute_objective_metrics`` / ``heuristic_quality_scores``
  保持评估看板数据完整。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from string import Template
from typing import Any, Callable, Optional

from src.adversarial_humanizer import (
    compute_objective_metrics,
    heuristic_quality_scores,
    run_humanizer_pipeline,
)
from src.schemas import GenerationConfig, OutputMode


CallLlmFn = Callable[..., str]


# ---------------------------------------------------------------------------
# 预设爆款段落蓝图
# ---------------------------------------------------------------------------

@dataclass
class BlueprintPreset:
    key: str
    name: str
    description: str
    paragraphs: list[str]


PRESET_BLUEPRINTS: dict[str, BlueprintPreset] = {
    "default_free": BlueprintPreset(
        key="default_free",
        name="默认自由结构",
        description="兼容初代 paragraph_count 滑块逻辑（无特定蓝图时按段落数自由生成）。",
        paragraphs=[],
    ),
    "controversy_bait": BlueprintPreset(
        key="controversy_bait",
        name="Reddit/X 争议引战流 (4段)",
        description="反常识暴论钩子 → 事实还原 → 观点+吐槽 → 争议提问引流评论区。",
        paragraphs=[
            "反常识暴论/痛点钩子：第一句就抛出与主流认知相悖的论断或直戳痛点的狠话，让人忍不住看下去或想反驳。",
            "原帖核心事实与细节还原：忠实交代事件的来龙去脉、关键数字与具体细节，铺垫可信度，为后面的观点蓄力。",
            "强行植入核心观点与犀利吐槽：把指定核心观点自然编织进来，配上一针见血的锐评，态度鲜明、不留情面。",
            "开放式争议提问引流评论区：用一个有火药味但不过界的开放问题收尾，制造站队与讨论，引诱评论区开撕。",
        ],
    ),
    "tifu_story": BlueprintPreset(
        key="tifu_story",
        name="TIFU 沉浸故事反转流 (4段)",
        description="倒叙悬念 → 冲突铺垫 → 神转折高潮 → 自嘲感悟 + TL;DR。",
        paragraphs=[
            "倒叙悬念开场：先用结果或最荒诞的一刻倒叙开场，制造强烈的「后来呢」悬念感。",
            "冲突升级与铺垫：交代前因后果与矛盾积累，让读者代入并替主角捏把汗，情绪逐步升温。",
            "神转折高潮与圈子黑话穿插：在情绪最高点抛出反转，自然穿插目标圈子的黑话/俚语，让转折更有网感。",
            "自嘲式感悟与 TL;DR 总结：用自嘲的口吻收束复盘，并给出 TL;DR 一句话总结，方便滑走的读者。",
        ],
    ),
    "geek_deepdive": BlueprintPreset(
        key="geek_deepdive",
        name="硬核极客深度拆解流 (3段)",
        description="行业痛点一针见血 → 技术/逻辑硬核对比 → 未来趋势预判。",
        paragraphs=[
            "行业痛点一针见血：开门见山指出行业/技术圈最扎心的那个真问题，拒绝隔靴搔痒的客套话。",
            "技术/逻辑硬核对比与观点植入：用具体的逻辑链、数据或技术细节做硬核对比，把核心观点焊死在论证里。",
            "未来趋势预判：基于前面的拆解，给出一句有信息量、敢下判断的趋势预测，收尾干净有力。",
        ],
    ),
}

CUSTOM_BLUEPRINT_KEY = "custom"
BLUEPRINT_ORDER = ["default_free", "controversy_bait", "tifu_story", "geek_deepdive"]


def list_presets() -> list[BlueprintPreset]:
    return [PRESET_BLUEPRINTS[k] for k in BLUEPRINT_ORDER if k in PRESET_BLUEPRINTS]


def resolve_paragraph_tasks(
    blueprint_key: str = "default_free",
    custom_blueprint: Optional[list[str]] = None,
) -> tuple[list[str], str]:
    """解析最终段落任务列表与蓝图名称。

    - ``blueprint_key`` 命中预设且预设含段落 -> 返回预设任务。
    - ``blueprint_key == "custom"`` 且 custom_blueprint 非空 -> 返回自定义任务。
    - 否则 -> 空列表（默认自由结构，按 paragraph_count 自由生成）。
    """
    custom = [t.strip() for t in (custom_blueprint or []) if t.strip()]
    if blueprint_key == CUSTOM_BLUEPRINT_KEY and custom:
        return custom, "自定义编排"
    if blueprint_key in PRESET_BLUEPRINTS and PRESET_BLUEPRINTS[blueprint_key].paragraphs:
        preset = PRESET_BLUEPRINTS[blueprint_key]
        return list(preset.paragraphs), preset.name
    return [], "默认自由结构"


def build_paragraph_blueprint_block(tasks: list[str]) -> str:
    """把段落任务列表组装为高权重 PARAGRAPH_BLUEPRINT_BLOCK。"""
    clean = [t.strip() for t in (tasks or []) if t.strip()]
    if not clean:
        return (
            "PARAGRAPH_BLUEPRINT_BLOCK:\n"
            "  (no fixed blueprint — structure the post freely, aiming for the target paragraph count.)"
        )
    lines = [
        "PARAGRAPH_BLUEPRINT_BLOCK (HIGHEST-PRIORITY structural constraint):",
        f"The post MUST be structured into exactly {len(clean)} paragraphs, in this exact order.",
        "Each paragraph must fulfill its assigned mission below; do not merge or skip missions.",
        "",
    ]
    for i, task in enumerate(clean, 1):
        lines.append(f"Paragraph {i} mission: {task}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 统一结果结构（兼容 HumanizerResult 的缓存 / 看板字段）
# ---------------------------------------------------------------------------

@dataclass
class StructureResult:
    mode: OutputMode
    final: str
    en_post: str
    zh_post: str
    zh_gloss: list[dict]
    analysis: str
    paragraph_tasks: list[str]
    blueprint_name: str
    stage1: str
    notes: str
    two_phase: bool
    intensity: str
    retried: bool
    scrub_notes: list[str]
    quality: dict
    metrics: dict

    def to_pipeline_dict(self) -> dict[str, Any]:
        """超集于 HumanizerResult.to_pipeline_dict()，额外携带双轨字段。"""
        return {
            "final": self.final,
            "stage1": self.stage1,
            "notes": self.notes,
            "two_phase": self.two_phase,
            "intensity": self.intensity,
            "retried": self.retried,
            "scrub_notes": list(self.scrub_notes),
            "quality": self.quality,
            # v2.0 双轨字段
            "mode": self.mode,
            "en_post": self.en_post,
            "zh_post": self.zh_post,
            "zh_gloss": self.zh_gloss,
            "analysis": self.analysis,
            "paragraph_tasks": list(self.paragraph_tasks),
            "blueprint_name": self.blueprint_name,
        }

    def metrics_for_ui(self) -> dict[str, Any]:
        out = dict(self.metrics)
        out["quality"] = self.quality
        out["intensity"] = self.intensity
        out["retried"] = self.retried
        return out


# ---------------------------------------------------------------------------
# Mode A：跨文化汉化 Prompt（英文原帖 -> 地道中文社媒爆款帖）
# ---------------------------------------------------------------------------

ZH_PERSONA_INSTRUCTIONS = {
    "严谨毒舌的技术老哥": (
        "像技术圈老炮用中文吐槽：专业、毒舌、不留情面，短句为主，爱用反问与阴阳怪气，"
        "但逻辑硬、不装懂、不卖弄术语。"
    ),
    "轻松幽默的 GenZ 网友": (
        "像中文互联网 GenZ：玩梗、夸张、口语化，多用「家人们」「谁懂啊」「绷不住了」"
        "这类网感口头禅，节奏快、有梗、自带弹幕感。"
    ),
    "客观中立的行业观察者": (
        "像中文行业观察号：克制、讲证据、不吹不黑，行文像有见地的知乎长文，"
        "避免标题党和情绪化，但也不端着。"
    ),
    "TIFU 戏剧化故事大王": (
        "像讲段子的故事大王：第一人称、自嘲、有戏剧张力，先铺垫再反转，细节生动、"
        "结尾抛金句，读起来像贴吧热帖，绝不端着说教。"
    ),
    "硅谷独立开发者/创业老兵": (
        "像熬过周期的独立开发者：直给、务实、有点疲惫但愿意分享血泪教训，反感融资话术，"
        "爱用大白话拆解增长、成本与踩坑，数据比形容词有说服力。"
    ),
    "贴吧/小红书神评嘴替": (
        "像评论区最能怼的神评嘴替：短句、玩梗、毒舌又好笑，网感拉满，专治各种不服，"
        "绝无客气场面话，一句顶十句。"
    ),
}

MODE_A_SYSTEM_TEMPLATE = Template(
    """You are a native Simplified-Chinese social-media ghostwriter. You receive an English Reddit post (and its top comments) and turn it into an authentic viral Chinese post that fits domestic communities (小红书 / 贴吧 / 知乎 网感).

ORIGINAL SUBREDDIT: $subreddit

WRITER PERSONA / TONE: $persona
Voice instructions (中文互联网语感): $zh_voice

$viewpoint_block

$control_block

$rag_block

$blueprint_block

HARD RULES:
1. Output the Chinese post body ONLY. No title, no preface, no translator notes, no markdown code fences.
2. 用中文互联网语感「重写」而不是逐句翻译：口语化、有梗、有情绪，读起来像真人发的爆款帖。
3. The core viewpoint above must drive the post's angle, but never quote it like a slogan or state it mechanically.
4. Weave Chinese internet slang / 黑话 naturally where fitting (from RAG zh_equivalent or general net-speak), never forced.
5. Strictly obey STANCE, TARGET AUDIENCE and BLACKLIST from the control block.
6. 排雷：禁止「众所周知」「综上所述」「在当今」「值得注意的是」等 AI 味官腔；多用短句、口头禅、反问与适度夸张。
7. 段落结构必须严格遵循 PARAGRAPH_BLUEPRINT_BLOCK 的任务划分（若无固定蓝图，按目标段落数自由切分）。"""
)


def build_mode_a_system_prompt(
    config: GenerationConfig,
    rag_block: str,
    tasks: list[str],
) -> str:
    """组装 Mode A（中文汉化）system prompt。"""
    if config.injected_viewpoint.strip():
        viewpoint_block = (
            "CORE VIEWPOINT (the post MUST embody this clearly and naturally):\n"
            + config.injected_viewpoint.strip()
        )
    else:
        viewpoint_block = (
            "CORE VIEWPOINT: none provided. Derive the most natural angle from the source itself."
        )
    return MODE_A_SYSTEM_TEMPLATE.safe_substitute(
        subreddit=config.subreddit,
        persona=config.tone,
        zh_voice=ZH_PERSONA_INSTRUCTIONS.get(
            config.tone, "用中文互联网的语感重写，口语化、有梗、有情绪。"
        ),
        viewpoint_block=viewpoint_block,
        control_block=config.prompt_control_block(),
        rag_block=(rag_block or "").strip() or "(no RAG lexicon injected)",
        blueprint_block=build_paragraph_blueprint_block(tasks),
    )


# ---------------------------------------------------------------------------
# Mode B：逐段中文对照回译 + 观点/俚语解析（英文成稿后的二次调用）
# ---------------------------------------------------------------------------

MODE_B_GLOSS_SYSTEM = """You are a bilingual (English / Simplified Chinese) editorial assistant for a content operations team.

You receive:
1. An optional list of paragraph missions (the blueprint the English post was supposed to follow).
2. The finished English Reddit post.

Produce two deliverables in Simplified Chinese:

1. A paragraph-by-paragraph Chinese back-translation (逐段中文对照回译). For each paragraph of the English post, give its faithful, colloquial Chinese meaning so a Chinese reviewer can verify content fidelity without seeing the original source. If a blueprint is provided, map paragraphs to those missions; otherwise, follow the post's natural paragraph breaks.

2. A final block titled 「🎯 观点植入与俚语运用解析」 that explains in plain Simplified Chinese:
   - Where and how the forced core viewpoint was woven in (观点植入);
   - Which Reddit slang / in-group jargon appeared and what it means in Chinese (俚语/黑话运用);
   - Which emotional stance and audience-calibration choices were made (立场与受众).

RULES:
- Emit ONLY the two markers below, with nothing before or after.
- Do NOT invent content that is not in the English post.
- Keep the back-translation faithful and concrete.

OUTPUT FORMAT:
===ZH_GLOSS===
【第1段｜<段落任务简述>】
中文对照：<第1段的中文回译>

【第2段｜<段落任务简述>】
中文对照：<第2段的中文回译>

(one block per paragraph)
===ANALYSIS===
🎯 观点植入与俚语运用解析
- 观点植入：...
- 俚语/黑话运用：...
- 立场与受众：...
"""

_ZH_GLOSS_MARKER = "===ZH_GLOSS==="
_ANALYSIS_MARKER = "===ANALYSIS==="


def build_mode_b_gloss_system_prompt(config: GenerationConfig, tasks: list[str]) -> str:
    """Mode B 回译/解析调用的 system prompt（当前为静态模板）。"""
    return MODE_B_GLOSS_SYSTEM


def build_mode_b_gloss_user_prompt(tasks: list[str], en_post: str) -> str:
    """Mode B 回译/解析调用的 user prompt：蓝图 + 英文成稿。"""
    clean = [t.strip() for t in (tasks or []) if t.strip()]
    if clean:
        missions = "\n".join(f"{i}. {t}" for i, t in enumerate(clean, 1))
        mission_text = "段落任务蓝图（按此映射逐段回译）：\n" + missions
    else:
        mission_text = "段落任务蓝图：无固定蓝图，请按英文成稿的自然段落逐段回译。"
    return mission_text + "\n\n英文成稿如下：\n\n" + en_post.strip()


# ---------------------------------------------------------------------------
# 结构化解析
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"```\s*$", "", t)
    return t.strip()


def _parse_gloss_blocks(part: str) -> list[dict]:
    """解析 【第N段｜任务】 + 中文对照 块。容错：无头时整段单块。"""
    blocks: list[dict] = []
    chunks = re.split(r"(?=【第\s*\d+\s*段)", part or "")
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"【第\s*(\d+)\s*段(?:[｜|]([^】]*))?】\s*(.*)", chunk, re.DOTALL)
        if not m:
            continue
        num = int(m.group(1))
        mission = (m.group(2) or "").strip()
        body = (m.group(3) or "").strip()
        zh_m = re.search(r"(?:中文对照|中文回译|回译)\s*[：:]\s*([\s\S]*)", body)
        zh = zh_m.group(1).strip() if zh_m else body
        blocks.append({"paragraph": num, "mission": mission, "zh": zh})
    if not blocks and (part or "").strip():
        blocks.append({"paragraph": 1, "mission": "", "zh": (part or "").strip()})
    return blocks


def parse_mode_b_gloss(raw: str) -> tuple[list[dict], str]:
    """解析 ===ZH_GLOSS=== / ===ANALYSIS=== 结构化输出。"""
    text = (raw or "").strip()
    gloss: list[dict] = []
    analysis = ""
    if _ZH_GLOSS_MARKER in text:
        after = text.split(_ZH_GLOSS_MARKER, 1)[1]
        if _ANALYSIS_MARKER in after:
            gloss_part, analysis_part = after.split(_ANALYSIS_MARKER, 1)
            analysis = _strip_fences(analysis_part)
        else:
            gloss_part = after
        gloss = _parse_gloss_blocks(gloss_part)
    else:
        # 容错：无标记时把整段当作单段对照
        gloss = _parse_gloss_blocks(text)
    return gloss, analysis


def parse_mode_a_output(raw: str) -> str:
    """解析 Mode A 中文成稿（允许可选 ===ZH_POST=== 标记包裹）。"""
    text = _strip_fences(raw)
    if "===ZH_POST===" in text:
        return text.split("===ZH_POST===", 1)[1].strip()
    return text


# ---------------------------------------------------------------------------
# 编排器：段落蓝图 + 双轨输出
# ---------------------------------------------------------------------------

def run_structure_pipeline(
    *,
    source: str,
    config: GenerationConfig,
    paragraph_tasks: list[str],
    blueprint_name: str,
    rag_block: str = "",
    call_llm: CallLlmFn,
    api_key: str,
    base_url: str,
    model: str,
    build_stage1_prompt: Callable[[GenerationConfig], str],
    build_stage2_prompt: Callable[[GenerationConfig], str],
    log: Optional[Callable[[str], None]] = None,
) -> StructureResult:
    """编排段落蓝图与双轨输出模式，返回统一 StructureResult。

    ``paragraph_tasks`` / ``blueprint_name`` 应为调用方（UI）已解析好的最终蓝图，
    避免与缓存键产生二次解析偏差。
    """

    def _log(msg: str) -> None:
        if log:
            log(msg)

    tasks = [t.strip() for t in (paragraph_tasks or []) if t.strip()]
    # 将最终蓝图写回 config，确保下游 Prompt 与缓存键一致
    config.paragraph_blueprint = list(tasks)
    if tasks:
        config.paragraph_count = len(tasks)

    if config.output_mode == "B_en_rewrite_zh_gloss":
        return _run_mode_b(
            source=source,
            config=config,
            tasks=tasks,
            blueprint_name=blueprint_name,
            call_llm=call_llm,
            api_key=api_key,
            base_url=base_url,
            model=model,
            build_stage1_prompt=build_stage1_prompt,
            build_stage2_prompt=build_stage2_prompt,
            log=_log,
        )
    return _run_mode_a(
        source=source,
        config=config,
        tasks=tasks,
        blueprint_name=blueprint_name,
        rag_block=rag_block,
        call_llm=call_llm,
        api_key=api_key,
        base_url=base_url,
        model=model,
        log=_log,
    )


def _run_mode_b(
    *,
    source: str,
    config: GenerationConfig,
    tasks: list[str],
    blueprint_name: str,
    call_llm: CallLlmFn,
    api_key: str,
    base_url: str,
    model: str,
    build_stage1_prompt: Callable[[GenerationConfig], str],
    build_stage2_prompt: Callable[[GenerationConfig], str],
    log: Optional[Callable[[str], None]] = None,
) -> StructureResult:
    """Mode B 旗舰：英文洗稿（含去 AI 味）+ 逐段中文对照 + 解析。"""

    def _log(msg: str) -> None:
        if log:
            log(msg)

    _log("🧩 `[INFO] Mode B 旗舰：英文洗稿（Stage1 + 去 AI 味）…`")
    humanizer_result = run_humanizer_pipeline(
        source=source,
        config=config,
        call_llm=call_llm,
        api_key=api_key,
        base_url=base_url,
        model=model,
        build_stage1_prompt=build_stage1_prompt,
        build_stage2_prompt=build_stage2_prompt,
        log=log,
    )
    en_post = humanizer_result.final

    _log("🀄 `[INFO] 生成逐段中文对照回译 + 观点/俚语解析…`")
    gloss_raw = call_llm(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=min(1.0, config.temperature + 0.05),
        system_prompt=build_mode_b_gloss_system_prompt(config, tasks),
        user_prompt=build_mode_b_gloss_user_prompt(tasks, en_post),
    )
    zh_gloss, analysis = parse_mode_b_gloss(gloss_raw)

    return StructureResult(
        mode=config.output_mode,
        final=en_post,
        en_post=en_post,
        zh_post="",
        zh_gloss=zh_gloss,
        analysis=analysis,
        paragraph_tasks=list(tasks),
        blueprint_name=blueprint_name,
        stage1=humanizer_result.stage1,
        notes=humanizer_result.notes,
        two_phase=humanizer_result.two_phase,
        intensity=humanizer_result.intensity,
        retried=humanizer_result.retried,
        scrub_notes=list(humanizer_result.scrub_notes),
        quality=humanizer_result.quality.to_dict(),
        metrics=humanizer_result.metrics.to_dict(),
    )


def _run_mode_a(
    *,
    source: str,
    config: GenerationConfig,
    tasks: list[str],
    blueprint_name: str,
    rag_block: str,
    call_llm: CallLlmFn,
    api_key: str,
    base_url: str,
    model: str,
    log: Optional[Callable[[str], None]] = None,
) -> StructureResult:
    """Mode A 跨文化汉化：英文原帖 -> 地道中文社媒爆款帖。"""

    def _log(msg: str) -> None:
        if log:
            log(msg)

    _log("🧩 `[INFO] Mode A 跨文化汉化：英文原帖 → 中文社媒爆款帖…`")
    system_prompt = build_mode_a_system_prompt(config, rag_block, tasks)
    zh_raw = call_llm(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=config.temperature,
        system_prompt=system_prompt,
        user_prompt=(
            "以下是抓取并清洗后的英文原帖（含高赞热评），请据此输出地道中文社媒爆款帖：\n\n"
            + source.strip()
        ),
    )
    zh_post = parse_mode_a_output(zh_raw)

    metrics = compute_objective_metrics(zh_post, list(config.blacklist_words))
    quality = heuristic_quality_scores(zh_post, config, metrics)

    return StructureResult(
        mode=config.output_mode,
        final=zh_post,
        en_post="",
        zh_post=zh_post,
        zh_gloss=[],
        analysis="",
        paragraph_tasks=list(tasks),
        blueprint_name=blueprint_name,
        stage1=zh_post,
        notes="",
        two_phase=False,
        intensity=config.resolved_de_ai_intensity(),
        retried=False,
        scrub_notes=[],
        quality=quality.to_dict(),
        metrics=metrics.to_dict(),
    )
