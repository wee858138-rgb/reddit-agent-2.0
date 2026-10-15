# -*- coding: utf-8 -*-
"""生成控制参数契约中枢。

所有生成相关函数必须接收 ``GenerationConfig``，禁止在函数间散乱传参。
字段命名以 v2.0 为准；通过 ``@property`` 保留初代别名兼容。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


OutputMode = Literal["A_zh_social", "B_en_rewrite_zh_gloss"]
DeAiIntensity = Literal["off", "weak", "medium", "strong"]


@dataclass
class GenerationConfig:
    """统一收纳初代受保护参数与二代扩展参数。"""

    # —— 初代必保 ——
    temperature: float = 0.8
    tone: str = "客观中立的行业观察者"
    injected_viewpoint: str = ""
    stance: str = "客观中立"
    target_audience: list[str] = field(default_factory=lambda: ["GenZ"])

    # —— 初代已有、一并契约化 ——
    subreddit: str = "r/technology"
    paragraph_count: int = 3
    enable_de_ai: bool = False
    blacklist_words: list[str] = field(default_factory=list)

    # —— 二代新增（后续步骤启用；本步先落契约）——
    output_mode: OutputMode = "B_en_rewrite_zh_gloss"
    paragraph_blueprint: list[str] = field(default_factory=list)
    de_ai_intensity: DeAiIntensity = "off"
    enable_rag: bool = False

    # ------------------------------------------------------------------
    # 初代字段别名（只读兼容）
    # ------------------------------------------------------------------

    @property
    def persona(self) -> str:
        """初代别名：数字人格 / 语气 → tone。"""
        return self.tone

    @property
    def sentiment(self) -> str:
        """初代别名：情感倾向 → stance。"""
        return self.stance

    @property
    def viewpoint(self) -> str:
        """初代别名：核心观点 → injected_viewpoint。"""
        return self.injected_viewpoint

    @property
    def audiences(self) -> list[str]:
        """初代别名：目标受众 → target_audience。"""
        return self.target_audience

    # ------------------------------------------------------------------
    # 派生辅助
    # ------------------------------------------------------------------

    def resolved_de_ai_intensity(self) -> DeAiIntensity:
        """合并初代开关与二代强度滑块。

        - 若 ``de_ai_intensity`` 非 off，以其为准；
        - 否则若 ``enable_de_ai`` 为 True，视为 medium（兼容初代勾选行为）。
        """
        if self.de_ai_intensity != "off":
            return self.de_ai_intensity
        if self.enable_de_ai:
            return "medium"
        return "off"

    def wants_de_ai(self) -> bool:
        """是否执行对抗性去 AI 味节点。"""
        return self.resolved_de_ai_intensity() != "off"

    def audience_label(self) -> str:
        """供 Prompt / UI 展示的受众文案。"""
        return "、".join(self.target_audience) if self.target_audience else "general Reddit readers"

    def blacklist_label(self) -> str:
        """供 Prompt 使用的禁用词列表文案。"""
        if not self.blacklist_words:
            return "(none)"
        return ", ".join(self.blacklist_words)

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict（不含 property 别名）。"""
        return asdict(self)

    def to_cache_dict(self) -> dict[str, Any]:
        """供缓存哈希使用的确定性序列化（不含 API Key）。

        - 列表字段排序以保证同一配置得到同一 hash
        - ``de_ai_intensity`` / ``enable_de_ai`` 归一化为 resolved 结果
        """
        intensity = self.resolved_de_ai_intensity()
        return {
            "temperature": round(float(self.temperature), 4),
            "tone": self.tone,
            "injected_viewpoint": self.injected_viewpoint.strip(),
            "stance": self.stance,
            "target_audience": sorted(self.target_audience),
            "subreddit": self.subreddit,
            "paragraph_count": int(self.paragraph_count),
            "blacklist_words": sorted(w.strip() for w in self.blacklist_words if w.strip()),
            "output_mode": self.output_mode,
            "paragraph_blueprint": list(self.paragraph_blueprint),
            "de_ai_intensity": intensity,
            "enable_de_ai": intensity != "off",
            "enable_rag": bool(self.enable_rag),
        }

    def prompt_control_block(self) -> str:
        """将 stance / target_audience / blacklist 拼成可注入 Prompt 的控制块。

        初代缺口补齐：这三个参数必须真正进入生成指令，而非仅停留在 UI。
        """
        audience = self.audience_label()
        blacklist = self.blacklist_label()
        lines = [
            "STANCE / SENTIMENT:",
            f"  {self.stance}",
            "  Embody this emotional posture throughout the post without naming it"
            " explicitly or turning it into a slogan.",
            "",
            "TARGET AUDIENCE:",
            f"  {audience}",
            "  Calibrate slang, formality, references and assumed shared knowledge"
            " for these readers.",
            "",
            "BLACKLIST (never use these words or close paraphrases):",
            f"  {blacklist}",
        ]
        if self.blacklist_words:
            lines.append(
                "  If a blacklisted term would otherwise appear, rewrite around it."
            )
        return "\n".join(lines)
