# -*- coding: utf-8 -*-
"""Reddit 原帖结构化清洗：零宽字符 / HTML 转义 / 高赞神评提纯。"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


@dataclass
class CleanedRedditPost:
    """清洗后的结构化 Reddit 帖子。"""

    post_id: str
    subreddit: str
    title: str
    body: str
    author: str = ""
    score: int = 0
    upvote_ratio: float = 0.0
    is_nsfw: bool = False
    top_comments: list[str] = field(default_factory=list)
    url: str = ""
    permalink: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CleanedRedditPost":
        return cls(
            post_id=str(data.get("post_id", "")),
            subreddit=str(data.get("subreddit", "")),
            title=str(data.get("title", "")),
            body=str(data.get("body", "")),
            author=str(data.get("author", "")),
            score=int(data.get("score", 0) or 0),
            upvote_ratio=float(data.get("upvote_ratio", 0.0) or 0.0),
            is_nsfw=bool(data.get("is_nsfw", False)),
            top_comments=list(data.get("top_comments") or []),
            url=str(data.get("url", "")),
            permalink=str(data.get("permalink", "")),
        )

    def format_for_llm(self, include_comments: bool = True) -> str:
        """将标题 + 正文 + 精选高赞热评组装成清晰的结构化素材文本。"""
        lines = [
            f"[Subreddit] r/{self.subreddit.lstrip('r/')}",
            f"[Title] {self.title}",
            f"[Author] u/{self.author}" if self.author else "[Author] unknown",
            f"[Score] {self.score} (upvote_ratio={self.upvote_ratio:.2f})",
            f"[NSFW] {'yes' if self.is_nsfw else 'no'}",
            "",
            "[Body]",
            self.body.strip() or "(empty selftext)",
        ]
        if include_comments and self.top_comments:
            lines.append("")
            lines.append("[Top Comments]")
            for i, c in enumerate(self.top_comments, 1):
                lines.append(f"{i}. {c}")
        return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# 清洗规则
# ---------------------------------------------------------------------------

_ZW_RE = re.compile(
    r"(&#x200[bB];|&#8203;|\u200b|\u200c|\u200d|\ufeff|\u00a0)"
)
_TRACKING_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:reddit\.com/clicks?/|out\.reddit\.com/|click\.redditmail\.com/)[^\s)>\]]+",
    re.IGNORECASE,
)
_GENERIC_UTM_RE = re.compile(
    r"(https?://[^\s)>\]]+?)(?:\?[^\s)>\]]*?(?:utm_[^=]+=[^&\s)>\]]+&?)+)",
    re.IGNORECASE,
)
_EDIT_TAIL_RE = re.compile(
    r"(?:\n|^)\s*(?:edit|update|eta)\s*\d*\s*[:：].*$",
    re.IGNORECASE | re.MULTILINE,
)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def clean_text(raw: Optional[str]) -> str:
    """清理零宽字符、HTML 转义、追踪外链与多余空白。"""
    if not raw:
        return ""
    text = str(raw)
    text = _ZW_RE.sub("", text)
    text = html.unescape(text)
    text = text.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    text = _TRACKING_URL_RE.sub("", text)
    # 去掉纯 utm 追踪参数（保留主链接）
    text = _GENERIC_UTM_RE.sub(r"\1", text)
    # 温和清理尾部 Edit 备注（保留正文中的 update 叙述不强删）
    # 仅当行以 Edit:/UPDATE: 开头时剔除
    text = _EDIT_TAIL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def _comment_body(node: dict[str, Any]) -> str:
    data = node.get("data") or {}
    body = clean_text(data.get("body") or "")
    author = str(data.get("author") or "")
    score = int(data.get("score") or 0)
    if not body:
        return ""
    return f"(+{score} by u/{author}) {body}"


def extract_top_comments(
    comments_listing: Any,
    *,
    top_n: int = 5,
) -> list[str]:
    """从 Reddit comments listing 提取高赞真人评论（过滤 AutoMod / deleted）。"""
    candidates: list[tuple[int, str]] = []

    children: list[Any] = []
    if isinstance(comments_listing, dict):
        children = (comments_listing.get("data") or {}).get("children") or []
    elif isinstance(comments_listing, list):
        # 有时传入的是整段 JSON 数组的第二项
        if comments_listing and isinstance(comments_listing[0], dict):
            if comments_listing[0].get("kind") == "Listing":
                children = (comments_listing[0].get("data") or {}).get("children") or []
            else:
                children = comments_listing

    for child in children:
        if not isinstance(child, dict):
            continue
        if child.get("kind") != "t1":
            continue
        data = child.get("data") or {}
        author = str(data.get("author") or "")
        body_raw = str(data.get("body") or "")
        if author.lower() == "automoderator":
            continue
        if author in ("[deleted]", "[removed]") or body_raw.strip() in (
            "[deleted]",
            "[removed]",
        ):
            continue
        body = clean_text(body_raw)
        if not body or len(body) < 2:
            continue
        score = int(data.get("score") or 0)
        candidates.append((score, _comment_body(child)))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in candidates[:top_n]]


def clean_reddit_payload(
    raw_json: Any,
    *,
    post_id: str = "",
    url: str = "",
    top_n_comments: int = 5,
) -> CleanedRedditPost:
    """将 Reddit `.json` 响应清洗为 CleanedRedditPost。"""
    post_data: dict[str, Any] = {}
    comments_listing: Any = None

    if isinstance(raw_json, list) and raw_json:
        # 标准 comments JSON： [post_listing, comments_listing]
        post_listing = raw_json[0] if isinstance(raw_json[0], dict) else {}
        children = (post_listing.get("data") or {}).get("children") or []
        if children and isinstance(children[0], dict):
            post_data = children[0].get("data") or {}
        if len(raw_json) > 1:
            comments_listing = raw_json[1]
    elif isinstance(raw_json, dict):
        # 单帖 info 或已是 post data
        if "title" in raw_json and "selftext" in raw_json:
            post_data = raw_json
        else:
            children = (raw_json.get("data") or {}).get("children") or []
            if children and isinstance(children[0], dict):
                post_data = children[0].get("data") or {}

    pid = post_id or str(post_data.get("id") or "")
    sub = str(post_data.get("subreddit") or "")
    if sub and not sub.startswith("r/"):
        sub_display = sub
    else:
        sub_display = sub.lstrip("r/")

    permalink = str(post_data.get("permalink") or "")
    full_url = url or (
        f"https://www.reddit.com{permalink}" if permalink else ""
    )

    top_comments = extract_top_comments(comments_listing, top_n=top_n_comments)

    return CleanedRedditPost(
        post_id=pid,
        subreddit=sub_display,
        title=clean_text(post_data.get("title") or ""),
        body=clean_text(post_data.get("selftext") or ""),
        author=str(post_data.get("author") or ""),
        score=int(post_data.get("score") or 0),
        upvote_ratio=float(post_data.get("upvote_ratio") or 0.0),
        is_nsfw=bool(post_data.get("over_18") or post_data.get("nsfw") or False),
        top_comments=top_comments,
        url=full_url,
        permalink=permalink,
    )


# ---------------------------------------------------------------------------
# 动态活体语料：清洗与风格提纯节点（Living Context）
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+")
_BAD_LINE_RE = re.compile(r"^\s*(?:edit|update|eta)\b", re.IGNORECASE)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "while",
    "with", "without", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "been", "being", "am", "have", "has", "had", "do", "does",
    "did", "will", "would", "can", "could", "should", "may", "might", "must",
    "to", "of", "in", "on", "at", "by", "from", "up", "down", "out", "over",
    "under", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "it", "its", "as", "we", "you", "he", "she", "they", "them",
    "his", "her", "their", "my", "your", "our", "i", "me", "what", "which",
    "who", "whom", "when", "where", "why", "how", "get", "got", "like",
    "really", "also", "still", "even", "just", "know", "think", "make", "made",
    "way", "one", "two", "new", "good", "bad", "right", "left", "use", "using",
    "used", "go", "going", "want", "need", "see", "s", "t", "don", "now",
}


def _clean_snippet(raw: str) -> str:
    if not raw:
        return ""
    s = _URL_RE.sub(" ", raw)
    s = _BAD_LINE_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) < 6:
        return ""
    if not re.search(r"[A-Za-z\u4e00-\u9fff]", s):
        return ""
    return s


def _denoise_snippets(corpus: list[Any]) -> list[str]:
    """第一层规则降噪：剔除 URL / AutoModerator / 乱码 / 无意义短评。"""
    out: list[str] = []
    seen: set[str] = set()
    for item in corpus:
        if isinstance(item, dict):
            title = str(item.get("title") or "")
            body = str(item.get("body") or "")
            comments = item.get("comments") or item.get("top_comments") or []
        elif hasattr(item, "to_dict"):
            d = item.to_dict()
            title = str(d.get("title") or "")
            body = str(d.get("body") or "")
            comments = d.get("top_comments") or d.get("comments") or []
        else:
            continue
        for raw in [title, body, *(comments or [])]:
            s = _clean_snippet(raw)
            if not s:
                continue
            key = s[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
    return out


def _local_extract(snippets: list[str]) -> dict[str, Any]:
    """第二层免 Key 兜底：本地高频词 / 句式分析器。"""
    text = " ".join(snippets)
    en_words = [w for w in re.findall(r"[a-z]{2,}", text.lower()) if w not in _STOPWORDS]
    freq = Counter(en_words)

    bigrams = [
        " ".join(pair)
        for pair in zip(en_words, en_words[1:])
        if pair[0] not in _STOPWORDS and pair[1] not in _STOPWORDS
    ]
    bg = Counter(bigrams)
    trending_topics = [t for t, _ in bg.most_common(10) if bg[t] >= 2]

    slang_pat = re.compile(
        r"\b(?:[A-Z]{2,}|[a-z]+[A-Z][a-zA-Z]*|[a-zA-Z0-9]*\d+[a-zA-Z0-9]*)\b"
    )
    slang_counter: Counter = Counter()
    slang_ctx: dict[str, str] = {}
    for sn in snippets:
        for m in slang_pat.findall(sn):
            low = m.lower()
            if len(m) < 3 or low in _STOPWORDS:
                continue
            slang_counter[low] += 1
            slang_ctx.setdefault(low, sn[:160])
    living_slang = []
    for term, c in slang_counter.most_common(12):
        if c < 2:
            break
        living_slang.append({"term": term, "context": slang_ctx[term]})

    hooks: list[str] = []
    for sn in snippets[:10]:
        sn = sn.strip()
        if len(sn) < 12:
            continue
        head = " ".join(sn.split()[:8])
        if head and head not in hooks:
            hooks.append(head)
    syntax_patterns = hooks[:6]

    if not living_slang:
        living_slang = [
            {"term": w, "context": f"当日高频词（出现 {c} 次）"}
            for w, c in freq.most_common(8)
        ]
    if not trending_topics:
        trending_topics = [w for w, _ in freq.most_common(8)]

    return {
        "trending_topics": trending_topics,
        "living_slang": living_slang,
        "syntax_patterns": syntax_patterns,
    }


def extract_living_context(
    corpus: list[Any],
    *,
    subreddit: str = "",
    llm_fn: Optional[Callable[[str], str]] = None,
) -> dict[str, Any]:
    """从当日版块热帖提纯《当日版块活体语料报告》。

    - 第一层（规则降噪）：剔除 URL / AutoModerator / 乱码 / 无意义短评。
    - 第二层（提纯）：若提供 ``llm_fn``（单参 prompt → 文本）则用 LLM 结构化提纯，
      否则用本地高频词/句式分析器兜底（免 Key 零报错）。
    """
    snippets = _denoise_snippets(corpus)
    if not snippets:
        return {"trending_topics": [], "living_slang": [], "syntax_patterns": []}
    if llm_fn is not None:
        try:
            return _llm_extract(llm_fn, snippets, subreddit)
        except Exception:  # noqa: BLE001
            pass
    return _local_extract(snippets)


def _llm_extract(
    llm_fn: Callable[[str], str], snippets: list[str], subreddit: str
) -> dict[str, Any]:
    combined = "\n".join(snippets[:60])[:12000]
    prompt = (
        f"你是 Reddit 版块 r/{subreddit} 的社区语料分析器。以下是该版块今天最热帖子的"
        "标题/正文/高赞评论。请提炼《当日版块活体语料报告》，只输出一个 JSON 对象，键为：\n"
        '  "trending_topics": 字符串数组（今日热议的具体话题/事件，3-6 个，中文表述）\n'
        '  "living_slang": [{"term": "鲜活黑话/缩写", "context": "语境解释（中文）"}]（3-8 个）\n'
        '  "syntax_patterns": 字符串数组（高赞帖子的真实句法结构/开头钩子/吐槽节奏范例，3-6 个）\n'
        "不要输出 JSON 以外的任何内容。\n\n"
        "===== 热帖语料 =====\n" + combined
    )
    raw = llm_fn(prompt)
    return _parse_llm_json(raw)


def _parse_llm_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    fence = re.search(r"\{[\s\S]*\}", text)
    data: dict[str, Any] = {}
    if fence:
        try:
            data = json.loads(fence.group(0))
        except json.JSONDecodeError:
            data = {}
    if not isinstance(data, dict):
        data = {}
    return {
        "trending_topics": [str(x) for x in (data.get("trending_topics") or [])][:8],
        "living_slang": _normalize_slang(data.get("living_slang")),
        "syntax_patterns": [str(x) for x in (data.get("syntax_patterns") or [])][:8],
    }


def _normalize_slang(raw: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in raw or []:
        if isinstance(item, str):
            out.append({"term": item, "context": ""})
        elif isinstance(item, dict):
            out.append(
                {
                    "term": str(item.get("term") or ""),
                    "context": str(item.get("context") or ""),
                }
            )
    return out[:10]
