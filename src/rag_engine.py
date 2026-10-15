# -*- coding: utf-8 -*-
"""轻量混合检索 RAG：JSON 词库 + 关键词/版块打分 + Prompt 注入块。"""

from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.cache_manager import CacheManager, get_cache_manager
from src.data_cleaner import _denoise_snippets

try:  # 可选：ChromaDB 本地向量库（未装则回退纯 Python TF-IDF）
    import chromadb
    _HAS_CHROMADB = True
except Exception:  # noqa: BLE001
    chromadb = None
    _HAS_CHROMADB = False

_DEFAULT_LEXICON = Path(__file__).resolve().parent.parent / "data" / "rag_lexicon.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class LexiconEntry:
    term: str
    subreddits: list[str] = field(default_factory=lambda: ["all"])
    en_meaning: str = ""
    zh_equivalent: str = ""
    example: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LexiconEntry":
        subs = data.get("subreddits") or ["all"]
        if isinstance(subs, str):
            subs = [s.strip() for s in re.split(r"[,，;；\s]+", subs) if s.strip()]
        return cls(
            term=str(data.get("term", "")).strip(),
            subreddits=[str(s).strip() for s in subs if str(s).strip()],
            en_meaning=str(data.get("en_meaning", "")).strip(),
            zh_equivalent=str(data.get("zh_equivalent", "")).strip(),
            example=str(data.get("example", "")).strip(),
        )


@dataclass
class ScoredEntry:
    entry: LexiconEntry
    score: float
    reasons: list[str] = field(default_factory=list)


class TfidfIndex:
    """纯 Python TF-IDF 余弦检索（免 numpy/chromadb，本地与云端零报错）。"""

    def __init__(self) -> None:
        self.docs: list[str] = []
        self._idf: dict[str, float] = {}
        self._vecs: list[dict[str, float]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = (text or "").lower()
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
        cjk = re.findall(r"[\u4e00-\u9fff]", text)
        for i in range(len(cjk) - 1):
            tokens.append(cjk[i] + cjk[i + 1])
        return [t for t in tokens if len(t) >= 2]

    def fit(self, docs: list[str]) -> None:
        self.docs = [d for d in docs if d and d.strip()]
        n = max(len(self.docs), 1)
        df: dict[str, int] = {}
        for d in self.docs:
            for term in set(self._tokenize(d)):
                df[term] = df.get(term, 0) + 1
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._vecs = [self._vectorize(d) for d in self.docs]

    def _vectorize(self, text: str) -> dict[str, float]:
        tf: dict[str, int] = {}
        for t in self._tokenize(text):
            tf[t] = tf.get(t, 0) + 1
        vec: dict[str, float] = {}
        norm = 0.0
        for t, c in tf.items():
            w = (1.0 + math.log(c)) * self._idf.get(t, 1.0)
            vec[t] = w
            norm += w * w
        norm = math.sqrt(norm) or 1.0
        return {t: w / norm for t, w in vec.items()}

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, str]]:
        if not self.docs:
            return []
        qv = self._vectorize(query)
        scored: list[tuple[float, str]] = []
        for i, v in enumerate(self._vecs):
            sim = sum(w * v[t] for t, w in qv.items() if t in v)
            scored.append((sim, self.docs[i]))
        scored.sort(key=lambda x: -x[0])
        return [(s, d) for s, d in scored[:top_k] if s > 0.0]


class RagEngine:
    """基于 ``data/rag_lexicon.json`` 的混合检索与持久化。"""

    def __init__(
        self,
        lexicon_path: Optional[Path | str] = None,
        cache: Optional[CacheManager] = None,
    ) -> None:
        self.path = Path(lexicon_path) if lexicon_path else _DEFAULT_LEXICON
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.version = 1
        self.updated_at = ""
        self.entries: list[LexiconEntry] = []
        self._cache = cache if cache is not None else get_cache_manager()
        self._index_cache: dict[str, TfidfIndex] = {}
        self.reload()

    def reload(self) -> None:
        with self._lock:
            if not self.path.exists():
                self.entries = []
                self.version = 1
                self.updated_at = _utc_now_iso()
                self._save_unlocked()
                return
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.version = int(data.get("version", 1) or 1)
            self.updated_at = str(data.get("updated_at") or "")
            self.entries = [
                LexiconEntry.from_dict(e) for e in (data.get("entries") or [])
            ]

    def _save_unlocked(self) -> None:
        payload = {
            "version": self.version,
            "updated_at": _utc_now_iso(),
            "entries": [e.to_dict() for e in self.entries],
        }
        self.updated_at = payload["updated_at"]
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def save(self) -> None:
        with self._lock:
            self._save_unlocked()

    def list_entries(self) -> list[LexiconEntry]:
        with self._lock:
            return list(self.entries)

    def add_entry(
        self,
        term: str,
        *,
        subreddits: Optional[list[str]] = None,
        en_meaning: str = "",
        zh_equivalent: str = "",
        example: str = "",
        upsert: bool = True,
    ) -> LexiconEntry:
        entry = LexiconEntry(
            term=term.strip(),
            subreddits=subreddits or ["all"],
            en_meaning=en_meaning.strip(),
            zh_equivalent=zh_equivalent.strip(),
            example=example.strip(),
        )
        if not entry.term:
            raise ValueError("term 不能为空")
        with self._lock:
            idx = next(
                (
                    i
                    for i, e in enumerate(self.entries)
                    if e.term.lower() == entry.term.lower()
                ),
                None,
            )
            if idx is not None:
                if not upsert:
                    raise ValueError(f"词条已存在：{entry.term}")
                self.entries[idx] = entry
            else:
                self.entries.append(entry)
            self._save_unlocked()
        return entry

    # ------------------------------------------------------------------
    # 混合检索
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_sub(subreddit: str) -> str:
        s = (subreddit or "").strip()
        if not s:
            return ""
        if s.lower() == "all":
            return "all"
        return s if s.startswith("r/") else f"r/{s}"

    def score_entry(
        self,
        entry: LexiconEntry,
        source_text: str,
        subreddit: str,
    ) -> ScoredEntry:
        text_l = (source_text or "").lower()
        term_l = entry.term.lower()
        score = 0.0
        reasons: list[str] = []

        # 关键词命中（整词 / 短语）
        if " " in term_l or "-" in term_l or ";" in term_l:
            if term_l in text_l:
                score += 5.0
                reasons.append("短语命中原文")
        else:
            if re.search(rf"\b{re.escape(term_l)}\b", text_l, flags=re.IGNORECASE):
                score += 5.0
                reasons.append("关键词命中原文")
            elif term_l in text_l:
                score += 2.5
                reasons.append("子串弱命中")

        # 释义/中文对应词回声（弱信号）
        for frag in (entry.en_meaning, entry.zh_equivalent):
            tokens = re.findall(r"[A-Za-z]{4,}|[\u4e00-\u9fff]{2,}", frag)
            hits = sum(1 for t in tokens[:6] if t.lower() in text_l)
            if hits:
                score += min(2.0, hits * 0.4)
                reasons.append(f"释义词回声×{hits}")

        # 版块关联
        target = self._norm_sub(subreddit)
        entry_subs = [self._norm_sub(s) for s in entry.subreddits] or ["all"]
        if "all" in entry_subs:
            score += 1.0
            reasons.append("通用词条")
        if target and target in entry_subs:
            score += 3.0
            reasons.append(f"匹配版块 {target}")
        elif target:
            # 同主题弱相关：programming ↔ technology
            soft_pairs = {
                "r/programming": {"r/technology"},
                "r/technology": {"r/programming"},
            }
            if soft_pairs.get(target, set()) & set(entry_subs):
                score += 1.2
                reasons.append("近邻版块弱相关")

        # 极客梗在技术版块略加权
        geek_terms = {
            "bike-shedding",
            "rubber ducking",
            "skill issue",
            "lgtm",
            "yak shaving",
        }
        if term_l in geek_terms and target in ("r/programming", "r/technology"):
            score += 0.8
            reasons.append("极客梗×技术版")

        return ScoredEntry(entry=entry, score=score, reasons=reasons)

    def retrieve(
        self,
        source_text: str,
        subreddit: str,
        top_k: int = 5,
        *,
        min_score: float = 1.0,
    ) -> list[ScoredEntry]:
        with self._lock:
            scored = [
                self.score_entry(e, source_text, subreddit) for e in self.entries
            ]
        scored = [s for s in scored if s.score >= min_score]
        scored.sort(key=lambda s: (-s.score, s.entry.term.lower()))
        # 若原文几乎没命中，仍按版块相关性回填通用热词
        if len(scored) < top_k:
            with self._lock:
                fallback = [
                    self.score_entry(e, "", subreddit) for e in self.entries
                ]
            fallback.sort(key=lambda s: (-s.score, s.entry.term.lower()))
            seen = {s.entry.term.lower() for s in scored}
            for s in fallback:
                if s.entry.term.lower() in seen:
                    continue
                if s.score <= 0:
                    continue
                scored.append(s)
                seen.add(s.entry.term.lower())
                if len(scored) >= top_k:
                    break
        return scored[:top_k]

    def retrieve_context(
        self,
        source_text: str,
        subreddit: str,
        top_k: int = 5,
        *,
        selected_syntax: Optional[list[str]] = None,
        selected_slang: Optional[list[str]] = None,
        selected_topics: Optional[list[str]] = None,
        corpus_key: Optional[str] = None,
    ) -> str:
        """生成可注入 Stage1/Stage2/ModeA/B 的完整 RAG 块（静态词典 + 用户勾选动态语料）。"""
        return self.retrieve_full_context(
            source_text,
            subreddit,
            top_k=top_k,
            selected_syntax=selected_syntax,
            selected_slang=selected_slang,
            selected_topics=selected_topics,
            corpus_key=corpus_key,
        )

    # ------------------------------------------------------------------
    # 动态热帖语料 RAG（Living Context）
    # ------------------------------------------------------------------

    def get_living_context(
        self, subreddit: str, corpus_key: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """读取目标版块的当日热帖语料池（含 24h fresh 标记）。"""
        return self._cache.get_subreddit_corpus(corpus_key or subreddit)

    def _living_index(self, subreddit: str, corpus: dict[str, Any]) -> Optional[TfidfIndex]:
        idx = self._index_cache.get(subreddit)
        if idx is not None:
            return idx
        snippets = _denoise_snippets(corpus.get("cleaned_corpus") or [])
        if not snippets:
            return None
        idx = TfidfIndex()
        idx.fit(snippets)
        self._index_cache[subreddit] = idx
        return idx

    def search_living(
        self, query: str, subreddit: str, top_k: int = 3
    ) -> list[dict[str, Any]]:
        corpus = self.get_living_context(subreddit)
        if not corpus or not corpus.get("fresh"):
            return []
        idx = self._living_index(subreddit, corpus)
        if idx is None:
            return []
        return [
            {"score": round(s, 4), "text": d}
            for s, d in idx.search(query, top_k=top_k)
        ]

    def retrieve_living_context(
        self, source_text: str, subreddit: str, top_k: int = 3
    ) -> dict[str, Any]:
        corpus = self.get_living_context(subreddit)
        if not corpus or not corpus.get("fresh"):
            return {"available": False, "snippets": [], "style": {}}
        hits = self.search_living(source_text, subreddit, top_k=top_k)
        return {
            "available": True,
            "snippets": hits,
            "style": corpus.get("extracted_style") or {},
            "fetched_at": corpus.get("fetched_at") or "",
            "age_seconds": corpus.get("age_seconds", 0),
        }

    def retrieve_full_context(
        self,
        source_text: str,
        subreddit: str,
        top_k: int = 5,
        *,
        selected_syntax: Optional[list[str]] = None,
        selected_slang: Optional[list[str]] = None,
        selected_topics: Optional[list[str]] = None,
        corpus_key: Optional[str] = None,
    ) -> str:
        """组装 RAG 注入块：静态词典命中 + 用户勾选的句法/黑话/热点（去臃肿、防跑题）。"""
        parts: list[str] = []

        corpus = self.get_living_context(subreddit, corpus_key=corpus_key)
        style: dict[str, Any] = {}
        if corpus and corpus.get("fresh"):
            style = corpus.get("extracted_style") or {}

        def _clean(lst):
            return [str(x).strip() for x in (lst or []) if x and str(x).strip()]

        if selected_topics is not None:
            topics = _clean(selected_topics)
        else:
            topics = style.get("trending_topics") or []
        if selected_slang is not None:
            slang_terms = _clean(selected_slang)
        else:
            slang_terms = [
                str(s.get("term", "")).strip()
                for s in (style.get("living_slang") or [])
                if isinstance(s, dict) and str(s.get("term", "")).strip()
            ]
        if selected_syntax is not None:
            syntax = _clean(selected_syntax)
        else:
            syntax = style.get("syntax_patterns") or []

        living_lines: list[str] = []
        if syntax:
            living_lines.append("[句法/钩子骨架（可迁移句式，仅模仿节奏，勿照抄事实）]")
            living_lines.extend(f"- {p}" for p in syntax)
        if slang_terms:
            slang_map = {
                str(s.get("term", "")): str(s.get("context", ""))
                for s in (style.get("living_slang") or [])
                if isinstance(s, dict)
            }
            living_lines.append("[鲜活黑话/俚语（仅作语感染色，语境不符则忽略）]")
            for t in slang_terms:
                ctx = slang_map.get(t, "")
                living_lines.append(f"- {t}" + (f" —— {ctx}" if ctx else ""))
        if topics:
            living_lines.append("[今日热点话题（按需借势，仅当与素材自然相关时顺带提及）]")
            living_lines.extend(f"- {t}" for t in topics)

        if living_lines:
            living_lines.insert(0, "LIVING_SUBREDDIT_RAG_CONTEXT:")
            living_lines.append("")
            parts.append("\n".join(living_lines))

        parts.append(self._static_context(source_text, subreddit, top_k=top_k))

        parts.append(
            "RAG_GUARDRAIL:\n"
            "  Treat the RAG corpus above as a tone/style dye ONLY. If a term, topic or "
            "  pattern does not fit the source material's context, ignore it outright. "
            "  Always obey the source material's facts and the core viewpoint; never let "
            "  RAG content override or derail the post."
        )

        return "\n".join(p for p in parts if p).rstrip()

    def _static_context(self, source_text: str, subreddit: str, top_k: int = 5) -> str:
        """静态基础黑话词典命中块。"""
        hits = self.retrieve(source_text, subreddit, top_k=top_k)
        if not hits:
            return (
                "RAG_LEXICON_CONTEXT:\n"
                "  (no lexicon hits — write natural Reddit vernacular without forced slang.)"
            )
        lines = [
            "RAG_LEXICON_CONTEXT:",
            "Use these Reddit-native terms ONLY where they fit naturally;"
            " never stuff slang that does not match the post's tone.",
            "",
        ]
        for i, hit in enumerate(hits, 1):
            e = hit.entry
            lines.append(f"{i}. TERM: {e.term}  (score={hit.score:.1f})")
            lines.append(f"   EN: {e.en_meaning}")
            lines.append(f"   ZH vibe: {e.zh_equivalent}")
            if e.example:
                lines.append(f"   Example: {e.example}")
            lines.append(f"   Subs: {', '.join(e.subreddits)}")
            lines.append("")
        return "\n".join(lines).rstrip()


_GLOBAL: Optional[RagEngine] = None
_GLOBAL_LOCK = threading.Lock()


def get_rag_engine(lexicon_path: Optional[Path | str] = None) -> RagEngine:
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is None:
            _GLOBAL = RagEngine(lexicon_path=lexicon_path)
        return _GLOBAL
