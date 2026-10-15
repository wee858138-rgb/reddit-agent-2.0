# -*- coding: utf-8 -*-
"""SQLite 本地缓存池：原帖预留表 + 生成结果参数哈希缓存。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.schemas import GenerationConfig

# 默认落在项目根目录 data/cache.db
_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "cache.db"

# 版块热帖语料池 24 小时缓存 TTL（秒）
CORPUS_TTL_SECONDS = 24 * 3600


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_source(source: str) -> str:
    """标准化素材文本，保证哈希稳定。"""
    text = (source or "").replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def compute_source_hash(source: str) -> str:
    normalized = normalize_source(source)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_param_hash(source: str, config: GenerationConfig) -> str:
    """param_hash = sha256(标准化素材 + GenerationConfig 排序序列化 JSON)。

    不含 API Key（Key 不在 GenerationConfig 中）。
    """
    payload = {
        "source": normalize_source(source),
        "config": config.to_cache_dict(),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CacheManager:
    """线程安全的 SQLite 缓存管理器（WAL + check_same_thread=False）。"""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS reddit_posts (
                post_id      TEXT PRIMARY KEY,
                url          TEXT NOT NULL,
                raw_json     TEXT,
                cleaned_json TEXT,
                fetched_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS generations (
                param_hash   TEXT PRIMARY KEY,
                source_hash  TEXT NOT NULL,
                config_json  TEXT NOT NULL,
                output_json  TEXT NOT NULL,
                created_at   TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_generations_source
                ON generations(source_hash);

            CREATE TABLE IF NOT EXISTS subreddit_corpus (
                subreddit            TEXT PRIMARY KEY,
                raw_posts_json       TEXT,
                cleaned_corpus_json  TEXT,
                extracted_style_json TEXT,
                fetched_at           TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # generations
    # ------------------------------------------------------------------

    def get_generation(self, param_hash: str) -> Optional[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT param_hash, source_hash, config_json, output_json, created_at "
                "FROM generations WHERE param_hash = ?",
                (param_hash,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "param_hash": row["param_hash"],
            "source_hash": row["source_hash"],
            "config": json.loads(row["config_json"]),
            "output": json.loads(row["output_json"]),
            "created_at": row["created_at"],
        }

    def put_generation(
        self,
        param_hash: str,
        source_hash: str,
        config: GenerationConfig | dict[str, Any],
        output: dict[str, Any],
    ) -> None:
        if isinstance(config, GenerationConfig):
            config_payload = config.to_cache_dict()
        else:
            config_payload = config
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO generations
                    (param_hash, source_hash, config_json, output_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(param_hash) DO UPDATE SET
                    source_hash = excluded.source_hash,
                    config_json = excluded.config_json,
                    output_json = excluded.output_json,
                    created_at  = excluded.created_at
                """,
                (
                    param_hash,
                    source_hash,
                    json.dumps(config_payload, ensure_ascii=False),
                    json.dumps(output, ensure_ascii=False),
                    _utc_now_iso(),
                ),
            )
            self._conn.commit()

    def lookup_or_none(
        self,
        source: str,
        config: GenerationConfig,
        *,
        bypass: bool = False,
    ) -> tuple[str, str, Optional[dict[str, Any]]]:
        """返回 (param_hash, source_hash, cached_output_or_None)。"""
        param_hash = compute_param_hash(source, config)
        source_hash = compute_source_hash(source)
        if bypass:
            return param_hash, source_hash, None
        hit = self.get_generation(param_hash)
        if hit is None:
            return param_hash, source_hash, None
        return param_hash, source_hash, hit["output"]

    # ------------------------------------------------------------------
    # reddit_posts（第 ⑤⑥ 步抓取时使用，此处先预留）
    # ------------------------------------------------------------------

    def get_reddit_post(self, post_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT post_id, url, raw_json, cleaned_json, fetched_at "
                "FROM reddit_posts WHERE post_id = ?",
                (post_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "post_id": row["post_id"],
            "url": row["url"],
            "raw": json.loads(row["raw_json"]) if row["raw_json"] else None,
            "cleaned": json.loads(row["cleaned_json"]) if row["cleaned_json"] else None,
            "fetched_at": row["fetched_at"],
        }

    def put_reddit_post(
        self,
        post_id: str,
        url: str,
        raw: Any = None,
        cleaned: Any = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO reddit_posts
                    (post_id, url, raw_json, cleaned_json, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(post_id) DO UPDATE SET
                    url = excluded.url,
                    raw_json = excluded.raw_json,
                    cleaned_json = excluded.cleaned_json,
                    fetched_at = excluded.fetched_at
                """,
                (
                    post_id,
                    url,
                    json.dumps(raw, ensure_ascii=False) if raw is not None else None,
                    json.dumps(cleaned, ensure_ascii=False) if cleaned is not None else None,
                    _utc_now_iso(),
                ),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # subreddit_corpus（版块热帖语料池，24h 缓存）
    # ------------------------------------------------------------------

    def get_subreddit_corpus(self, subreddit: str) -> Optional[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT subreddit, raw_posts_json, cleaned_corpus_json, "
                "extracted_style_json, fetched_at "
                "FROM subreddit_corpus WHERE subreddit = ?",
                (subreddit,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        try:
            fetched = datetime.fromisoformat(row["fetched_at"])
            age = (datetime.now(timezone.utc) - fetched).total_seconds()
        except (TypeError, ValueError):
            age = CORPUS_TTL_SECONDS  # 解析失败视为已过期
        return {
            "subreddit": row["subreddit"],
            "raw_posts": json.loads(row["raw_posts_json"]) if row["raw_posts_json"] else [],
            "cleaned_corpus": json.loads(row["cleaned_corpus_json"]) if row["cleaned_corpus_json"] else [],
            "extracted_style": json.loads(row["extracted_style_json"]) if row["extracted_style_json"] else {},
            "fetched_at": row["fetched_at"],
            "fresh": age < CORPUS_TTL_SECONDS,
            "age_seconds": int(age),
        }

    def put_subreddit_corpus(
        self,
        subreddit: str,
        raw_posts: Any,
        cleaned_corpus: Any,
        extracted_style: Any,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO subreddit_corpus
                    (subreddit, raw_posts_json, cleaned_corpus_json, extracted_style_json, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(subreddit) DO UPDATE SET
                    raw_posts_json = excluded.raw_posts_json,
                    cleaned_corpus_json = excluded.cleaned_corpus_json,
                    extracted_style_json = excluded.extracted_style_json,
                    fetched_at = excluded.fetched_at
                """,
                (
                    subreddit,
                    json.dumps(raw_posts, ensure_ascii=False),
                    json.dumps(cleaned_corpus, ensure_ascii=False),
                    json.dumps(extracted_style, ensure_ascii=False),
                    _utc_now_iso(),
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# Streamlit 会话级单例辅助
_GLOBAL: Optional[CacheManager] = None
_GLOBAL_LOCK = threading.Lock()


def get_cache_manager(db_path: Optional[Path | str] = None) -> CacheManager:
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is None:
            _GLOBAL = CacheManager(db_path=db_path)
        return _GLOBAL
