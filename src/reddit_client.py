# -*- coding: utf-8 -*-
"""合规 Reddit 抓取客户端：多通道反 403 降级 + 版块热帖批量抓取。

通道优先级：
  A. 可选官方 OAuth（侧边栏配置 Client ID / Secret 后启用）
  B. ``curl_cffi`` Chrome TLS 指纹伪装 + 真实浏览器 Header（www / old 的 .json）
  C. Reddit RSS 订阅源兜底（对反爬极宽松，用 ElementTree 解析 Atom）

若 ``curl_cffi`` 未安装则自动降级为普通 ``requests``（仍带浏览器 Header）。
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from src.cache_manager import CacheManager, get_cache_manager
from src.data_cleaner import (
    CleanedRedditPost,
    clean_reddit_payload,
    clean_text,
    extract_top_comments,
)

try:  # 可选依赖：Chrome TLS 指纹伪装
    from curl_cffi import requests as curl_requests
    _HAS_CURL_CFFI = True
except Exception:  # noqa: BLE001
    curl_requests = None
    _HAS_CURL_CFFI = False

DEFAULT_USER_AGENT = "web:reddit_agent_mvp:v2.0 (by /u/creator_ops)"
CHROME_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

PUBLIC_JSON_TMPL = (
    "https://{host}.reddit.com/comments/{post_id}.json?limit=15&sort=top&raw_json=1"
)
OAUTH_JSON_TMPL = "https://oauth.reddit.com/comments/{post_id}?limit=15&sort=top&raw_json=1"
RSS_TMPL = "https://{host}.reddit.com/comments/{post_id}.rss"
SUBREDDIT_JSON_TMPL = (
    "https://{host}.reddit.com/r/{sub}/hot.json?limit={limit}&t={time_filter}&raw_json=1"
)
SUBREDDIT_RSS_TMPL = "https://{host}.reddit.com/r/{sub}/hot/.rss"

BROWSER_HEADERS = {
    "User-Agent": CHROME_USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

_ATOM = "{http://www.w3.org/2005/Atom}"

# 匹配标准 /comments/{id}/、redd.it/{id}、old.reddit、以及路径中的帖子 id
_POST_ID_RE = re.compile(
    r"(?:redd\.it/|/comments/|/gallery/)([a-z0-9]{5,8})\b", re.IGNORECASE
)
_BARE_ID_RE = re.compile(r"^[a-z0-9]{5,8}$", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_r_prefix(name: str) -> str:
    """安全去除 r/ 前缀（不用 str.lstrip('r/') 以免误伤 r/rust 这类版块）。"""
    s = (name or "").strip()
    return s[2:] if s.lower().startswith("r/") else s


def _html_to_text(html_text: str) -> str:
    """RSS <content> 中的 HTML 转纯文本。"""
    t = _HTML_TAG_RE.sub(" ", html_text or "")
    t = (
        t.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
    )
    return clean_text(t)


@dataclass
class FetchResult:
    cleaned: CleanedRedditPost
    from_cache: bool
    source_channel: str  # oauth | www_json | old_json | www_rss | old_rss | cache
    raw_json: Any = None


class RedditClientError(RuntimeError):
    """抓取或解析失败。"""


class RedditClient:
    """免 Key 开箱即用的 Reddit 抓取器（多通道反 403 降级）。"""

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        cache: Optional[CacheManager] = None,
        client_id: str = "",
        client_secret: str = "",
        username: str = "",
        password: str = "",
        proxy: str = "",
        max_retries: int = 4,
        timeout: float = 20.0,
    ) -> None:
        self.user_agent = user_agent
        self.cache = cache if cache is not None else get_cache_manager()
        self.client_id = (client_id or "").strip()
        self.client_secret = (client_secret or "").strip()
        self.username = (username or "").strip()
        self.password = (password or "").strip()
        self.proxy = (proxy or "").strip() or None
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": self.user_agent, "Accept": "application/json"}
        )
        self._oauth_token: Optional[str] = None
        self._oauth_expiry: float = 0.0

    # ------------------------------------------------------------------
    # URL 解析
    # ------------------------------------------------------------------

    @staticmethod
    def extract_post_id(url_or_id: str) -> str:
        text = (url_or_id or "").strip()
        if not text:
            raise RedditClientError("空的 Reddit URL / post_id")
        if _BARE_ID_RE.match(text):
            return text.lower()

        parsed = urlparse(text if "://" in text else "https://" + text.lstrip("/"))
        path = parsed.path or ""
        combined = f"{parsed.netloc}{path}"
        m = _POST_ID_RE.search(combined) or _POST_ID_RE.search(text)
        if not m:
            raise RedditClientError(
                f"无法从链接解析 post_id：{url_or_id!r}。"
                "请粘贴完整帖子链接（含 /comments/xxxxx/）或 redd.it 短链。"
            )
        return m.group(1).lower()

    @staticmethod
    def normalize_post_url(url_or_id: str, post_id: Optional[str] = None) -> str:
        pid = post_id or RedditClient.extract_post_id(url_or_id)
        return f"https://www.reddit.com/comments/{pid}/"

    # ------------------------------------------------------------------
    # OAuth（通道 A，可选）
    # ------------------------------------------------------------------

    def oauth_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _ensure_oauth_token(self) -> str:
        if self._oauth_token and time.time() < self._oauth_expiry - 30:
            return self._oauth_token
        if not self.oauth_configured():
            raise RedditClientError("未配置 Reddit OAuth 凭证")

        auth = (self.client_id, self.client_secret)
        data = (
            {"grant_type": "password", "username": self.username, "password": self.password}
            if self.username and self.password
            else {"grant_type": "client_credentials"}
        )
        resp = self._session.post(
            "https://www.reddit.com/api/v1/access_token",
            data=data,
            auth=auth,
            timeout=self.timeout,
            proxies=self._proxies(),
        )
        if resp.status_code != 200:
            raise RedditClientError(
                f"OAuth 令牌获取失败 HTTP {resp.status_code}: {resp.text[:200]}"
            )
        payload = resp.json()
        token = payload.get("access_token")
        if not token:
            raise RedditClientError("OAuth 响应缺少 access_token")
        self._oauth_token = token
        self._oauth_expiry = time.time() + float(payload.get("expires_in", 3600))
        return token

    # ------------------------------------------------------------------
    # HTTP 底层：proxy + curl_cffi 双实现 + 退避
    # ------------------------------------------------------------------

    def _proxies(self) -> Optional[dict[str, str]]:
        if not self.proxy:
            return None
        return {"http": self.proxy, "https": self.proxy}

    def _do_get(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        use_curl: bool = True,
    ):
        proxies = self._proxies()
        merged = dict(BROWSER_HEADERS)
        if headers:
            merged.update(headers)
        if use_curl and _HAS_CURL_CFFI:
            try:
                return curl_requests.get(
                    url,
                    headers=merged,
                    impersonate="chrome124",
                    proxies=proxies,
                    timeout=self.timeout,
                    allow_redirects=True,
                )
            except Exception:  # noqa: BLE001
                pass  # 降级到 requests
        return self._session.get(
            url,
            headers=merged,
            proxies=proxies,
            timeout=self.timeout,
            allow_redirects=True,
        )

    def _get_with_backoff(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        use_curl: bool = True,
    ):
        """带 429/5xx 退避的 GET；403 直接抛出让上层切通道。"""
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._do_get(url, headers=headers, use_curl=use_curl)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(min(2 ** attempt, 8))
                continue

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else (2 ** attempt)
                except ValueError:
                    wait = 2 ** attempt
                time.sleep(min(max(wait, 1.0), 60.0))
                last_err = RedditClientError("HTTP 429 Too Many Requests")
                continue
            if resp.status_code >= 500:
                last_err = RedditClientError(f"HTTP {resp.status_code}")
                time.sleep(min(2 ** attempt, 8))
                continue
            if resp.status_code == 403:
                raise RedditClientError(
                    f"Reddit 请求被拦截 HTTP 403: {resp.text[:200]}"
                )
            if resp.status_code != 200:
                raise RedditClientError(
                    f"Reddit 请求失败 HTTP {resp.status_code}: {resp.text[:240]}"
                )
            return resp
        raise RedditClientError(f"重试耗尽：{last_err}")

    @staticmethod
    def _json_or_raise(resp) -> Any:
        """把「200 但非 JSON（如被拦截的 HTML 页）」转成 RedditClientError，触发降级。"""
        try:
            return resp.json()
        except ValueError as exc:
            raise RedditClientError(f"响应不是合法 JSON：{exc}") from exc

    # ------------------------------------------------------------------
    # 单帖：JSON 多通道（OAuth → www/old .json）
    # ------------------------------------------------------------------

    def _fetch_raw_json(self, post_id: str) -> tuple[Any, str]:
        if self.oauth_configured():
            try:
                resp = self._get_with_backoff(
                    OAUTH_JSON_TMPL.format(post_id=post_id),
                    headers={"Authorization": f"bearer {self._ensure_oauth_token()}"},
                    use_curl=False,
                )
                return self._json_or_raise(resp), "oauth"
            except RedditClientError:
                pass
        for host in ("www", "old"):
            try:
                resp = self._get_with_backoff(
                    PUBLIC_JSON_TMPL.format(host=host, post_id=post_id)
                )
                return self._json_or_raise(resp), f"{host}_json"
            except RedditClientError:
                continue
        raise RedditClientError("所有 .json 通道均失败（可能被 403 拦截）")

    # ------------------------------------------------------------------
    # 单帖：RSS 兜底（通道 C）
    # ------------------------------------------------------------------

    def _fetch_rss_post(self, post_id: str, host: str = "www") -> CleanedRedditPost:
        url = RSS_TMPL.format(host=host, post_id=post_id)
        resp = self._get_with_backoff(
            url,
            headers={"Accept": "application/rss+xml, application/xml, text/xml, */*"},
            use_curl=False,  # RSS 对普通请求宽松，curl_cffi 指纹反而易 403
        )
        try:
            return self._parse_rss_post(resp.text, post_id, url)
        except Exception as exc:  # noqa: BLE001
            raise RedditClientError(f"RSS 解析失败：{exc}") from exc

    def _parse_rss_post(self, xml_text: str, post_id: str, url: str) -> CleanedRedditPost:
        root = ET.fromstring(xml_text)
        title = clean_text(root.findtext(f"{_ATOM}title") or "")
        comments: list[str] = []
        for entry in root.findall(f"{_ATOM}entry"):
            author = entry.findtext(f"{_ATOM}author/{_ATOM}name") or ""
            content = _html_to_text(entry.findtext(f"{_ATOM}content") or "")
            if author.lower() == "automoderator" or author in ("[deleted]", "[removed]"):
                continue
            if content and len(content) >= 2:
                comments.append(content)
        return CleanedRedditPost(
            post_id=post_id,
            subreddit="",
            title=title,
            body="",  # comments RSS 不含正文 selftext
            author="",
            score=0,
            upvote_ratio=0.0,
            is_nsfw=False,
            top_comments=comments[:15],
            url=url,
            permalink=f"/comments/{post_id}/",
        )

    def _fetch_post_cleaned(
        self, post_id: str, url: str, top_n: int
    ) -> tuple[CleanedRedditPost, str, Any]:
        try:
            raw, channel = self._fetch_raw_json(post_id)
            cleaned = clean_reddit_payload(
                raw, post_id=post_id, url=url, top_n_comments=top_n
            )
            return cleaned, channel, raw
        except RedditClientError:
            pass
        for host in ("www", "old"):
            try:
                cleaned = self._fetch_rss_post(post_id, host)
                return cleaned, f"{host}_rss", None
            except RedditClientError:
                continue
        raise RedditClientError("帖子抓取失败：所有通道（JSON + RSS）均被拦截")

    # ------------------------------------------------------------------
    # 对外 API：单帖
    # ------------------------------------------------------------------

    def fetch_post(
        self,
        url_or_id: str,
        *,
        bypass_cache: bool = False,
        include_top_comments: int = 5,
    ) -> FetchResult:
        post_id = self.extract_post_id(url_or_id)
        canonical_url = self.normalize_post_url(url_or_id, post_id)

        if not bypass_cache:
            cached = self.cache.get_reddit_post(post_id)
            if cached and cached.get("cleaned"):
                return FetchResult(
                    cleaned=CleanedRedditPost.from_dict(cached["cleaned"]),
                    from_cache=True,
                    source_channel="cache",
                    raw_json=cached.get("raw"),
                )

        cleaned, channel, raw = self._fetch_post_cleaned(
            post_id, canonical_url, include_top_comments
        )
        self.cache.put_reddit_post(
            post_id=post_id,
            url=canonical_url,
            raw=raw,
            cleaned=cleaned.to_dict(),
        )
        return FetchResult(
            cleaned=cleaned,
            from_cache=False,
            source_channel=channel,
            raw_json=raw,
        )

    # ------------------------------------------------------------------
    # 对外 API：版块热帖批量抓取
    # ------------------------------------------------------------------

    def _fetch_top_comments(self, post_id: str, top_n: int = 5) -> list[str]:
        for host in ("www", "old"):
            try:
                resp = self._get_with_backoff(
                    PUBLIC_JSON_TMPL.format(host=host, post_id=post_id)
                )
                raw = self._json_or_raise(resp)
                comments_listing = raw[1] if isinstance(raw, list) and len(raw) > 1 else None
                return extract_top_comments(comments_listing, top_n=top_n)
            except RedditClientError:
                continue
        return []

    def fetch_subreddit_hot(
        self,
        subreddit: str,
        limit: int = 15,
        time_filter: str = "day",
        *,
        include_comments: bool = True,
        max_comment_posts: int = 10,
    ) -> list[CleanedRedditPost]:
        """抓取目标版块当天最热帖子（多通道降级），可选附带高赞评论。"""
        sub = _strip_r_prefix(subreddit)
        posts: list[CleanedRedditPost] = []

        for host in ("www", "old"):
            try:
                resp = self._get_with_backoff(
                    SUBREDDIT_JSON_TMPL.format(
                        host=host, sub=sub, limit=limit, time_filter=time_filter
                    )
                )
                data = self._json_or_raise(resp)
                posts = self._parse_subreddit_json(data, sub)
                if posts:
                    break
            except RedditClientError:
                continue

        if not posts:
            for host in ("www", "old"):
                try:
                    resp = self._get_with_backoff(
                        SUBREDDIT_RSS_TMPL.format(
                            host=host, sub=sub, limit=limit, time_filter=time_filter
                        ),
                        headers={"Accept": "application/rss+xml, application/xml, text/xml, */*"},
                        use_curl=False,
                    )
                    posts = self._parse_subreddit_rss(resp.text, sub)
                    if posts:
                        break
                except RedditClientError:
                    continue

        if not posts:
            raise RedditClientError(f"版块 r/{sub} 热帖抓取失败（多通道均被拦截）")

        posts = posts[:limit]
        if include_comments:
            for p in posts[:max_comment_posts]:
                if p.top_comments:
                    continue
                p.top_comments = self._fetch_top_comments(p.post_id, top_n=5)
                time.sleep(1.0)
        return posts

    def _parse_subreddit_json(self, data: Any, sub: str) -> list[CleanedRedditPost]:
        posts: list[CleanedRedditPost] = []
        children = ((data or {}).get("data") or {}).get("children") or []
        for child in children:
            if not isinstance(child, dict) or child.get("kind") != "t3":
                continue
            d = child.get("data") or {}
            title = clean_text(d.get("title") or "")
            body = clean_text(d.get("selftext") or "")
            if not title and not body:
                continue
            permalink = str(d.get("permalink") or "")
            posts.append(
                CleanedRedditPost(
                    post_id=str(d.get("id") or ""),
                    subreddit=_strip_r_prefix(str(d.get("subreddit") or sub)),
                    title=title,
                    body=body,
                    author=str(d.get("author") or ""),
                    score=int(d.get("score") or 0),
                    upvote_ratio=float(d.get("upvote_ratio") or 0.0),
                    is_nsfw=bool(d.get("over_18") or d.get("nsfw") or False),
                    top_comments=[],
                    url=f"https://www.reddit.com{permalink}" if permalink else "",
                    permalink=permalink,
                )
            )
        return posts

    def _parse_subreddit_rss(self, xml_text: str, sub: str) -> list[CleanedRedditPost]:
        try:
            root = ET.fromstring(xml_text)
        except Exception as exc:  # noqa: BLE001
            raise RedditClientError(f"版块 RSS 解析失败：{exc}") from exc
        posts: list[CleanedRedditPost] = []
        for entry in root.findall(f"{_ATOM}entry"):
            title = clean_text(entry.findtext(f"{_ATOM}title") or "")
            content = _html_to_text(entry.findtext(f"{_ATOM}content") or "")
            author = entry.findtext(f"{_ATOM}author/{_ATOM}name") or ""
            link_el = entry.find(f"{_ATOM}link")
            link = link_el.get("href", "") if link_el is not None else ""
            entry_id = entry.findtext(f"{_ATOM}id") or ""
            m = _POST_ID_RE.search(entry_id + " " + link)
            pid = m.group(1).lower() if m else ""
            if not title and not content:
                continue
            posts.append(
                CleanedRedditPost(
                    post_id=pid,
                    subreddit=sub,
                    title=title,
                    body=content,
                    author=author,
                    score=0,
                    upvote_ratio=0.0,
                    is_nsfw=False,
                    top_comments=[],
                    url=link,
                    permalink="",
                )
            )
        return posts


def parse_reddit_url(url: str) -> str:
    """便捷函数：仅解析 post_id。"""
    return RedditClient.extract_post_id(url)
