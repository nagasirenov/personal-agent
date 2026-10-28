"""Bounded, model-selected public web lookup independent of the model transport.

Every hop of a fetch (the first URL and each redirect, at most MAX_REDIRECTS) is re-validated: http or
https only, no credentials in the URL, a small set of ports, and every resolved address must be a
public (globally routable) IP.  Page text is untrusted data and is labeled as such.
"""
from __future__ import annotations

import asyncio
import codecs
import html
import json
from html.parser import HTMLParser
import ipaddress
import re
import socket
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import httpcore
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree

import httpx


_MAX_RESPONSE = 32 * 1024 * 1024  # safety ceiling: a bigger body is cut here and processed, never rejected
_MAX_PARSE_CHARS = 8_000_000  # the HTML parser looks at most at this much decoded text
_MAX_TEXT = 12000
_MAX_META_VALUE = 500
_MAX_JSON_LD_BLOCKS = 3
_MAX_JSON_LD_CHARS = 60000
_META_KEYS = {
    "description": "description", "author": "author", "keywords": "keywords",
    "og:title": "og_title", "og:description": "og_description", "og:type": "og_type",
    "og:site_name": "og_site_name", "og:url": "og_url", "og:image": "og_image",
    "og:video": "og_video", "og:video:url": "og_video", "og:video:secure_url": "og_video",
    "og:audio": "og_audio", "article:author": "article_author",
    "article:published_time": "published_time", "twitter:title": "twitter_title",
    "twitter:description": "twitter_description", "twitter:image": "twitter_image",
    "twitter:creator": "twitter_creator", "twitter:player": "twitter_player",
}
_OEMBED_KEYS = ("title", "author_name", "author_url", "provider_name", "type", "thumbnail_url",
                "width", "height", "version")
_SEARCH_URL = "https://www.bing.com/search"
_ALLOWED_PORTS = {None, 80, 443, 8080, 8443}
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_MAX_URL_LENGTH = 2048
MAX_REDIRECTS = 5
MAX_SEARCHES = 4
MAX_OPENS = 4
OPEN_DEADLINE_SECONDS = 20.0
_UNTRUSTED = "untrusted_web_content"
_OPEN_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PersonalAgent/0.1",
                 "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,text/plain;q=0.8,*/*;q=0.5",
                 "Accept-Language": "en-US,en;q=0.8,ru;q=0.7"}


def _web_url_ok(url: str) -> bool:
    try:
        parts = urlsplit(url)
        host = parts.hostname
        return (parts.scheme in {"http", "https"} and bool(host) and not parts.username
                and not parts.password and parts.port in _ALLOWED_PORTS
                and not host.endswith((".local", ".internal", ".localhost")))
    except ValueError:
        return False


class WebLookupDenied(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class WebLookupRequest:
    kind: str
    query: str = ""
    result_id: int = 0
    url: str = ""


def validate_search_query(query: object) -> str:
    """Return a clean public search query or raise WebLookupDenied."""
    if not isinstance(query, str) or not 2 <= len(query.strip()) <= 180 or any(ord(c) < 32 for c in query):
        raise WebLookupDenied("invalid_web_query")
    # Do not send obvious local paths or credential-like strings to an external engine.
    if re.search(r"[A-Za-z]:\\|\\\\|\b(?:sk-[A-Za-z0-9_-]{16,}|Bearer\s+\S+|\d{5,16}:[A-Za-z0-9_-]{20,})", query, re.I):
        raise WebLookupDenied("private_web_query")
    return query.strip()


def validate_result_id(result_id: object) -> int:
    if type(result_id) is int and 1 <= result_id <= 5:
        return result_id
    raise WebLookupDenied("invalid_result_id")


def validate_web_url(url: object) -> str:
    """A user- or model-supplied page URL: syntax only; hosts are resolved and checked per hop."""
    if (not isinstance(url, str) or not url.strip() or len(url.strip()) > _MAX_URL_LENGTH
            or any(ord(c) < 33 or ord(c) == 127 for c in url.strip())):
        raise WebLookupDenied("invalid_web_url")
    url = url.strip()
    if not _web_url_ok(url):
        raise WebLookupDenied("web_url_denied")
    return url


class _VisibleHTML(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "form", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.in_title = False
        self.title_parts: list[str] = []
        self.parts: list[str] = []
        self.text_len = 0
        self.title_len = 0
        self.meta: dict[str, str] = {}
        self.json_ld: list[str] = []
        self.oembed_url = ""
        self._ld_parts: list[str] | None = None
        self.ld_len = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self.skip += 1
        if tag == "script":
            kind = dict(attrs).get("type") or ""
            if kind.lower().split(";")[0].strip() == "application/ld+json":
                self._ld_parts = []
                self.ld_len = 0
        elif tag == "meta":
            self._take_meta(dict(attrs))
        elif tag == "link":
            self._take_link(dict(attrs))
        if tag == "title":
            self.in_title = True
        if tag in {"p", "h1", "h2", "h3", "li", "article", "br", "tr"}:
            self.parts.append("\n")

    def _take_meta(self, attrs: dict) -> None:
        key = (attrs.get("property") or attrs.get("name") or attrs.get("itemprop") or "").strip().lower()
        field = _META_KEYS.get(key)
        content = attrs.get("content")
        if field and content and field not in self.meta and _clip(content):
            self.meta[field] = _clip(content)

    def _take_link(self, attrs: dict) -> None:
        rel = (attrs.get("rel") or "").lower().split()
        href = (attrs.get("href") or "").strip()
        if not href or len(href) > _MAX_URL_LENGTH:
            return
        if "canonical" in rel and "canonical_url" not in self.meta:
            self.meta["canonical_url"] = href[:_MAX_META_VALUE]
        elif ("alternate" in rel and not self.oembed_url
              and (attrs.get("type") or "").lower().startswith("application/json+oembed")):
            self.oembed_url = href

    def handle_endtag(self, tag):
        if tag == "script" and self._ld_parts is not None:
            raw = "".join(self._ld_parts).strip()
            self._ld_parts = None
            if raw and len(raw) <= _MAX_JSON_LD_CHARS and len(self.json_ld) < _MAX_JSON_LD_BLOCKS * 2:
                self.json_ld.append(raw)
        if tag in self._SKIP and self.skip:
            self.skip -= 1
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self._ld_parts is not None:
            if self.ld_len <= _MAX_JSON_LD_CHARS:
                self._ld_parts.append(data)
                self.ld_len += len(data)
        elif self.in_title:
            if self.title_len < 300:
                self.title_parts.append(data)
                self.title_len += len(data)
        elif not self.skip and self.text_len < 18000:
            self.parts.append(data)
            self.text_len += len(data)

    def text(self):
        return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", "".join(self.parts))).strip()[:_MAX_TEXT]

    def title(self):
        return re.sub(r"\s+", " ", "".join(self.title_parts)).strip()[:240]


def _plain_text(text: str) -> str:
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\r\n", "\n").split("\n"))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()[:_MAX_TEXT]


def _decode(payload: bytes, content_type: str) -> str:
    match = re.search(r"charset=([\w.-]+)", content_type, re.I)
    if match:
        try:
            return payload.decode(codecs.lookup(match.group(1)).name, errors="replace")
        except LookupError:
            pass
    return payload.decode("utf-8", errors="replace")


async def _resolve_addresses(host: str, port: int) -> list[str]:
    infos = await asyncio.wait_for(asyncio.to_thread(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM),
                                   timeout=3)
    return [info[4][0] for info in infos]


def _global_address(raw: str) -> bool:
    address = ipaddress.ip_address(raw.split("%", 1)[0])
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not address.is_multicast


async def _public_url(url: str) -> bool:
    """True only for a syntactically allowed http(s) URL whose host resolves to public addresses only."""
    if not _web_url_ok(url):
        return False
    try:
        parts = urlsplit(url)
        port = parts.port or (443 if parts.scheme == "https" else 80)
        addresses = await _resolve_addresses(parts.hostname, port)
        return bool(addresses) and all(_global_address(address) for address in addresses)
    except (ValueError, OSError, asyncio.TimeoutError):
        return False


class PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, backend: httpcore.AsyncNetworkBackend | None = None) -> None:
        self._backend = backend or httpcore._backends.auto.AutoBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ) -> httpcore.AsyncNetworkStream:
        try:
            addresses = await _resolve_addresses(host, port)
        except (OSError, asyncio.TimeoutError):
            raise httpcore.ConnectError(f"DNS resolution failed for {host}") from None
        if not addresses or not all(_global_address(addr) for addr in addresses):
            raise httpcore.ConnectError("Blocked non-global address (DNS rebinding protection)")
        return await self._backend.connect_tcp(
            addresses[0], port, timeout=timeout, local_address=local_address, socket_options=socket_options
        )

    async def sleep(self, delay: float) -> None:
        await self._backend.sleep(delay)


class PinnedAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=self._pool._ssl_context,
            network_backend=PinnedNetworkBackend(),
        )


def _find_edge_executable() -> str | None:
    for candidate in (
        os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)") + r"\Microsoft\Edge\Application\msedge.exe",
        os.environ.get("PROGRAMFILES", r"C:\Program Files") + r"\Microsoft\Edge\Application\msedge.exe",
        os.environ.get("LOCALAPPDATA", r"C:\Users\Default\AppData\Local") + r"\Microsoft\Edge\Application\msedge.exe",
    ):
        if Path(candidate).is_file():
            return candidate
    found = shutil.which("msedge")
    return found if found and Path(found).is_file() else None


async def _render_headless_edge(url: str, timeout: float = 12.0) -> str | None:
    if not await _public_url(url):
        return None
    edge = _find_edge_executable()
    if not edge:
        return None
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with tempfile.TemporaryDirectory(prefix="edge-headless-") as temp_profile:
        cmd = [
            edge,
            "--headless=new",
            "--disable-gpu",
            "--dump-dom",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={temp_profile}",
            url,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                creationflags=flags,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            if proc.returncode == 0 and out:
                return out[:_MAX_PARSE_CHARS].decode("utf-8", errors="replace")
        except (OSError, asyncio.TimeoutError):
            return None
    return None


async def _bounded_get(client: httpx.AsyncClient, url: str, **kwargs) -> tuple[bytes, str]:
    async with client.stream("GET", url, follow_redirects=False, **kwargs) as response:
        response.raise_for_status()
        body, content_type, _ = await _read_body(response)
        return body, content_type


async def _read_body(response: httpx.Response) -> tuple[bytes, str, bool]:
    """Read up to the safety ceiling.  A bigger body is cut there (flagged), never rejected."""
    chunks = []
    size = 0
    truncated = False
    async for chunk in response.aiter_bytes():
        if size + len(chunk) > _MAX_RESPONSE:
            chunk = chunk[:_MAX_RESPONSE - size]
            truncated = True
        size += len(chunk)
        chunks.append(chunk)
        if truncated:
            break
    return b"".join(chunks), response.headers.get("content-type", ""), truncated


async def _fetch_page(client: httpx.AsyncClient, url: str) -> tuple[bytes, str, str, bool]:
    """GET with manual redirects; each hop is validated again.

    Returns (body, content type, final URL, body_truncated)."""
    for hop in range(MAX_REDIRECTS + 1):
        if not await _public_url(url):
            raise WebLookupDenied("web_url_denied" if hop == 0 else "web_redirect_denied")
        async with client.stream("GET", url, follow_redirects=False) as response:
            if response.status_code in _REDIRECT_STATUSES:
                location = response.headers.get("location", "")
                if not location or len(location) > _MAX_URL_LENGTH or any(ord(c) < 32 for c in location):
                    raise WebLookupDenied("web_redirect_invalid")
                if hop == MAX_REDIRECTS:
                    raise WebLookupDenied("web_too_many_redirects")
                url = urljoin(url, location)
                continue
            response.raise_for_status()
            body, content_type, truncated = await _read_body(response)
            return body, content_type, url, truncated
    raise WebLookupDenied("web_too_many_redirects")  # pragma: no cover - loop always returns or raises


def _clip(value: object, limit: int = _MAX_META_VALUE) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value))).strip()[:limit]


def _shrink_json(value, depth: int = 0):
    """Bounded copy of parsed JSON-LD: shallow, short strings, few list items."""
    if isinstance(value, str):
        return _clip(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if depth >= 5:
        return None
    if isinstance(value, list):
        return [_shrink_json(item, depth + 1) for item in value[:10]]
    if isinstance(value, dict):
        return {str(key)[:60]: _shrink_json(item, depth + 1)
                for key, item in list(value.items())[:40] if key != "@context"}
    return None


def _parse_page(text: str) -> tuple[str, str, dict, list, str]:
    """(visible text, title, meta fields, JSON-LD blocks, oEmbed JSON URL) from an HTML page."""
    parser = _VisibleHTML()
    parser.feed(text[:_MAX_PARSE_CHARS])
    json_ld = []
    for raw in parser.json_ld[:_MAX_JSON_LD_BLOCKS]:
        try:
            json_ld.append(_shrink_json(json.loads(raw)))
        except (ValueError, RecursionError):
            continue
    return parser.text(), parser.title(), parser.meta, json_ld, parser.oembed_url


async def _fetch_oembed(client: httpx.AsyncClient, page_url: str, href: str) -> dict | None:
    """Fetch a discovered oEmbed JSON document through the same per-hop public-URL validation."""
    try:
        url = urljoin(page_url, href)
        if len(url) > _MAX_URL_LENGTH or not _web_url_ok(url):
            return None
        body, content_type, _, _ = await _fetch_page(client, url)
        data = json.loads(_decode(body, content_type))
    except (httpx.HTTPError, WebLookupDenied, ValueError, UnicodeError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    result = {key: _clip(data[key]) if isinstance(data[key], str) else data[key]
              for key in _OEMBED_KEYS if key in data and isinstance(data[key], (str, int, float))}
    if isinstance(data.get("html"), str):
        embed_text = _clip(re.sub(r"<[^>]+>", " ", data["html"]), 1200)
        if embed_text:
            result["embed_text"] = embed_text
    return result or None


class WebLookupSession:
    """Per-turn search references and budgets. No query or retrieved page is persisted as service state."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client
        self.results: list[dict] = []
        self.searches = 0
        self.opens = 0

    async def execute(self, request: WebLookupRequest) -> dict:
        if request.kind == "search":
            return await self._search(request)
        if request.kind == "open":
            return await self._open(request)
        raise WebLookupDenied("unknown_web_kind")

    def _client_context(self, **kwargs):
        if self.client is not None:
            return _BorrowedClient(self.client)
        return httpx.AsyncClient(transport=PinnedAsyncHTTPTransport(), timeout=10.0, trust_env=False, **kwargs)

    async def _search(self, request: WebLookupRequest) -> dict:
        if self.searches >= MAX_SEARCHES:
            return {"kind": "web_search", "ok": False, "error": {"code": "web_search_budget"}}
        self.searches += 1
        self.results = []
        try:
            async with self._client_context(headers={"User-Agent": "PersonalAgent/0.1 (public web search)"}) as client:
                payload, _ = await _bounded_get(client, _SEARCH_URL, params={"q": request.query, "format": "rss"})
            root = ElementTree.fromstring(payload)
            if root.tag != "rss":
                raise WebLookupDenied("web_search_invalid_response")
            for item in root.findall("./channel/item"):
                url = (item.findtext("link") or "").strip()
                if not _web_url_ok(url) or len(url) > _MAX_URL_LENGTH:
                    continue
                if any(result["url"] == url for result in self.results):
                    continue
                self.results.append({"result_id": len(self.results) + 1,
                                     "title": html.unescape((item.findtext("title") or "")[:240]),
                                     "url": url,
                                     "snippet": re.sub(r"<[^>]+>", " ", html.unescape(item.findtext("description") or ""))[:800],
                                     "published": item.findtext("pubDate")})
                if len(self.results) == 5:
                    break
            return {"kind": "web_search", "ok": bool(self.results), "query": request.query,
                    "results": self.results, "content_trust": _UNTRUSTED,
                    "searched_at": datetime.now(timezone.utc).isoformat(),
                    **({"error": {"code": "web_no_results"}} if not self.results else {})}
        except (httpx.HTTPError, ElementTree.ParseError, WebLookupDenied, UnicodeError) as exc:
            self.results = []
            code = exc.code if isinstance(exc, WebLookupDenied) else "web_search_unavailable"
            return {"kind": "web_search", "ok": False, "error": {"code": code}}

    async def _open(self, request: WebLookupRequest) -> dict:
        if self.opens >= MAX_OPENS:
            return {"kind": "web_open", "ok": False, "error": {"code": "web_open_budget"}}
        if request.url:
            source: dict = {"url": request.url}
        else:
            if not 1 <= request.result_id <= len(self.results):
                return {"kind": "web_open", "ok": False, "error": {"code": "unknown_search_result"}}
            source = dict(self.results[request.result_id - 1])
        self.opens += 1
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            async with self._client_context(headers=_OPEN_HEADERS) as client:
                payload, content_type, final_url, body_truncated = await asyncio.wait_for(
                    _fetch_page(client, source["url"]), OPEN_DEADLINE_SECONDS)
                kind = content_type.split(";", 1)[0].strip().lower()
                text = _decode(payload, content_type)
                meta: dict = {}
                json_ld: list = []
                oembed_href = ""
                if "html" in kind:
                    content, title, meta, json_ld, oembed_href = await asyncio.to_thread(_parse_page, text)
                    if not content.strip() and not json_ld and self.client is None:
                        remaining = max(1.0, OPEN_DEADLINE_SECONDS - (loop.time() - started))
                        rendered = await _render_headless_edge(final_url, timeout=min(remaining, 12.0))
                        if rendered:
                            r_content, r_title, r_meta, r_json_ld, _ = await asyncio.to_thread(_parse_page, rendered)
                            if len(r_content.strip()) > len(content.strip()):
                                content = r_content
                            if r_title and not title:
                                title = r_title
                            meta = {**r_meta, **meta}
                            json_ld = r_json_ld or json_ld
                    if title and not source.get("title"):
                        source["title"] = title
                elif kind == "text/plain":
                    content = _plain_text(text)
                else:
                    raise WebLookupDenied("web_page_unsupported_format")
                oembed = None
                if oembed_href:
                    remaining = max(1.0, OPEN_DEADLINE_SECONDS - (loop.time() - started))
                    try:
                        oembed = await asyncio.wait_for(_fetch_oembed(client, final_url, oembed_href), remaining)
                    except asyncio.TimeoutError:
                        oembed = None
            if not content and not meta and not json_ld and not oembed and not source.get("title"):
                raise WebLookupDenied("web_page_no_text")
            if not source.get("title"):
                title = meta.get("og_title") or meta.get("twitter_title") or (oembed or {}).get("title")
                if title:
                    source["title"] = title[:240]
            result = {"kind": "web_open", "ok": True, "source": source, "text": content,
                      "content_type": kind, "content_trust": _UNTRUSTED,
                      "retrieved_at": datetime.now(timezone.utc).isoformat(),
                      "truncated": len(content) >= _MAX_TEXT or body_truncated,
                      "body_truncated": body_truncated}
            if meta:
                result["meta"] = meta
            if json_ld:
                result["json_ld"] = json_ld
            if oembed:
                result["oembed"] = oembed
            if final_url != source["url"]:
                result["final_url"] = final_url
            return result
        except asyncio.TimeoutError:
            return {"kind": "web_open", "ok": False, "source": source, "error": {"code": "web_page_timeout"}}
        except (httpx.HTTPError, WebLookupDenied, UnicodeError) as exc:
            code = exc.code if isinstance(exc, WebLookupDenied) else "web_page_unavailable"
            return {"kind": "web_open", "ok": False, "source": source, "error": {"code": code}}


class _BorrowedClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, *args):
        return False
