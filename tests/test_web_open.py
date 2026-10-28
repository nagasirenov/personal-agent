"""web_open by URL: manual redirects with per-hop validation, public-IP only, text/plain, budgets."""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import httpx

from personal_agent.web_lookup import (
    MAX_OPENS, WebLookupDenied, WebLookupRequest, WebLookupSession, validate_web_url,
)

PUBLIC = {"a.example.org": ["93.184.216.34"], "b.example.org": ["93.184.216.35"], "c.example.org": ["8.8.8.8"],
          "d.example.org": ["8.8.4.4"], "rebind.example.org": ["10.0.0.7"], "mixed.example.org": ["8.8.8.8", "127.0.0.1"]}
HTML = b"<html><head><title>Real title</title></head><body><p>Hello MARKER_42 world</p></body></html>"


def resolver():
    async def resolve(host: str, port: int) -> list[str]:
        if host in PUBLIC:
            return PUBLIC[host]
        import ipaddress
        ipaddress.ip_address(host)  # literals resolve to themselves; unknown names fail like NXDOMAIN
        return [host]
    return resolve


def session_for(routes: dict, seen: list | None = None) -> tuple[WebLookupSession, httpx.AsyncClient]:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(str(request.url))
        route = routes.get(str(request.url))
        if route is None:
            raise AssertionError(f"unexpected request {request.url}")
        return route(request) if callable(route) else route

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    return WebLookupSession(client), client


def redirect(location: str, status: int = 302) -> httpx.Response:
    return httpx.Response(status, headers={"location": location})


class WebOpenUrlTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch("personal_agent.web_lookup._resolve_addresses", new=resolver())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_url_validation(self):
        self.assertEqual(validate_web_url(" http://a.example.org/x "), "http://a.example.org/x")
        for bad in ("ftp://a.example.org/x", "file:///C:/x", "javascript:alert(1)", "https://u:p@a.example.org/",
                    "https://a.example.org:22/", "https://printer.local/", "https://a.example.org/ x", "", 5, None,
                    "https://" + "a" * 3000):
            with self.subTest(url=bad), self.assertRaises(WebLookupDenied):
                validate_web_url(bad)

    async def test_user_url_http_and_https_are_fetched_and_titled(self):
        for url in ("https://a.example.org/page", "http://a.example.org/page"):
            session, client = session_for({url: httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"},
                                                                content=HTML)})
            async with client:
                result = await session.execute(WebLookupRequest("open", url=url))
            with self.subTest(url=url):
                self.assertTrue(result["ok"], result)
                self.assertIn("MARKER_42", result["text"])
                self.assertEqual(result["source"], {"url": url, "title": "Real title"})
                self.assertEqual(result["content_trust"], "untrusted_web_content")
                self.assertNotIn("final_url", result)

    async def test_redirect_chain_is_followed_and_every_hop_is_validated(self):
        seen: list[str] = []
        session, client = session_for({
            "https://a.example.org/start": redirect("https://b.example.org/one"),
            "https://b.example.org/one": redirect("/two", 301),  # relative target
            "https://b.example.org/two": redirect("http://c.example.org/final", 307),
            "http://c.example.org/final": httpx.Response(200, headers={"content-type": "text/html"}, content=HTML),
        }, seen)
        checked: list[str] = []
        original = resolver()

        async def spy(host, port):
            checked.append(host)
            return await original(host, port)

        with patch("personal_agent.web_lookup._resolve_addresses", new=spy):
            async with client:
                result = await session.execute(WebLookupRequest("open", url="https://a.example.org/start"))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["final_url"], "http://c.example.org/final")
        self.assertEqual(len(seen), 4)
        self.assertEqual(checked, ["a.example.org", "b.example.org", "b.example.org", "c.example.org"])

    async def test_more_than_five_redirects_is_refused(self):
        routes = {f"https://a.example.org/{i}": redirect(f"https://a.example.org/{i + 1}") for i in range(10)}
        seen: list[str] = []
        session, client = session_for(routes, seen)
        async with client:
            result = await session.execute(WebLookupRequest("open", url="https://a.example.org/0"))
        self.assertEqual(result["error"]["code"], "web_too_many_redirects")
        self.assertEqual(len(seen), 6)

    async def test_short_link_chain_of_five_hops_resolves_and_each_hop_is_validated(self):
        routes = {"https://a.example.org/s": redirect("https://b.example.org/1"),
                  "https://b.example.org/1": redirect("https://c.example.org/2", 301),
                  "https://c.example.org/2": redirect("/3?utm=x", 302),
                  "https://c.example.org/3?utm=x": redirect("https://d.example.org/4", 307),
                  "https://d.example.org/4": redirect("https://b.example.org/video/5", 308),
                  "https://b.example.org/video/5": httpx.Response(
                      200, headers={"content-type": "text/html"}, content=HTML)}
        session, client = session_for(routes)
        async with client:
            result = await session.execute(WebLookupRequest("open", url="https://a.example.org/s"))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["final_url"], "https://b.example.org/video/5")

    async def test_redirect_to_private_or_loopback_or_metadata_targets_is_denied_without_a_request(self):
        for target in ("http://127.0.0.1/admin", "http://192.168.1.5/", "http://169.254.169.254/latest/meta-data",
                       "http://[::1]/", "http://rebind.example.org/", "http://mixed.example.org/",
                       "http://2130706433/", "ftp://a.example.org/x", "https://user:pw@a.example.org/"):
            seen: list[str] = []
            session, client = session_for({"https://a.example.org/start": redirect(target)}, seen)
            async with client:
                result = await session.execute(WebLookupRequest("open", url="https://a.example.org/start"))
            with self.subTest(target=target):
                self.assertFalse(result["ok"])
                self.assertEqual(result["error"]["code"], "web_redirect_denied")
                self.assertEqual(seen, ["https://a.example.org/start"])  # the private hop was never requested

    async def test_user_url_to_private_address_is_denied_without_a_request(self):
        for url in ("http://127.0.0.1:8080/", "https://10.1.2.3/", "http://localhost/", "http://rebind.example.org/",
                    "http://[fe80::1]/", "http://[::ffff:127.0.0.1]/"):
            seen: list[str] = []
            session, client = session_for({}, seen)
            async with client:
                result = await session.execute(WebLookupRequest("open", url=url))
            with self.subTest(url=url):
                self.assertEqual(result["error"]["code"], "web_url_denied")
                self.assertEqual(seen, [])

    async def test_redirect_without_location_and_http_errors(self):
        session, client = session_for({"https://a.example.org/x": httpx.Response(302),
                                       "https://a.example.org/y": httpx.Response(404)})
        async with client:
            self.assertEqual((await session.execute(WebLookupRequest("open", url="https://a.example.org/x")))
                             ["error"]["code"], "web_redirect_invalid")
            self.assertEqual((await session.execute(WebLookupRequest("open", url="https://a.example.org/y")))
                             ["error"]["code"], "web_page_unavailable")

    async def test_text_plain_is_supported_and_other_types_are_not(self):
        session, client = session_for({
            "https://a.example.org/notes.txt": httpx.Response(
                200, headers={"content-type": "text/plain; charset=windows-1251"},
                content="Первая строка\r\n\r\n\r\n\r\nВторая   строка".encode("cp1251")),
            "https://a.example.org/data.bin": httpx.Response(200, headers={"content-type": "application/pdf"},
                                                              content=b"%PDF"),
        })
        async with client:
            plain = await session.execute(WebLookupRequest("open", url="https://a.example.org/notes.txt"))
            other = await session.execute(WebLookupRequest("open", url="https://a.example.org/data.bin"))
        self.assertTrue(plain["ok"], plain)
        self.assertEqual(plain["text"], "Первая строка\n\nВторая строка")
        self.assertEqual(plain["content_type"], "text/plain")
        self.assertEqual(other["error"]["code"], "web_page_unsupported_format")

    async def test_huge_page_is_truncated_not_refused_and_budget_counts_user_urls(self):
        big = (b"<html><head><title>Huge</title><meta property='og:title' content='Big og'></head><body><p>start MARK</p>"
               + b"<p>filler filler filler</p>" * (1024 * 1024))  # ~27 MB
        session, client = session_for({
            "https://a.example.org/big": httpx.Response(200, headers={"content-type": "text/html"}, content=big),
        })
        async with client:
            result = await session.execute(WebLookupRequest("open", url="https://a.example.org/big"))
            self.assertTrue(result["ok"], result)
            self.assertIn("MARK", result["text"])
            self.assertTrue(result["truncated"])
            self.assertFalse(result["body_truncated"])
            self.assertEqual(result["meta"]["og_title"], "Big og")
            for _ in range(MAX_OPENS - 1):
                await session.execute(WebLookupRequest("open", url="https://a.example.org/big"))
            self.assertEqual((await session.execute(WebLookupRequest("open", url="https://a.example.org/big")))
                             ["error"]["code"], "web_open_budget")

    async def test_body_beyond_safety_ceiling_is_cut_and_flagged(self):
        with patch("personal_agent.web_lookup._MAX_RESPONSE", 4096):
            session, client = session_for({
                "https://a.example.org/b": httpx.Response(200, headers={"content-type": "text/plain"},
                                                          content=b"line one\n" + b"x" * 100000)})
            async with client:
                result = await session.execute(WebLookupRequest("open", url="https://a.example.org/b"))
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["body_truncated"])
        self.assertTrue(result["truncated"])
        self.assertTrue(result["text"].startswith("line one"))

    async def test_meta_opengraph_canonical_and_json_ld_are_extracted_for_js_pages(self):
        page = b"""<html><head><title>  Clip   page </title>
          <meta name="description" content="Short &amp; sweet">
          <meta property="og:title" content="OG title"><meta property="og:description" content="OG desc"/>
          <meta property="og:video" content="https://cdn.example.org/v.mp4"><meta property="og:image" content="https://cdn.example.org/i.jpg">
          <meta name="twitter:title" content="TW title"><meta name="author" content="Some Author">
          <meta property="og:title" content="second ignored">
          <link rel="canonical" href="https://a.example.org/c/1"/>
          <script type="application/ld+json">{"@context":"https://schema.org","@type":"VideoObject","name":"LD name","author":{"name":"LD author"},"description":"%s"}</script>
          <script type="application/ld+json">{not json</script>
          <script>var secret = 1;</script></head><body><div id="root"></div></body></html>""" % (b"d" * 5000)
        session, client = session_for({"https://a.example.org/v": httpx.Response(
            200, headers={"content-type": "text/html"}, content=page)})
        async with client:
            result = await session.execute(WebLookupRequest("open", url="https://a.example.org/v"))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["text"], "")
        meta = result["meta"]
        self.assertEqual(meta["description"], "Short & sweet")
        self.assertEqual(meta["og_title"], "OG title")
        self.assertEqual(meta["og_description"], "OG desc")
        self.assertEqual(meta["og_video"], "https://cdn.example.org/v.mp4")
        self.assertEqual(meta["og_image"], "https://cdn.example.org/i.jpg")
        self.assertEqual(meta["twitter_title"], "TW title")
        self.assertEqual(meta["author"], "Some Author")
        self.assertEqual(meta["canonical_url"], "https://a.example.org/c/1")
        self.assertEqual(result["source"]["title"], "Clip page")
        self.assertEqual(len(result["json_ld"]), 1)
        self.assertEqual(result["json_ld"][0]["name"], "LD name")
        self.assertNotIn("@context", result["json_ld"][0])
        self.assertLessEqual(len(result["json_ld"][0]["description"]), 500)
        self.assertNotIn("secret", str(result))

    async def test_oembed_discovery_is_fetched_with_the_same_validation(self):
        page = (b"<html><head><title>T</title>"
                b'<link rel="alternate" type="application/json+oembed" href="/oembed?url=x&amp;f=json">'
                b"</head><body><p>body text</p></body></html>")
        seen: list[str] = []
        oembed = httpx.Response(200, headers={"content-type": "application/json"}, content=(
            b'{"title":"Clip caption","author_name":"Creator","author_url":"https://x.example.org/u",'
            b'"provider_name":"Site","type":"video","html":"<blockquote><p>caption #tag</p></blockquote><script>x</script>"}'))
        session, client = session_for({"https://a.example.org/p": httpx.Response(
            200, headers={"content-type": "text/html"}, content=page),
            "https://a.example.org/oembed?url=x&f=json": oembed}, seen)
        async with client:
            result = await session.execute(WebLookupRequest("open", url="https://a.example.org/p"))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["oembed"]["title"], "Clip caption")
        self.assertEqual(result["oembed"]["author_name"], "Creator")
        self.assertIn("caption #tag", result["oembed"]["embed_text"])
        self.assertEqual(len(seen), 2)

    async def test_oembed_to_private_host_or_broken_json_does_not_fail_the_page(self):
        for href, response in (("http://127.0.0.1/o.json", None),
                               ("/o", httpx.Response(200, headers={"content-type": "application/json"}, content=b"{oops"))):
            page = (f'<html><head><link rel="alternate" type="application/json+oembed" href="{href}">'
                    f"</head><body><p>body</p></body></html>").encode()
            routes = {"https://a.example.org/p": httpx.Response(200, headers={"content-type": "text/html"}, content=page)}
            seen: list[str] = []
            if response is not None:
                routes["https://a.example.org/o"] = response
            session, client = session_for(routes, seen)
            async with client:
                result = await session.execute(WebLookupRequest("open", url="https://a.example.org/p"))
            with self.subTest(href=href):
                self.assertTrue(result["ok"], result)
                self.assertNotIn("oembed", result)
                self.assertNotIn("http://127.0.0.1/o.json", seen)


class WebOpenToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_accepts_exactly_one_of_result_id_or_url(self):
        from tests.test_tools import turn_for
        from tests.test_service_observation_integration import ObservationHarness
        harness = ObservationHarness()
        try:
            turn = turn_for(harness)
            both = await turn.web_open({"result_id": 1, "url": "https://a.example.org/"})
            neither = await turn.web_open({})
            bad = await turn.web_open({"url": "file:///C:/Windows/win.ini"})
            self.assertEqual(both["error"]["code"], "invalid_arguments")
            self.assertEqual(neither["error"]["code"], "invalid_arguments")
            self.assertEqual(bad["error"]["code"], "web_url_denied")
            with patch.object(turn.web, "execute", new=AsyncMock(return_value={"ok": True})) as execute:
                await turn.web_open({"url": "https://a.example.org/doc"})
            self.assertEqual(execute.await_args.args[0], WebLookupRequest("open", url="https://a.example.org/doc"))
        finally:
            await harness.close()

    async def test_pinned_network_backend_blocks_dns_rebinding(self):
        import httpcore
        from personal_agent.web_lookup import PinnedNetworkBackend
        backend = PinnedNetworkBackend()
        with self.assertRaises(httpcore.ConnectError):
            await backend.connect_tcp("127.0.0.1", 80)
        with self.assertRaises(httpcore.ConnectError):
            await backend.connect_tcp("rebind.example.org", 80)

    async def test_render_headless_edge_guards_and_execution(self):
        from personal_agent.web_lookup import _render_headless_edge
        # Denied URL
        self.assertIsNone(await _render_headless_edge("http://127.0.0.1/private"))

        # Mocked Edge execution on public URL
        class MockProc:
            returncode = 0
            async def communicate(self):
                return b"<html><body><p>Rendered JS Text</p></body></html>", b""

        with patch("personal_agent.web_lookup._public_url", new=AsyncMock(return_value=True)), \
             patch("personal_agent.web_lookup._find_edge_executable", return_value="msedge.exe"), \
             patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=MockProc())):
            result = await _render_headless_edge("https://a.example.org/spa")
            self.assertIn("Rendered JS Text", result)


if __name__ == "__main__":
    unittest.main()
