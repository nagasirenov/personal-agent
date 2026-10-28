"""Presentation policy, independent from capture and model inference."""
from __future__ import annotations

import re


def plain_text(text: str) -> str:
    """Remove presentation markup while preserving code, paths and link targets."""
    blocks = []

    def keep_code(match):
        blocks.append(match.group(2).rstrip("\n"))
        return f"\x00CODE{len(blocks) - 1}\x00"

    text = re.sub(r"^\s*(```|~~~)[^\n]*\n(.*?)^\s*\1\s*$", keep_code, text, flags=re.M | re.S)

    def keep_inline(match):
        blocks.append(match.group(1))
        return f"\x00CODE{len(blocks) - 1}\x00"

    text = re.sub(r"`([^`\n]+)`", keep_inline, text)
    text = re.sub(r"^\s*```[^\n]*\n?", "", text, flags=re.M)
    text = re.sub(r"^\s*~~~[^\n]*\n?", "", text, flags=re.M)
    text = re.sub(r"!?(\[([^\]\n]+)\])\(([^)\n]+)\)", r"\2 (\3)", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:---+|\*\*\*+)\s*$", "", text, flags=re.M)
    for marker in ("**", "__", "~~"):
        escaped = re.escape(marker)
        text = re.sub(escaped + r"([^\n]+?)" + escaped, r"\1", text)
    text = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", text)
    for index, block in enumerate(blocks):
        text = text.replace(f"\x00CODE{index}\x00", block)
    return text.strip()


def format_response(text: str, preferences: dict) -> str:
    if preferences.get("response_format") == "plain_text" or preferences.get("markdown_allowed") is False:
        return plain_text(text)
    return text
