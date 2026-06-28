#!/usr/bin/env python3
"""Validate VitePress prev/next pager order in built docs HTML."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "docs" / ".vitepress" / "dist"
CONFIG = ROOT / "docs" / ".vitepress" / "config.ts"
PAGER_RE = re.compile(
    r'<a class="VPLink link pager-link (?P<kind>prev|next)" href="(?P<href>[^"]+)"',
)
BASE_RE = re.compile(r"base:\s*['\"]([^'\"]+)['\"]")
SECTIONS = {
    "guide": ["getting-started", "methods", "benchmarks"],
    "examples": [
        "basic",
        "non-uniform-inputs",
        "save-load",
        "bootstrap",
        "multi-output",
        "xarray",
        "hdmr",
        "advanced-workflow",
        "efast",
        "dgsm",
        "pce",
        "hsic",
        "pawn",
        "batch_reactor",
    ],
}


def _load_base() -> str:
    if not CONFIG.exists():
        raise FileNotFoundError(f"Missing VitePress config: {CONFIG}")
    match = BASE_RE.search(CONFIG.read_text(encoding="utf-8"))
    if not match:
        raise ValueError("Could not determine VitePress base path from docs/.vitepress/config.ts")
    return match.group(1).rstrip("/")


def _expected_href(base: str, section: str, slug: str) -> str:
    return f"{base}/{section}/{slug}.html"


def _check_section(base: str, section: str, slugs: list[str]) -> list[str]:
    failures: list[str] = []
    for index, slug in enumerate(slugs):
        html_path = DIST / section / f"{slug}.html"
        if not html_path.exists():
            failures.append(f"Missing built page: {html_path.relative_to(ROOT)}")
            continue

        content = html_path.read_text(encoding="utf-8")
        pager_links = {
            match.group("kind"): match.group("href") for match in PAGER_RE.finditer(content)
        }
        page_href = _expected_href(base, section, slug)

        for kind, href in pager_links.items():
            if href == page_href:
                failures.append(f"{section}/{slug}.html has a self-referential {kind} link")

        expected_prev = None if index == 0 else _expected_href(base, section, slugs[index - 1])
        expected_next = (
            None if index == len(slugs) - 1 else _expected_href(base, section, slugs[index + 1])
        )

        actual_prev = pager_links.get("prev")
        actual_next = pager_links.get("next")

        if actual_prev != expected_prev:
            failures.append(
                (
                    f"{section}/{slug}.html prev mismatch: "
                    f"expected {expected_prev!r}, got {actual_prev!r}"
                )
            )
        if actual_next != expected_next:
            failures.append(
                (
                    f"{section}/{slug}.html next mismatch: "
                    f"expected {expected_next!r}, got {actual_next!r}"
                )
            )
    return failures


def main() -> int:
    if not DIST.exists():
        print(f"Missing VitePress build output: {DIST}", file=sys.stderr)
        return 1

    base = _load_base()
    failures: list[str] = []
    for section, slugs in SECTIONS.items():
        failures.extend(_check_section(base, section, slugs))

    if failures:
        print("VitePress pager check failed.", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("VitePress pager OK for guide and examples sections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
