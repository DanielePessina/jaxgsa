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
LINK_RE = re.compile(r"link:\s*['\"]([^'\"]+)['\"]")
# Sections of the sidebar that the pager check covers, in sidebar order.
CHECKED_SECTIONS = ("guide", "examples")


def _read_config() -> str:
    """Read the VitePress config file.

    Returns:
        The text of ``docs/.vitepress/config.ts``.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not CONFIG.exists():
        raise FileNotFoundError(f"Missing VitePress config: {CONFIG}")
    return CONFIG.read_text(encoding="utf-8")


def _load_base(config_text: str) -> str:
    """Get the site base path from the config text.

    Args:
        config_text: The text of the VitePress config file.

    Returns:
        The base path without a trailing slash.

    Raises:
        ValueError: If the config has no ``base`` field.
    """
    match = BASE_RE.search(config_text)
    if not match:
        raise ValueError("Could not determine VitePress base path from docs/.vitepress/config.ts")
    return match.group(1).rstrip("/")


def _load_section_slugs(config_text: str, section: str) -> list[str]:
    """Get the page slugs of one sidebar group, in sidebar order.

    The function reads the group that the config keys as ``'/<section>/'``. It
    then takes the ``link`` field of every entry in that group.

    Args:
        config_text: The text of the VitePress config file.
        section: The section name, such as ``"guide"``.

    Returns:
        The page slugs of the group, in the order the sidebar lists them.

    Raises:
        ValueError: If the group is missing, if the group is empty, or if an
            entry has a link that does not belong to the section.
    """
    key = f"'/{section}/'"
    start = config_text.find(key)
    if start == -1:
        raise ValueError(
            f"Could not find sidebar group {key} in docs/.vitepress/config.ts. "
            "The pager check needs that group."
        )

    open_bracket = config_text.find("[", start)
    if open_bracket == -1:
        raise ValueError(
            f"Could not read the entries of sidebar group {key} in docs/.vitepress/config.ts."
        )
    # Match brackets so a nested array, such as a collapsible sub-group, stays
    # inside the slice. A plain search for "]" would stop at the inner array
    # and drop every page after it without saying so.
    depth = 0
    close_bracket = -1
    for pos in range(open_bracket, len(config_text)):
        char = config_text[pos]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                close_bracket = pos
                break
    if close_bracket == -1:
        raise ValueError(
            f"Sidebar group {key} in docs/.vitepress/config.ts has no closing bracket."
        )

    prefix = f"/{section}/"
    slugs: list[str] = []
    for link in LINK_RE.findall(config_text[open_bracket:close_bracket]):
        if not link.startswith(prefix) or link == prefix:
            raise ValueError(
                f"Sidebar group {key} has the unexpected link {link!r}. "
                f"Every link of the group must start with {prefix!r} and name a page."
            )
        slugs.append(link[len(prefix) :])

    if not slugs:
        raise ValueError(f"Sidebar group {key} in docs/.vitepress/config.ts has no links.")
    return slugs


def _load_sections(config_text: str) -> dict[str, list[str]]:
    """Get the checked sidebar groups and their page slugs.

    Args:
        config_text: The text of the VitePress config file.

    Returns:
        A mapping of section name to page slugs, in sidebar order.

    Raises:
        ValueError: If a checked group is missing or cannot be read.
    """
    return {section: _load_section_slugs(config_text, section) for section in CHECKED_SECTIONS}


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
    """Check the pager links of every sidebar page.

    Returns:
        0 if all pager links are correct, 1 if not.
    """
    if not DIST.exists():
        print(f"Missing VitePress build output: {DIST}", file=sys.stderr)
        return 1

    try:
        config_text = _read_config()
        base = _load_base(config_text)
        sections = _load_sections(config_text)
    except (FileNotFoundError, ValueError) as error:
        print("VitePress pager check failed.", file=sys.stderr)
        print(f"  - {error}", file=sys.stderr)
        return 1

    failures: list[str] = []
    for section, slugs in sections.items():
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
