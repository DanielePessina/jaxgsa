#!/usr/bin/env python3
"""Validate documentation coverage for the public jaxgsa surface.

Two levels. The first checks that every name in ``jaxgsa.__all__`` has a
heading or anchor in ``docs/api/index.md``. The second checks that every
public name each of those namespaces exports is written down somewhere under
``docs/``, so a new estimator keyword or result field cannot ship undocumented.

The second level reads the installed package rather than the source tree,
because a namespace re-exports names it does not define.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_DOC = ROOT / "docs" / "api" / "index.md"
DOCS_DIR = ROOT / "docs"
INIT_FILE = ROOT / "src" / "jaxgsa" / "__init__.py"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
HTML_ID_RE = re.compile(r'<a\s+id="([^"]+)"', re.IGNORECASE)


def _strip_inline_markup(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def _normalize(text: str) -> str:
    text = _strip_inline_markup(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _load_exports() -> list[str]:
    tree = ast.parse(INIT_FILE.read_text(encoding="utf-8"), filename=str(INIT_FILE))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if not isinstance(node.value, (ast.List, ast.Tuple)):
                        raise ValueError("__all__ must be a literal list or tuple")
                    exports: list[str] = []
                    for elt in node.value.elts:
                        if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                            raise ValueError("__all__ entries must be literal strings")
                        exports.append(elt.value)
                    return exports
    raise ValueError("Could not find __all__ in src/jaxgsa/__init__.py")


def _build_required_entries() -> list[str]:
    return sorted(_load_exports())


def _load_doc_tokens() -> tuple[set[str], str]:
    if not API_DOC.exists():
        raise FileNotFoundError(f"Missing API reference page: {API_DOC}")

    content = API_DOC.read_text(encoding="utf-8")
    tokens: set[str] = set()

    for line in content.splitlines():
        heading_match = HEADING_RE.match(line)
        if heading_match:
            tokens.add(_normalize(heading_match.group(2)))

    for html_id in HTML_ID_RE.findall(content):
        tokens.add(_normalize(html_id))

    normalized_doc = _normalize(content)
    return {token for token in tokens if token}, normalized_doc


def _entry_variants(entry: str) -> set[str]:
    variants = {
        _normalize(entry),
        _normalize(entry.replace(".", " ")),
        _normalize(entry.replace("_", " ")),
        _normalize(entry.replace(".", " ").replace("_", " ")),
    }
    if "." in entry:
        owner, member = entry.split(".", 1)
        variants.add(_normalize(f"{owner} {member}"))
        variants.add(_normalize(f"{owner} {member.replace('_', ' ')}"))
        variants.add(_normalize(member))
        variants.add(_normalize(member.replace("_", " ")))
    return {variant for variant in variants if variant}


def _member_coverage() -> list[str]:
    """Return ``"namespace.name"`` for every public member no doc page names.

    Returns:
        Sorted list of undocumented members. Empty when every public name of
        every exported namespace appears in some file under ``docs/``.
    """
    import importlib
    import inspect

    corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(DOCS_DIR.rglob("*.md"))
        if ".vitepress" not in path.parts
    )

    jaxgsa = importlib.import_module("jaxgsa")
    undocumented: list[str] = []
    for namespace in _load_exports():
        module = getattr(jaxgsa, namespace, None)
        if not inspect.ismodule(module):
            continue
        for member in getattr(module, "__all__", ()):
            if member.startswith("_"):
                continue
            if member not in corpus:
                undocumented.append(f"{namespace}.{member}")
    return sorted(undocumented)


def main() -> int:
    required_entries = _build_required_entries()
    doc_tokens, normalized_doc = _load_doc_tokens()

    missing = []
    for entry in required_entries:
        variants = _entry_variants(entry)
        if not any(variant in doc_tokens or variant in normalized_doc for variant in variants):
            missing.append(entry)

    if missing:
        print("API docs coverage check failed.", file=sys.stderr)
        print("Missing anchors/headings for:", file=sys.stderr)
        for entry in missing:
            print(f"  - {entry}", file=sys.stderr)
        return 1

    undocumented = _member_coverage()
    if undocumented:
        print("API docs coverage check failed.", file=sys.stderr)
        print("Public names no page under docs/ mentions:", file=sys.stderr)
        for entry in undocumented:
            print(f"  - {entry}", file=sys.stderr)
        print(
            "Document each on its method page, or drop it from that "
            "namespace's __all__ if it was never meant to be public.",
            file=sys.stderr,
        )
        return 1

    print(
        f"API namespace coverage OK: {len(required_entries)} entries documented "
        f"in {API_DOC.relative_to(ROOT)}, and every public member of each is "
        "named under docs/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
