#!/usr/bin/env python3
"""Validate API docs coverage for the public gsax surface."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_DOC = ROOT / "docs" / "api" / "index.md"
INIT_FILE = ROOT / "src" / "gsax" / "__init__.py"
CLASS_FILES = {
    "Problem": ROOT / "src" / "gsax" / "problem.py",
    "SamplingResult": ROOT / "src" / "gsax" / "sampling.py",
    "SAResult": ROOT / "src" / "gsax" / "sobol" / "_result.py",
    "HDMRResult": ROOT / "src" / "gsax" / "hdmr" / "_result.py",
    "PCEResult": ROOT / "src" / "gsax" / "pce" / "_result.py",
    "EFASTResult": ROOT / "src" / "gsax" / "efast" / "_result.py",
    "DGSMResult": ROOT / "src" / "gsax" / "dgsm" / "_result.py",
    "PAWNResult": ROOT / "src" / "gsax" / "pawn" / "_result.py",
    "HSICResult": ROOT / "src" / "gsax" / "hsic" / "_result.py",
}
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
    raise ValueError("Could not find __all__ in src/gsax/__init__.py")


def _collect_class_contracts(path: Path, class_name: str) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            fields: set[str] = set()
            properties: set[str] = set()
            methods: set[str] = set()
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    name = item.target.id
                    if not name.startswith("_"):
                        fields.add(name)
                elif isinstance(item, ast.FunctionDef):
                    if item.name.startswith("_"):
                        continue
                    decorators = {
                        getattr(decorator, "id", None)
                        for decorator in item.decorator_list
                        if isinstance(decorator, ast.Name)
                    }
                    if "property" in decorators:
                        properties.add(item.name)
                    else:
                        methods.add(item.name)
            return {"fields": fields, "properties": properties, "methods": methods}
    raise ValueError(f"Could not find class {class_name} in {path}")


def _build_required_entries() -> list[str]:
    exports = _load_exports()
    required = set(exports)
    for class_name, path in CLASS_FILES.items():
        contracts = _collect_class_contracts(path, class_name)
        required.add(class_name)
        for field in contracts["fields"]:
            required.add(f"{class_name}.{field}")
        for prop in contracts["properties"]:
            required.add(f"{class_name}.{prop}")
        for method in contracts["methods"]:
            required.add(f"{class_name}.{method}")
    return sorted(required)


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

    print(
        f"API docs coverage OK: {len(required_entries)} entries documented "
        f"in {API_DOC.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
