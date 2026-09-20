# Vendored anti-slop provenance

- Source repository: unknown; the plugin was supplied by the local
  `install-anti-slop` Codex skill rather than fetched from a repository.
- Source snapshot: aggregate SHA-256
  `254c9bbbeef2af535480d0c1a9fcbf4f6b0a9246a06ff8c21a52831a39ede21c`
  over the skill's `assets/anti-slop/` files, sorted by source path.
- Source location at installation:
  `.agents/skills/install-anti-slop/assets/anti-slop/`.
- Installed entry point: `web/tools/oxlint/anti-slop/index.ts`.
- Optional Effect entry point: `web/tools/oxlint/anti-slop/effect/index.ts`;
  not registered because `web/package.json` has no direct `effect` dependency.
- Installed on: 2026-09-18.
- Intentional deviations: none. Generic rules and bundled vendored assets were
  copied unchanged; this provenance file was added beside the entry point.

The nested `vendor/eslint-stylistic/UPSTREAM.md` records the provenance of the
readable-spacing rule's upstream Stylistic source.
