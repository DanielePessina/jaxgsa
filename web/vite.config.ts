import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

/**
 * Read the Python package version from the repository's pyproject.toml.
 *
 * The studio footer names the jaxgsa release it mirrors. Reading the version
 * here, at build time, keeps one source of truth: a release bumps
 * pyproject.toml only, and web/package.json never needs a hand edit.
 */
function jaxgsaVersion(): string {
  const pyproject = readFileSync(new URL("../pyproject.toml", import.meta.url), "utf8");
  const project = pyproject.split(/^\[project\]\s*$/m)[1]?.split(/^\[/m)[0] ?? "";
  const match = /^version\s*=\s*"([^"]+)"/m.exec(project);

  if (!match) {
    throw new Error("vite.config.ts: no [project] version found in ../pyproject.toml");
  }

  return match[1];
}

export default defineConfig(({ command }) => ({
  define: {
    __JAXGSA_VERSION__: JSON.stringify(jaxgsaVersion()),
  },
  // The repository is a GitHub Pages project site. Keep dev at `/`, while
  // production assets resolve below `/jaxgsa/` instead of `/assets`.
  base: process.env.VITE_BASE ?? (command === "build" ? "/jaxgsa/" : "/"),
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.spec.ts"],
    testTimeout: 60_000,
    hookTimeout: 60_000,
  },
}));
