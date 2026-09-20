import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig(({ command }) => ({
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
