import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/postcss";

const conditions = ["onnxruntime-web-use-extern-wasm", "module", "browser", "development|production"];

export default defineConfig({
  // Relative assets make the same ZIP work under any repository or custom domain.
  base: "./",
  plugins: [react(), {
    name: "local-development-csp",
    apply: "serve",
    transformIndexHtml: html => html.replace(/<meta http-equiv="Content-Security-Policy"[^>]*>/, ""),
  }],
  resolve: { conditions },
  // Discover the lazily imported worker runtime before the first scan. Otherwise
  // Vite restarts a fresh development page when it first optimizes ORT.
  optimizeDeps: { include: ["onnxruntime-web/all"] },
  css: { postcss: { plugins: [tailwindcss()] } },
  worker: {
    format: "es",
    plugins: () => [{ name: "ort-self-hosted-runtime", config: () => ({ resolve: { conditions } }) }],
  },
  build: { outDir: "docs", emptyOutDir: true, chunkSizeWarningLimit: 1100 },
});
