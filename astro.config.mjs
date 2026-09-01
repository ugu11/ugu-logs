import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeCitation from "rehype-citation";

// GitHub Pages project site: https://ugu11.github.io/ugu-logs/
export default defineConfig({
  site: "https://ugu11.github.io",
  base: "/ugu-logs",
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        rehypeKatex,
        [
          rehypeCitation,
          {
            bibliography: "src/content/blog/references.bib",
            csl: "vancouver",
            linkCitations: true,
          },
        ],
      ],
    }),
  },
});
