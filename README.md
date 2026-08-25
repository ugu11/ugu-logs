# ugu-logs

A simple blog for writing about AI topics and projects, built with [Astro](https://astro.build)
+ TypeScript and hosted for free on GitHub Pages.

Live at: https://ugu11.github.io/ugu-logs/

## Writing a post

Add a markdown file to `src/content/blog/`, e.g. `src/content/blog/my-new-post.md`:

```md
---
title: "My New Post"
description: "One sentence describing the post."
date: 2026-08-25
tags: ["llms", "notes"]
---

Your content here, in regular markdown.
```

The filename becomes the post's URL slug (`/blog/my-new-post/`). Set `draft: true`
in the frontmatter to keep a post out of the build until it's ready.

## Local development

```bash
npm install
npm run dev
```

Open http://localhost:4321.

## Deploying

Pushing to `master` triggers `.github/workflows/deploy.yml`, which builds the site
and deploys it to GitHub Pages automatically.

One-time setup on GitHub: in the repo's **Settings → Pages**, set **Source** to
**GitHub Actions**.

## Project structure

```
src/
  content/blog/     markdown posts
  content.config.ts post frontmatter schema
  layouts/          shared page layout
  pages/            routes (home, post page, about, RSS feed)
  styles/           global CSS
```
