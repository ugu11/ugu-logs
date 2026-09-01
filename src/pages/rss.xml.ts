import rss from "@astrojs/rss";
import type { APIContext } from "astro";
import { getCollection } from "astro:content";
import { url } from "../lib/url";

export async function GET(context: APIContext) {
  const posts = (await getCollection("blog", ({ data }) => !data.draft)).sort(
    (a, b) => b.data.date.valueOf() - a.data.date.valueOf()
  );

  return rss({
    title: "ugu-logs",
    description: "AI notes and projects, written up as I go.",
    site: new URL(url(), context.site ?? "https://ugu11.github.io"),
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.date,
      link: url(`blog/${post.id}/`),
    })),
  });
}
