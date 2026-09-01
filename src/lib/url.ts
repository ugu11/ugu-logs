const base = import.meta.env.BASE_URL.replace(/\/+$/, "");

/** Build a site-absolute URL, tolerating a base with or without a trailing slash. */
export function url(path = ""): string {
  return `${base}/${path.replace(/^\/+/, "")}`;
}
