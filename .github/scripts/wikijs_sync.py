#!/usr/bin/env python3
"""
Publish a project's docs folder (Markdown, Quarto .qmd and Jupyter .ipynb)
to a Wiki.js v3 site through its REST API.

    WIKI_URL=https://documentation.jjsm.science \
    WIKI_API_KEY=... WIKI_SITE_ID=<site uuid> \
    python wikijs_sync.py --src docs --prefix astronomy/varistar --tags astronomy,python

  docs/index.md            -> /astronomy/varistar
  docs/install.md          -> /astronomy/varistar/install
  docs/tutorial.ipynb      -> /astronomy/varistar/tutorial   (rendered by Quarto)
  docs/api/core.qmd        -> /astronomy/varistar/api/core   (rendered by Quarto)

Notebooks are rendered with `quarto render --to gfm`, then:
  - math is moved into code ($x$ -> $`x`$, $$...$$ -> ```math) so Wiki.js'
    Markdown doesn't mangle it; the site's KaTeX script renders it
  - Quarto's cell wrapper <div>/::: lines are removed
  - figures are embedded as base64 data URIs, so no asset upload is needed
  - the first "# Title" (or YAML title) becomes the Wiki.js page title
  - a Page Tools block (GitHub / Edit / Cite buttons) is added on top

Only the Python standard library is used. Quarto must be on PATH for
.qmd/.ipynb files (and Jupyter too if you pass --execute).
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

EMBED_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_with_quarto(src: Path, execute: bool) -> tuple[str, Path]:
    """Render .qmd/.ipynb to GitHub-flavoured markdown. Returns (markdown, base_dir)."""
    # resolve() so macOS' /var -> /private/var symlink doesn't confuse Quarto's output path
    tmp = Path(tempfile.mkdtemp(prefix="wikisync-")).resolve()
    work = tmp / src.name
    shutil.copy2(src, work)
    # copy sibling files the notebook may reference (images, data, helper .py modules)
    for sib in src.parent.iterdir():
        if sib.is_file() and sib != src and sib.suffix.lower() not in {".md", ".mdx", ".qmd", ".ipynb"}:
            shutil.copy2(sib, tmp / sib.name)
    out = work.with_suffix(".md")
    cmd = ["quarto", "render", str(work), "--to", "gfm", "--output", out.name,
           "-M", "fig-format:png", "-M", "wrap:none"]
    if execute:
        cmd.append("--execute")
    elif src.suffix == ".qmd":
        cmd += ["--no-execute"]
    subprocess.run(cmd, check=True, cwd=tmp)
    return out.read_text(encoding="utf-8"), tmp


def protect_math(md: str) -> str:
    """Wiki.js v3 Markdown has no math and mangles {}, [], ^, ~ inside $...$.
    Move math into code, which the site's KaTeX script renders:
        $$...$$ (own lines) -> ```math fence      $x$ -> $`x`$
    Fenced code blocks and inline code are left untouched."""
    out = []
    parts = re.split(r"(^(?:```|~~~)[^\n]*\n.*?^(?:```|~~~)[ \t]*$)", md, flags=re.S | re.M)
    for i, part in enumerate(parts):
        if i % 2 == 1:                      # fenced code block
            out.append(part)
            continue
        # display math: $$ ... $$ (may span lines)
        def display(m):
            before = part[:m.start()].rsplit("\n", 1)[-1]
            after = part[m.end():].split("\n", 1)[0]
            tex = m.group(1).strip()
            if before.strip() or after.strip():       # inside a sentence
                return "`$$" + " ".join(tex.split()) + "$$`"
            return "```math\n" + tex + "\n```"
        part = re.sub(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", display, part, flags=re.S)
        # inline math, skipping inline code spans
        segs = re.split(r"(`+[^`]*?`+)", part)
        for j in range(0, len(segs), 2):
            segs[j] = re.sub(r"(?<![\\$\w])\$(?=\S)([^$\n]+?)(?<=\S)\$(?![\w$`])",
                             lambda m: "$`" + m.group(1) + "`$", segs[j])
        out.append("".join(segs))
    md = "".join(out)
    return re.sub(r"\n{3,}", "\n\n", md)


def strip_cell_wrappers(md: str) -> str:
    lines = []
    for line in md.splitlines():
        s = line.strip()
        if re.fullmatch(r"<div[^>]*class=\"[^\"]*(cell|quarto|panel-tabset)[^\"]*\"[^>]*>", s):
            continue
        if s == "</div>":
            continue
        if re.fullmatch(r":::+\s*(\{[^}]*\})?\s*", s):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def embed_images(md: str, base: Path) -> str:
    def repl(m):
        alt, target = m.group(1), m.group(2).strip()
        url = target.split()[0].strip("<>")
        if re.match(r"^(https?:|data:|/)", url):
            return m.group(0)
        path = (base / urllib.parse.unquote(url)).resolve()
        if not path.is_file():
            print(f"  ! image not found: {url}", file=sys.stderr)
            return m.group(0)
        mime = mimetypes.guess_type(path.name)[0] or ""
        if mime not in EMBED_TYPES:
            print(f"  ! cannot embed {mime or path.suffix} image ({url}); use png/jpg", file=sys.stderr)
            return m.group(0)
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f"![{alt}](data:{mime};base64,{b64})"
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md)


def split_title(md: str, fallback: str) -> tuple[str, str]:
    title = None
    fm = re.match(r"^---\n(.*?)\n---\n", md, flags=re.S)
    if fm:
        t = re.search(r"^title:\s*[\"']?(.+?)[\"']?\s*$", fm.group(1), flags=re.M)
        if t:
            title = t.group(1)
        md = md[fm.end():]
    if not title:
        h = re.match(r"^\s*#\s+(.+?)\s*#*\s*\n", md)
        if h:
            title = h.group(1)
            md = md[h.end():]
    return (title or fallback), md.lstrip()


ADMONITIONS = {"note": "NOTE", "info": "NOTE", "tip": "TIP", "important": "IMPORTANT",
               "warning": "WARNING", "caution": "WARNING", "danger": "CAUTION"}


def front_matter(text: str) -> dict:
    fm = re.match(r"^---\n(.*?)\n---(\n|$)", text, flags=re.S)
    out = {}
    if fm:
        for line in fm.group(1).splitlines():
            m = re.match(r"^(\w[\w-]*):\s*[\"']?(.*?)[\"']?\s*$", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def source_front_matter(path: Path) -> dict:
    """Front matter of a .md/.mdx/.qmd file, or of a notebook's first cell."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        cells = json.loads(text).get("cells", [])
        text = "".join(cells[0].get("source", "")) if cells else ""
    return front_matter(text)


def from_docusaurus(md: str) -> str:
    """Docusaurus MDX -> plain Markdown Wiki.js understands."""
    # MDX import/export lines
    md = re.sub(r"^(import|export)\s.*$\n?", "", md, flags=re.M)
    # :::tip Title ... :::  ->  > [!TIP] Title
    def adm(m):
        kind = ADMONITIONS.get(m.group(1).lower(), "NOTE")
        title = (m.group(2) or "").strip().strip("[]")
        body = m.group(3).strip("\n")
        head = f"> [!{kind}]" + (f" {title}" if title else "")
        return head + "\n" + "\n".join("> " + ln if ln.strip() else ">" for ln in body.splitlines()) + "\n"
    md = re.sub(r"^:::(note|info|tip|important|warning|caution|danger)[ \t]*([^\n]*)\n(.*?)^:::[ \t]*$",
                adm, md, flags=re.S | re.M | re.I)
    # leftover JSX components are not supported
    for tag in sorted(set(re.findall(r"<([A-Z][A-Za-z0-9]*)\b", md))):
        print(f"  ! MDX component <{tag}> is not supported by Wiki.js and will show as text", file=sys.stderr)
    return md


def rewrite_links(md: str, page_dir: str) -> str:
    """Relative Docusaurus links (./loading-data, ../api.md) -> absolute wiki paths."""
    def repl(m):
        text, url = m.group(1), m.group(2)
        if re.match(r"^(https?:|mailto:|data:|#|/)", url):
            return m.group(0)
        target, _, anchor = url.partition("#")
        target = re.sub(r"\.(mdx?|qmd|ipynb)$", "", target)
        parts = [p for p in page_dir.split("/") if p]
        for seg in target.split("/"):
            if seg in ("", "."):
                continue
            if seg == "..":
                parts = parts[:-1]
            else:
                parts.append(seg)
        if parts and parts[-1] in {"index", "intro", "README", "readme"}:
            parts = parts[:-1]
        return f"[{text}](/{'/'.join(parts)}{'#' + anchor if anchor else ''})"
    return re.sub(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)", repl, md)


def drop_duplicate_h1(md: str, title: str) -> str:
    m = re.match(r"^\s*#\s+(.+?)\s*\n", md)
    if m and m.group(1).strip().lower() == title.strip().lower():
        return md[m.end():].lstrip()
    return md


def build_page(src: Path, execute: bool) -> tuple[str, str]:
    fallback = src.stem.replace("-", " ").replace("_", " ").title()
    if src.suffix in {".qmd", ".ipynb"}:
        md, base = render_with_quarto(src, execute)
        try:
            # interactive outputs (plotly, widgets) need JS, which Wiki.js strips
            if re.search(r"<script\b", md):
                print(f"  ! {src.name}: interactive output (<script>) won't render on Wiki.js", file=sys.stderr)
            if re.search(r"<style\b", md):
                print(f"  ! {src.name}: HTML table/output with <style>; styling will be lost", file=sys.stderr)
            if 'class="panel-tabset"' in md:
                print(f"  ! {src.name}: panel-tabset flattened into consecutive sections", file=sys.stderr)
            for ref in sorted(set(re.findall(r"\?@[\w:-]+", md))):
                print(f"  ! {src.name}: unresolved cross-reference {ref} (cross-page @refs don't resolve)", file=sys.stderr)
            md = strip_cell_wrappers(md)
            # Quarto callouts come out as GitHub alerts; move their "### Title" onto the alert line
            md = re.sub(r"^> \[!(\w+)\]\n>\n> #{1,6} (.+)\n(>\n)?", r"> [!\1] \2\n", md, flags=re.M)
            md = protect_math(md)
            md = embed_images(md, base)
        finally:
            shutil.rmtree(base, ignore_errors=True)
    else:
        md = src.read_text(encoding="utf-8")
        if src.suffix == ".mdx":
            md = from_docusaurus(md)
        md = protect_math(md)
        md = embed_images(md, src.parent)
    title, body = split_title(md, fallback)
    return title, drop_duplicate_h1(body, title)


# --------------------------------------------------------------------------
# Wiki.js API
# --------------------------------------------------------------------------

class Wiki:
    def __init__(self, url, key, site):
        self.base = url.rstrip("/") + "/_api/sites/" + site
        self.key = key

    def _req(self, method, path, body=None, query=None):
        url = self.base + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "X-API-Key": self.key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            if e.code == 404 and method == "GET":
                return None
            detail = e.read().decode(errors="replace")[:500]
            raise SystemExit(f"{method} {url} -> HTTP {e.code}: {detail}")

    def find(self, path):
        res = self._req("GET", "/pages", query={"path": path})
        if not res:
            return None
        if isinstance(res, dict):
            for key in ("pages", "items", "data", "results"):
                if isinstance(res.get(key), list):
                    res = res[key]
                    break
        if isinstance(res, dict):
            return res if res.get("path", path) == path else None
        for p in res:
            if p.get("path") == path:
                return p
        return None

    def upsert(self, path, title, content, tags, dry_run=False):
        page = self.find(path)
        body = {"path": path, "title": title, "editor": "markdown", "content": content,
                "reasonForChange": "Synced from GitHub"}
        if tags:
            body["tags"] = tags
        if dry_run:
            print(f"  [dry-run] {'update' if page else 'create'} /{path} ({title})")
            return
        if page:
            self._req("PUT", f"/pages/{page['id']}", body)
            print(f"  updated /{path}")
        else:
            body["publishState"] = "published"
            self._req("POST", "/pages", body)
            print(f"  created /{path}")


# --------------------------------------------------------------------------

def wiki_path(prefix: str, rel: Path) -> str:
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1].lower() in {"index", "readme"}:
        parts = parts[:-1]
    return "/".join([p for p in [prefix.strip("/")] + parts if p])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="docs", help="docs folder in the repo")
    ap.add_argument("--prefix", required=True, help="wiki path prefix, e.g. astronomy/varistar")
    ap.add_argument("--tags", default="", help="comma-separated tags added to every page")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""),
                    help="owner/name for the Page Tools block (defaults to $GITHUB_REPOSITORY)")
    ap.add_argument("--branch", default=os.environ.get("GITHUB_REF_NAME", "main"), help="branch for edit links")
    ap.add_argument("--no-pagetools", action="store_true", help="do not add the Page Tools block on top")
    ap.add_argument("--execute", action="store_true", help="execute notebooks before rendering")
    ap.add_argument("--dry-run", action="store_true", help="render but do not publish")
    ap.add_argument("--only", nargs="*", help="only these files (relative to --src)")
    args = ap.parse_args()

    src = Path(args.src)
    files = sorted(p for p in src.rglob("*")
                   if p.suffix in {".md", ".mdx", ".qmd", ".ipynb"}
                   and ".ipynb_checkpoints" not in p.parts and not p.name.startswith("_")
                   and not any(part.endswith("_files") for part in p.parts))
    if args.only:
        wanted = {str(Path(o)) for o in args.only}
        files = [f for f in files if str(f.relative_to(src)) in wanted]

    wiki = None
    if not args.dry_run:
        try:
            wiki = Wiki(os.environ["WIKI_URL"], os.environ["WIKI_API_KEY"], os.environ["WIKI_SITE_ID"])
        except KeyError as e:
            raise SystemExit(f"missing environment variable {e}")
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    for f in files:
        rel = f.relative_to(src)
        meta = source_front_matter(f)
        slug = meta.get("slug", "")
        if slug.startswith("/"):                      # Docusaurus absolute slug ("/" = docs home)
            path = "/".join(p for p in [args.prefix.strip("/"), slug.strip("/")] if p)
        elif slug:
            path = wiki_path(args.prefix, rel.parent / slug)
        else:
            path = wiki_path(args.prefix, rel)
        print(f"{rel} -> /{path}")
        title, content = build_page(f, args.execute)
        content = rewrite_links(content, path if slug == "/" or rel.stem in {"index", "README"} else path.rsplit("/", 1)[0])
        if not args.no_pagetools:
            attrs = f'repo="{args.repo}" file="{f.as_posix()}" branch="{args.branch}"' if args.repo else ''
            tag = f"::block-pagetools{{{attrs}}}" if attrs else "::block-pagetools"
            content = f"{tag}\n::\n\n" + content
        if args.dry_run:
            out = Path("wikisync-preview") / rel.with_suffix(".md")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f"<!-- title: {title} -->\n{content}", encoding="utf-8")
            print(f"  [dry-run] wrote {out}")
        else:
            wiki.upsert(path, title, content, tags)


if __name__ == "__main__":
    main()