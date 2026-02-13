#!/usr/bin/env python3
import argparse
import html
import json
import sys
import os
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from app.nithin_corpus_utils import (
    analyze_posts,
    normalize_text,
    parse_date,
    read_corpus,
    tokenize,
    write_corpus,
)


class NitterParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_item = False
        self.item_depth = 0
        self.current_text: list[str] = []
        self.current_date: str | None = None
        self.posts: list[dict] = []
        self.in_content = False
        self.content_depth = 0
        self.in_date = False
        self.date_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_attr = attrs_dict.get("class", "")
        if "timeline-item" in class_attr:
            self.in_item = True
            self.item_depth = 1
            self.current_text = []
            self.current_date = None
            return

        if self.in_item:
            self.item_depth += 1

        if "tweet-content" in class_attr:
            self.in_content = True
            self.content_depth = 1
            self.current_text = []
            return
        if self.in_content:
            self.content_depth += 1

        if "tweet-date" in class_attr:
            self.in_date = True
            self.date_depth = 1
            return
        if self.in_date:
            self.date_depth += 1

        if self.in_date and tag == "a":
            title = attrs_dict.get("title")
            if title:
                self.current_date = title

    def handle_endtag(self, tag):
        if self.in_content:
            self.content_depth -= 1
            if self.content_depth <= 0:
                self.in_content = False
                self.content_depth = 0
        if self.in_date:
            self.date_depth -= 1
            if self.date_depth <= 0:
                self.in_date = False
                self.date_depth = 0
        if self.in_item:
            self.item_depth -= 1
            if self.item_depth <= 0:
                text = html.unescape("".join(self.current_text)).strip()
                if text:
                    self.posts.append({
                        "text": text,
                        "created_at": self.current_date
                    })
                self.in_item = False
                self.item_depth = 0
                self.current_text = []
                self.current_date = None

    def handle_data(self, data):
        if self.in_content:
            self.current_text.append(data)


class LinkedInHtmlParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_content = False
        self.depth = 0
        self.buffer: list[str] = []
        self.posts: list[dict] = []
        self.capture_classes = {
            "feed-shared-update-v2__commentary",
            "update-components-text",
            "break-words",
        }

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_attr = attrs_dict.get("class", "")
        if any(cls in class_attr for cls in self.capture_classes):
            self.in_content = True
            self.depth = 1
            self.buffer = []
            return
        if self.in_content:
            self.depth += 1

    def handle_endtag(self, tag):
        if not self.in_content:
            return
        self.depth -= 1
        if self.depth <= 0:
            text = html.unescape("".join(self.buffer)).strip()
            if text:
                self.posts.append({"text": text, "created_at": None})
            self.in_content = False
            self.depth = 0
            self.buffer = []

    def handle_data(self, data):
        if self.in_content:
            self.buffer.append(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape public posts with explicit opt-in flags (use only where allowed)"
    )
    parser.add_argument("--platform", choices=["x", "linkedin"], required=True)
    parser.add_argument("--profile", help="Profile handle (e.g., Nithin0dha)")
    parser.add_argument("--mode", choices=["nitter", "html"], default="nitter")
    parser.add_argument("--html", help="Path to saved HTML file or folder (manual save)")
    parser.add_argument("--nitter-instance", default="https://nitter.net")
    parser.add_argument("--max-posts", type=int, default=50)
    parser.add_argument("--out", default="data/nithin_corpus.jsonl")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--since", help="Keep posts on/after date (YYYY-MM-DD)")
    parser.add_argument("--until", help="Keep posts on/before date (YYYY-MM-DD)")
    parser.add_argument("--update-style", action="store_true")
    parser.add_argument("--force-update-style", action="store_true", help="Override style guide lock")
    parser.add_argument("--i-acknowledge-terms", action="store_true")
    parser.add_argument("--i-acknowledge-risk", action="store_true")
    return parser.parse_args()


def ensure_acknowledged(args: argparse.Namespace) -> bool:
    if not (args.i_acknowledge_terms and args.i_acknowledge_risk):
        print(
            "Refusing to scrape without explicit opt-in. Re-run with "
            "--i-acknowledge-terms and --i-acknowledge-risk.",
            file=sys.stderr,
        )
        return False
    return True


def fetch_url(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120 Safari/537.36"
        },
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        return response.read().decode("utf-8", errors="ignore")


def parse_nitter_profile(profile: str, instance: str, max_posts: int) -> list[dict]:
    url = f"{instance.rstrip('/')}/{profile}"
    html_text = fetch_url(url)
    parser = NitterParser()
    parser.feed(html_text)
    posts = []
    for entry in parser.posts[:max_posts]:
        posts.append({
            "platform": "x",
            "text": entry.get("text"),
            "created_at": entry.get("created_at"),
            "id": None,
            "source": "nitter"
        })
    return posts


def parse_html_files(html_path: str, platform: str, max_posts: int) -> list[dict]:
    path = Path(html_path)
    files = []
    if path.is_dir():
        files = list(path.rglob("*.html"))
    elif path.is_file():
        files = [path]

    if not files:
        raise FileNotFoundError("No HTML files found to parse.")

    posts: list[dict] = []
    for file_path in files:
        content = file_path.read_text(errors="ignore")
        if platform == "x":
            parser = NitterParser()
        else:
            parser = LinkedInHtmlParser()
        parser.feed(content)
        for entry in parser.posts:
            text = entry.get("text") if isinstance(entry, dict) else entry
            created_at = entry.get("created_at") if isinstance(entry, dict) else None
            posts.append({
                "platform": platform,
                "text": text,
                "created_at": created_at,
                "id": None,
                "source": "html_saved"
            })
            if len(posts) >= max_posts:
                return posts
    return posts


def update_style(posts: list[dict], out_path: Path, append: bool, force: bool) -> bool:
    style_path = Path("data/nithin_style_guide.json")
    style = json.loads(style_path.read_text()) if style_path.exists() else {}
    if style.get("locked") and not force:
        print("Style guide is locked. Skipping update. Use --force-update-style to override.")
        return False
    corpus_posts = posts
    if append:
        corpus_posts = read_corpus(out_path)
    style["derived"] = analyze_posts(corpus_posts)
    style_path.write_text(json.dumps(style, indent=2))
    return True


def main() -> int:
    args = parse_args()
    if not ensure_acknowledged(args):
        return 2

    if args.platform == "linkedin" and args.mode == "nitter":
        print(
            "Automated LinkedIn scraping is disabled. "
            "Use LinkedIn export or provide saved HTML with --mode html.",
            file=sys.stderr,
        )
        return 2

    posts: list[dict] = []
    if args.mode == "nitter":
        if not args.profile:
            print("Missing --profile for nitter mode.", file=sys.stderr)
            return 2
        posts = parse_nitter_profile(args.profile, args.nitter_instance, args.max_posts)
    elif args.mode == "html":
        if not args.html:
            print("Missing --html path for html mode.", file=sys.stderr)
            return 2
        posts = parse_html_files(args.html, args.platform, args.max_posts)

    since_dt = parse_date(args.since) if args.since else None
    until_dt = parse_date(args.until) if args.until else None

    # Normalize and filter
    cleaned = []
    seen = set()
    skipped_no_date = 0
    skipped_out_of_range = 0
    for post in posts:
        post["text"] = normalize_text(post.get("text", ""))
        if not post["text"]:
            continue
        if len(tokenize(post["text"])) < args.min_words:
            continue

        if since_dt or until_dt:
            dt = parse_date(post.get("created_at"))
            if not dt:
                skipped_no_date += 1
                continue
            if since_dt and dt < since_dt:
                skipped_out_of_range += 1
                continue
            if until_dt and dt > until_dt:
                skipped_out_of_range += 1
                continue

        key = post["platform"] + ":" + post["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(post)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_corpus(out_path, cleaned, append=args.append)

    style_updated = False
    if args.update_style:
        style_updated = update_style(cleaned, out_path, append=args.append, force=args.force_update_style)

    print(f"Scraped {len(cleaned)} posts -> {out_path}")
    if since_dt or until_dt:
        print(f"Date filter: since={args.since or 'n/a'} until={args.until or 'n/a'}")
        print(f"Skipped (no date): {skipped_no_date}")
        print(f"Skipped (out of range): {skipped_out_of_range}")
    if style_updated:
        print("Updated data/nithin_style_guide.json with derived stats")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
