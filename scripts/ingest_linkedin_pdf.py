#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import os
from datetime import date, datetime, timedelta
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


NOISE_LINES = {
    "like",
    "comment",
    "repost",
    "send",
    "following",
    "message",
    "all activity",
    "posts",
    "comments",
    "videos",
    "images",
    "more",
    "me",
    "for business",
    "reactivate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest LinkedIn activity PDF export and update style guide"
    )
    parser.add_argument("--pdf", required=True, help="Path to LinkedIn activity PDF export")
    parser.add_argument("--out", default="data/nithin_corpus.jsonl", help="Output JSONL corpus path")
    parser.add_argument("--append", action="store_true", help="Append to existing corpus instead of overwriting")
    parser.add_argument("--min-words", type=int, default=3, help="Minimum words per post to keep")
    parser.add_argument("--since", help="Keep posts on/after date (YYYY-MM-DD)")
    parser.add_argument("--until", help="Keep posts on/before date (YYYY-MM-DD)")
    parser.add_argument("--reference-date", help="Reference date for relative timestamps (YYYY-MM-DD)")
    parser.add_argument("--update-style", action="store_true", help="Update style guide with derived stats")
    parser.add_argument("--force-update-style", action="store_true", help="Override style guide lock")
    return parser.parse_args()


def run_pdftotext(pdf_path: str) -> str:
    result = subprocess.run(
        ["pdftotext", pdf_path, "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr.strip()}")
    return result.stdout


def is_time_line(line: str) -> bool:
    cleaned = line.replace("•", "").strip()
    if not cleaned:
        return False

    if parse_date(cleaned):
        return True

    rel = re.fullmatch(r"(\d+)\s*(d|w|mo|m|y|yr|h|hr|hrs|hours)", cleaned, re.IGNORECASE)
    return rel is not None


def parse_activity_date(label: str, reference_date: date) -> datetime | None:
    cleaned = label.replace("•", "").strip()
    cleaned = cleaned.replace("ago", "").strip()

    absolute = parse_date(cleaned)
    if absolute:
        return absolute

    match = re.fullmatch(r"(\d+)\s*(d|w|mo|m|y|yr|h|hr|hrs|hours)", cleaned, re.IGNORECASE)
    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2).lower()
    ref_dt = datetime(reference_date.year, reference_date.month, reference_date.day)

    if unit in {"d"}:
        return ref_dt - timedelta(days=value)
    if unit in {"w"}:
        return ref_dt - timedelta(weeks=value)
    if unit in {"h", "hr", "hrs", "hours"}:
        return ref_dt - timedelta(hours=value)
    if unit in {"y", "yr"}:
        try:
            return ref_dt.replace(year=ref_dt.year - value)
        except ValueError:
            # Feb 29 fallback
            return ref_dt.replace(month=2, day=28, year=ref_dt.year - value)
    if unit in {"mo", "m"}:
        year = ref_dt.year
        month = ref_dt.month - value
        while month <= 0:
            month += 12
            year -= 1
        day = min(ref_dt.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
        return datetime(year, month, day)

    return None


def clean_content(lines: list[str]) -> str:
    cleaned: list[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if text.lower() in NOISE_LINES:
            continue
        if text.lower().startswith("premium:"):
            continue
        if text.lower().startswith("reactivate"):
            continue
        if re.fullmatch(r"\d+:\d+.*", text.lower()):
            continue
        if re.fullmatch(r"[\d,]+", text):
            continue
        if re.search(r"[\d,]+ comments", text.lower()):
            continue
        if re.search(r"[\d,]+ reposts", text.lower()):
            continue
        if re.search(r"and [\d,]+ others", text.lower()):
            continue
        if re.fullmatch(r"\d+x", text.lower()):
            continue
        cleaned.append(text)

    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    return "\n".join(cleaned).strip()


def extract_posts(text: str, reference_date: date) -> list[dict]:
    lines = [line.strip() for line in text.splitlines()]
    posts: list[dict] = []
    i = 0
    while i < len(lines):
        if lines[i] == "Nithin Kamath":
            date_line_idx = None
            for j in range(i + 1, min(i + 12, len(lines))):
                if is_time_line(lines[j]):
                    date_line_idx = j
                    break
            if date_line_idx is None:
                i += 1
                continue
            raw_date = lines[date_line_idx]
            dt = parse_activity_date(raw_date, reference_date)

            content_lines = []
            k = date_line_idx + 1
            while k < len(lines) and lines[k] != "Nithin Kamath":
                content_lines.append(lines[k])
                k += 1

            text_block = clean_content(content_lines)
            if text_block:
                posts.append({
                    "platform": "linkedin",
                    "text": text_block,
                    "created_at": dt.isoformat() if dt else None,
                    "id": None,
                    "source": "linkedin_pdf",
                })
            i = k
        else:
            i += 1
    return posts


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    reference_date = date.today()
    if args.reference_date:
        ref = parse_date(args.reference_date)
        if not ref:
            raise ValueError("Invalid --reference-date, expected YYYY-MM-DD")
        reference_date = ref.date()

    text = run_pdftotext(str(pdf_path))
    posts = extract_posts(text, reference_date)

    since_dt = parse_date(args.since) if args.since else None
    until_dt = parse_date(args.until) if args.until else None

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
        style_path = Path("data/nithin_style_guide.json")
        style = json.loads(style_path.read_text()) if style_path.exists() else {}
        if style.get("locked") and not args.force_update_style:
            print("Style guide is locked. Skipping update. Use --force-update-style to override.")
        else:
            corpus_posts = cleaned
            if args.append:
                corpus_posts = read_corpus(out_path)
            style["derived"] = analyze_posts(corpus_posts)
            style_path.write_text(json.dumps(style, indent=2))
            style_updated = True

    print(f"Ingested {len(cleaned)} posts -> {out_path}")
    if since_dt or until_dt:
        print(f"Date filter: since={args.since or 'n/a'} until={args.until or 'n/a'}")
        print(f"Skipped (no date): {skipped_no_date}")
        print(f"Skipped (out of range): {skipped_out_of_range}")
    if style_updated:
        print("Updated data/nithin_style_guide.json with derived stats")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
