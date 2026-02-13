import json
import re
from collections import Counter
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "this", "that",
    "these", "those", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "as", "at", "by", "from",
    "it", "its", "we", "our", "you", "your", "i", "me", "my", "us",
}


def normalize_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def is_emoji(char: str) -> bool:
    code = ord(char)
    ranges = [
        (0x1F300, 0x1FAFF),  # Misc symbols and pictographs
        (0x2600, 0x26FF),    # Misc symbols
        (0x2700, 0x27BF),    # Dingbats
        (0x1F1E6, 0x1F1FF),  # Flags
    ]
    return any(start <= code <= end for start, end in ranges)


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None

    cleaned = value.strip()
    cleaned = cleaned.replace("·", "").replace("UTC", "UTC").replace("  ", " ")

    # RFC 2822 / Twitter created_at
    try:
        dt = parsedate_to_datetime(cleaned)
        if dt:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    # ISO variants
    try:
        iso = cleaned.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S %Z",
        "%Y-%m-%d %H:%M",
        "%b %d, %Y",
        "%b %d, %Y %H:%M",
        "%b %d, %Y %I:%M %p",
        "%b %d, %Y %I:%M %p %Z",
        "%d %b %Y",
        "%d %b %Y %H:%M",
        "%d %b %Y %I:%M %p",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

    return None


def analyze_posts(posts: list[dict]) -> dict:
    per_platform = {"x": [], "linkedin": []}
    for post in posts:
        if post.get("platform") in per_platform:
            per_platform[post["platform"]].append(post)

    derived = {
        "analysis_date": str(date.today()),
        "sample_size": {p: len(items) for p, items in per_platform.items()},
        "avg_words_per_post": {},
        "avg_sentence_words": {},
        "question_rate": {},
        "emoji_rate": {},
        "link_rate": {},
        "common_openers": {},
        "common_closers": {},
        "common_phrases": {}
    }

    for platform, items in per_platform.items():
        if not items:
            continue
        word_counts = []
        sentence_lengths = []
        question_posts = 0
        emoji_posts = 0
        link_posts = 0
        opener_counter = Counter()
        closer_counter = Counter()
        phrase_counter = Counter()

        for post in items:
            text = normalize_text(post.get("text", ""))
            if not text:
                continue
            word_counts.append(len(tokenize(text)))
            sentences = split_sentences(text)
            for sentence in sentences:
                sentence_lengths.append(len(tokenize(sentence)))
            if "?" in text:
                question_posts += 1
            if any(is_emoji(ch) for ch in text):
                emoji_posts += 1
            if "http" in post.get("text", "") or "www." in post.get("text", ""):
                link_posts += 1

            words = tokenize(text)
            if len(words) >= 3:
                opener_counter[" ".join(words[:3])] += 1
                closer_counter[" ".join(words[-3:])] += 1

            for i in range(len(words) - 2):
                phrase = " ".join(words[i:i+3])
                if all(w in STOPWORDS for w in phrase.split()):
                    continue
                phrase_counter[phrase] += 1

        derived["avg_words_per_post"][platform] = round(sum(word_counts) / max(len(word_counts), 1), 2)
        derived["avg_sentence_words"][platform] = round(sum(sentence_lengths) / max(len(sentence_lengths), 1), 2)
        derived["question_rate"][platform] = round(question_posts / max(len(items), 1), 3)
        derived["emoji_rate"][platform] = round(emoji_posts / max(len(items), 1), 3)
        derived["link_rate"][platform] = round(link_posts / max(len(items), 1), 3)
        derived["common_openers"][platform] = [k for k, _ in opener_counter.most_common(8)]
        derived["common_closers"][platform] = [k for k, _ in closer_counter.most_common(8)]
        derived["common_phrases"][platform] = [k for k, _ in phrase_counter.most_common(12)]

    return derived


def write_corpus(path: Path, posts: list[dict], append: bool):
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for post in posts:
            f.write(json.dumps(post, ensure_ascii=False) + "\n")


def read_corpus(path: Path) -> list[dict]:
    if not path.exists():
        return []
    posts: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                posts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return posts
