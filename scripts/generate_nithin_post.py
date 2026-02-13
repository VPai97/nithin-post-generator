#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from app.nithin_post_generator import NithinPostGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate X/LinkedIn drafts in Nithin Kamath's public voice"
    )
    parser.add_argument("--platform", choices=["x", "linkedin"], required=True, help="Target platform")
    parser.add_argument("--context", required=True, help="Core context for the post")
    parser.add_argument("--facts", action="append", default=[], help="Fact to include (repeatable)")
    parser.add_argument("--angle", default=None, help="Angle/stance (optional)")
    parser.add_argument("--cta", default=None, help="Optional CTA or question")
    parser.add_argument("--thread", action="store_true", help="Generate an X thread")
    parser.add_argument("--variants", type=int, default=3, help="Number of variants")
    parser.add_argument("--max-chars", type=int, default=None, help="Override per-post max chars (X only)")
    parser.add_argument("--allow-research", action="store_true", default=True, help="Enable research (default: on)")
    parser.add_argument("--research-query", default=None, help="Research query (optional)")
    parser.add_argument("--auto-research", action="store_true", default=True, help="Auto research if context is short (default: on)")
    parser.add_argument("--proofread", action="store_true", default=True, help="Proofread output (default: on)")
    parser.add_argument("--disable-research", action="store_true", help="Disable research")
    parser.add_argument("--disable-proofread", action="store_true", help="Disable proofreading")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generator = NithinPostGenerator()

    allow_research = args.allow_research and not args.disable_research
    proofread = args.proofread and not args.disable_proofread

    result = generator.generate(
        context=args.context,
        platform=args.platform,
        facts=args.facts,
        angle=args.angle,
        cta=args.cta,
        thread=args.thread,
        variants=args.variants,
        max_chars=args.max_chars,
        allow_research=allow_research,
        research_query=args.research_query,
        auto_research=args.auto_research,
        proofread=proofread
    )

    print(result.text)

    if result.warnings:
        print("\nWarnings:", file=sys.stderr)
        for warning in result.warnings:
            print(f"- {warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
