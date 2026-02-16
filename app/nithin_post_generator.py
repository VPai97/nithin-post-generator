import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from anthropic import Anthropic
except ImportError:  # Allows fallback templates without the dependency
    Anthropic = None

try:
    import language_tool_python
except ImportError:
    language_tool_python = None

import requests

from app.research_client import ResearchClient, ResearchResult


@dataclass
class GeneratedPost:
    text: str
    warnings: list[str]
    metadata: dict


class NithinPostGenerator:
    """Generate X/LinkedIn drafts in Nithin Kamath's public voice."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.style = self._load_json("nithin_style_guide.json")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        self.llm_provider = None
        if Anthropic is not None and api_key:
            self.client = Anthropic(api_key=api_key)
            self.llm_provider = "anthropic"

        self.ollama_model = os.environ.get("OLLAMA_MODEL")
        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if self.client is None and self.ollama_model:
            self.llm_provider = "ollama"

        self.lt_tool = None
        if language_tool_python is not None:
            try:
                self.lt_tool = language_tool_python.LanguageToolPublicAPI("en-US")
            except Exception:
                self.lt_tool = None
        self.research = ResearchClient()

    def _load_json(self, filename: str) -> dict:
        path = self.data_dir / filename
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def is_available(self) -> bool:
        return self.llm_provider in {"anthropic", "ollama"}

    def generate(
        self,
        context: str,
        platform: str,
        facts: list[str],
        angle: Optional[str] = None,
        cta: Optional[str] = None,
        thread: bool = False,
        variants: int = 3,
        max_chars: Optional[int] = None,
        allow_research: bool = True,
        research_query: Optional[str] = None,
        auto_research: bool = True,
        proofread: bool = True
    ) -> GeneratedPost:
        warnings: list[str] = []
        research_results: list[ResearchResult] = []
        research_used = False
        research_summary = ""
        research_query_used = None

        if allow_research:
            if not self.research.is_available():
                warnings.append("Research requested but search API key not configured")
            else:
                research_query_used = self._pick_research_query(context, research_query, auto_research)
                if research_query_used:
                    research_results = self.research.search(research_query_used, max_results=5)
                    if research_results:
                        research_used = True
                        research_summary = self._summarize_research(research_results, context)
                    else:
                        warnings.append("Research returned no results")
                else:
                    warnings.append("Research skipped (context sufficient or no query provided)")

        if not self.is_available():
            text = self._offline_generate(
                context=context,
                platform=platform,
                facts=facts,
                angle=angle,
                cta=cta,
                thread=thread,
                variants=variants,
                max_chars=max_chars
            )
            if proofread:
                proofread_text = self._proofread(text, platform, thread)
                if proofread_text:
                    text = proofread_text
                else:
                    warnings.append("Proofread step failed, returning original draft")
            warnings.extend(self._basic_warnings(text, platform, max_chars, thread))
            return GeneratedPost(
                text=text,
                warnings=warnings + ["No LLM configured. Returned a style-based offline draft."],
                metadata={
                    "platform": platform,
                    "thread": thread,
                    "variants": variants,
                    "llm": False,
                    "research_used": research_used,
                    "research_query": research_query_used,
                    "research_summary": research_summary,
                    "sources": self._format_sources(research_results)
                }
            )

        system_prompt = self._build_system_prompt(platform, thread, variants, max_chars)
        user_prompt = self._build_user_prompt(
            context,
            facts,
            angle,
            cta,
            research_results,
            research_summary
        )

        try:
            text = self._llm_generate(system_prompt, user_prompt, max_tokens=1200)
        except Exception as exc:
            text = self._fallback_template(context, platform, facts, angle, cta, thread)
            return GeneratedPost(
                text=text,
                warnings=warnings + [f"LLM error: {exc}. Returned a structured draft template."],
                metadata={
                    "platform": platform,
                    "thread": thread,
                    "variants": 1,
                    "llm": False,
                    "research_used": research_used,
                    "research_query": research_query_used,
                    "research_summary": research_summary,
                    "sources": self._format_sources(research_results)
                }
            )

        if proofread:
            proofread_text = self._proofread(text, platform, thread)
            if proofread_text:
                text = proofread_text
            else:
                warnings.append("Proofread step failed, returning original draft")

        warnings.extend(self._basic_warnings(text, platform, max_chars, thread))
        return GeneratedPost(
            text=text,
            warnings=warnings,
            metadata={
                "platform": platform,
                "thread": thread,
                "variants": variants,
                "llm": True,
                "research_used": research_used,
                "research_query": research_query_used,
                "research_summary": research_summary,
                "sources": self._format_sources(research_results)
            }
        )

    def _build_system_prompt(self, platform: str, thread: bool, variants: int, max_chars: Optional[int]) -> str:
        style = self.style
        platform_rules = style.get("platforms", {}).get(platform, {})

        max_chars_rule = max_chars or platform_rules.get("max_chars")
        word_target = platform_rules.get("target_words") or platform_rules.get("single_post_words")
        derived = style.get("derived", {})
        observed_openers = derived.get("common_openers", {}).get(platform, [])[:5]
        observed_closers = derived.get("common_closers", {}).get(platform, [])[:5]
        observed_phrases = derived.get("common_phrases", {}).get(platform, [])[:8]
        avg_sentence_words = derived.get("avg_sentence_words", {}).get(platform)
        question_rate = derived.get("question_rate", {}).get(platform)

        system_prompt = f"""You are ghostwriting public posts for Nithin Kamath (CEO of Zerodha).
Write in his public voice: clear, practical, data-backed, candid, and humble.

Tone:
{", ".join(style.get("tone", []))}

Do:
{chr(10).join("- " + d for d in style.get("do", []))}

Don't:
{chr(10).join("- " + d for d in style.get("dont", []))}

Language & formatting:
{chr(10).join("- " + r for r in style.get("language", {}).get("formatting", []))}
Preferred abbreviations: {", ".join(style.get("language", {}).get("preferred_abbreviations", []))}

Signature phrases (use sparingly when it fits):
{", ".join(style.get("signature_phrases", []))}

Guardrails:
{chr(10).join("- " + g for g in style.get("guardrails", []))}

Platform: {platform.upper()}
Thread: {"yes" if thread else "no"}
Target words: {word_target}
Max chars per post: {max_chars_rule}

Observed patterns from recent public posts (use lightly; don't force):
- Common openers: {", ".join(observed_openers) if observed_openers else "n/a"}
- Common closers: {", ".join(observed_closers) if observed_closers else "n/a"}
- Common phrases: {", ".join(observed_phrases) if observed_phrases else "n/a"}
- Avg sentence words: {avg_sentence_words if avg_sentence_words is not None else "n/a"}
- Question rate: {question_rate if question_rate is not None else "n/a"}

Output format:
- Provide {variants} distinct variants.
- Separate each variant with a blank line and the line: ---"""

        if platform == "x":
            if thread:
                system_prompt += "\n- For threads, label each tweet as '1/N', '2/N', etc."
            else:
                system_prompt += "\n- For single posts, output a single tweet per variant."
        else:
            system_prompt += "\n- For LinkedIn, use 3-6 short paragraphs."

        return system_prompt

    def _build_user_prompt(
        self,
        context: str,
        facts: list[str],
        angle: Optional[str],
        cta: Optional[str],
        research_results: list[ResearchResult],
        research_summary: str
    ) -> str:
        facts_block = "\n".join(f"- {fact}" for fact in facts) if facts else "(none provided)"
        angle_block = angle if angle else "(none)"
        cta_block = cta if cta else "(none)"
        research_block = ""
        if research_results:
            research_lines = []
            for i, item in enumerate(research_results, start=1):
                snippet = item.snippet.strip()
                if len(snippet) > 280:
                    snippet = snippet[:277] + "..."
                research_lines.append(f"[{i}] {item.title} — {snippet} (Source: {item.url})")
            research_block = "\n".join(research_lines)

        return f"""Context:
{context}

Facts to include (only these can be stated as facts):
{facts_block}

Angle / stance:
{angle_block}

Optional CTA or question:
{cta_block}

Research snippets (use only if needed; cite with [#] when you use them):
{research_block if research_block else "(none)"}

Research summary (if helpful):
{research_summary if research_summary else "(none)"}

If a key fact is missing, insert [ADD FACT] placeholder. Do not invent numbers."""

    def _pick_research_query(
        self,
        context: str,
        research_query: Optional[str],
        auto_research: bool
    ) -> Optional[str]:
        if research_query:
            return research_query.strip()
        if not auto_research:
            return None
        if len(context.split()) < 20:
            return context.strip()
        return None

    def _summarize_research(
        self,
        results: list[ResearchResult],
        context: str
    ) -> str:
        if not results or not self.is_available():
            return ""

        sources_block = "\n".join(
            f"[{i+1}] {r.title}: {r.snippet}" for i, r in enumerate(results)
        )

        system_prompt = (
            "You are a research assistant. Summarize the sources into 3-5 bullets. "
            "Use only the provided snippets. Do not add new facts."
        )
        user_prompt = f"""Context:
{context}

Sources:
{sources_block}

Return 3-5 concise bullets."""

        try:
            return self._llm_generate(system_prompt, user_prompt, max_tokens=300)
        except Exception:
            return ""

    def _proofread(self, draft: str, platform: str, thread: bool) -> Optional[str]:
        if self.is_available():
            system_prompt = (
                "You are a careful editor. Fix grammar, spelling, and punctuation only. "
                "Do not change meaning, tone, or add/remove facts. Preserve citations like [1]. "
                "Keep thread numbering as-is."
            )
            user_prompt = f"Proofread this {platform} draft:\n\n{draft}"

            try:
                edited = self._llm_generate(system_prompt, user_prompt, max_tokens=900)
                if not edited:
                    return None
                if len(edited) > len(draft) * 1.2:
                    return None
                return edited
            except Exception:
                return None

        if self.lt_tool is None:
            return None

        try:
            corrected = self.lt_tool.correct(draft)
            if len(corrected) > len(draft) * 1.2:
                return None
            return corrected
        except Exception:
            return None

    def _llm_generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text.strip()

        if self.llm_provider == "ollama":
            payload = {
                "model": self.ollama_model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {"num_predict": max_tokens}
            }
            url = f"{self.ollama_host.rstrip('/')}/api/chat"
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {})
            return (message.get("content") or "").strip()

        raise RuntimeError("No LLM provider configured")

    def _format_sources(self, results: list[ResearchResult]) -> list[dict]:
        return [
            {"title": r.title, "url": r.url, "snippet": r.snippet}
            for r in results
        ]

    def _basic_warnings(self, text: str, platform: str, max_chars: Optional[int], thread: bool) -> list[str]:
        warnings: list[str] = []
        if platform != "x":
            return warnings

        per_post_limit = max_chars or self.style.get("platforms", {}).get("x", {}).get("max_chars", 280)

        if thread:
            for line in text.splitlines():
                cleaned = line.strip()
                if not cleaned:
                    continue
                if cleaned[0].isdigit() and "/" in cleaned:
                    if len(cleaned) > per_post_limit:
                        warnings.append(f"Tweet exceeds {per_post_limit} chars: {cleaned[:60]}...")
        else:
            if len(text) > per_post_limit:
                warnings.append(f"Post exceeds {per_post_limit} chars.")

        return warnings

    def _offline_generate(
        self,
        context: str,
        platform: str,
        facts: list[str],
        angle: Optional[str],
        cta: Optional[str],
        thread: bool,
        variants: int,
        max_chars: Optional[int]
    ) -> str:
        variants = max(1, int(variants or 1))
        outputs: list[str] = []
        for i in range(variants):
            rng = self._rng_for(context, platform, str(i + 1))
            if platform == "x":
                post = self._build_x_offline(
                    rng=rng,
                    context=context,
                    facts=facts,
                    angle=angle,
                    cta=cta,
                    thread=thread,
                    max_chars=max_chars
                )
            else:
                post = self._build_linkedin_offline(
                    rng=rng,
                    context=context,
                    facts=facts,
                    angle=angle,
                    cta=cta
                )
            outputs.append(post)

        if len(outputs) == 1:
            return outputs[0]
        return "\n\n---\n\n".join(
            f"Variant {idx + 1}:\n{post}" for idx, post in enumerate(outputs)
        )

    def _rng_for(self, *parts: str) -> random.Random:
        seed_text = "|".join(p or "" for p in parts)
        seed = int(hashlib.md5(seed_text.encode("utf-8")).hexdigest(), 16) % (2**32)
        return random.Random(seed)

    def _clean_phrases(self, phrases: list[str]) -> list[str]:
        if not phrases:
            return []
        bad_terms = [
            "nitter",
            "hls",
            "enable hls",
            "replies",
            "comments",
            "link in",
            "link to",
            "zerodha com",
            "reposted",
            "video piped",
            "com z",
            "kamath reposted"
        ]
        cleaned: list[str] = []
        for phrase in phrases:
            value = phrase.strip()
            if not value:
                continue
            lower = value.lower()
            if any(term in lower for term in bad_terms):
                continue
            if not re.search(r"[a-zA-Z]", value):
                continue
            if len(value.split()) > 10:
                continue
            cleaned.append(value)
        return cleaned

    def _pick(self, rng: random.Random, options: list[str], fallback: str) -> str:
        if not options:
            return fallback
        return rng.choice(options)

    def _shorten(self, text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]).rstrip() + "..."

    def _build_hook(self, context: str, opener: str) -> str:
        base = context.strip() if context.strip() else "[ADD CONTEXT]"
        if not opener:
            return base
        opener = opener.strip()
        if not opener.endswith((".", "?", ":")):
            opener = opener + ":"
        return f"{opener} {base}"

    def _format_facts_short(self, facts: list[str], max_items: int = 2) -> str:
        if not facts:
            return "[ADD FACT]"
        items = [f for f in facts if f.strip()]
        if not items:
            return "[ADD FACT]"
        items = items[:max_items]
        if len(items) == 1:
            return f"Data: {items[0]}"
        return "Data: " + " | ".join(items)

    def _takeaway_line(self, rng: random.Random, angle: Optional[str]) -> str:
        prefixes = ["Takeaway", "Net", "So what", "Bottom line", "The question to ask"]
        prefix = self._pick(rng, prefixes, "Takeaway")
        content = angle.strip() if angle else "[ADD TAKEAWAY]"
        return f"{prefix}: {content}"

    def _default_question(self, rng: random.Random) -> str:
        questions = [
            "What are you seeing on this?",
            "Curious to hear other views.",
            "Am I missing something here?",
            "Would love counterpoints."
        ]
        return self._pick(rng, questions, "Curious to hear other views.")

    def _build_x_offline(
        self,
        rng: random.Random,
        context: str,
        facts: list[str],
        angle: Optional[str],
        cta: Optional[str],
        thread: bool,
        max_chars: Optional[int]
    ) -> str:
        style = self.style.get("derived", {})
        openers = self._clean_phrases(style.get("common_openers", {}).get("x", []))
        signature = [p for p in self.style.get("signature_phrases", [])]
        openers = openers + signature
        fallback_openers = [
            "Quick thought",
            "Short note",
            "A simple way to look at this",
            "One thing to keep in mind",
            "The question to ask"
        ]
        opener = self._pick(rng, openers, self._pick(rng, fallback_openers, "Quick thought"))

        hook = self._build_hook(self._shorten(context.strip(), 28), opener)
        fact_line = self._format_facts_short(facts)
        takeaway = self._takeaway_line(rng, angle)
        cta_line = cta.strip() if cta else ""

        question_rate = style.get("question_rate", {}).get("x", 0.18)
        if not cta_line and rng.random() < question_rate:
            cta_line = self._default_question(rng)

        if thread:
            segments = [hook]
            if fact_line:
                segments.append(fact_line)
            if takeaway:
                segments.append(takeaway)
            if cta_line:
                segments.append(cta_line)
            segments = segments[:5]
            count = len(segments)
            per_post_limit = max_chars or self.style.get("platforms", {}).get("x", {}).get("max_chars", 280)
            tweets = []
            for idx, segment in enumerate(segments):
                numbered = f"{idx + 1}/{count} {segment}"
                if len(numbered) > per_post_limit:
                    numbered = numbered[: max(per_post_limit - 1, 1)].rstrip() + "…"
                tweets.append(numbered)
            return "\n\n".join(tweets)

        lines = [hook, fact_line, takeaway]
        if cta_line:
            lines.append(cta_line)

        per_post_limit = max_chars or self.style.get("platforms", {}).get("x", {}).get("max_chars", 280)
        text = "\n".join(line for line in lines if line)
        if len(text) <= per_post_limit:
            return text

        # Drop optional CTA first if needed
        if cta_line:
            lines = [hook, fact_line, takeaway]
            text = "\n".join(line for line in lines if line)
        if len(text) <= per_post_limit:
            return text

        # Shorten hook if still long
        short_hook = self._shorten(hook, 18)
        lines = [short_hook, fact_line, takeaway]
        text = "\n".join(line for line in lines if line)
        if len(text) > per_post_limit:
            text = text[: max(per_post_limit - 1, 1)].rstrip() + "…"
        return text

    def _build_linkedin_offline(
        self,
        rng: random.Random,
        context: str,
        facts: list[str],
        angle: Optional[str],
        cta: Optional[str]
    ) -> str:
        style = self.style.get("derived", {})
        openers = self._clean_phrases(style.get("common_openers", {}).get("linkedin", []))
        signature = [p for p in self.style.get("signature_phrases", [])]
        fallback_openers = [
            "Here are a few thoughts",
            "Sharing a quick observation",
            "A candid note",
            "Some context",
            "A simple way to look at this"
        ]
        opener = self._pick(rng, openers + signature, self._pick(rng, fallback_openers, "Here are a few thoughts"))

        hook = self._build_hook(self._shorten(context.strip(), 50), opener)
        paragraphs = [hook]

        if facts:
            if len(facts) == 1:
                paragraphs.append(f"Data point: {facts[0]}")
            else:
                fact_lines = "\n".join(f"- {fact}" for fact in facts[:4])
                paragraphs.append(f"Data points:\n{fact_lines}")
        else:
            paragraphs.append("[ADD FACT]")

        takeaway = angle.strip() if angle else "[ADD TAKEAWAY]"
        paragraphs.append(f"What I'd take from this: {takeaway}.")

        if cta and cta.strip():
            paragraphs.append(cta.strip())
        else:
            question_rate = style.get("question_rate", {}).get("linkedin", 0.16)
            if rng.random() < question_rate:
                paragraphs.append(self._default_question(rng))

        return "\n\n".join(p for p in paragraphs if p)


# Singleton instance
_nithin_generator: Optional[NithinPostGenerator] = None


def get_nithin_generator() -> NithinPostGenerator:
    global _nithin_generator
    if _nithin_generator is None:
        _nithin_generator = NithinPostGenerator()
    return _nithin_generator
