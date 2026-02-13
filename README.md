# Nithin Kamath Post Generator

Generate X and LinkedIn drafts in Nithin Kamath's public voice, using a local style guide derived from recent posts.

## Setup
```bash
export ANTHROPIC_API_KEY="your_key"
export RESEARCH_PROVIDER="tavily" # or serper, brave
export RESEARCH_API_KEY="your_search_key"
export OLLAMA_MODEL="llama3.1" # optional local LLM
export OLLAMA_HOST="http://localhost:11434" # optional
```

## Run the web app
```bash
./run.sh
```

## CLI usage
```bash
python scripts/generate_nithin_post.py \
  --platform x \
  --context "Users are asking whether F&O participation is rising among new investors." \
  --facts "F&O participation is up year-over-year (internal data)" \
  --facts "SEBI released a consultation paper on F&O risk controls last week" \
  --angle "Cautious: participation is up, but risk awareness is lagging" \
  --cta "What else should we publish to make this clearer?" \
  --thread
```

## Ingest LinkedIn PDF (last-year window)
```bash
python scripts/ingest_linkedin_pdf.py \
  --pdf /path/to/Activity_Nithin_Kamath_LinkedIn.pdf \
  --since 2025-02-13 \
  --until 2026-02-13 \
  --reference-date 2026-02-13 \
  --update-style
```

## Ingest Nitter PDF (X)
```bash
python scripts/ingest_nitter_pdf.py \
  --pdf /path/to/Nithin_Kamath_nitter.pdf \
  --since 2025-02-13 \
  --until 2026-02-13 \
  --update-style \
  --append
```

## Notes
- The generator only uses facts you provide.
- If a key fact is missing, it inserts `[ADD FACT]` placeholders.
- Drafts should be reviewed before publishing.
- Style guide is currently locked. Use `--force-update-style` to override during ingestion.
- Research is optional and requires a provider + API key.
- If no API key is available, you can run a local model via Ollama (`OLLAMA_MODEL`).
- For grammar-only proofreading without an LLM, the app uses LanguageTool (public API).
