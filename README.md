# mcnulty

Team-10 harness for Harness Arena: an agent that solves the incident tasks of the Arena Task API with
`google/gemma-4-31b-it`, correctly and cheaply.

```bash
cp .env.example .env    # fill in ARENA_API_KEY and OPENROUTER_API_KEY
uv sync
uv run mcnulty run --note "<hypothesis>" --tasks t01
```

Every run writes all requests and responses to `runs/<UTC time>/events.jsonl`. Platform spec: `docs/ARENA_DOCS.md`.
