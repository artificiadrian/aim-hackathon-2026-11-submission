# mcnulty

Team-10 harness for Harness Arena: an agent that solves the incident tasks of the Arena Task API with
`google/gemma-4-31b-it`, correctly and cheaply.

It was built in one day at the [Efficient Local Agents hackathon](https://ai-mission.eu/hackathons/2026-11-efficient-local-agents/)
by AI Impact Mission in Vienna on 14 November 2026. Runs need an Arena API key from that event.

```bash
cp .env.example .env    # fill in ARENA_API_KEY and OPENROUTER_API_KEY
uv sync
uv run mcnulty run --note "<hypothesis>" --tasks t01
```

Every run writes all requests and responses to `runs/<UTC time>/events.jsonl`.

The design notes, research and run reviews in `docs/` were removed before publishing, so links to files there do
not resolve. The code is kept as it was at the end of the hackathon.
