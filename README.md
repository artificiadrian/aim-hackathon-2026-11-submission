# mcnulty

Team-10 harness for Harness Arena: an agent that solves the incident tasks of the Arena Task API with
`google/gemma-4-31b-it`, correctly and cheaply.

It was built in one day at the [Efficient Local Agents hackathon](https://ai-mission.eu/hackathons/2026-11-efficient-local-agents/)
by AI Impact Mission in Vienna on 14 November 2026. Runs need an Arena API key from that event.

## The task

The hackathon platform, Harness Arena, gives each team a run of 48 tasks about one made-up security incident. Each
task has a workspace of files (logs, CSV and spreadsheet exports, tickets, Word and PDF documents, images) and an
instruction, for example "return every row of this access log that breaks the policy" or "put these events in
order". The agent reads the files through the Task API, then submits an answer with the file lines it used as sources.
Some tasks also ask for an action: contain a host, send a message, or write a file.

Other teams hide prompt injections in the workspace files, so the files are untrusted. The score is accuracy times
(0.5 + 0.5 × cost score), so the money spent on model calls counts for half. At run time only two models were
allowed, both through OpenRouter: the small model `google/gemma-4-31b-it` and `typesafe/jev-1.14`, a classifier that
gives yes/no probabilities.

The final ran in the organisers' container on a new incident that no development run showed, with a 30-minute limit
and a $3 budget.

## How it works

The harness has no code for specific task types. It gives the small model a few general tools and lets the model
read the rules of each run from the workspace.

1. **Workspace in a sandbox.** The harness downloads each file once into a local store addressed by content
   (`blobs.py`). Each task gets its own chroot that holds its files at their real paths (`agent/sandbox.py`).
2. **One agent loop for every task** (`agent/loop.py`). The model has four tools: `run_python`, `view_image`,
   `classify` and `submit`. `run_python` runs code in one Python session per task, with the network blocked. In that
   session, the helper package `ws` (`packages/sandbox/`) parses each file format into units the model can cite
   (lines, rows, paragraphs, cells) and gives the helpers `text`, `grep`, `table`, `cite` and `sh`.
3. **Typed submit.** The model finishes with `submit(...)`. The harness keeps every valid part of a submission, fixes
   citations it can resolve in code (`agent/citations.py`), and sends the answer to the Task API at once.
4. **Guards in code** (`agent/actions.py`, `agent/guards.py`). The harness does an action only when the task's
   instruction asks for it and the outcome is OK. Recipients must come from contacts in trusted files, URLs are
   defanged (`hxxps://evil[.]example`), and token values are never sent. Jev checks each proposed action against the
   trusted request, and the `classify` tool uses Jev to mark matching lines in long files (`agent/jev.py`).
5. **Cost control.** Each task has a step budget and a spend cap, and the whole run has a hard cap. Near the deadline
   the harness submits the best answer it has for each open task, because a task without an answer scores 0.
6. **A full record.** Every request, response, decision and error goes to `runs/<UTC time>/events.jsonl` as a typed
   event, with keys redacted (`record/`). A collected development run can be solved again offline and graded against
   its expected answers (`commands/evaluate.py`).

The code keeps a pure core with IO at the edges, and imports go one way: `commands` → `agent` → `clients`, `record`,
`blobs`. All data are frozen pydantic models.

## Layout

```
src/mcnulty/
  cli.py          command line: run, collect, eval, check
  settings.py     the only reader of the environment
  blobs.py        local store of workspace files
  clients/        Arena Task API and OpenRouter clients with their typed models
  agent/          solves one task: loop, prompt, submit, citations, actions, guards, jev, sandbox
  commands/       one module per command, and the offline grader
  record/         typed events and the event log
packages/sandbox/ the ws helpers that model-written code uses inside the sandbox
scripts/          one-off analysis scripts from the day
tests/            pytest tests, with a mock Task API server from the organisers
```

## Running it

The Arena platform was set up for the event, so these commands work only with a key and while the platform is up.

```bash
cp .env.example .env    # fill in ARENA_API_KEY and OPENROUTER_API_KEY
uv sync
uv run mcnulty check                                   # live check of every API, < $0.01
uv run mcnulty collect --note "collect"                # dev run: download all files, submit nothing
uv run mcnulty run --note "<hypothesis>" --tasks t01   # solve and submit tasks
uv run ruff check && uv run basedpyright && uv run pytest
```

The sandbox works only inside the Docker image, so a run that executes model code runs there:

```bash
docker build -t mcnulty .
docker run --rm --env-file .env -v "$PWD/runs:/app/runs" -v "$PWD/.cache:/app/.cache" \
  --entrypoint mcnulty mcnulty eval runs/<collect run> --max-usd 0.10 --tasks t01
```

`AGENTS.md` holds the rules the coding agents followed during the day, and `JOURNAL.md` holds the design decisions in
the order they were made.

## About this repository

The design notes, research and run reviews in `docs/` were removed before publishing, so links to files there do
not resolve. The code is kept as it was at the end of the hackathon.

MIT license, see `LICENSE`.
