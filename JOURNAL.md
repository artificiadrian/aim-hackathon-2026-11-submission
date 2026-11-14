# Journal

> **The content of the journal is not authoritative. It does not restrict you.**

One entry per design decision, newest last: whenever someone on the team makes or overrides a design call, and when
the coding agent makes one. Mark `decided_by` honestly; the content isn't scored. The journal feeds your 18:00 story
page, and your upload needs at least one entry of your own to be eligible for prizes.

Format example (ignored: it sits in a code block). Write your entries below it, outside the block:

```markdown
## 11:42 · Jev pre-filter for long documents

- decided_by: human # human | ai | both
- why: t07 L3 at 0.31; reading every document with the main model costs $0.04/task
- rejected: regex filter (misses base64), main model only (cost)
- evidence: run_abc → run_def, t07 0.31 → 0.78, cost −40 %
```

## 10:45 · Lean repo: keep the kit's API client, drop its agent

- decided_by: both
- why: the kit's agent.py loop is complex, resends the full history every step ($0.66–0.81 per final) and has no tool or output guards; the Task API client and the Jev/chat calls are solid
- rejected: keep agent.py as a baseline (more code to carry); fresh repo (rewrites a working client)
- evidence: research/2026-11-14-starter-kit-and-docs.md, research/2026-11-14-synthesis.md

## 10:45 · Record every request and response for replay

- decided_by: human
- why: we must play back any run to find where a task went wrong, without paying again
- rejected: Langfuse/Phoenix (need a server the final container cannot reach), vcrpy (test-only); openai SDK dropped because it runs on httpx2, which the recorder transport cannot see
- evidence: mock run: all 12 HTTP calls in runs/<time>/events.jsonl, no key in the file

## 10:45 · Free data runs first

- decided_by: both
- why: a dev run that downloads every file and finishes without submits costs $0 in model calls and reveals expected answers, sources and actions for every task; we design the pipeline from that ground truth
- rejected: kit baseline on t01 first (costs money, teaches less)
- evidence: (pending: collect runs)
