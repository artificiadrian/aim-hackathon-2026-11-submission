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

## 10:49 · First dev attack t10-p1 posted early
- decided_by: both
- why: a dev attack joins the pool only at the next half hour; posting early gives placement and hit data sooner. Carrier `audit_log_detail` lands on cited evidence in about 4 of 10 tasks; goal `url_exfil` hits even when the victim only quotes the URL.
- rejected: waiting for a polished attack (no data until then), `email_body` / document carriers (rarely on cited evidence)
- evidence: t10-p1 accepted, active from 11:00 local; placements and hits to follow from GET /payloads

## 11:20 · Explicit recorder object, typed events, src layout

- decided_by: human
- why: a global recorder (contextvars, module state, start()) hides who logs what; a Recorder object that carries its task context and logs pydantic events is plain to read and to replay
- rejected: transport hook and module-level state (implicit); the kit's dict-based code style
- evidence: tests/test_mock_run.py: every event parses back as a typed Envelope, no key in events.jsonl

## 11:10 · Three principles: keep it simple, type everything, record it all

- decided_by: human
- why: one small harness that we can read, check and replay beats a clever one; typed models catch API drift at the boundary; full records let us debug any task after the run
- rejected: raw dicts across functions, global state, stdout-only results
- evidence: mcnulty check passes on typed request and response models (run_DaQJ7V011XKX, $0.00015); every task and run end is a typed event

## 11:40 · Generic agent over per-family solvers (bitter lesson)

- decided_by: human
- why: the final runs on an unseen incident with new wording; code fitted to our 26 dev families can be free but wrong, and a wrong answer scores 0. Small generic capabilities (typed parsing, search, table query, one agent loop, generic guards) carry over
- rejected: per-family deterministic solvers and a dev-phrasing router (prior knowledge that breaks on the final)
- evidence: docs/research/2026-11-14-data-task-families.md (rules and formats change per seed); docs/design.md

## 12:00 · Code mode on a real filesystem, sandbox in Docker

- decided_by: both
- why: the experiment showed Gemma writes working Python (59/63 scripts ran) and fails on loop friction; a real filesystem with plain Python and pandas removes the helper API the model tripped on; chroot + second user in the same Docker image isolates scripts from our keys and behaves the same locally and in the final
- rejected: a `ws` helper API (all failed scripts bypassed it), native macOS sandboxing (no chroot, differs from the final), plain-text views of binary files (maybe later, decide with data)
- evidence: docs/research/2026-11-14-code-mode-experiment.md (4/11 correct, $0.0073 per task); docs/plan.md

## 12:30 · Sandbox: one root per task, hard links to a copy made once per container

- decided_by: both
- why: the model writes to a scratch folder (`/tmp`, kept between steps) but cannot change the workspace files or its own Python; it gets pandas, numpy, openpyxl, odfpy, python-docx, pypdf, pillow, pyyaml and tzdata. Docker's overlay filesystem copies a file up on every hard link, so the container's own root (one Python for harness and sandbox, without /app, uv and the system folders) is copied once into the writable layer (4.7 s); each task's root is then a hard-link copy (0.1 s, 4 MB) under the final's 4 GB disk limit
- rejected: a second root filesystem in the image (320 MB duplicate); a hand-picked list of folders to copy (one missing library breaks an import only at run time); hard links straight from the image (320 MB per task); one shared root (tasks run in parallel at the same paths); a pool of slots (more moving parts); blocking processes (Python is enough; the network block in sitecustomize is for a clear error, the hard limits are chroot, uid, read-only files, empty environment and the gateway-only network)
- evidence: tests/test_sandbox.py in the image (11 passed); a real task with 393 files ready in 0.5 s

## 13:10 · run_python output contract and run safety before more evals

- decided_by: both
- why: the reviews found the final run can lose every task: one print loop fills the harness memory (container killed
  in 1-8 s), a detached child hangs a task, nothing watches the 30 minutes, and the answer goes out without a guard.
  The user stopped an eval: fix a large batch first, then measure once.
- rejected: spill files the model can open (pi's bash pattern: costs a step, recompute is cheap here); one uid per
  task and a script semaphore (later, need measuring); a model call to gate actions (cost)
- evidence: docs/research/2026-11-14-review-findings.md (SB1-SB3, SG1, SG4, OP1-OP7), docs/research/2026-11-14-pi-code-mode.md;
  flood and detached-child tests pass in the image; defang changes 0 of 3,184 expected answer strings


## 13:40 · Action gate from trusted text only

- decided_by: ai
- why: an injected "quarantine X" or "lift the quarantine" in an analysis task made the harness act; an out-of-scope
  action fails the task even when the tool rejects it (ARENA_DOCS). Allowed kinds now come from the instruction and
  the open ticket only (words, not paths, scopes or negations); inbox tasks allow what the message requests (the model
  judges it, the recipient and token checks still apply); a release needs the agent named. Closed tickets go to the
  model as history.
- rejected: a model yes/no call per task (cost); no gate (review SG2, HIGH)
- evidence: the gate decides right on every distinct dev instruction (containment, rotation, send, draft, inbox
  tasks allowed; ordering, patient zero, DPA, board triage, status checks blocked); tests in tests/test_guards.py

## 14:00 · Citations fixed in code after submit

- decided_by: ai
- why: correct answers lost 30 % on citations. The model cites csv rows by line, the platform matches the record
  (first column); refusals cited nothing, but every inbox task expects its message (33 of 33 in dev)
- rejected: a resolver that adds rows by answer value (review MF2: 6 in 10 precise, over-citing is penalised)
- evidence: on the stopped eval's own submissions, offline: t01 0/1 -> 1/1, t07 0/10 -> 10/10, t10 0/1 -> 1/1, no
  extra citations; on all dev data every expected csv record is reachable from a line (1,178), the archive log with
  a repeating first column stays by line (94)
