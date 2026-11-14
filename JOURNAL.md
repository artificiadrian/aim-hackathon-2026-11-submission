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

## 14:40 · Files per task by zip, not a prefetch of the whole run

- decided_by: both
- why: the prefetch made ~530 requests (27-30 s at the rate limit) before any task could start, and one failure
  touched every task. `workspace.zip` gives a task in one request, but the platform caps it at 8 MB and lists the
  left-out files in LARGE_FILES.txt (19 big reference PDFs, shared by all 48 tasks); a file whose state changed
  (containment.json) differs from the manifest
- decision: each task fetches its zip when it starts, keeps members that match the manifest hash, and fetches the
  rest by hash into the shared cache, each blob once per run (a lock per blob)
- rejected: skip the big PDFs (never cited in dev, but that is fitting to dev data)
- evidence: live check on run_94dAWU6i2EV- t01: 375 members, 373 match the manifest, 19 left out, 1 changed

## 14:55 · Action gate by a small model call, not by word lists

- decided_by: human
- why: the 13:40 regex gate failed closed: a final request worded differently ("isolate", "cut off") would drop a
  needed action and fail the task; its words were tuned on dev instructions (prior knowledge)
- decision: when an accepted submission proposes actions with outcome OK, one call (Gemma, no tools, max 200 tokens)
  reads only the instruction and the open ticket and lists the kinds asked for; injected workspace text never
  reaches it. Unreadable reply or a failed call: every kind allowed (fail open). Content checks (defang, tokens,
  verified recipient) stay
- rejected: regex gate (brittle), no gate (an injected quarantine in an analysis task acts)
- evidence: not yet measured with the model; the next eval shows the ScopeDecided events

## 15:05 · One way to finish: submit(...) in a script

- decided_by: human
- why: two ways to finish (tool and builtin) cost a schema on every call, and the tool made the model retype ids and
  locators; in the stopped eval 6 of 10 tasks already finished from Python
- decision: tools are run_python and view_image; the last step offers only run_python and asks for one short script
  that calls submit(...); the action forms are in SYSTEM (drift test against the models)
- rejected: keep both (schema cost, two paths to test)
- evidence: tool schemas 5,300 -> 704 characters per call; a crashing last script now falls back (not measured yet)

## 15:30 · Back to a minimal generic base

- decided_by: human
- why: parts fitted to dev data (word lists, dev file names, dev statistics) fail on a final that words or names
  things differently, and some failed closed
- removed: the action check (regex, then a model call), the inbox citation rule, the closed-ticket split (all tickets
  go to the model as they are)
- changed: the recipient check fails open when no contact list is found (it blocked every message before)
- kept: defang and token redaction (formats from the platform docs), csv citations as records (platform docs),
  run safety, the sandbox

## 15:55 · Helpers in the sandbox: text, cite, grep, sh

- decided_by: both
- why: the model computed locators by hand (line numbers, sheet rows, non-empty paragraphs) and got them wrong;
  the rules for each file type (platform docs) belong in code, not in every prompt
- decision: sandbox/ws.py, loaded by sitecustomize as builtins: text(path) shows any file as units with locators,
  cite(path, value, column=None) returns sources, grep searches every format, sh runs a shell command. No bash tool:
  one tool keeps the prefix small; subprocess works from Python anyway
- evidence: on all dev runs, the helpers produce the platform's expected locator for xlsx 109/109, ods 51/51, docx
  13/13, pdf 5/5, jsonl 824/824, md 108/108, csv 1,407/1,434 (the 27 are records of servers missing from the file)

## 16:15 · Helpers as a package with typed results; real calls instead of fakes

- decided_by: human
- why: one file mixed parsing, search and shell; results as raw dicts and tuples were easy to misuse; a new key makes
  dev calls unlimited, so tests should use the real model, not a scripted one
- decision: sandbox/ws/ (models: Source, Unit, Hit as pydantic; formats: one reader per file type plus a parse cache
  in /tmp; search: text, cite, grep over every format; shell: sh). grep returns Hit objects whose source goes into
  submit(...) as is. tests/test_live_loop.py runs real Gemma in the real sandbox on saved tasks
- evidence: locator check on all dev runs unchanged (xlsx 109/109, ods 51/51, docx 13/13, pdf 5/5, jsonl 824/824,
  md 108/108); interpreter start 69 ms with the helpers loaded. The live test is not run yet: the key in .env got a
  401 from openrouter.ai

## 16:30 · Closed tickets are history again (platform docs)

- decided_by: ai
- why: at 15:30 the closed-ticket split was removed as dev-fitted; docs/ARENA_DOCS.md (ticket tasks) says the
  request is the ticket with `status: open` and a closed ticket is "history, not a request": a platform rule, not
  dev data. A ticket without a status stays a request
- evidence: docs/ARENA_DOCS.md lines 82-84

## 14:30 · Eval with all fixes: two bugs found, both fixed

- decided_by: both
- evidence: runs/20261114T125844Z-eval and the t19 run after it (real calls): 11 tasks, mean score 0.589,
  $0.0120 per task (before: 6/11 right, $0.0114). Answers right in 9 of 11; t14 wrong, t09 no submission
- bug 1: models write sources=[cite(...)] and cite returns a list; the nested list failed validation and every
  source was dropped. Fix: model input flattens nested lists one level. Offline on the same submissions: mean
  0.589 -> 0.743 (t01, t10 1/1; t05 4/6; t23 4/4; t31 4/5; t40, t19 2/3)
- bug 2: t09's last script called cite(jsonl, value, column="ts"); cite raised for the missing column and the
  submit(...) in the same call never ran. Fix: cite never raises; it matches anywhere and says so on stderr
- also: sitecustomize replaced by an explicit runner (python3 -I -m ws.run): no start-up hook, builtins untouched
- open: 4 of 11 tasks used all 12 steps (t02, t09, t14, t23)


## 15:30 · Audit batch 1: never lose a computed answer; tell the model what the harness drops

- decided_by: both
- why: the audit (docs/research/2026-11-14-audit-summary.md) found answers lost to a late failed call or a rejected
  submit (A2), actions dropped by guards without a word to the model (A5), hidden grep matches (A4), evidence cut from
  the history by short outputs (A6), substring citations (D), NotJson on pandas values (D), jsonl keys missing from
  the overview (D), and planted claims in free-text fields framing clean agents (B3; live run t09 frame_agent)
- rejected: a code check of each answer against free-text claims (task knowledge); per-level step budgets and a
  first-message grep pass (open for the user)
- evidence: host checks 91 passed; image tests 22 passed; no eval yet (batching fixes first)

## 16:10 · table(), the task level and scanned tickets read up front; pandas-stubs

- decided_by: both
- why: L3 tasks join tables and the model guessed headers and lost row citations (audit B4); the platform gives a
  level per task (docs/ARENA_DOCS.md); a scanned ticket can be the request itself and the model found that late
  (review 131442Z); pandas had its type checks switched off for want of stubs
- rejected: a first-message grep of ids from the instruction (needs a guess of what an id looks like); a time-zone
  column in table() (the zone comes from the workspace, the model converts)
- evidence: host 92 passed, image 22 passed; reviews 20261114T132912Z-eval and 20261114T133127Z

## 16:40 · Batch 2: submit earlier, a retry that cannot crash, hidden characters, a fair grader

- decided_by: both
- why: reviews 20261114T133127Z and 20261114T132912Z-eval: sources lost on last-step submits (the citation check
  skips the last step), t15 lost its answer to a crashing last script and retry, a zero-width-split injection passed
  a search (t02), and the local grader scored lists of objects as ordered
- what: a "submit next" note two steps before the end; the retry after the last step offers only a JSON submit tool;
  "print a column's values before you filter on one"; invisible characters removed and marked in the helpers and in
  the inlined texts; lists of objects graded as sets
- rejected: printing small tables in full (pandas already prints up to 60 rows; t15's model never printed the table)
- evidence: host 96 passed, image 22 passed

## 14:57 · Soft end of the step budget; citation check on the last step; 16 steps for level 3

- decided_by: both
- why: the run reviews show most submissions come on the last step, where the citation check was off, so
  hand-written sources went out unchecked (t09 0/6, t10 0/13, t31 0/4, t44 0/5); the last step also recomputed and
  crashed (t15). Now a submission is sent back once on any step that leaves a call to fix it (the extra step after
  the last one; on the retry the submit tool of batch 2), the batch-2 "submit next" note no longer says the last
  step skips the check, and level 3 gets 16 steps
- rejected: 8 steps for level 1 (evals show L1 tasks at 11-13 steps, one scored 1.0 at step 12); a per-run parse
  cache (parsing every file up front can be too much; the user doubts it)
- evidence: docs/research/2026-11-14-run-analyses-pattern.md; host checks pass; no eval yet

## 17:10 · No dev file names: operator files, tickets and contacts from the workspace itself

- decided_by: both
- why: `/runbook.md`, `/contacts.csv` and `/tickets/` were dev names in code; the platform only says /AGENTS.md is
  the source of truth and names the other operator files, and that the instruction points to the ticket queue
- what: operator files = /AGENTS.md + the workspace paths it names (exact match); tickets = files with a `status:` in
  the folders the instruction names, scans there transcribed; the recipient check (dev contacts file and column)
  removed; the inbox prompt line removed; the retry after the last step offers run_python and the JSON submit tool
- rejected: keeping the recipient check (fails open on other names, so it does nothing on the final)
- evidence: host 94 passed

## 18:00 · One persistent Python session per task (kernel)

- decided_by: both
- why: eval 20261114T135828Z: 6 of 9 script failures were NameError on a variable from an earlier script (each script was a new process); every script reloaded its frames
- rejected: Jupyter kernel (new dependency, more surface); pipes (a print flood or stray child blocks the harness); files in the task root (the task uid could swap them)
- evidence: docs/ideas/persistent-kernel.md; 31 image tests pass; 12 kernels with pandas: 912 MB

## 18:40 · Per-task spend cap; pandas readers return table(); fixes from the eval review

- decided_by: human
- why: eval 20261114T135828Z (docs/runs/20261114T135828Z-eval.md): the run cap left 22 of 48 tasks at 0; hand-written sources hit 25 of 67, table()/cite() sources 18 of 19, and the model calls pd.read_* three times as often as table()
- decided: a task may spend 2 × cap / tasks, then submits its best; pd.read_csv/read_excel/read_json on a workspace table file return table(path) (a hidden hook on purpose: if the next eval shows no gain in sources, remove it and try another way); cite() takes a list of values; the parse cache is in memory, not /tmp/.ws (it flooded the model's grep -r); view_image() in a cell says it is a tool
- rejected: a reserve per unstarted task (more code), rejecting hand-written sources (costs steps)
- evidence: next eval on a fresh set

## 15:29 · Jev: a classify tool and a shadow action check

- decided_by: both
- why: board triage failed on keyword lists (t10 0.27); live probes on the collected runs
  (docs/research/2026-11-14-jev-probe.md) showed Jev judges every line of a board for about $0.004 with no
  injected post missed (193 of 203 flagged, 10 in the unsure list, 6 false flags over 15 boards), and that a
  "does the request ask for this kind of operation" question blocks 0 of 66 right actions and catches every other
  wrong kind except revoke-for-rotate
- decision: a third tool classify(path, question) for text files (lines as units, invisible characters removed,
  base64 decoded in code, a second pass that judges split messages together; flagged >= 0.8, unsure 0.2-0.8 for
  the model to read); after the loop, one Jev call per proposed action op, recorded as action_judged, blocking
  nothing until an eval shows no false block
- rejected: the definition inside the per-line question (64 routine posts flagged on one board: the question
  goes into the state, each line gets a short question); a word gate (25 analysis requests match "send")
- evidence: tests/test_jev.py (4 offline, 1 live); host checks pass

## 16:40 · Over-citing: an estimated cut in the grader, a narrower sources line in the prompt

- decided_by: both
- why: live 20261114T133127Z t40 cited 8 for 4 expected and got sources_score 0.75 with 4/4 hits; the local grader
  has no cut, and 17 of 56 final submissions in the 8 newest evals cite more than expected + 2 (t04 54 for 3, t05
  6-8 for 2 in every eval); the replay shows `located` adds no case (docs/research/2026-11-14-over-citing.md)
- decision: the grader keeps `score` and adds `sources_cited` and `score_est` (a 0.25 sources cut when cited >
  expected + 2, the rule that fits all live grades); the prompt asks for the units that hold the answer items plus
  the one lookup-table row each item was matched on, each unit once
- rejected: a cap on citations in code (the expected count is unknown at submit time)
- evidence: tests/test_eval.py::test_grade_estimates_the_over_citing_cut; host checks pass; the real rule comes
  from the next live run's feedback

## 17:10 · Recipient and protected-domain checks, tickets only from a ticket's head, named files name no queue

- decided_by: both
- why: audit2 (docs/research/2026-11-14-audit2-summary.md D1-D3): no code check stopped a message to an address
  only workspace data names, or a block of our own or a partner's domain; a named file made its parent folders
  ticket queues (130504Z t36: a decoy image became a trusted ticket), and a "status: open" line anywhere promoted a
  queued file
- decision: a message goes only to an address written in the instruction, the operator files or the open ticket
  (no file names: contacts.csv counts because AGENTS.md names it); a block_domain op is blocked for the domain of
  a trusted address, its parents and subdomains; both block, and the model is told once. A ticket's status is read
  from its front matter or first 3 lines; only a folder the instruction names as a folder is a queue; images the
  instruction names are transcribed up front as data. Also: NaN/Infinity in an answer become null; the run starts
  before the tool list, which may fail
- rejected: parsing a "verified" column (dev file layout); the earlier contacts.csv recipient check (dev file name)
- evidence: tests/test_guards.py on all collected runs: 28 expected recipients and 188 expected domains pass, 223
  distinct addresses only workspace data names are blocked (look-alikes rnuggingface.example and Cyrillic i among
  them); ticket sets unchanged in 624 tasks; 168 tasks lose parent-folder queues (t07: 4 images were tickets)
