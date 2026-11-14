"""Citations the harness can fix in code after ``submit``, from the task's own files. Pure functions.

- csv: the platform matches ``record`` (the first column). Models cite the line they printed; a line becomes the
  record that starts or ends on it when the first column is a unique id. Files whose first column repeats (a time
  column) stay cited by line.
- inbox: a task that handles an inbox message rests on that message, whatever the outcome (33 of 33 such dev tasks
  cite it), so its files are cited when the model did not.
"""

import csv
import io
import re

from mcnulty.clients.arena import Source

INBOX_TASK = re.compile(r"\binbox\b", re.IGNORECASE)


def line_records(text: str):
    """Physical line -> first-column value for every record of a csv, or empty when that column is not unique."""
    reader = csv.reader(io.StringIO(text))
    records: dict[int, str] = {}
    seen: set[str] = set()
    start = 1
    for i, row in enumerate(reader):
        end = reader.line_num
        if i > 0 and row and row[0].strip():  # line 1 is the header
            value = row[0].strip()
            if value in seen:
                return dict[int, str]()
            seen.add(value)
            records |= {start: value, end: value}
        start = end + 1
    return records


def as_record(source: Source, records: dict[str, dict[int, str]]):
    lines = records.get(source.path, {})
    if source.line is None or source.record is not None or source.line not in lines:
        return source
    return Source(path=source.path, record=lines[source.line])


def fixed_sources(
    sources: tuple[Source, ...],
    records: dict[str, dict[int, str]],
    instruction: str,
    paths: frozenset[str],
):
    """Sources with csv lines as records and the inbox message added for inbox tasks; duplicates removed."""
    fixed = [as_record(s, records) for s in sources]
    if INBOX_TASK.search(instruction):
        cited = {s.path for s in fixed}
        fixed += [Source(path=p) for p in sorted(paths) if p.startswith("/inbox/") and p not in cited]
    return tuple(dict.fromkeys(fixed))
