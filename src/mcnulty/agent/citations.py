"""Citations the harness can fix in code after ``submit``, from the task's own files. Pure functions.

- csv: the platform matches ``record`` (the first column). Models cite the line they printed; a line becomes the
  record that starts or ends on it when the first column is a unique id. Files whose first column repeats (a time
  column) stay cited by line.
"""

import csv
import io

from mcnulty.clients.arena import Source


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


def fixed_sources(sources: tuple[Source, ...], records: dict[str, dict[int, str]]):
    """Sources with csv lines as records; duplicates removed."""
    return tuple(dict.fromkeys(as_record(s, records) for s in sources))
