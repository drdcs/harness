"""Step 1 - parse a notebook into numbered cells. Pure Python, no LLM.

Databricks .py source:  split on "# COMMAND ----------", keep the real file line numbers.
Jupyter .ipynb:         one cell per JSON cell, lines numbered continuously across cells.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

SEPARATOR = "# COMMAND ----------"
HEADER = "# Databricks notebook source"
MAGIC = re.compile(r"^# MAGIC ?")
MAGIC_KINDS = {"%md": "markdown", "%sql": "sql", "%run": "run", "%sh": "shell", "%pip": "pip", "%scala": "scala"}


@dataclass
class Cell:
    id: int                 # 1-based position in the notebook
    kind: str               # python | markdown | sql | run | ... | commented
    lines: dict[int, str]   # line number -> raw source text

    @property
    def start(self):
        return min(self.lines)

    @property
    def end(self):
        return max(self.lines)


def classify(texts, cell_type="code"):
    body = [MAGIC.sub("", t).strip() for t in texts if t.strip()]
    if cell_type == "markdown":
        return "markdown"
    if body and body[0].split()[0] in MAGIC_KINDS:
        return MAGIC_KINDS[body[0].split()[0]]
    if body and all(t.startswith("#") for t in body):
        return "commented"          # every line is a comment -> dead code, never executes
    return "python"


def trim(numbered):
    """Drop leading/trailing blank lines so a cell's range is exactly its code."""
    while numbered and not numbered[0][1].strip():
        numbered.pop(0)
    while numbered and not numbered[-1][1].strip():
        numbered.pop()
    return numbered


def chunks_from_source(text):
    chunk = []
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip() == SEPARATOR:
            yield "code", chunk
            chunk = []
        elif not (number == 1 and line.strip() == HEADER):
            chunk.append((number, line))
    yield "code", chunk


def chunks_from_ipynb(text):
    number = 0
    for cell in json.loads(text)["cells"]:
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        chunk = []
        for line in source.splitlines():
            number += 1
            chunk.append((number, line))
        yield cell["cell_type"], chunk


def parse(path) -> list[Cell]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    chunks = chunks_from_ipynb(text) if path.suffix == ".ipynb" else chunks_from_source(text)
    cells = []
    for cell_type, chunk in chunks:
        chunk = trim(chunk)
        if chunk:
            cells.append(Cell(len(cells) + 1, classify([t for _, t in chunk], cell_type), dict(chunk)))
    return cells


def render(cells) -> str:
    """What the LLM sees: every line carries its own address, so it copies cites instead of counting."""
    out = []
    for cell in cells:
        out.append(f"=== C{cell.id} [{cell.kind}] ===")
        out += [f"C{cell.id} L{n} | {t}" for n, t in cell.lines.items()]
    return "\n".join(out)
