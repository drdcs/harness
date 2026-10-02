"""Steps 2-3 - ask the LLM for cited claims, then verify every cite against the parsed notebook."""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, NativeOutput, PromptedOutput, RunContext, ToolOutput

from notebook import Cell, render

CONFIG = Path(__file__).with_name("config.yaml")
MODES = {"native": NativeOutput, "prompted": PromptedOutput, "tool": ToolOutput}

DOCS = {
    "manager": {
        "title": "Manager overview",
        "reader": "a non-technical finance manager. Plain business language, no code jargon, but quote names in backticks.",
        "sections": {
            "What it does": "the business purpose and the outcome of a run",
            "Inputs and outputs": "which files it reads and which tables, reports or emails it produces",
            "What could go wrong": "business risks: wrong allocations, money left unapplied, silent failures",
        },
    },
    "developer": {
        "title": "Developer guide",
        "reader": "a developer new to this code who must maintain it.",
        "sections": {
            "Structure": "the stages of the notebook and which cells implement each",
            "Data flow": "how data moves from source files through dataframes to the written tables",
            "Rules applied": "every business rule, threshold and priority order, with its exact value",
        },
    },
    "agent": {
        "title": "AI agent brief",
        "reader": "an AI coding agent that will debug or change this notebook. Be terse and exact.",
        "sections": {
            "Invariants": "conditions that must stay true, e.g. balances and reconciliation totals",
            "Risky areas": "reassigned constants, dead or commented code, hidden dependencies, unguarded side effects",
            "How to check a change is safe": "concrete checks that already exist in the code and what they verify",
        },
    },
}

RULES = """You document a Databricks notebook. It is given as lines "C<cell> L<line> | code".
- Each claim states ONE fact in one or two sentences and has at least one cite.
- A cite is {cell, start, end}, copied from the line prefixes, pointing at the lines that prove the claim. Keep cites narrow (under 20 lines).
- Wrap identifiers, columns, tables and constants in backticks exactly as written in the code.
- Any number you state must appear in the cited lines.
- Cells marked [commented] and lines starting with # do not run; if you cite them, say they are commented out.
- Code pulled in by %run is not visible; say so instead of guessing.
- If a constant is reassigned later, describe the value in effect where it is used."""

TICKED = re.compile(r"`([^`]+)`")
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?!\w)")
DEAD_WORDS = re.compile(r"comment|disabled|inactive|dead|unused|not (?:run|execut)|parked|legacy|old", re.I)


class Cite(BaseModel):
    cell: int = Field(description="N from the C<N> prefix")
    start: int = Field(description="first L number")
    end: int = Field(description="last L number, equal to start for one line")


class Claim(BaseModel):
    text: str
    cites: list[Cite] = Field(min_length=1)


class Section(BaseModel):
    title: str
    claims: list[Claim]


class Doc(BaseModel):
    sections: list[Section]


@dataclass
class Deps:
    cells: dict[int, Cell]
    dropped: list = field(default_factory=list)


def numbers(text):
    return {float(n.replace(",", "")) for n in NUMBER.findall(text)}


def check(claim: Claim, cells: dict[int, Cell]) -> list[str]:
    """Return why a claim is not grounded in its cited code (empty list = verified)."""
    errors, cited = [], []
    for c in claim.cites:
        c.start, c.end = sorted((c.start, c.end))
        owner = next((x for x in cells.values() if x.start <= c.start and c.end <= x.end), None)
        if owner and c.cell != owner.id:  # line numbers are unique, so a wrong cell id is safely repaired
            c.cell = owner.id
        cell = cells.get(c.cell)
        if not cell:
            errors.append(f"cell C{c.cell} does not exist")
        elif c.start < cell.start or c.end > cell.end:
            errors.append(f"C{c.cell} spans L{cell.start}-L{cell.end}, cite L{c.start}-L{c.end} is outside it")
        elif c.end - c.start > 20:
            errors.append(f"cite C{c.cell} L{c.start}-L{c.end} is too broad, narrow it")
        else:
            cited += [t for n, t in cell.lines.items() if c.start <= n <= c.end]
    if errors:
        return errors
    code = "\n".join(cited)
    if not code.strip():
        return ["cited lines are blank"]
    for name in TICKED.findall(claim.text):
        if name not in code and not all(w in code for w in re.findall(r"\w+", name)):
            errors.append(f"`{name}` does not appear in the cited lines")
    missing = numbers(TICKED.sub(" ", claim.text)) - numbers(code)
    if missing:
        errors.append(f"numbers {sorted(missing)} do not appear in the cited lines")
    live = [t for t in cited if t.strip() and not (t.lstrip().startswith("#") and not t.startswith("# MAGIC"))]
    if not live and not DEAD_WORDS.search(claim.text):
        errors.append("all cited lines are commented out; say so or cite live code")
    return errors


def verify(ctx: RunContext[Deps], doc: Doc) -> Doc:
    """Pydantic AI output validator: bounce bad cites back to the model, drop what is still bad at the end."""
    failed = [(s.title, c, e) for s in doc.sections for c in s.claims if (e := check(c, ctx.deps.cells))]
    if failed and ctx.retry < ctx.max_retries:
        raise ModelRetry("These claims failed the citation check. Fix the cite or wording, or remove them, "
                         "then return the whole document again:\n"
                         + "\n".join(f"- [{t}] {c.text!r}: {'; '.join(e)}" for t, c, e in failed))
    ctx.deps.dropped = [{"section": t, "text": c.text, "errors": e} for t, c, e in failed]
    for s in doc.sections:
        s.claims = [c for c in s.claims if not check(c, ctx.deps.cells)]
    return doc


def load_config():
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["model"] = os.getenv("HARNESS_MODEL", cfg["model"])
    provider = cfg["model"].split(":", 1)[0]
    cfg["model_settings"] = {**cfg.get("model_settings", {}), **cfg.get("provider_settings", {}).get(provider, {})}
    for name, value in (cfg.get("api_keys") or {}).items():
        os.environ.setdefault(name, str(value))
    if provider == "ollama":
        os.environ.setdefault("OLLAMA_BASE_URL", cfg.get("ollama_base_url", "http://localhost:11434/v1"))
    return cfg


def generate(cells: list[Cell], audience: str, cfg: dict) -> dict:
    spec = DOCS[audience]
    agent = Agent(
        cfg["model"],
        output_type=MODES[cfg.get("output_mode", "tool")](Doc),
        deps_type=Deps,
        instructions=[RULES, lambda ctx: "NOTEBOOK:\n" + render(ctx.deps.cells.values())],
        retries=cfg.get("retries", 2),
        model_settings=cfg.get("model_settings", {}),
    )
    agent.output_validator(verify)
    sections = "\n".join(f'- "{title}": {what}' for title, what in spec["sections"].items())
    prompt = f"Reader: {spec['reader']}\nWrite \"{spec['title']}\" with exactly these sections, 3-7 claims each:\n{sections}"
    deps = Deps({c.id: c for c in cells})
    result = agent.run_sync(prompt, deps=deps)
    usage = result.usage() if callable(result.usage) else result.usage
    reply = result.response  # model/provider as reported by the API response, not by config
    called = f"{reply.provider_name}:{reply.model_name} @ {reply.provider_url}"
    print(f"[harness] {audience}: configured={cfg['model']} called={called} "
          f"requests={usage.requests} tokens={usage.input_tokens + usage.output_tokens}", flush=True)
    return {"title": spec["title"], **result.output.model_dump(), "dropped": deps.dropped, "model": called,
            "requests": usage.requests, "tokens": usage.input_tokens + usage.output_tokens}


def ref(cite) -> str:
    return f"C{cite['cell']} L{cite['start']}" + (f"-{cite['end']}" if cite["end"] != cite["start"] else "")


def to_markdown(doc: dict, notebook: str) -> str:
    out = [f"# {doc['title']}", f"_Source: `{notebook}` · every claim cites [cell line] in the notebook_", ""]
    for s in doc["sections"]:
        out += [f"## {s['title']}", ""] + [f"- {c['text']} " + " ".join(f"[{ref(x)}]" for x in c["cites"])
                                          for c in s["claims"]] + [""]
    return "\n".join(out)
