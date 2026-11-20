"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================


from collections import defaultdict
from dataclasses import dataclass
from textwrap import wrap

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi


_YEAR = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_COMPANIES: dict[str, str] = {}
_PROPER_NAME = re.compile(r"\b[A-Z][a-zA-Z&.-]*(?:\s+[A-Z][a-zA-Z&.-]*)+")
_BROKER = re.compile(r"\b(?:securities|capital|research|partners|advisors|advisers)\.?$", re.I)
_STOP_WORDS = set("""
    a an the at by and or to of for from on in as its s did does do was were
    is are it what who which how much many had has have hold held end year
    ended fiscal december june march september usd dollars million millions
    billion billions percent percentage number value report reported generate
    spend company's company companies more higher lower between give set city
""".split())
_ALIASES = (
    (r"\bnet accounts receivable\b", "accounts receivable net"),
    (r"\bheadcount\b", "employees"),
    (r"\bemployees\b", "employees"),
    (r"\bcfo\b", "chief financial officer"),
    (r"\bceo\b", "chief executive officer"),
    (r"\baudit firm\b", "independent auditor"),
    (r"\baudited\b", "auditor"),
    (r"\bheadquartered\b|\bheadquarters\b", "headquarters"),
    (r"\beurope middle east and africa\b", "emea"),
    (r"\bgreater china\b", "greater china"),
)
_TEXT_BLOCK = re.compile(r"^(?:h[1-6]|p|li|table|blockquote|pre|dt|dd)$")


@dataclass(frozen=True)
class RetrievalIndex:
    chunks: list[Document]
    model: BM25Okapi | None
    texts: list[str]
    groups: dict[tuple[str, str], list[int]]


def _doc_metadata(soup: BeautifulSoup, file_name: str) -> dict:
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    heading = soup.find("h1")
    company = heading.get_text(" ", strip=True) if heading else title.split("|")[0]
    company = re.sub(r"\s*\([^()]*\)\s*$", "", company).strip()
    year = _YEAR.search(title)
    if not year:
        year = _YEAR.search(soup.get_text(" ", strip=True))
    return {
        "source": file_name,
        "company": company,
        "fiscal_year": int(year[1]) if year else None,
        "doc_type": _document_type(title),
    }


def _document_type(text: str) -> str:
    if re.search(r"\b(?:earnings|conference|results)\s+call\b", text, re.I):
        return "earnings_call"
    if re.search(r"\b(?:results review|research report|target price|(?:investment )?rating|forecast\w*|broker)\b", text, re.I):
        return "research_report"
    return "annual_report"


def _company_names(question: str) -> list[str]:
    names = {}
    normalized = normalize(question)
    has_known = any(f" {key} " in normalized for key in _COMPANIES)
    comparison = bool(re.search(r"\bwhich company\b|\bcompare\b", question, re.I))
    for match in _PROPER_NAME.finditer(question):
        words = match[0].split()
        while words and words[0].lower() in _STOP_WORDS:
            words.pop(0)
        name = " ".join(words).rstrip(".")
        key = company_key(name)
        if key in _COMPANIES:
            names[key] = _COMPANIES[key]
        elif len(words) > 1 and not _BROKER.search(name):
            # An unknown issuer can resemble a known one. Preserve the full span.
            tail = question[match.end():]
            possessive = bool(re.match(r"['?](?:s\b)?", tail))
            issuer_position = re.match(r"\s+(?:hold|report|announce|generate|earn|in|or|and)\b|[?,.]|$", tail)
            if possessive or (issuer_position and (comparison or not has_known)):
                names[key] = name

    for key, name in _COMPANIES.items():
        if f" {key} " in normalized and not any(
            key != other and f" {key} " in f" {other} " for other in names
        ):
            names[key] = name
    return list(names.values())


def _tokens(text: str) -> list[str]:
    words = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    for pattern, replacement in _ALIASES:
        words = re.sub(pattern, replacement, words)
    return [
        word[:-1] if word.endswith("s") and len(word) > 4 and not word.endswith(("ss", "us")) else word
        for word in words.split() if word not in _STOP_WORDS and not word.isdigit()
    ]


def _query_terms(question: str, names: list[str]) -> list[str]:
    for name in names:
        pattern = r"\W+".join(map(re.escape, company_key(name).split()))
        question = re.sub(rf"\b{pattern}\b", " ", question, flags=re.I)
    question = re.sub(r"\b(?:FY|Q)\d+\b", " ", question, flags=re.I)
    return _tokens(question)


def _pack(lines: list[str], header: str, limit: int) -> list[str]:
    prefix = header + "\n" if header else ""
    capacity = limit - len(prefix)
    blocks = []
    current = []
    length = 0
    for line in lines:
        for part in wrap(line, width=capacity) if len(line) > capacity else [line]:
            if current and length + 1 + len(part) > capacity:
                blocks.append(prefix + "\n".join(current))
                current, length = [], 0
            length += len(part) + bool(current)
            current.append(part)
    if current:
        blocks.append(prefix + "\n".join(current))
    return blocks


def _eligible_year(chunk: Document, year: int | None, doc_type: str) -> bool:
    if year is None:
        return True
    fiscal_year = chunk.metadata["fiscal_year"]
    if doc_type == "annual_report":
        return fiscal_year is not None and fiscal_year - 1 <= year <= fiscal_year
    return year == fiscal_year or str(year) in _YEAR.findall(chunk.page_content)


def _lexical_score(terms: list[str], text: str) -> tuple[float, float]:
    term_set = set(terms)
    coverage = len(term_set.intersection(text.split())) / max(1, len(term_set))
    phrase = " ".join(terms)
    phrase_match = bool(phrase) and f" {phrase} " in f" {text} "
    return coverage, float(phrase_match)


def table_to_text(table_html: str) -> str:
    """Serialize each HTML table row with the column headers and caption."""
    table = BeautifulSoup(table_html, "html.parser").find("table")
    if table is None:
        return ""
    caption = table.find("caption")
    lines = [caption.get_text(" ", strip=True)] if caption else []
    headers = []
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        values = [cell.get_text(" ", strip=True) for cell in cells]
        if not values:
            continue
        if row.find_parent("thead") or (not headers and all(cell.name == "th" for cell in cells) and len(cells) > 1):
            headers = values
            continue
        label = values[0]
        fields = [
            f"{headers[i]}: {value}" if i < len(headers) and headers[i] else value
            for i, value in enumerate(values[1:], start=1) if value
        ]
        line = " | ".join([label, *fields])
        if len(line) <= MAX_LINE_CHARS:
            lines.append(line)
        else:
            lines.extend(f"{label} | {field}" for field in fields)
    return "\n".join(lines)


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Keep sections, table captions and complete rows together where possible."""
    soup = BeautifulSoup(html, "html.parser")
    metadata = _doc_metadata(soup, file_name)
    _COMPANIES[company_key(metadata["company"])] = metadata["company"]
    chunks = []
    pending = []
    section = ""
    body = soup.body or soup

    def emit(lines: list[str], heading: str, is_table: bool = False) -> None:
        header = heading[:MAX_CHUNK_CHARS // 4]
        for content in _pack(lines, header, MAX_CHUNK_CHARS):
            chunks.append(Document(page_content=content, metadata={
                **metadata, "section": heading, "table": is_table,
            }))

    for node in body.find_all(_TEXT_BLOCK):
        if node.find_parent(["table", "li", "blockquote"]):
            continue
        if node.name == "table":
            emit(pending, section)
            pending = []
            table_lines = table_to_text(str(node)).splitlines()
            if table_lines:
                caption = table_lines.pop(0) if node.caption else section
                emit(table_lines, caption, True)
        elif node.name.startswith("h"):
            emit(pending, section)
            section = node.get_text(" ", strip=True)
            pending = [section] if section else []
        else:
            text = node.get_text(" ", strip=True)
            if text:
                if pending == [section]:
                    pending.clear()
                pending.append(text)
    emit(pending, section)
    return chunks


def parse_query(question: str) -> dict:
    """Identify the issuers, year requested, and document type."""
    year = _YEAR.search(question)
    return {
        "companies": _company_names(question),
        "fiscal_year": int(year[1]) if year else None,
        "doc_type": _document_type(question),
    }


def build_index(chunks: list[Document]) -> RetrievalIndex:
    """Build one lexical index and group its rows by issuer and document type."""
    groups = defaultdict(list)
    tokenized = []
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        groups[company_key(meta["company"]), meta["doc_type"]].append(i)
        tokenized.append(_tokens(chunk.page_content) or [""])
    model = BM25Okapi(tokenized) if tokenized else None
    return RetrievalIndex(chunks, model, [" ".join(row) for row in tokenized], dict(groups))


def _rank(index: RetrievalIndex, terms: list[str], candidates: list[int],
          year: int | None, doc_type: str) -> list[tuple[float, float, int]]:
    scores = index.model.get_batch_scores(terms, candidates)
    ranked = []
    for i, score in zip(candidates, scores):
        chunk = index.chunks[i]
        coverage, phrase = _lexical_score(terms, index.texts[i])
        exact_year = year is not None and chunk.metadata["fiscal_year"] == year
        table = chunk.metadata.get("table", False)
        score += 12 * coverage + 6 * phrase + 2 * exact_year
        if doc_type == "annual_report" and table:
            score += 3
        ranked.append((score, coverage, i))
    return sorted(ranked, reverse=True)


def _retrieve(index: RetrievalIndex, question: str, *, diversify: bool, abstain: bool) -> list[Document]:
    if index.model is None:
        return []
    query = parse_query(question)
    names, year, doc_type = query["companies"], query["fiscal_year"], query["doc_type"]
    terms = _query_terms(question, names)
    groups = []
    for name in names:
        options = index.groups.get((company_key(name), doc_type), [])
        eligible = [i for i in options if _eligible_year(index.chunks[i], year, doc_type)]
        if not eligible and abstain:
            return []
        if eligible:
            groups.append(eligible)
    if not groups:
        if abstain and names:
            return []
        groups = [[i for i, chunk in enumerate(index.chunks) if chunk.metadata["doc_type"] == doc_type]]
    ranked_groups = [_rank(index, terms, group, year, doc_type) for group in groups]
    # Annual-report questions usually name the metric directly. Calls and
    # research questions use more framing words, so use a lower threshold.
    minimum_coverage = 2 / 3 if doc_type == "annual_report" else 0.40
    if abstain and any(
        not group or max(item[1] for item in group) < minimum_coverage
        for group in ranked_groups
    ):
        return []

    ranked = sorted((item for group in ranked_groups for item in group), reverse=True)
    first = sorted((group[0] for group in ranked_groups if group), reverse=True) if diversify else []
    selected = {item[2] for item in first[:MAX_RETRIEVED]}
    for _, _, i in ranked:
        if len(selected) == MAX_RETRIEVED:
            break
        selected.add(i)
    return [index.chunks[i] for _, _, i in ranked if i in selected]


def retrieve(index: RetrievalIndex, question: str) -> list[Document]:
    """Rank eligible chunks and reserve space for every comparison issuer."""
    return _retrieve(index, question, diversify=True, abstain=True)


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
