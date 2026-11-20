"""Run the public ablations. Pass --answers to use the fixed DeepSeek generator."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import hw4


def flat_chunks(corpus, normalize_tables):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for filename, html in corpus.items():
        soup = BeautifulSoup(html, "html.parser")
        metadata = hw4._doc_metadata(soup, filename)
        if normalize_tables:
            for table in soup.find_all("table"):
                table.replace_with(hw4.table_to_text(str(table)))
        text = soup.get_text(" ", strip=True)
        chunks.extend(Document(page_content=part, metadata=metadata)
                      for part in splitter.split_text(text))
    return chunks


def global_bm25(chunks):
    model = BM25Okapi([chunk.page_content.lower().split() for chunk in chunks])

    def retrieve(question):
        return list(model.get_top_n(question.lower().split(), chunks, n=hw4.MAX_RETRIEVED))

    return retrieve


def lexical_rank(index, question):
    query = hw4.parse_query(question)
    terms = hw4._query_terms(question, query["companies"])
    ranked = hw4._rank(index, terms, list(range(len(index.chunks))),
                       query["fiscal_year"], query["doc_type"])
    return [index.chunks[item[2]] for item in ranked[:hw4.MAX_RETRIEVED]]


def evaluate(label, retrieve, questions, gold, folder, with_answers):
    rows = []
    for qid, question in questions.items():
        chunks = retrieve(question)
        answerable = bool(gold[qid]["evidence"])
        rows.append({
            "id": qid, "category": gold[qid]["category"], "question": question,
            "retrieved": ";".join(chunk.metadata["source"] for chunk in chunks),
            "answerable": answerable,
            "hit": hw4.retrieval_hit(chunks, gold[qid]) if answerable else "",
            "answer": "", "correct": "",
        })
        rows[-1]["chunks"] = chunks

    if with_answers:
        def answer(row):
            return hw4.generate(row["question"], row["chunks"])

        with ThreadPoolExecutor(max_workers=8) as pool:
            for row, response in zip(rows, pool.map(answer, rows)):
                row["answer"] = response
                row["correct"] = hw4.answer_correct(response, gold[row["id"]])

    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    metrics = {
        "variant": label,
        "recall_at_5": sum(row["hit"] for row in answerable) / len(answerable),
        "kept": sum(bool(row["retrieved"]) for row in answerable) / len(answerable),
        "rejected": sum(not row["retrieved"] for row in unanswerable) / len(unanswerable),
        "accuracy": sum(row["correct"] for row in rows) / len(rows) if with_answers else "",
    }
    for row in rows:
        del row["chunks"]
    write_csv(folder / f"{label}.csv", rows)
    return metrics


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=Path("public_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("ablation_results"))
    parser.add_argument("--answers", action="store_true")
    args = parser.parse_args()
    folder = args.data_folder
    questions = hw4.load_json(folder / "questions.json")
    gold = hw4.load_json(folder / "ground_truth.json")
    corpus = hw4.load_corpus(folder / "corpus")
    structured = [chunk for filename, html in corpus.items()
                  for chunk in hw4.chunk_document(filename, html)]
    index = hw4.build_index(structured)
    variants = [
        ("B0", global_bm25(flat_chunks(corpus, normalize_tables=False))),
        ("V1", global_bm25(flat_chunks(corpus, normalize_tables=True))),
        ("V2", global_bm25(structured)),
        ("V3", lambda question: lexical_rank(index, question)),
        ("V4", lambda question: hw4._retrieve(index, question, diversify=False, abstain=False)),
        ("V5", lambda question: hw4._retrieve(index, question, diversify=True, abstain=False)),
        ("V6", lambda question: hw4.retrieve(index, question)),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [evaluate(label, retrieve, questions, gold, args.output_dir, args.answers)
               for label, retrieve in variants]
    write_csv(args.output_dir / "summary.csv", metrics)
    for row in metrics:
        print(row)


if __name__ == "__main__":
    main()
