# FTEC5660 Homework 4

## Homework 4 solution

~~~mermaid
flowchart LR
    A[HTML company reports] --> B[Extract metadata and label table cells]
    B --> C[Split by section and table]
    C --> D[Build BM25 index and metadata groups]
    Q[Analyst question] --> P[Parse issuers, year and document type]
    D --> R[Filter and rank eligible chunks]
    P --> R
    R --> S{Enough evidence for each issuer?}
    S -- Yes --> T[Send up to five chunks to the fixed generator]
    S -- No --> U[Return an empty list]
    T --> V[Generated answer]
    U --> W[NOT_FOUND]
~~~

I serialize each table row with its column headers and repeat the caption when a table spans several chunks. Prose chunks retain their section titles, and each chunk records the source, issuer, report year, and document type. The index combines BM25 with groups by issuer and document type. The query parser uses names discovered in the corpus, patterns for unseen issuers, and question wording for the requested year and document type. Annual-report eligibility includes the year covered by the prior-year column. Ranking combines BM25, query-term coverage, phrase matches, and bonuses for tables and matching report years. Comparisons reserve space for each issuer. If a requested issuer or period has no eligible chunks, or the eligible chunks have low coverage, the retriever returns an empty list.

### Public ablation

Each row changes one component relative to the preceding row. Recall@5 requires all annotated evidence groups for a question to have a match. Kept measures the fraction of answerable questions with a nonempty result. Rejected measures the fraction of unanswerable questions that return an empty list.

| Variant | Change | Recall@5 | Kept | Rejected | Answer accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| B0 | Flatten HTML, recursive splitting, whitespace BM25 across all chunks | 23.9% | 100% | 0% | 41.7% |
| V1 | Label table rows before recursive splitting | 23.9% | 100% | 0% | 41.7% |
| V2 | Use section- and table-aware chunking | 26.1% | 100% | 0% | 41.7% |
| V3 | Use query-aware lexical ranking | 47.8% | 100% | 0% | 56.7% |
| V4 | Filter by issuer, document type and eligible year | 100% | 100% | 0% | 96.7% |
| V5 | Reserve a result for each issuer in comparisons | 100% | 100% | 0% | 96.7% |
| V6 | Enable abstention | 100% | 100% | 100% | 100.0% |

- **V1.** Table labels expose which year and metric each value belongs to. Whole-corpus BM25 still returned similarly worded tables for other issuers, and both recall and accuracy stayed flat.
- **V2.** Section boundaries and repeated captions give shorter evidence passages enough context to be retrieved on their own. The improvement is small while the search pool is still global.
- **V3.** Normalized terms and phrase matching help with differences in wording, such as “net accounts receivable” versus “accounts receivable, net.” Shared financial labels still rank highly across issuers.
- **V4.** Filtering achieves the largest recall gain. For q051, the global search retrieved receivables passages without the requested issuer's FY2025 figure. Filtering kept eligible Northgate Robotics filings, including the passage with the exact figure.
- **V5.** Per-issuer reservation guarantees room for both issuers in comparison questions. V4 already covered both companies for every public comparison, so the measured recall stayed at its ceiling.
- **V6.** Filtering alone retrieved several FY2025 passages for q019, even though they lacked the requested lobbying expense. The coverage check returns an empty list for that question and raises overall accuracy from 96.7% to 100%. This rule can miss a disclosed figure if the report and question describe it with different terms.

The final public run also kept the row and column labels for all graded table cells, passed metadata and content checks on all 40 documents, and identified all three query fields correctly for all 60 questions. The official runner produced results.csv in 9.5 seconds and scored 60 of 60 answers correctly, for 75 of 75 points on the public Tasks 1 through 4. The answer accuracy column is measured with the fixed DeepSeek generator for each ablation variant.

### Reproduce

On Windows, create a project virtual environment and run the offline ablation with its interpreter.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe ablation.py
~~~

Set DEEPSEEK_API_KEY in the local .env file to run the fixed generator.

~~~powershell
.\.venv\Scripts\python.exe hw4.py --data-folder public_test --output results.csv
.\.venv\Scripts\python.exe ablation.py --answers
~~~

The ablation script writes ablation_results/summary.csv and one CSV per variant. The official runner writes results.csv.

## Reflection

Ten years from now, I want to be someone people can trust with decisions that affect them. If an AI agent can produce a polished analysis in seconds, I still want to be able to explain what I relied on, whose interests I considered, and what would make me change my mind. I still want to understand how financial decisions are made and build tools that make those decisions easier to scrutinize. This assignment made that concrete for me. A convincing answer about a company is worth very little if the evidence comes from another company or another year.

Beyond my work, I want to be known as someone who keeps promises and makes time for people, especially when there is nothing to gain from it. I am willing to struggle through a difficult conversation or learn a subject slowly enough to notice where my understanding breaks down. An agent could draft an apology or summarize a chapter. I would still want to listen to the person I hurt and sit with the ideas I find hard. In both cases, the time I spend changes how I respond the next time.

I think judgment, patience, and the willingness to admit uncertainty would still make me valuable. I would define myself by the commitments I keep when I am tired, wrong, or under pressure, and by how I treat the people who bear the consequences of my choices. Tomorrow, I can start with one small habit. Before asking an agent to help me make a decision, I will write down what I think and what evidence would change my mind. After reading its answer, I will check one important claim against the original source and explain my final decision in my own words.
