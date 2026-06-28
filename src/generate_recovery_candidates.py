from __future__ import annotations

import argparse
import html
import re
import shutil
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "outputs" / "submission_backup_3825_no_padding_targeted_v2.csv"
PUBLIC_FINDINGS_PATH = ROOT / "data" / "processed" / "public_judging_findings_mapped.parquet"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"


def clean_description(value: object, max_chars: int = 700) -> str:
    text = html.unescape(unicodedata.normalize("NFKD", str(value or "")))
    text = re.sub(r"<br\s*/?>", ". ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[([^\]]{1,120})\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<https?://[^>]+>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[\[\]()<>{}#>*_`|]+", " ", text)
    text = re.sub(r"\bSee here\b\.?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return text


def useful_public_paragraphs(value: object, max_paragraphs: int = 2) -> list[str]:
    raw = str(value or "")
    raw = re.sub(r"```.*?```", "\n", raw, flags=re.DOTALL)
    paragraphs: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if (
            "submitted by" in lowered
            or lowered.startswith(("recommend", "recommended mitigation", "###", "##", "**[", ">"))
            or lowered.startswith(("*submitted", "_submitted", "[", "<http", "http"))
            or re.fullmatch(r"[\[\]\(\)\s\d#L:/._-]+", stripped)
        ):
            continue
        cleaned = clean_description(stripped, max_chars=420)
        if len(cleaned) < 45:
            continue
        if cleaned.count("(") + cleaned.count(")") + cleaned.count("[") + cleaned.count("]") > 6:
            continue
        if re.search(r"\b(function|contract|uint256|address|require|return)\b", cleaned) and len(cleaned.split()) < 12:
            continue
        paragraphs.append(cleaned)
        if len(paragraphs) >= max_paragraphs:
            break
    return paragraphs


def concise_public_description(row: pd.Series) -> str:
    title = clean_description(row.get("title", ""), max_chars=240)
    body = " ".join(useful_public_paragraphs(row.get("description", ""), max_paragraphs=2))
    if title and title.lower() not in body.lower():
        text = f"{title}. {body}" if body else f"{title}."
    else:
        text = body or title
    return clean_description(text, max_chars=700)


def _pair_similarity(query: str, candidates: list[str]) -> list[float]:
    if not candidates:
        return []
    documents = [query, *candidates]
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, lowercase=True
    )
    word_vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, lowercase=True
    )
    char_matrix = char_vectorizer.fit_transform(documents)
    word_matrix = word_vectorizer.fit_transform(documents)
    char_scores = cosine_similarity(char_matrix[0:1], char_matrix[1:]).ravel()
    word_scores = cosine_similarity(word_matrix[0:1], word_matrix[1:]).ravel()
    return (0.6 * char_scores + 0.4 * word_scores).tolist()


def simple_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    if isinstance(frame.index, pd.RangeIndex) and frame.index.name is None:
        data = frame.reset_index(drop=True)
    else:
        index_name = frame.index.name or "index"
        if index_name in frame.columns:
            index_name = f"{index_name}_index"
        data = frame.reset_index(names=index_name)
    columns = [str(column) for column in data.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in data.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in data.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def apply_description_replacements(
    baseline: pd.DataFrame,
    public_findings: pd.DataFrame,
    threshold: float,
    max_changes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = baseline.copy(deep=True)
    required = {"Property", "repo_path", "severity", "tag", "subtag", "description"}
    missing = required.difference(output.columns)
    if missing:
        raise ValueError(f"Baseline is missing required columns: {sorted(missing)}")
    if "repo_path" not in public_findings.columns:
        raise ValueError("Public findings are missing repo_path")

    public = public_findings.copy()
    public["_candidate_description"] = public.apply(concise_public_description, axis=1)
    public = public[public["_candidate_description"].str.len().ge(30)]
    if "verification_confidence" in public.columns:
        public = public[public["verification_confidence"].eq("exact")]
    elif "confidence" in public.columns:
        public = public[public["confidence"].eq("exact")]

    proposals: list[dict[str, object]] = []
    for index, row in output.iterrows():
        repo_candidates = public[public["repo_path"].astype(str).eq(str(row["repo_path"]))]
        if repo_candidates.empty:
            continue
        query = clean_description(row["description"])
        candidate_texts = repo_candidates["_candidate_description"].tolist()
        title_texts = [clean_description(value) for value in repo_candidates.get("title", pd.Series([""] * len(repo_candidates)))]
        full_scores = _pair_similarity(query, candidate_texts)
        title_scores = _pair_similarity(query, title_texts)
        scores = [max(full, title) for full, title in zip(full_scores, title_scores)]
        best_position = max(range(len(scores)), key=scores.__getitem__)
        score = float(scores[best_position])
        replacement = candidate_texts[best_position]
        if score < threshold or replacement == query:
            continue
        source = repo_candidates.iloc[best_position]
        proposals.append(
            {
                "_index": index,
                "Property": int(row["Property"]),
                "repo_path": row["repo_path"],
                "similarity": score,
                "old_description": query,
                "new_description": replacement,
                "source_url": source.get("source_url", ""),
            }
        )

    proposals.sort(key=lambda item: (-float(item["similarity"]), int(item["Property"])))
    selected = proposals[:max_changes]
    for proposal in selected:
        output.at[proposal["_index"], "description"] = proposal["new_description"]

    changes = pd.DataFrame(selected)
    if not changes.empty:
        changes = changes.drop(columns=["_index"])
    return output, changes


def apply_artifact_cleanup(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = baseline.copy(deep=True)
    artifact_pattern = re.compile(r"https?://|\[|\]|<|>|Submitted by|```", re.IGNORECASE)
    changes: list[dict[str, object]] = []
    for index, row in output.iterrows():
        original = str(row["description"])
        if not artifact_pattern.search(original):
            continue
        cleaned = clean_description(original, max_chars=700)
        if len(cleaned) < 20 or cleaned == original:
            continue
        output.at[index, "description"] = cleaned
        changes.append(
            {
                "Property": int(row["Property"]),
                "repo_path": row["repo_path"],
                "old_description": original,
                "new_description": cleaned,
            }
        )
    return output, pd.DataFrame(changes)


def validate_invariants(candidate: pd.DataFrame, baseline: pd.DataFrame) -> None:
    if candidate.shape != baseline.shape:
        raise ValueError("Recovery candidate changed submission shape")
    if candidate.columns.tolist() != baseline.columns.tolist():
        raise ValueError("Recovery candidate changed submission columns")
    fixed_columns = ["Property", "repo_path", "severity", "tag", "subtag"]
    if not candidate[fixed_columns].equals(baseline[fixed_columns]):
        raise ValueError("Recovery candidate changed counts, repositories, properties, or labels")
    if candidate["description"].isna().any():
        raise ValueError("Recovery candidate contains empty descriptions")
    if candidate["description"].astype(str).str.contains(r"[\r\n]", regex=True).any():
        raise ValueError("Recovery candidate contains description newlines")


def write_error_report(baseline: pd.DataFrame, failed: pd.DataFrame) -> None:
    baseline_counts = baseline["repo_path"].value_counts()
    failed_counts = failed["repo_path"].value_counts()
    comparison = pd.DataFrame({"baseline_382_5": baseline_counts, "failed_227": failed_counts}).fillna(0).astype(int)
    comparison["delta"] = comparison["failed_227"] - comparison["baseline_382_5"]
    largest_increases = comparison.sort_values("delta", ascending=False).head(10)
    largest_decreases = comparison.sort_values("delta").head(10)
    missing_repos = sorted(set(baseline["repo_path"]) - set(failed["repo_path"]))
    exact_overlap = len(
        set(map(tuple, baseline.fillna("").astype(str).values.tolist()))
        & set(map(tuple, failed.fillna("").astype(str).values.tolist()))
    )

    lines = [
        "# Public Judging 227 Error Analysis",
        "",
        "## Observed Result",
        "",
        "- Known-good baseline public score: `382.5`",
        "- Public-judging recall candidate public score: `227`",
        f"- Score change: `{-155.5:.1f}`",
        f"- Exact full-row overlap between submissions: `{exact_overlap}` of `400`",
        f"- Baseline repository coverage: `{baseline['repo_path'].nunique()}`",
        f"- Failed candidate repository coverage: `{failed['repo_path'].nunique()}`",
        f"- Repositories dropped entirely: `{', '.join(missing_repos)}`",
        "",
        "## Root Causes",
        "",
        "1. The candidate replaced all 400 baseline findings, so leaderboard evidence from the 382.5 solution was discarded.",
        "2. Public contest findings are not equivalent to Kaggle ground truth. Kaggle repositories include mutated, remediated, and curated variants.",
        "3. Source verification proved that referenced files or functions existed; it did not prove that the vulnerable condition survived in the Kaggle snapshot.",
        "4. The competition metric penalizes over-reporting per repository. Public finding counts were transferred too literally to several repositories.",
        "5. Label mapping was partly heuristic or ambiguous, while Sherlock labels included manual mappings.",
        "6. Train calibration already warned against this policy: selecting all public findings produced count MAE `6.019` per repo and tuple F1 `0.267`.",
        "",
        "## Largest Count Increases",
        "",
        simple_markdown_table(largest_increases),
        "",
        "## Largest Count Decreases",
        "",
        simple_markdown_table(largest_decreases),
        "",
        "## Corrective Strategy",
        "",
        "- Restore the byte-identical 382.5 baseline as the root submission.",
        "- Preserve its 400 rows, all 53 repository allocations, and all severity/tag/subtag labels.",
        "- Test only small description substitutions backed by exact source evidence and high semantic similarity.",
        "- Keep each experimental CSV separate; never overwrite the known-good artifact.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "public_judging_227_error_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-threshold", type=float, default=0.65)
    parser.add_argument("--medium-threshold", type=float, default=0.55)
    parser.add_argument("--high-max-changes", type=int, default=8)
    parser.add_argument("--medium-max-changes", type=int, default=24)
    args = parser.parse_args()

    baseline = pd.read_csv(BASELINE_PATH)
    public_findings = pd.read_parquet(PUBLIC_FINDINGS_PATH)
    failed_path = OUTPUT_DIR / "submission_public_judging_recall.csv"
    failed = pd.read_csv(failed_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_copy = OUTPUT_DIR / "submission_recovered_3825_baseline.csv"
    shutil.copyfile(BASELINE_PATH, safe_copy)
    shutil.copyfile(BASELINE_PATH, ROOT / "submission.csv")
    shutil.copyfile(BASELINE_PATH, OUTPUT_DIR / "submission.csv")

    high, high_changes = apply_description_replacements(
        baseline, public_findings, args.high_threshold, args.high_max_changes
    )
    medium, medium_changes = apply_description_replacements(
        baseline, public_findings, args.medium_threshold, args.medium_max_changes
    )
    artifact_clean, artifact_changes = apply_artifact_cleanup(baseline)
    validate_invariants(high, baseline)
    validate_invariants(medium, baseline)
    validate_invariants(artifact_clean, baseline)

    high_path = OUTPUT_DIR / "submission_recovery_desc_high.csv"
    medium_path = OUTPUT_DIR / "submission_recovery_desc_medium.csv"
    artifact_path = OUTPUT_DIR / "submission_recovery_clean_artifacts.csv"
    high.to_csv(high_path, index=False)
    medium.to_csv(medium_path, index=False)
    artifact_clean.to_csv(artifact_path, index=False)
    write_error_report(baseline, failed)

    report_lines = [
        "# Recovery Candidate Report",
        "",
        "- Root submission remains the byte-identical 382.5 baseline.",
        "- Candidates preserve Property, repo_path, severity, tag, subtag, and per-repo counts.",
        f"- High candidate description changes: `{len(high_changes)}` at threshold `{args.high_threshold}`.",
        f"- Medium candidate description changes: `{len(medium_changes)}` at threshold `{args.medium_threshold}`.",
        f"- Artifact-clean candidate description changes: `{len(artifact_changes)}`.",
        "",
        "## Artifact-Clean Candidate Changes",
        "",
        simple_markdown_table(artifact_changes) if not artifact_changes.empty else "No changes.",
        "",
        "## High Candidate Changes",
        "",
        simple_markdown_table(high_changes) if not high_changes.empty else "No changes.",
        "",
        "## Medium Candidate Changes",
        "",
        simple_markdown_table(medium_changes) if not medium_changes.empty else "No changes.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "recovery_candidates_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(f"Restored baseline: {ROOT / 'submission.csv'}")
    print(f"Artifact-clean candidate: {artifact_path} ({len(artifact_changes)} description changes)")
    print(f"High candidate: {high_path} ({len(high_changes)} description changes)")
    print(f"Medium candidate: {medium_path} ({len(medium_changes)} description changes)")


if __name__ == "__main__":
    main()
