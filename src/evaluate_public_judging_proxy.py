import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "test_public_finding_matches.parquet"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_processed.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "public_judging_findings_mapped.parquet"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "public_judging_proxy_report.md"

LABEL_COLUMNS = ["severity", "tag", "subtag"]
DEFAULT_SIMILARITY_THRESHOLDS = (0.45, 0.55, 0.65)
CONFIDENCE_ORDER = {"rejected": 0, "weak": 1, "strong": 2, "exact": 3}
MAPPED_EXTRA_COLUMNS = [
    "raw_severity",
    "raw_tag",
    "raw_subtag",
    "severity_mapping_method",
    "tag_mapping_method",
    "subtag_mapping_method",
    "nearest_train_repo",
    "nearest_train_similarity",
    "label_mapping_confidence",
    "labels_complete",
]


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _label_key(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _text(value).lower()))


def legalize_label(value: Any, legal_values: Iterable[str]) -> str | None:
    lookup = {_label_key(item): str(item) for item in legal_values if _label_key(item)}
    return lookup.get(_label_key(value))


def _legal_sets(train: pd.DataFrame, supplied: dict[str, Iterable[str]] | None) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for column in LABEL_COLUMNS:
        values = supplied.get(column, []) if supplied is not None else train.get(column, pd.Series(dtype=str))
        result[column] = {_text(value) for value in values if _text(value)}
    return result


def _finding_text(frame: pd.DataFrame) -> pd.Series:
    descriptions = frame.get("description", pd.Series("", index=frame.index)).fillna("").map(_text)
    titles = frame.get("title", pd.Series("", index=frame.index)).fillna("").map(_text)
    return (titles + " " + descriptions).str.strip()


def _nearest_descriptions(findings: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(
        {
            "nearest_train_index": [-1] * len(findings),
            "nearest_train_repo": [""] * len(findings),
            "nearest_train_similarity": [0.0] * len(findings),
        },
        index=findings.index,
    )
    for column in LABEL_COLUMNS:
        empty[f"nearest_{column}"] = ""
    if findings.empty or train.empty:
        return empty

    train_text = _finding_text(train)
    finding_text = _finding_text(findings)
    if not train_text.str.strip().any() or not finding_text.str.strip().any():
        return empty
    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        train_matrix = vectorizer.fit_transform(train_text)
    except ValueError:
        return empty
    finding_matrix = vectorizer.transform(finding_text)
    similarities = finding_matrix @ train_matrix.T

    rows = []
    for position in range(len(findings)):
        row = similarities.getrow(position)
        if row.nnz:
            best_position = int(row.data.argmax())
            train_position = int(row.indices[best_position])
            similarity = float(row.data[best_position])
        else:
            train_position = -1
            similarity = 0.0
        record = {
            "nearest_train_index": train_position,
            "nearest_train_repo": (
                _text(train.iloc[train_position].get("repo_path", "")) if train_position >= 0 else ""
            ),
            "nearest_train_similarity": similarity,
        }
        for column in LABEL_COLUMNS:
            record[f"nearest_{column}"] = (
                _text(train.iloc[train_position].get(column, "")) if train_position >= 0 else ""
            )
        rows.append(record)
    return pd.DataFrame(rows, index=findings.index)


RULES = [
    (
        r"\b(reentr|nonreentrant|checks? effects interactions?|external call)\b",
        "Reentrancy",
        "Violating CEI / Missing nonReentrant",
    ),
    (r"\b(precision|rounding|division before|loss of precision|decimal)\b", "Arithmetic", "Precision Loss"),
    (
        r"\b(access control|unauthorized|permission|onlyowner|missing role|privilege)\b",
        "Access Control",
        "Invalid Validation",
    ),
]


def _rule_labels(text: str, legal: dict[str, set[str]]) -> tuple[str | None, str | None]:
    for pattern, tag, subtag in RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return legalize_label(tag, legal["tag"]), legalize_label(subtag, legal["subtag"])
    return None, None


def map_public_findings(
    findings: pd.DataFrame,
    reference_train: pd.DataFrame,
    legal_labels: dict[str, Iterable[str]] | None = None,
    similarity_threshold: float = 0.55,
) -> pd.DataFrame:
    mapped = _normalize_confidence(findings).reset_index(drop=True)
    reference = reference_train.copy().reset_index(drop=True)
    legal = _legal_sets(reference, legal_labels)
    nearest = _nearest_descriptions(mapped, reference).reset_index(drop=True)
    for column in nearest.columns:
        mapped[column] = nearest[column]

    finding_texts = _finding_text(mapped)
    raw_columns: dict[str, list[str]] = {}
    for column in LABEL_COLUMNS:
        preferred = f"raw_{column}"
        supplied_values = mapped.get(column, pd.Series("", index=mapped.index))
        preferred_values = mapped.get(preferred, pd.Series("", index=mapped.index))
        raw_columns[column] = [
            _text(raw) or _text(supplied)
            for raw, supplied in zip(preferred_values, supplied_values)
        ]
        mapped[preferred] = raw_columns[column]

    output: dict[str, list[str]] = {column: [] for column in LABEL_COLUMNS}
    methods: dict[str, list[str]] = {column: [] for column in LABEL_COLUMNS}
    confidences = []
    for position in range(len(mapped)):
        rule_tag, rule_subtag = _rule_labels(finding_texts.iloc[position], legal)
        row_confidences = []
        for column in LABEL_COLUMNS:
            direct = legalize_label(raw_columns[column][position], legal[column])
            nearest_value = legalize_label(mapped.iloc[position][f"nearest_{column}"], legal[column])
            similarity = float(mapped.iloc[position]["nearest_train_similarity"])
            rule_value = rule_tag if column == "tag" else rule_subtag if column == "subtag" else None
            if direct is not None:
                value, method, confidence = direct, "direct", 1.0
            elif nearest_value is not None and similarity >= similarity_threshold:
                value, method, confidence = nearest_value, "nearest_description", similarity
            elif rule_value is not None:
                value, method, confidence = rule_value, "rule", 0.65
            else:
                value, method, confidence = "", "unresolved", 0.0
            output[column].append(value)
            methods[column].append(method)
            row_confidences.append(confidence)
        confidences.append(min(row_confidences))

    for column in LABEL_COLUMNS:
        mapped[column] = output[column]
        mapped[f"{column}_mapping_method"] = methods[column]
        illegal = set(mapped[column]) - legal[column] - {""}
        if illegal:
            raise ValueError(f"Illegal mapped {column} labels: {sorted(illegal)}")
    mapped["label_mapping_confidence"] = confidences
    mapped["labels_complete"] = mapped[LABEL_COLUMNS].ne("").all(axis=1)
    return mapped


def multiset_f1(true_items: list[Any], predicted_items: list[Any]) -> float:
    truth = Counter(true_items)
    predicted = Counter(predicted_items)
    if not truth and not predicted:
        return 1.0
    if not truth or not predicted:
        return 0.0
    overlap = sum((truth & predicted).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(truth.values())
    return 2 * precision * recall / (precision + recall)


def repository_metrics(truth: pd.DataFrame, predicted: pd.DataFrame) -> dict[str, float | int]:
    true_count = len(truth)
    pred_count = len(predicted)
    metrics: dict[str, float | int] = {
        "true_count": true_count,
        "pred_count": pred_count,
        "count_error": pred_count - true_count,
        "count_abs_error": abs(pred_count - true_count),
    }
    for column in LABEL_COLUMNS:
        metrics[f"{column}_multiset_f1"] = multiset_f1(
            truth.get(column, pd.Series(dtype=str)).tolist(),
            predicted.get(column, pd.Series(dtype=str)).tolist(),
        )
    metrics["tuple_multiset_f1"] = multiset_f1(
        list(truth[LABEL_COLUMNS].itertuples(index=False, name=None)),
        list(predicted[LABEL_COLUMNS].itertuples(index=False, name=None)),
    )
    return metrics


def _confidence_value(value: Any) -> int:
    return CONFIDENCE_ORDER.get(_text(value).lower(), 0)


def _normalize_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "verification_confidence" not in normalized.columns:
        normalized["verification_confidence"] = normalized.get(
            "confidence", pd.Series("unknown", index=normalized.index)
        )
    normalized["verification_confidence"] = (
        normalized["verification_confidence"].fillna("unknown").map(_text).str.lower()
    )
    return normalized


def evaluate_train_proxy(
    public_findings: pd.DataFrame,
    train: pd.DataFrame,
    legal_labels: dict[str, Iterable[str]] | None = None,
    similarity_threshold: float = 0.55,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    public = _normalize_confidence(public_findings).reset_index(drop=True)
    truth = train.copy().reset_index(drop=True)

    mapped_parts = []
    for repo in sorted(set(public.get("repo_path", pd.Series(dtype=str))) & set(truth["repo_path"])):
        repo_public = public[public["repo_path"] == repo].copy()
        reference = truth[truth["repo_path"] != repo].copy()
        mapped_parts.append(
            map_public_findings(repo_public, reference, legal_labels, similarity_threshold)
        )
    mapped = (
        pd.concat(mapped_parts, ignore_index=True)
        if mapped_parts
        else map_public_findings(public.iloc[0:0], truth.iloc[0:0], legal_labels, similarity_threshold)
    )

    confidence_names = sorted(
        {_text(value).lower() or "unknown" for value in mapped.get("verification_confidence", [])},
        key=lambda item: (-_confidence_value(item), item),
    )
    policies = [(name, {name}) for name in confidence_names]
    policies.append(("all", set(confidence_names)))

    per_repo_rows = []
    summary_rows = []
    repos = sorted(set(mapped.get("repo_path", pd.Series(dtype=str))))
    for policy_name, accepted in policies:
        policy_rows = []
        for repo in repos:
            true_repo = truth[truth["repo_path"] == repo]
            policy_repo = mapped[
                (mapped["repo_path"] == repo)
                & mapped["verification_confidence"].fillna("unknown").str.lower().isin(accepted)
            ]
            pred_repo = policy_repo[policy_repo["labels_complete"]]
            metrics = repository_metrics(true_repo, pred_repo)
            metrics["pred_count"] = len(policy_repo)
            metrics["count_error"] = len(policy_repo) - len(true_repo)
            metrics["count_abs_error"] = abs(len(policy_repo) - len(true_repo))
            record = {
                "verification_confidence": policy_name,
                "repo_path": repo,
                **metrics,
            }
            per_repo_rows.append(record)
            policy_rows.append(record)
        if policy_rows:
            policy_frame = pd.DataFrame(policy_rows)
            summary_rows.append(
                {
                    "verification_confidence": policy_name,
                    "repository_count": len(policy_frame),
                    "count_mae": float(policy_frame["count_abs_error"].mean()),
                    "severity_multiset_f1": float(policy_frame["severity_multiset_f1"].mean()),
                    "tag_multiset_f1": float(policy_frame["tag_multiset_f1"].mean()),
                    "subtag_multiset_f1": float(policy_frame["subtag_multiset_f1"].mean()),
                    "tuple_multiset_f1": float(policy_frame["tuple_multiset_f1"].mean()),
                }
            )
    per_repo = pd.DataFrame(per_repo_rows)
    summary = pd.DataFrame(summary_rows)
    return mapped, per_repo, summary


def _empty_mapped() -> pd.DataFrame:
    columns = [
        "repo_path",
        "severity",
        "tag",
        "subtag",
        "description",
        "verification_confidence",
        *MAPPED_EXTRA_COLUMNS,
    ]
    return pd.DataFrame(columns=list(dict.fromkeys(columns)))


def _write_blocked_report(report_path: Path, findings_path: Path, reason: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Public Judging Proxy Report",
                "",
                "## Status: Blocked",
                "",
                f"- Reason: {reason}",
                f"- Expected upstream findings: `{findings_path}`",
                "- No proxy thresholds were calibrated and no labels were invented.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_report(
    report_path: Path,
    mapped: pd.DataFrame,
    calibration: pd.DataFrame,
    recommended_threshold: float,
    recommended_confidence: str,
) -> None:
    lines = [
        "# Public Judging Proxy Report",
        "",
        "## Status: Complete" if not calibration.empty else "## Status: Calibration unavailable",
        "",
        f"- Mapped findings: `{len(mapped)}`",
        f"- Complete legal label tuples: `{int(mapped['labels_complete'].sum())}`",
    ]
    if calibration.empty:
        lines.extend(
            [
                "- No public findings matched train repository identifiers, so no leakage-free proxy calibration was possible.",
                f"- Operational default description threshold: `{recommended_threshold:.2f}`",
                f"- Operational default verification confidence: `{recommended_confidence}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- Recommended description threshold: `{recommended_threshold:.2f}`",
                f"- Recommended verification confidence: `{recommended_confidence}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Mapping Methods",
            "",
            "| Label | Method | Count |",
            "| --- | --- | ---: |",
        ]
    )
    for column in LABEL_COLUMNS:
        counts = mapped[f"{column}_mapping_method"].value_counts()
        for method, count in counts.items():
            lines.append(f"| {column} | {method} | {int(count)} |")
    if not calibration.empty:
        lines.extend(
            [
                "",
                "## Train-Repository Proxy",
                "",
                "| Threshold | Verification confidence | Repositories | Count MAE | Tuple multiset F1 |",
                "| ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in calibration.itertuples(index=False):
            lines.append(
                f"| {row.similarity_threshold:.2f} | {row.verification_confidence} | "
                f"{int(row.repository_count)} | {row.count_mae:.4f} | {row.tuple_multiset_f1:.4f} |"
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    findings_path: Path | str = DEFAULT_FINDINGS_PATH,
    train_path: Path | str = DEFAULT_TRAIN_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    report_path: Path | str = DEFAULT_REPORT_PATH,
) -> pd.DataFrame:
    findings_path = Path(findings_path)
    train_path = Path(train_path)
    output_path = Path(output_path)
    report_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not findings_path.exists():
        mapped = _empty_mapped()
        mapped.to_parquet(output_path, index=False)
        _write_blocked_report(report_path, findings_path, "upstream verified findings are absent")
        return mapped
    findings = pd.read_parquet(findings_path)
    if findings.empty:
        mapped = _empty_mapped()
        mapped.to_parquet(output_path, index=False)
        _write_blocked_report(report_path, findings_path, "upstream verified findings contain no parsed findings")
        return mapped
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train data: {train_path}")

    train = pd.read_parquet(train_path)
    legal = _legal_sets(train, None)
    train_repos = set(train["repo_path"])
    proxy_findings = findings[findings.get("repo_path", pd.Series(dtype=str)).isin(train_repos)].copy()
    calibration_parts = []
    for threshold in DEFAULT_SIMILARITY_THRESHOLDS:
        _, _, summary = evaluate_train_proxy(proxy_findings, train, legal, threshold)
        if not summary.empty:
            summary["similarity_threshold"] = threshold
            calibration_parts.append(summary)
    calibration = pd.concat(calibration_parts, ignore_index=True) if calibration_parts else pd.DataFrame()

    if calibration.empty:
        recommended_threshold = 0.55
        recommended_confidence = "exact"
    else:
        ranked = calibration.assign(
            selection_score=calibration["tuple_multiset_f1"] - 0.01 * calibration["count_mae"]
        ).sort_values(
            ["selection_score", "similarity_threshold", "verification_confidence"],
            ascending=[False, False, True],
        )
        best = ranked.iloc[0]
        recommended_threshold = float(best["similarity_threshold"])
        recommended_confidence = str(best["verification_confidence"])

    mapped = map_public_findings(findings, train, legal, recommended_threshold)
    mapped.to_parquet(output_path, index=False)
    _write_report(
        report_path,
        mapped,
        calibration,
        recommended_threshold,
        recommended_confidence,
    )
    return mapped


def main() -> int:
    parser = argparse.ArgumentParser(description="Map public judging findings and evaluate a train proxy.")
    parser.add_argument("--findings", type=Path, default=DEFAULT_FINDINGS_PATH)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    mapped = run(args.findings, args.train, args.output, args.report)
    print(f"Mapped findings: {len(mapped)}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
