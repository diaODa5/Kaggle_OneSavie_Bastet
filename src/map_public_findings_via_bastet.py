import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd


LABEL_COLUMNS = ["severity", "tag", "subtag"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINDINGS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test_public_finding_matches.parquet"
)
DEFAULT_EXTERNAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "external_bastet_findings_mapped.parquet"
)
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw_kaggle" / "train.csv"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "public_judging_findings_bastet_mapped.parquet"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "public_judging_bastet_mapping_report.md"
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_contest(value: Any) -> str:
    text = _text(value).lower().replace("\\", "/").rstrip("/")
    text = text.rsplit("/", 1)[-1]
    text = re.sub(r"\.git$", "", text)
    text = re.sub(r"-(?:findings|judging)(?:-main)?$", "", text)
    return text


def normalize_issue_id(value: Any) -> str:
    match = re.search(r"\b([hm])[-_ ]*0*(\d+)\b", _text(value), re.IGNORECASE)
    if not match:
        return ""
    return f"{match.group(1).upper()}-{int(match.group(2))}"


def _label_key(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _text(value).lower()))


def _label_parts(value: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                _label_key(part)
                for part in _text(value).split(",")
                if _label_key(part)
            }
        )
    )


def _legal_catalog(train: pd.DataFrame) -> dict[str, Any]:
    lookups = {}
    for column in LABEL_COLUMNS:
        values = train.get(column, pd.Series(dtype=str)).map(_text)
        lookups[column] = {
            _label_key(value): value
            for value in values
            if _label_key(value)
        }
    tuple_lookup = {}
    for row in train.reindex(columns=LABEL_COLUMNS).itertuples(index=False):
        severity, tag, subtag = (_text(value) for value in row)
        if not severity or not tag or not subtag:
            continue
        key = (
            _label_key(severity),
            _label_parts(tag),
            _label_parts(subtag),
        )
        tuple_lookup[key] = (severity, tag, subtag)
    return {"lookups": lookups, "tuples": tuple_lookup}


def _legalize(value: Any, lookup: dict[str, str]) -> str:
    return lookup.get(_label_key(value), "")


def _source_rank(value: Any) -> int:
    name = _text(value).lower().replace("\\", "/").rsplit("/", 1)[-1]
    if name == "dataset_0831.csv":
        return 2
    if name == "dataset.csv":
        return 1
    return 0


def _normalized_similarity(left: Any, right: Any) -> float:
    left_text = _label_key(left)
    right_text = _label_key(right)
    if not left_text or not right_text:
        return 0.0
    return SequenceMatcher(None, left_text, right_text).ratio()


def _public_text(row: pd.Series) -> str:
    return " ".join(
        part
        for part in (
            _text(row.get("title")),
            _text(row.get("root_cause")),
            _text(row.get("description")),
        )
        if part
    )


def _public_description(row: pd.Series) -> str:
    title = _text(row.get("title"))
    root_cause = _text(row.get("root_cause")) or _text(row.get("description"))
    if title and root_cause and _label_key(title) != _label_key(root_cause):
        return f"{title}\n\n{root_cause}"
    return title or root_cause


def _external_text(row: pd.Series) -> str:
    return " ".join(
        part
        for part in (
            _text(row.get("description")),
            _text(row.get("report_text")),
        )
        if part
    )


def _prepare_external(
    external: pd.DataFrame,
    catalog: dict[str, Any],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[
        tuple[str, str],
        dict[tuple[str, str, str], dict[str, Any]],
    ] = {}
    lookups = catalog["lookups"]
    for _, row in external.iterrows():
        contest_key = normalize_contest(row.get("repo_path"))
        issue_key = (
            normalize_issue_id(row.get("report_path"))
            or normalize_issue_id(row.get("report_text"))
            or normalize_issue_id(row.get("description"))
        )
        labels = tuple(
            _legalize(row.get(column), lookups[column])
            for column in LABEL_COLUMNS
        )
        if not contest_key or not issue_key or not all(labels):
            continue
        key = (contest_key, issue_key)
        candidate = {
            "severity": labels[0],
            "tag": labels[1],
            "subtag": labels[2],
            "source_rank": _source_rank(row.get("source_csv")),
            "source_csv": _text(row.get("source_csv")),
            "text": _external_text(row),
        }
        existing = grouped.setdefault(key, {}).get(labels)
        if existing is None or candidate["source_rank"] > existing["source_rank"]:
            grouped[key][labels] = candidate
    return {key: list(candidates.values()) for key, candidates in grouped.items()}


def _resolve_exact_candidates(
    candidates: list[dict[str, Any]],
    catalog: dict[str, Any],
    public_text: str,
) -> dict[str, Any]:
    candidate_count = len(candidates)
    if candidate_count == 1:
        selected = candidates[0]
        return {
            **{column: selected[column] for column in LABEL_COLUMNS},
            "mapping_method": "external_exact",
            "mapping_confidence": 1.0,
            "mapping_ambiguity": False,
            "mapping_candidate_count": 1,
        }

    severities = {_label_key(candidate["severity"]) for candidate in candidates}
    tag_parts = tuple(
        sorted(
            {
                part
                for candidate in candidates
                for part in _label_parts(candidate["tag"])
            }
        )
    )
    subtag_parts = tuple(
        sorted(
            {
                part
                for candidate in candidates
                for part in _label_parts(candidate["subtag"])
            }
        )
    )
    merged = None
    if len(severities) == 1:
        merged = catalog["tuples"].get(
            (next(iter(severities)), tag_parts, subtag_parts)
        )
    if merged is not None:
        return {
            "severity": merged[0],
            "tag": merged[1],
            "subtag": merged[2],
            "mapping_method": "external_exact_merged",
            "mapping_confidence": 0.98,
            "mapping_ambiguity": False,
            "mapping_candidate_count": candidate_count,
        }

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate["source_rank"],
            _normalized_similarity(public_text, candidate["text"]),
            candidate["severity"],
            candidate["tag"],
            candidate["subtag"],
        ),
        reverse=True,
    )
    selected = ranked[0]
    similarity = _normalized_similarity(public_text, selected["text"])
    return {
        **{column: selected[column] for column in LABEL_COLUMNS},
        "mapping_method": "external_exact_ambiguous",
        "mapping_confidence": min(0.94, 0.82 + 0.12 * similarity),
        "mapping_ambiguity": True,
        "mapping_candidate_count": candidate_count,
    }


def _existing_legal_resolution(
    finding: pd.Series,
    catalog: dict[str, Any],
) -> dict[str, Any] | None:
    labels = {
        column: _legalize(
            finding.get(column),
            catalog["lookups"][column],
        )
        for column in LABEL_COLUMNS
    }
    if not all(labels.values()):
        return None
    return {
        **labels,
        "mapping_method": "existing_legal",
        "mapping_confidence": 1.0,
        "mapping_ambiguity": False,
        "mapping_candidate_count": 0,
    }


def _train_fallback_resolution(
    finding: pd.Series,
    train: pd.DataFrame,
    catalog: dict[str, Any],
    threshold: float,
) -> dict[str, Any] | None:
    public_text = _public_text(finding)
    public_severity = _legalize(
        finding.get("severity"),
        catalog["lookups"]["severity"],
    )
    best: tuple[float, pd.Series] | None = None
    for _, candidate in train.iterrows():
        candidate_severity = _legalize(
            candidate.get("severity"),
            catalog["lookups"]["severity"],
        )
        if public_severity and candidate_severity != public_severity:
            continue
        similarity = _normalized_similarity(
            public_text,
            candidate.get("description"),
        )
        if best is None or similarity > best[0]:
            best = (similarity, candidate)
    if best is None or best[0] < threshold:
        return None
    similarity, candidate = best
    labels = {
        column: _legalize(
            candidate.get(column),
            catalog["lookups"][column],
        )
        for column in LABEL_COLUMNS
    }
    if not all(labels.values()):
        return None
    return {
        **labels,
        "mapping_method": "train_text_fallback",
        "mapping_confidence": similarity,
        "mapping_ambiguity": False,
        "mapping_candidate_count": 0,
    }


def _external_fallback_resolution(
    finding: pd.Series,
    external: pd.DataFrame,
    catalog: dict[str, Any],
    threshold: float,
) -> dict[str, Any] | None:
    contest_key = normalize_contest(finding.get("contest"))
    public_text = _public_text(finding)
    public_severity = _legalize(
        finding.get("severity"),
        catalog["lookups"]["severity"],
    )
    best: tuple[float, int, dict[str, str]] | None = None
    for _, candidate in external.iterrows():
        if normalize_contest(candidate.get("repo_path")) != contest_key:
            continue
        labels = {
            column: _legalize(
                candidate.get(column),
                catalog["lookups"][column],
            )
            for column in LABEL_COLUMNS
        }
        if not all(labels.values()):
            continue
        if public_severity and labels["severity"] != public_severity:
            continue
        similarity = max(
            _normalized_similarity(public_text, candidate.get("description")),
            _normalized_similarity(public_text, candidate.get("report_text")),
        )
        rank = _source_rank(candidate.get("source_csv"))
        if best is None or (similarity, rank) > (best[0], best[1]):
            best = (similarity, rank, labels)
    if best is None or best[0] < threshold:
        return None
    return {
        **best[2],
        "mapping_method": "external_text_same_contest",
        "mapping_confidence": best[0],
        "mapping_ambiguity": False,
        "mapping_candidate_count": 1,
    }


def map_public_findings(
    findings: pd.DataFrame,
    external: pd.DataFrame,
    train: pd.DataFrame,
    fallback_threshold: float = 0.90,
) -> pd.DataFrame:
    catalog = _legal_catalog(train)
    external_by_key = _prepare_external(external, catalog)
    rows = []
    for _, finding in findings.iterrows():
        row = finding.to_dict()
        key = (
            normalize_contest(finding.get("contest")),
            normalize_issue_id(finding.get("issue_id")),
        )
        candidates = external_by_key.get(key, [])
        if candidates:
            resolution = _resolve_exact_candidates(
                candidates,
                catalog,
                _public_text(finding),
            )
        else:
            resolution = _existing_legal_resolution(finding, catalog)
            if (
                resolution is None
                and _text(finding.get("platform")).lower() == "sherlock"
            ):
                resolution = _external_fallback_resolution(
                    finding,
                    external,
                    catalog,
                    fallback_threshold,
                )
            if (
                resolution is None
                and _text(finding.get("platform")).lower() == "sherlock"
            ):
                resolution = _train_fallback_resolution(
                    finding,
                    train,
                    catalog,
                    fallback_threshold,
                )
            if resolution is None:
                resolution = {
                    "severity": _legalize(
                        finding.get("severity"),
                        catalog["lookups"]["severity"],
                    ),
                    "tag": "",
                    "subtag": "",
                    "mapping_method": "unresolved",
                    "mapping_confidence": 0.0,
                    "mapping_ambiguity": False,
                    "mapping_candidate_count": 0,
                }
        row.update(resolution)
        row["description"] = _public_description(finding)
        row["labels_complete"] = all(row[column] for column in LABEL_COLUMNS)
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_columns(
    frame: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{source_name} missing required columns: {', '.join(missing)}"
        )


def _validate_output_labels(
    mapped: pd.DataFrame,
    train: pd.DataFrame,
) -> None:
    for column in LABEL_COLUMNS:
        legal = {_text(value) for value in train[column] if _text(value)}
        illegal = {
            _text(value)
            for value in mapped[column]
            if _text(value) and _text(value) not in legal
        }
        if illegal:
            raise ValueError(
                f"Illegal output {column} labels: {sorted(illegal)}"
            )


def _write_report(
    mapped: pd.DataFrame,
    report_path: Path,
    fallback_threshold: float,
) -> None:
    methods = mapped.get(
        "mapping_method",
        pd.Series("", index=mapped.index),
    ).fillna("")
    exact_count = int(methods.str.startswith("external_exact").sum())
    ambiguous_count = int(
        mapped.get(
            "mapping_ambiguity",
            pd.Series(False, index=mapped.index),
        ).fillna(False).astype(bool).sum()
    )
    sherlock_fallback_count = int(
        methods.isin(
            {"external_text_same_contest", "train_text_fallback"}
        ).sum()
    )
    unresolved_count = int(methods.eq("unresolved").sum())
    lines = [
        "# Public Judging Bastet Mapping Report",
        "",
        "## Summary",
        "",
        f"- Findings: `{len(mapped)}`",
        f"- Exact contest + issue mappings: `{exact_count}`",
        f"- Ambiguous mappings: `{ambiguous_count}`",
        f"- Sherlock fallbacks: `{sherlock_fallback_count}`",
        f"- Unresolved findings: `{unresolved_count}`",
        f"- Complete legal label tuples: `{int(mapped['labels_complete'].sum())}`",
        f"- High-threshold fallback minimum: `{fallback_threshold:.2f}`",
        "",
        "## Mapping Methods",
        "",
        "| Method | Count |",
        "| --- | ---: |",
    ]
    for method, count in methods.value_counts().sort_index().items():
        lines.append(f"| {method or 'blank'} | {int(count)} |")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Code4rena exact matching uses normalized contest and normalized H/M issue ID.",
            "- External text fallback is restricted to the same normalized contest.",
            "- `dataset_0831.csv` has source priority when legal versions cannot be safely merged.",
            "- Output labels are either exact train enumerations or empty.",
            "- Descriptions use the public final finding title and public root cause/body.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    findings_path: Path | str = DEFAULT_FINDINGS_PATH,
    external_path: Path | str = DEFAULT_EXTERNAL_PATH,
    train_path: Path | str = DEFAULT_TRAIN_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    report_path: Path | str = DEFAULT_REPORT_PATH,
    fallback_threshold: float = 0.90,
) -> pd.DataFrame:
    findings_path = Path(findings_path)
    external_path = Path(external_path)
    train_path = Path(train_path)
    output_path = Path(output_path)
    report_path = Path(report_path)
    for path in (findings_path, external_path, train_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required mapping input: {path}")

    findings = pd.read_parquet(findings_path)
    external = pd.read_parquet(external_path)
    train = pd.read_csv(train_path)
    _validate_columns(
        findings,
        {"platform", "contest", "issue_id", "title", "severity", "description"},
        "Public findings",
    )
    _validate_columns(
        external,
        {
            "repo_path",
            "report_path",
            "severity",
            "tag",
            "subtag",
            "description",
            "source_csv",
        },
        "External Bastet findings",
    )
    _validate_columns(
        train,
        {"severity", "tag", "subtag", "description"},
        "Train CSV",
    )

    mapped = map_public_findings(
        findings,
        external,
        train,
        fallback_threshold=fallback_threshold,
    )
    _validate_output_labels(mapped, train)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_parquet(output_path, index=False)
    _write_report(mapped, report_path, fallback_threshold)
    return mapped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Map public judging findings through exact Bastet labels."
    )
    parser.add_argument("--findings", type=Path, default=DEFAULT_FINDINGS_PATH)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL_PATH)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=0.90,
    )
    args = parser.parse_args()
    mapped = run(
        findings_path=args.findings,
        external_path=args.external,
        train_path=args.train,
        output_path=args.output,
        report_path=args.report,
        fallback_threshold=args.fallback_threshold,
    )
    print(f"Mapped findings: {len(mapped)}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
