import argparse
import configparser
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ZIP_PATH = PROJECT_ROOT / "data" / "raw_kaggle" / "train.zip"
DEFAULT_TRAIN_CSV_PATH = PROJECT_ROOT / "data" / "raw_kaggle" / "train.csv"
DEFAULT_EXTERNAL_FINDINGS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "external_bastet_findings_mapped.parquet"
)
DEFAULT_HIGH_CONFIDENCE_PAIRS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "high_confidence_train_hash_external_pairs.csv"
)
DEFAULT_OFFICIAL_MATCHES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "official_external_repo_matches.parquet"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "train_public_judging_calibration.parquet"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "reports" / "train_public_judging_calibration_report.md"
)

DEPENDENCY_REPOSITORIES = {
    "chainlink",
    "ds-test",
    "forge-std",
    "openzeppelin-contracts",
    "openzeppelin-contracts-upgradeable",
    "prb-math",
    "safe-contracts",
    "solmate",
}
ACCEPTED_FINGERPRINT_CONFIDENCE = {"high", "very_high"}
MAPPING_COLUMNS = [
    "train_repo_path",
    "origin_url",
    "contest",
    "external_repo_path",
    "mapping_source",
    "mapping_confidence",
    "rejection_reason",
]
LABEL_COLUMNS = ["severity", "tag", "subtag"]
MATCH_COLUMNS = [
    "truth_index",
    "candidate_index",
    "description_similarity",
    "tuple_agreement",
    "match_score",
]


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _github_owner_repo(url: str) -> tuple[str, str]:
    match = re.search(
        r"(?:github\.com[:/])(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+?)(?:\.git)?/?$",
        _text(url),
        flags=re.IGNORECASE,
    )
    if not match:
        return "", ""
    return match.group("owner"), match.group("repo")


def _contest_slug(repo: str) -> str:
    slug = _text(repo).removesuffix(".git")
    return re.sub(r"-(?:findings|judging)$", "", slug, flags=re.IGNORECASE)


def parse_main_project_origin(config_text: str) -> dict[str, str]:
    parser = configparser.RawConfigParser(interpolation=None)
    try:
        parser.read_string(config_text)
    except configparser.Error:
        return {
            "origin_url": "",
            "contest": "",
            "external_repo_path": "",
            "mapping_source": "",
            "mapping_confidence": "",
            "rejection_reason": "invalid_git_config",
        }

    section = 'remote "origin"'
    origin_url = parser.get(section, "url", fallback="").strip()
    owner, repo = _github_owner_repo(origin_url)
    if not origin_url:
        reason = "origin_missing"
    elif not owner or not repo:
        reason = "unsupported_origin"
    elif repo.lower() in DEPENDENCY_REPOSITORIES:
        reason = "dependency_origin"
    else:
        reason = ""
    contest = _contest_slug(repo) if not reason else ""
    return {
        "origin_url": origin_url,
        "contest": contest,
        "external_repo_path": f"repos/{contest}" if contest else "",
        "mapping_source": "git_origin" if contest else "",
        "mapping_confidence": "exact" if contest else "",
        "rejection_reason": reason,
    }


def read_train_origin_evidence(
    train_zip_path: Path | str,
    train_repo_paths: Iterable[str],
) -> pd.DataFrame:
    wanted = {str(repo) for repo in train_repo_paths}
    records: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(train_zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = info.filename.replace("\\", "/").split("/")
            if (
                len(parts) != 4
                or parts[0] != "train"
                or parts[1] not in wanted
                or parts[2:] != [".git", "config"]
            ):
                continue
            parsed = parse_main_project_origin(
                zf.read(info).decode("utf-8", errors="replace")
            )
            records[parts[1]] = {"train_repo_path": parts[1], **parsed}

    rows = []
    for repo in sorted(wanted):
        rows.append(
            records.get(
                repo,
                {
                    "train_repo_path": repo,
                    "origin_url": "",
                    "contest": "",
                    "external_repo_path": "",
                    "mapping_source": "",
                    "mapping_confidence": "",
                    "rejection_reason": "git_config_missing",
                },
            )
        )
    return pd.DataFrame(rows, columns=MAPPING_COLUMNS)


def _best_train_pairs(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty:
        return {}
    rows = frame.copy()
    for column in ("agreement_ratio", "matched_finding_count"):
        if column not in rows:
            rows[column] = 0
    rows["_confidence_rank"] = (
        rows.get("confidence", pd.Series("", index=rows.index))
        .astype(str)
        .str.lower()
        .map({"very_high": 2, "high": 1})
        .fillna(0)
    )
    rows = rows[rows["_confidence_rank"] > 0].sort_values(
        ["_confidence_rank", "agreement_ratio", "matched_finding_count"],
        ascending=False,
        na_position="last",
    )
    return {
        str(row["train_repo_path"]): row
        for _, row in rows.drop_duplicates("train_repo_path").iterrows()
    }


def _best_official_matches(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty:
        return {}
    rows = frame.copy()
    if "split" in rows:
        rows = rows[rows["split"].astype(str).str.lower() == "train"]
    rows["_confidence"] = rows.get(
        "confidence", pd.Series("", index=rows.index)
    ).astype(str).str.lower()
    rows = rows[rows["_confidence"].isin(ACCEPTED_FINGERPRINT_CONFIDENCE)]
    rows["_confidence_rank"] = rows["_confidence"].map({"very_high": 2, "high": 1})
    rows = rows.sort_values(
        ["_confidence_rank", "score"],
        ascending=False,
        na_position="last",
    )
    return {
        str(row["repo_path"]): row
        for _, row in rows.drop_duplicates("repo_path").iterrows()
    }


def resolve_train_repo_mapping(
    train_repo_paths: Iterable[str],
    origin_evidence: pd.DataFrame,
    high_confidence_pairs: pd.DataFrame,
    official_matches: pd.DataFrame,
) -> pd.DataFrame:
    origin_by_repo = (
        {
            str(row["train_repo_path"]): row
            for _, row in origin_evidence.iterrows()
        }
        if not origin_evidence.empty
        else {}
    )
    pairs_by_repo = _best_train_pairs(high_confidence_pairs)
    official_by_repo = _best_official_matches(official_matches)
    rows = []
    for repo in sorted({str(value) for value in train_repo_paths}):
        origin = origin_by_repo.get(repo)
        if origin is not None and _text(origin.get("external_repo_path")):
            rows.append({column: _text(origin.get(column)) for column in MAPPING_COLUMNS})
            continue

        pair = pairs_by_repo.get(repo)
        if pair is not None:
            external_repo = _text(pair.get("top_external_repo_path"))
            rows.append(
                {
                    "train_repo_path": repo,
                    "origin_url": _text(origin.get("origin_url")) if origin is not None else "",
                    "contest": external_repo.removeprefix("repos/"),
                    "external_repo_path": external_repo,
                    "mapping_source": "high_confidence_train_pair",
                    "mapping_confidence": _text(pair.get("confidence")).lower(),
                    "rejection_reason": "",
                }
            )
            continue

        official = official_by_repo.get(repo)
        if official is not None:
            external_repo = _text(official.get("matched_external_repo_path"))
            rows.append(
                {
                    "train_repo_path": repo,
                    "origin_url": _text(origin.get("origin_url")) if origin is not None else "",
                    "contest": external_repo.removeprefix("repos/"),
                    "external_repo_path": external_repo,
                    "mapping_source": "official_train_fingerprint",
                    "mapping_confidence": _text(official.get("confidence")).lower(),
                    "rejection_reason": "",
                }
            )
            continue

        rows.append(
            {
                "train_repo_path": repo,
                "origin_url": _text(origin.get("origin_url")) if origin is not None else "",
                "contest": "",
                "external_repo_path": "",
                "mapping_source": "unresolved",
                "mapping_confidence": "",
                "rejection_reason": (
                    _text(origin.get("rejection_reason"))
                    if origin is not None
                    else "no_mapping_evidence"
                ),
            }
        )
    return pd.DataFrame(rows, columns=MAPPING_COLUMNS)


def _normalized_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _text(value).lower()))


def _canonical_multilabel(value: Any) -> tuple[str, ...]:
    labels = {
        _normalized_text(part)
        for part in re.split(r"[,;|]+", _text(value))
        if _normalized_text(part)
    }
    return tuple(sorted(labels))


def _canonical_label(value: Any, column: str) -> Any:
    if column in {"tag", "subtag"}:
        return _canonical_multilabel(value)
    return _normalized_text(value)


def _finding_text(frame: pd.DataFrame) -> pd.Series:
    title = frame.get("title", pd.Series("", index=frame.index)).fillna("").map(_text)
    description = (
        frame.get("description", pd.Series("", index=frame.index))
        .fillna("")
        .map(_text)
    )
    return (title + " " + description).str.strip()


def deduplicate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy().reset_index(drop=True)
    rows = candidates.copy()
    rows["_dedup_text"] = _finding_text(rows).map(_normalized_text)
    for column in LABEL_COLUMNS:
        values = rows.get(column, pd.Series("", index=rows.index))
        rows[f"_dedup_{column}"] = [
            _canonical_label(value, column) for value in values
        ]
    rows = rows.drop_duplicates(
        subset=[
            "_dedup_text",
            "_dedup_severity",
            "_dedup_tag",
            "_dedup_subtag",
        ],
        keep="first",
    )
    return rows.drop(
        columns=[
            "_dedup_text",
            "_dedup_severity",
            "_dedup_tag",
            "_dedup_subtag",
        ]
    ).reset_index(drop=True)


def _description_similarity_matrix(
    truth: pd.DataFrame,
    candidates: pd.DataFrame,
) -> np.ndarray:
    if truth.empty or candidates.empty:
        return np.zeros((len(truth), len(candidates)), dtype=float)
    true_text = _finding_text(truth)
    candidate_text = _finding_text(candidates)
    if not true_text.str.strip().any() or not candidate_text.str.strip().any():
        return np.zeros((len(truth), len(candidates)), dtype=float)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
    )
    try:
        matrix = vectorizer.fit_transform(
            pd.concat([true_text, candidate_text], ignore_index=True)
        )
    except ValueError:
        return np.zeros((len(truth), len(candidates)), dtype=float)
    true_matrix = matrix[: len(truth)]
    candidate_matrix = matrix[len(truth) :]
    return np.clip((true_matrix @ candidate_matrix.T).toarray(), 0.0, 1.0)


def _tuple_agreement(
    truth_row: pd.Series,
    candidate_row: pd.Series,
) -> float:
    agreements = [
        _canonical_label(truth_row.get(column, ""), column)
        == _canonical_label(candidate_row.get(column, ""), column)
        for column in LABEL_COLUMNS
    ]
    return sum(agreements) / len(agreements)


def greedy_one_to_one_matches(
    truth: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    true_rows = truth.reset_index(drop=True)
    candidate_rows = candidates.reset_index(drop=True)
    similarities = _description_similarity_matrix(true_rows, candidate_rows)
    possible = []
    for truth_index in range(len(true_rows)):
        for candidate_index in range(len(candidate_rows)):
            similarity = float(similarities[truth_index, candidate_index])
            tuple_agreement = _tuple_agreement(
                true_rows.iloc[truth_index],
                candidate_rows.iloc[candidate_index],
            )
            possible.append(
                {
                    "truth_index": truth_index,
                    "candidate_index": candidate_index,
                    "description_similarity": similarity,
                    "tuple_agreement": tuple_agreement,
                    "match_score": 0.85 * similarity + 0.15 * tuple_agreement,
                }
            )
    possible.sort(
        key=lambda row: (
            -row["match_score"],
            -row["description_similarity"],
            -row["tuple_agreement"],
            row["truth_index"],
            row["candidate_index"],
        )
    )
    used_truth: set[int] = set()
    used_candidates: set[int] = set()
    selected = []
    for row in possible:
        if (
            row["truth_index"] in used_truth
            or row["candidate_index"] in used_candidates
        ):
            continue
        used_truth.add(row["truth_index"])
        used_candidates.add(row["candidate_index"])
        selected.append(row)
    return pd.DataFrame(selected, columns=MATCH_COLUMNS)


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


def repository_metrics(
    truth: pd.DataFrame,
    predicted: pd.DataFrame,
) -> dict[str, float | int]:
    true_rows = truth.reset_index(drop=True)
    predicted_rows = predicted.reset_index(drop=True)
    metrics: dict[str, float | int] = {}
    for column in LABEL_COLUMNS:
        true_items = [
            _canonical_label(value, column)
            for value in true_rows.get(column, pd.Series("", index=true_rows.index))
        ]
        predicted_items = [
            _canonical_label(value, column)
            for value in predicted_rows.get(
                column, pd.Series("", index=predicted_rows.index)
            )
        ]
        metrics[f"{column}_multiset_f1"] = multiset_f1(
            true_items,
            predicted_items,
        )
    true_tuples = [
        tuple(_canonical_label(row.get(column, ""), column) for column in LABEL_COLUMNS)
        for _, row in true_rows.iterrows()
    ]
    predicted_tuples = [
        tuple(_canonical_label(row.get(column, ""), column) for column in LABEL_COLUMNS)
        for _, row in predicted_rows.iterrows()
    ]
    metrics["tuple_multiset_f1"] = multiset_f1(true_tuples, predicted_tuples)
    return metrics


STRATEGIES = (
    "all",
    "description_exactish_0.70",
    "description_exactish_0.85",
    "high_only",
    "medium_only",
)


def _candidate_max_similarities(
    truth: pd.DataFrame,
    candidates: pd.DataFrame,
) -> np.ndarray:
    similarities = _description_similarity_matrix(truth, candidates)
    if similarities.size == 0:
        return np.zeros(len(candidates), dtype=float)
    return similarities.max(axis=0)


def _select_strategy_candidates(
    strategy: str,
    candidates: pd.DataFrame,
    maximum_similarities: np.ndarray,
) -> pd.DataFrame:
    if strategy == "all":
        return candidates.copy().reset_index(drop=True)
    if strategy.startswith("description_exactish_"):
        threshold = float(strategy.rsplit("_", 1)[1])
        return candidates.loc[maximum_similarities >= threshold].reset_index(drop=True)
    severity = candidates.get(
        "severity", pd.Series("", index=candidates.index)
    ).map(_normalized_text)
    if strategy == "high_only":
        return candidates.loc[severity == "high"].reset_index(drop=True)
    if strategy == "medium_only":
        return candidates.loc[severity == "medium"].reset_index(drop=True)
    raise ValueError(f"Unknown strategy: {strategy}")


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _evaluate_repository_strategies(
    mapping_row: pd.Series,
    truth: pd.DataFrame,
    public_candidates: pd.DataFrame,
) -> list[dict[str, Any]]:
    candidates = deduplicate_candidates(public_candidates)
    maximum_similarities = _candidate_max_similarities(truth, candidates)
    records = []
    for strategy in STRATEGIES:
        selected = _select_strategy_candidates(
            strategy,
            candidates,
            maximum_similarities,
        )
        matches = greedy_one_to_one_matches(truth, selected)
        similarities = (
            matches["description_similarity"].to_numpy(dtype=float)
            if not matches.empty
            else np.array([], dtype=float)
        )
        metrics = repository_metrics(truth, selected)
        selected_count = len(selected)
        true_count = len(truth)
        exactish_070 = int((similarities >= 0.70).sum())
        exactish_085 = int((similarities >= 0.85).sum())
        records.append(
            {
                "strategy": strategy,
                "train_repo_path": _text(mapping_row.get("train_repo_path")),
                "contest": _text(mapping_row.get("contest")),
                "external_repo_path": _text(
                    mapping_row.get("external_repo_path")
                ),
                "mapping_source": _text(mapping_row.get("mapping_source")),
                "mapping_confidence": _text(
                    mapping_row.get("mapping_confidence")
                ),
                "mapping_rejection_reason": _text(
                    mapping_row.get("rejection_reason")
                ),
                "true_count": true_count,
                "public_candidate_count": len(candidates),
                "selected_candidate_count": selected_count,
                "count_error": selected_count - true_count,
                "count_abs_error": abs(selected_count - true_count),
                "matched_pair_count": len(matches),
                "top_description_similarity": (
                    float(similarities.max()) if similarities.size else 0.0
                ),
                "description_match_count_0_70": exactish_070,
                "description_match_count_0_85": exactish_085,
                "description_precision_0_70": _safe_ratio(
                    exactish_070, selected_count
                ),
                "description_recall_0_70": _safe_ratio(
                    exactish_070, true_count
                ),
                "description_precision_0_85": _safe_ratio(
                    exactish_085, selected_count
                ),
                "description_recall_0_85": _safe_ratio(
                    exactish_085, true_count
                ),
                **metrics,
            }
        )
    return records


def summarize_strategies(calibration: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        group = calibration[calibration["strategy"] == strategy]
        if group.empty:
            continue
        selected_count = int(group["selected_candidate_count"].sum())
        true_count = int(group["true_count"].sum())
        exactish_070 = int(group["description_match_count_0_70"].sum())
        exactish_085 = int(group["description_match_count_0_85"].sum())
        rows.append(
            {
                "strategy": strategy,
                "repository_count": len(group),
                "mapped_repository_count": int(
                    group["external_repo_path"].ne("").sum()
                ),
                "total_true_count": true_count,
                "total_public_candidate_count": int(
                    group["public_candidate_count"].sum()
                ),
                "total_selected_candidate_count": selected_count,
                "count_mae": float(group["count_abs_error"].mean()),
                "description_precision_0_70": _safe_ratio(
                    exactish_070, selected_count
                ),
                "description_recall_0_70": _safe_ratio(
                    exactish_070, true_count
                ),
                "description_precision_0_85": _safe_ratio(
                    exactish_085, selected_count
                ),
                "description_recall_0_85": _safe_ratio(
                    exactish_085, true_count
                ),
                "severity_multiset_f1": float(
                    group["severity_multiset_f1"].mean()
                ),
                "tag_multiset_f1": float(group["tag_multiset_f1"].mean()),
                "subtag_multiset_f1": float(
                    group["subtag_multiset_f1"].mean()
                ),
                "tuple_multiset_f1": float(
                    group["tuple_multiset_f1"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def recommend_precision_strategy(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "all"
    eligible = summary[summary["total_selected_candidate_count"] > 0].copy()
    if eligible.empty:
        return "all"
    preference = {
        "description_exactish_0.85": 5,
        "description_exactish_0.70": 4,
        "high_only": 3,
        "medium_only": 2,
        "all": 1,
    }
    eligible["_preference"] = eligible["strategy"].map(preference).fillna(0)
    ranked = eligible.sort_values(
        [
            "description_precision_0_85",
            "description_precision_0_70",
            "tuple_multiset_f1",
            "_preference",
            "description_recall_0_85",
        ],
        ascending=False,
    )
    return str(ranked.iloc[0]["strategy"])


def _read_csv_if_present(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_parquet_if_present(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def write_report(
    calibration: pd.DataFrame,
    mapping: pd.DataFrame,
    summary: pd.DataFrame,
    recommended_strategy: str,
    report_path: Path | str,
) -> None:
    report_path = Path(report_path)
    lines = [
        "# Train Public Judging Calibration Report",
        "",
        "## Leakage Boundary",
        "",
        "- Calibration split: `train` only",
        "- Ground-truth labels: `data/raw_kaggle/train.csv` only",
        "- Repository identity archive: `data/raw_kaggle/train.zip` only",
        "- External candidates: mapped public report findings only",
        "- Test labels read: `no`",
        "- Test CSV or test ZIP used: `no`",
        "",
        "## Repository Mapping",
        "",
        f"- Train repositories: `{mapping['train_repo_path'].nunique()}`",
        f"- Resolved repositories: `{int(mapping['external_repo_path'].ne('').sum())}`",
        f"- Unresolved repositories: `{int(mapping['external_repo_path'].eq('').sum())}`",
        "",
        "| Mapping source | Repositories |",
        "| --- | ---: |",
    ]
    for source, count in mapping["mapping_source"].value_counts().items():
        lines.append(f"| {source} | {int(count)} |")

    selected = summary[summary["strategy"] == recommended_strategy]
    rationale = ""
    if not selected.empty:
        row = selected.iloc[0]
        rationale = (
            f"0.85 similarity precision proxy {row['description_precision_0_85']:.3f}, "
            f"recall proxy {row['description_recall_0_85']:.3f}, "
            f"tuple multiset F1 {row['tuple_multiset_f1']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Strategy Calibration",
            "",
            "- Exact-ish policies compare train descriptions against public report title/description text.",
            "- High/Medium policies retain only the corresponding mapped severity.",
            "",
            "| Strategy | Selected | Count MAE | Desc P@0.85 | Desc R@0.85 | Severity F1 | Tag F1 | Subtag F1 | Tuple F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {int(row.total_selected_candidate_count)} | "
            f"{row.count_mae:.3f} | {row.description_precision_0_85:.3f} | "
            f"{row.description_recall_0_85:.3f} | "
            f"{row.severity_multiset_f1:.3f} | {row.tag_multiset_f1:.3f} | "
            f"{row.subtag_multiset_f1:.3f} | {row.tuple_multiset_f1:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Recommended precision strategy",
            "",
            f"- Strategy: `{recommended_strategy}`",
            f"- Evidence: {rationale or 'No mapped candidates were available.'}",
            "- This recommendation is calibration evidence from train repositories, not a test-label estimate.",
            "",
            "## Per-Repository Artifact",
            "",
            f"- Rows: `{len(calibration)}`",
            "- The parquet contains true/public/selected counts, count error, top description similarity, and severity/tag/subtag/tuple multiset F1 for every strategy and train repository.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    train_zip_path: Path | str = DEFAULT_TRAIN_ZIP_PATH,
    train_csv_path: Path | str = DEFAULT_TRAIN_CSV_PATH,
    external_findings_path: Path | str = DEFAULT_EXTERNAL_FINDINGS_PATH,
    high_confidence_pairs_path: Path | str = DEFAULT_HIGH_CONFIDENCE_PAIRS_PATH,
    official_matches_path: Path | str = DEFAULT_OFFICIAL_MATCHES_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    report_path: Path | str = DEFAULT_REPORT_PATH,
) -> pd.DataFrame:
    train_zip_path = Path(train_zip_path)
    train_csv_path = Path(train_csv_path)
    external_findings_path = Path(external_findings_path)
    high_confidence_pairs_path = Path(high_confidence_pairs_path)
    official_matches_path = Path(official_matches_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    for required in (train_zip_path, train_csv_path, external_findings_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing required calibration input: {required}")

    train = pd.read_csv(train_csv_path)
    required_train_columns = {
        "repo_path",
        "severity",
        "tag",
        "subtag",
        "description",
    }
    missing_train = sorted(required_train_columns - set(train.columns))
    if missing_train:
        raise ValueError(
            f"Train CSV missing required columns: {', '.join(missing_train)}"
        )
    external = pd.read_parquet(external_findings_path)
    required_external_columns = {
        "repo_path",
        "severity",
        "tag",
        "subtag",
        "description",
    }
    missing_external = sorted(required_external_columns - set(external.columns))
    if missing_external:
        raise ValueError(
            "External findings missing required columns: "
            + ", ".join(missing_external)
        )

    train_repo_paths = sorted(train["repo_path"].astype(str).unique())
    origin_evidence = read_train_origin_evidence(
        train_zip_path,
        train_repo_paths,
    )
    mapping = resolve_train_repo_mapping(
        train_repo_paths,
        origin_evidence,
        _read_csv_if_present(high_confidence_pairs_path),
        _read_parquet_if_present(official_matches_path),
    )

    records = []
    external_repo_values = external["repo_path"].astype(str)
    for _, mapping_row in mapping.iterrows():
        train_repo = mapping_row["train_repo_path"]
        external_repo = mapping_row["external_repo_path"]
        truth = train[train["repo_path"].astype(str) == train_repo].copy()
        candidates = (
            external[external_repo_values == external_repo].copy()
            if external_repo
            else external.iloc[0:0].copy()
        )
        records.extend(
            _evaluate_repository_strategies(
                mapping_row,
                truth,
                candidates,
            )
        )
    calibration = pd.DataFrame(records)
    summary = summarize_strategies(calibration)
    recommended_strategy = recommend_precision_strategy(summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_parquet(output_path, index=False)
    write_report(
        calibration,
        mapping,
        summary,
        recommended_strategy,
        report_path,
    )
    return calibration


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate mapped public report findings against train repositories "
            "without reading test labels."
        )
    )
    parser.add_argument("--train-zip", type=Path, default=DEFAULT_TRAIN_ZIP_PATH)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV_PATH)
    parser.add_argument(
        "--external-findings",
        type=Path,
        default=DEFAULT_EXTERNAL_FINDINGS_PATH,
    )
    parser.add_argument(
        "--high-confidence-pairs",
        type=Path,
        default=DEFAULT_HIGH_CONFIDENCE_PAIRS_PATH,
    )
    parser.add_argument(
        "--official-matches",
        type=Path,
        default=DEFAULT_OFFICIAL_MATCHES_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    calibration = run(
        train_zip_path=args.train_zip,
        train_csv_path=args.train_csv,
        external_findings_path=args.external_findings,
        high_confidence_pairs_path=args.high_confidence_pairs,
        official_matches_path=args.official_matches,
        output_path=args.output,
        report_path=args.report,
    )
    print(f"Calibration rows: {len(calibration)}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
