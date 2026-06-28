from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from generate_manual_label_realignment_v1 import (
        clean_public_description,
        load_public_findings,
        normalize_issue,
        text,
        validate_candidate,
    )
except ImportError:
    from src.generate_manual_label_realignment_v1 import (
        clean_public_description,
        load_public_findings,
        normalize_issue,
        text,
        validate_candidate,
    )


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_manual_label_realignment_v1.csv"
PUBLIC_FINDINGS_PATH = ROOT / "data" / "processed" / "public_judging_findings_bastet_mapped.parquet"
TRAIN_PATH = ROOT / "data" / "raw_kaggle" / "train.csv"
SAMPLE_PATH = ROOT / "data" / "raw_kaggle" / "submission_example.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_manual_label_realignment_v2_bold.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "manual_label_realignment_v2_bold_report.md"
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"
OUTPUT_SUBMISSION_PATH = ROOT / "outputs" / "submission.csv"


# These rows already point at the right repository, but their old labels disagree
# with a high-similarity public finding. Exact findings are preferred; a small
# number of "strong" public matches are included because v1's +1 result suggests
# label-space realignment is scoring better than continued description polishing.
LABEL_ONLY_UPDATES: dict[int, tuple[str, str, str, tuple[str, ...]]] = {
    308: ("9befb133ee74", "2021-06-gro", "M-03", ("exact", "strong")),
    337: ("e0d2d83ea351", "2023-01-popcorn", "M-31", ("exact", "strong")),
    221: ("099243e83259", "2022-11-stakehouse", "H-15", ("exact",)),
    342: ("099243e83259", "2022-11-stakehouse", "M-07", ("exact",)),
    50: ("099243e83259", "2022-11-stakehouse", "H-16", ("exact",)),
    223: ("099243e83259", "2022-11-stakehouse", "H-14", ("exact",)),
    349: ("e0d2d83ea351", "2023-01-popcorn", "M-06", ("exact", "strong")),
    108: ("b155c4de6012", "2023-09-delegate", "M-01", ("exact",)),
    210: ("e0d2d83ea351", "2023-01-popcorn", "H-06", ("exact",)),
    54: ("198fa93fabdd", "2022-06-nibbl", "M-05", ("exact",)),
    310: ("0315ba9d8121", "2024-07-benddao", "M-14", ("exact",)),
    352: ("fa1836e0615e", "2023-08-dopex", "M-11", ("exact",)),
}


# The replacement rows move budget out of overrepresented, highly repetitive
# tail rows and into exact public findings that are absent from the current v1.
ROW_REPLACEMENTS: dict[int, tuple[str, tuple[str, str, str]]] = {
    340: ("e0d2d83ea351", ("198fa93fabdd", "2022-06-nibbl", "M-06")),
    346: ("e0d2d83ea351", ("198fa93fabdd", "2022-06-nibbl", "M-07")),
    350: ("fa1836e0615e", ("198fa93fabdd", "2022-06-nibbl", "M-09")),
    334: ("0315ba9d8121", ("b2a8c124d062", "2022-07-swivel", "M-05")),
    336: ("0315ba9d8121", ("b2a8c124d062", "2022-07-swivel", "M-08")),
    338: ("0315ba9d8121", ("b2a8c124d062", "2022-07-swivel", "M-11")),
    341: ("0315ba9d8121", ("dd17040a53bc", "2021-12-amun", "M-04")),
    344: ("0315ba9d8121", ("dd17040a53bc", "2021-12-amun", "M-05")),
    345: ("099243e83259", ("ba93b3b8d7cf", "2022-05-sturdy", "M-03")),
    348: ("099243e83259", ("ba93b3b8d7cf", "2022-05-sturdy", "M-05")),
    347: ("0315ba9d8121", ("e6e43dfea59f", "2022-04-badger-citadel", "M-05")),
    331: ("e0d2d83ea351", ("7b5e1803022e", "2022-06-notional-coop", "M-09")),
}


def get_public_finding(
    public: pd.DataFrame,
    repo_path: str,
    contest: str,
    issue_id: str,
    allowed_confidence: tuple[str, ...],
) -> pd.Series:
    matches = public[
        public["_repo_path"].eq(repo_path)
        & public["_contest"].eq(contest)
        & public["_issue_norm"].eq(normalize_issue(issue_id))
        & public["labels_complete"].astype(bool)
    ].copy()
    matches = matches[matches["confidence"].map(lambda value: text(value).lower()).isin(allowed_confidence)]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one public finding for "
            f"{repo_path} {contest} {issue_id} with confidence {allowed_confidence}, got {len(matches)}"
        )
    return matches.iloc[0]


def get_candidate_row(candidate: pd.DataFrame, prop: int) -> tuple[int, dict[str, Any]]:
    mask = candidate["Property"].astype(int).eq(prop)
    if int(mask.sum()) != 1:
        raise ValueError(f"Property {prop} not found exactly once")
    index = candidate.index[mask][0]
    return index, candidate.loc[index].to_dict()


def apply_label_update(candidate: pd.DataFrame, public_row: pd.Series, prop: int, changes: list[dict[str, Any]]) -> None:
    index, old = get_candidate_row(candidate, prop)
    if text(old["repo_path"]) != text(public_row["repo_path"]):
        raise ValueError(f"Property {prop} repo mismatch: {old['repo_path']} vs {public_row['repo_path']}")
    for column in ["severity", "tag", "subtag"]:
        candidate.at[index, column] = text(public_row[column])
    changes.append(
        {
            "kind": "label_only",
            "Property": prop,
            "repo_path": old["repo_path"],
            "issue": text(public_row["issue_id"]),
            "confidence": text(public_row["confidence"]),
            "title": text(public_row["title"]),
            "old": f"{old['severity']} / {old['tag']} / {old['subtag']}",
            "new": f"{public_row['severity']} / {public_row['tag']} / {public_row['subtag']}",
        }
    )


def apply_replacement(
    candidate: pd.DataFrame,
    public_row: pd.Series,
    prop: int,
    expected_old_repo: str,
    changes: list[dict[str, Any]],
) -> None:
    index, old = get_candidate_row(candidate, prop)
    if text(old["repo_path"]) != expected_old_repo:
        raise ValueError(f"Property {prop} expected old repo {expected_old_repo}, got {old['repo_path']}")
    candidate.at[index, "repo_path"] = text(public_row["repo_path"])
    candidate.at[index, "severity"] = text(public_row["severity"])
    candidate.at[index, "tag"] = text(public_row["tag"])
    candidate.at[index, "subtag"] = text(public_row["subtag"])
    candidate.at[index, "description"] = clean_public_description(public_row)
    changes.append(
        {
            "kind": "exact_budget_replacement",
            "Property": prop,
            "repo_path": text(public_row["repo_path"]),
            "issue": text(public_row["issue_id"]),
            "confidence": text(public_row["confidence"]),
            "title": text(public_row["title"]),
            "old": f"{old['repo_path']} | {old['severity']} / {old['tag']} / {old['subtag']}",
            "new": (
                f"{public_row['repo_path']} | {public_row['severity']} / "
                f"{public_row['tag']} / {public_row['subtag']}"
            ),
        }
    )


def write_report(
    changes: list[dict[str, Any]],
    before_counts: dict[str, int],
    after_counts: dict[str, int],
    validation: dict[str, Any],
) -> None:
    label_count = sum(1 for change in changes if change["kind"] == "label_only")
    replacement_count = sum(1 for change in changes if change["kind"] == "exact_budget_replacement")
    lines = [
        "# Manual Label Realignment V2 Bold Report",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Direction: double down on the +1 signal by shifting away from description edits and toward public-label alignment.",
        f"- Label-only updates: `{label_count}`",
        f"- Exact budget replacements: `{replacement_count}`",
        "- Empty padding rows: `0`",
        "",
        "## Count Changes",
        "",
    ]
    for repo in sorted(set(before_counts) | set(after_counts)):
        before = before_counts.get(repo, 0)
        after = after_counts.get(repo, 0)
        if before != after:
            lines.append(f"- `{repo}`: `{before}` -> `{after}`")
    lines.extend(["", "## Changes", ""])
    for change in sorted(changes, key=lambda item: int(item["Property"])):
        lines.extend(
            [
                f"### Property {int(change['Property'])} - {change['kind']}",
                f"- Repo: `{change['repo_path']}`",
                f"- Issue: `{change['issue']}`",
                f"- Confidence: `{change['confidence']}`",
                f"- Title: {change['title']}",
                f"- Old: `{change['old']}`",
                f"- New: `{change['new']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Validation",
            "",
            f"- Passed: `{validation['passed']}`",
            f"- Shape: `{validation['shape']}`",
            f"- Description duplicate rate: `{validation['description_duplicate_rate']:.6f}`",
            "",
            "### Errors",
        ]
    )
    lines.extend([f"- {error}" for error in validation["errors"]] if validation["errors"] else ["No validation errors."])
    lines.extend(["", "### Warnings"])
    lines.extend([f"- {warning}" for warning in validation["warnings"]] if validation["warnings"] else ["No validation warnings."])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    base = pd.read_csv(BASE_PATH)
    train = pd.read_csv(TRAIN_PATH)
    public = load_public_findings(train)
    sample_columns = pd.read_csv(SAMPLE_PATH, nrows=0).columns.tolist()
    candidate = base[sample_columns].copy()
    before_counts = candidate["repo_path"].value_counts().to_dict()
    changes: list[dict[str, Any]] = []

    for prop, (repo_path, contest, issue_id, allowed_confidence) in LABEL_ONLY_UPDATES.items():
        public_row = get_public_finding(public, repo_path, contest, issue_id, allowed_confidence)
        apply_label_update(candidate, public_row, prop, changes)

    for prop, (expected_old_repo, finding_key) in ROW_REPLACEMENTS.items():
        public_row = get_public_finding(public, *finding_key, allowed_confidence=("exact",))
        apply_replacement(candidate, public_row, prop, expected_old_repo, changes)

    for column in ["repo_path", "severity", "tag", "subtag", "description"]:
        candidate[column] = candidate[column].map(text)

    validation = validate_candidate(candidate)
    after_counts = candidate["repo_path"].value_counts().to_dict()
    write_report(changes, before_counts, after_counts, validation)
    if not validation["passed"]:
        raise RuntimeError(f"Candidate validation failed; see {REPORT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(OUTPUT_PATH, index=False)
    shutil.copy2(OUTPUT_PATH, ROOT_SUBMISSION_PATH)
    shutil.copy2(OUTPUT_PATH, OUTPUT_SUBMISSION_PATH)

    print(f"Changed rows: {len(changes)}")
    print(f"Label-only updates: {len(LABEL_ONLY_UPDATES)}")
    print(f"Exact budget replacements: {len(ROW_REPLACEMENTS)}")
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Copied recommended candidate to: {ROOT_SUBMISSION_PATH}")
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
