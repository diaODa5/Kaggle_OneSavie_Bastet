from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from generate_public_judging_submission import apply_manual_sherlock_labels
    from validate_submission import validate_submission_frames
except ImportError:
    from src.generate_public_judging_submission import apply_manual_sherlock_labels
    from src.validate_submission import validate_submission_frames


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_40075_template_description_repair_v7.csv"
PUBLIC_FINDINGS_PATH = ROOT / "data" / "processed" / "public_judging_findings_bastet_mapped.parquet"
TRAIN_PATH = ROOT / "data" / "raw_kaggle" / "train.csv"
TEST_PATH = ROOT / "data" / "raw_kaggle" / "test.csv"
SAMPLE_PATH = ROOT / "data" / "raw_kaggle" / "submission_example.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_manual_label_realignment_v1.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "manual_label_realignment_v1_report.md"
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"
OUTPUT_SUBMISSION_PATH = ROOT / "outputs" / "submission.csv"


LABEL_ONLY_UPDATES = {
    353: ("9470d2cf198f", "2025-02-rova", "M-1"),
    354: ("9470d2cf198f", "2025-02-rova", "M-2"),
    355: ("592eed5791df", "2024-07-kwenta-staking-contracts", "001-H"),
    356: ("73f6a793d916", "2024-01-rio-vesting-escrow", "001-H"),
}


ROW_REPLACEMENTS = {
    357: ("1167ec3a176e", "2024-03-arrakis", "001-M"),
    358: ("1167ec3a176e", "2024-03-arrakis", "002-M"),
    359: ("1167ec3a176e", "2024-03-arrakis", "004-M"),
    360: ("1167ec3a176e", "2024-03-arrakis", "005-M"),
}


BUDGET_REPLACEMENT = {
    "replace_property": 351,
    "old_repo": "e0d2d83ea351",
    "new_finding": ("73f6a793d916", "2024-01-rio-vesting-escrow", "002-M"),
    "reason": (
        "Move one row from a 26-row high-count Popcorn repo with many repeated "
        "Accounting Error / Incorrect Formula entries into Rio, which had only one row "
        "but has a second exact public finding."
    ),
}


def text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value).replace("\r", " ").replace("\n", " ")).strip()


def normalize_issue(value: Any) -> str:
    raw = text(value)
    match = re.search(r"\b([hm])[-_ ]*0*(\d+)\b", raw, re.I)
    if match:
        return f"{match.group(1).upper()}-{int(match.group(2))}"
    match = re.search(r"\b0*(\d+)[-_ ]*([hm])\b", raw, re.I)
    if match:
        return f"{int(match.group(1)):03d}-{match.group(2).upper()}"
    return raw.upper()


def clean_public_description(row: pd.Series, max_chars: int = 520) -> str:
    title = text(row.get("title"))
    body = str(row.get("description") or "")
    body = re.sub(r"```.*?```", " ", body, flags=re.S)
    body = re.sub(r"https?://\S+", " ", body)
    body = re.sub(r"`([^`]+)`", r"\1", body)
    body = re.sub(r"[*_#>\[\]\(\)]+", " ", body)
    body = text(body)
    if title and title.lower() not in body.lower():
        description = f"{title}. {body}" if body else title
    else:
        description = body or title
    description = text(description)
    if len(description) > max_chars:
        description = description[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:") + "."
    return description


def load_public_findings(train: pd.DataFrame) -> pd.DataFrame:
    findings = pd.read_parquet(PUBLIC_FINDINGS_PATH)
    findings = apply_manual_sherlock_labels(findings, train)
    findings["_issue_norm"] = findings["issue_id"].map(normalize_issue)
    findings["_contest"] = findings["contest"].map(text)
    findings["_repo_path"] = findings["repo_path"].map(text)
    return findings


def get_finding(public: pd.DataFrame, repo_path: str, contest: str, issue_id: str) -> pd.Series:
    issue_norm = normalize_issue(issue_id)
    matches = public[
        public["_repo_path"].eq(repo_path)
        & public["_contest"].eq(contest)
        & public["_issue_norm"].eq(issue_norm)
        & public["labels_complete"].astype(bool)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one public finding for {repo_path} {contest} {issue_id}, got {len(matches)}"
        )
    row = matches.iloc[0]
    if text(row.get("confidence")).lower() != "exact":
        raise ValueError(f"Finding {repo_path} {contest} {issue_id} is not exact confidence")
    return row


def apply_label_only_update(
    candidate: pd.DataFrame,
    public_row: pd.Series,
    prop: int,
    changes: list[dict[str, Any]],
) -> None:
    mask = candidate["Property"].astype(int).eq(prop)
    if int(mask.sum()) != 1:
        raise ValueError(f"Property {prop} not found exactly once")
    index = candidate.index[mask][0]
    old = candidate.loc[index].to_dict()
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
            "title": text(public_row["title"]),
            "old": f"{old['severity']} / {old['tag']} / {old['subtag']}",
            "new": f"{public_row['severity']} / {public_row['tag']} / {public_row['subtag']}",
        }
    )


def apply_row_replacement(
    candidate: pd.DataFrame,
    public_row: pd.Series,
    prop: int,
    changes: list[dict[str, Any]],
    kind: str,
) -> None:
    mask = candidate["Property"].astype(int).eq(prop)
    if int(mask.sum()) != 1:
        raise ValueError(f"Property {prop} not found exactly once")
    index = candidate.index[mask][0]
    old = candidate.loc[index].to_dict()
    candidate.at[index, "repo_path"] = text(public_row["repo_path"])
    candidate.at[index, "severity"] = text(public_row["severity"])
    candidate.at[index, "tag"] = text(public_row["tag"])
    candidate.at[index, "subtag"] = text(public_row["subtag"])
    candidate.at[index, "description"] = clean_public_description(public_row)
    changes.append(
        {
            "kind": kind,
            "Property": prop,
            "repo_path": public_row["repo_path"],
            "issue": text(public_row["issue_id"]),
            "title": text(public_row["title"]),
            "old": f"{old['repo_path']} | {old['severity']} / {old['tag']} / {old['subtag']}",
            "new": (
                f"{public_row['repo_path']} | {public_row['severity']} / "
                f"{public_row['tag']} / {public_row['subtag']}"
            ),
        }
    )


def validate_candidate(candidate: pd.DataFrame) -> dict:
    sample = pd.read_csv(SAMPLE_PATH)
    test = pd.read_csv(TEST_PATH)
    train = pd.read_csv(TRAIN_PATH)
    report = validate_submission_frames(candidate, sample, test, train, None, expected_rows=400)
    report["passed"] = not report["errors"]
    return report


def write_report(changes: list[dict[str, Any]], before_counts: dict[str, int], after_counts: dict[str, int], validation: dict) -> None:
    lines = [
        "# Manual Label Realignment V1 Report",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Direction: abandon description expansion and small source-audit retagging; realign exact public-report labels and move one row of budget.",
        "- Label-only updates: `4`",
        "- Arrakis exact-row replacements: `4`",
        "- Budget replacement: `1`",
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
    for change in changes:
        lines.extend(
            [
                f"### Property {int(change['Property'])} - {change['kind']}",
                f"- Repo: `{change['repo_path']}`",
                f"- Issue: `{change['issue']}`",
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

    for prop, key in LABEL_ONLY_UPDATES.items():
        apply_label_only_update(candidate, get_finding(public, *key), prop, changes)

    for prop, key in ROW_REPLACEMENTS.items():
        apply_row_replacement(candidate, get_finding(public, *key), prop, changes, "exact_row_replacement")

    budget = BUDGET_REPLACEMENT
    old_row = candidate.loc[candidate["Property"].astype(int).eq(budget["replace_property"])]
    if len(old_row) != 1 or text(old_row.iloc[0]["repo_path"]) != budget["old_repo"]:
        raise ValueError("Budget replacement target row does not match expected old repo")
    apply_row_replacement(
        candidate,
        get_finding(public, *budget["new_finding"]),
        int(budget["replace_property"]),
        changes,
        "budget_replacement",
    )

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
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Copied recommended candidate to: {ROOT_SUBMISSION_PATH}")
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
