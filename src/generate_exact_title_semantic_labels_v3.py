from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from generate_manual_label_realignment_v1 import (
        load_public_findings,
        normalize_issue,
        text,
        validate_candidate,
    )
except ImportError:
    from src.generate_manual_label_realignment_v1 import (
        load_public_findings,
        normalize_issue,
        text,
        validate_candidate,
    )


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_manual_label_realignment_v2_bold.csv"
TRAIN_PATH = ROOT / "data" / "raw_kaggle" / "train.csv"
SAMPLE_PATH = ROOT / "data" / "raw_kaggle" / "submission_example.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_exact_title_semantic_labels_v3.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "exact_title_semantic_labels_v3_report.md"
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"
OUTPUT_SUBMISSION_PATH = ROOT / "outputs" / "submission.csv"


# Curated after checking exact-title containment, mapping provenance, and semantic
# agreement between each title and its mapped tag/subtag.
LABEL_UPDATES: dict[int, tuple[str, str, str]] = {
    1: ("03196f805abb", "2021-09-sushimiso", "H-01"),
    14: ("4ef0b4d678ab", "2022-12-caviar", "H-01"),
    44: ("0315ba9d8121", "2024-07-benddao", "H-04"),
    45: ("0315ba9d8121", "2024-07-benddao", "H-05"),
    46: ("0315ba9d8121", "2024-07-benddao", "H-06"),
    67: ("4ef0b4d678ab", "2022-12-caviar", "H-02"),
    69: ("4ef0b4d678ab", "2022-12-caviar", "M-01"),
    105: ("af8e4af75dc2", "2022-10-inverse", "M-07"),
    134: ("dd17040a53bc", "2021-12-amun", "M-06"),
    147: ("e6e43dfea59f", "2022-04-badger-citadel", "M-02"),
    154: ("98b7c1691811", "2022-01-xdefi", "H-01"),
    176: ("03196f805abb", "2021-09-sushimiso", "M-01"),
    177: ("7100a75d6ebd", "2022-07-juicebox", "M-03"),
    185: ("7100a75d6ebd", "2022-07-juicebox", "M-09"),
    232: ("e169bbefc6e2", "2022-04-jpegd", "H-09"),
    237: ("0315ba9d8121", "2024-07-benddao", "M-01"),
    243: ("e169bbefc6e2", "2022-04-jpegd", "M-06"),
    316: ("0315ba9d8121", "2024-07-benddao", "M-17"),
}


ALLOWED_MAPPING_METHODS = {"external_exact", "external_exact_merged"}


def normalize_match_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text(value).lower()).strip()


def resolve_finding(
    public: pd.DataFrame,
    repo_path: str,
    contest: str,
    issue_id: str,
) -> pd.Series:
    matches = public[
        public["_repo_path"].eq(repo_path)
        & public["_contest"].eq(contest)
        & public["_issue_norm"].eq(normalize_issue(issue_id))
        & public["labels_complete"].astype(bool)
        & public["confidence"].map(lambda value: text(value).lower()).eq("exact")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one exact public finding for {repo_path} {contest} {issue_id}, got {len(matches)}"
        )
    finding = matches.iloc[0]
    if bool(finding.get("mapping_ambiguity", False)):
        raise ValueError(f"Ambiguous label mapping for {repo_path} {contest} {issue_id}")
    if text(finding.get("mapping_method")) not in ALLOWED_MAPPING_METHODS:
        raise ValueError(
            f"Unsupported mapping method for {repo_path} {contest} {issue_id}: "
            f"{finding.get('mapping_method')}"
        )
    return finding


def build_candidate() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    base = pd.read_csv(BASE_PATH)
    train = pd.read_csv(TRAIN_PATH)
    public = load_public_findings(train)
    sample_columns = pd.read_csv(SAMPLE_PATH, nrows=0).columns.tolist()
    candidate = base[sample_columns].copy()
    changes: list[dict[str, Any]] = []

    for prop, finding_key in LABEL_UPDATES.items():
        mask = candidate["Property"].astype(int).eq(prop)
        if int(mask.sum()) != 1:
            raise ValueError(f"Property {prop} not found exactly once")
        index = candidate.index[mask][0]
        old = candidate.loc[index].to_dict()
        finding = resolve_finding(public, *finding_key)

        if text(old["repo_path"]) != text(finding["repo_path"]):
            raise ValueError(
                f"Property {prop} repo mismatch: {old['repo_path']} vs {finding['repo_path']}"
            )
        if text(old["severity"]) != text(finding["severity"]):
            raise ValueError(
                f"Property {prop} severity would change: {old['severity']} vs {finding['severity']}"
            )

        normalized_title = normalize_match_text(finding["title"])
        normalized_description = normalize_match_text(old["description"])
        title_contained = bool(normalized_title and normalized_title in normalized_description)
        if not title_contained:
            raise ValueError(f"Property {prop} does not contain the exact public title")

        candidate.at[index, "tag"] = text(finding["tag"])
        candidate.at[index, "subtag"] = text(finding["subtag"])
        changes.append(
            {
                "Property": prop,
                "finding_key": finding_key,
                "repo_path": text(finding["repo_path"]),
                "issue": text(finding["issue_id"]),
                "title": text(finding["title"]),
                "confidence": text(finding["confidence"]).lower(),
                "mapping_method": text(finding["mapping_method"]),
                "mapping_ambiguity": bool(finding.get("mapping_ambiguity", False)),
                "title_contained": title_contained,
                "old_tag": text(old["tag"]),
                "new_tag": text(finding["tag"]),
                "old_subtag": text(old["subtag"]),
                "new_subtag": text(finding["subtag"]),
            }
        )

    for column in ["repo_path", "severity", "tag", "subtag", "description"]:
        candidate[column] = candidate[column].map(text)
    return candidate, changes


def write_report(changes: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    lines = [
        "# Exact-Title Semantic Labels V3 Report",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Leaderboard baseline: `402.13`",
        "- Strategy: freeze row allocation and descriptions; update only semantically corroborated exact-title tags/subtags.",
        f"- Label updates: `{len(changes)}`",
        "- Repository count changes: `0`",
        "- Description changes: `0`",
        "",
        "## Changes",
        "",
    ]
    for change in changes:
        lines.extend(
            [
                f"### Property {change['Property']}",
                f"- Repo: `{change['repo_path']}`",
                f"- Issue: `{change['issue']}`",
                f"- Title: {change['title']}",
                f"- Mapping: `{change['mapping_method']}`, confidence `{change['confidence']}`",
                f"- Tag: `{change['old_tag']}` -> `{change['new_tag']}`",
                f"- Subtag: `{change['old_subtag']}` -> `{change['new_subtag']}`",
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
        ]
    )
    if validation["errors"]:
        lines.extend(["", "### Errors", *[f"- {error}" for error in validation["errors"]]])
    if validation["warnings"]:
        lines.extend(["", "### Warnings", *[f"- {warning}" for warning in validation["warnings"]]])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    candidate, changes = build_candidate()
    validation = validate_candidate(candidate)
    write_report(changes, validation)
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
