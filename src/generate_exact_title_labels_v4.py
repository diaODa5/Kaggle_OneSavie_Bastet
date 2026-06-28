from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from generate_exact_title_semantic_labels_v3 import (
        load_public_findings,
        normalize_match_text,
        resolve_finding,
        text,
        validate_candidate,
    )
except ImportError:
    from src.generate_exact_title_semantic_labels_v3 import (
        load_public_findings,
        normalize_match_text,
        resolve_finding,
        text,
        validate_candidate,
    )


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_exact_title_semantic_labels_v3.csv"
TRAIN_PATH = ROOT / "data" / "raw_kaggle" / "train.csv"
SAMPLE_PATH = ROOT / "data" / "raw_kaggle" / "submission_example.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_exact_title_labels_v4.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "exact_title_labels_v4_report.md"
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"
OUTPUT_SUBMISSION_PATH = ROOT / "outputs" / "submission.csv"


LABEL_UPDATES: dict[int, tuple[str, str, str]] = {
    11: ("1d8f1f43cde9", "2022-03-volt", "H-01"),
    19: ("8a20eb1e521f", "2024-06-badger", "M-01"),
    39: ("e6e43dfea59f", "2022-04-badger-citadel", "H-01"),
    49: ("099243e83259", "2022-11-stakehouse", "H-17"),
    52: ("198fa93fabdd", "2022-06-nibbl", "M-01"),
    110: ("b2a8c124d062", "2022-07-swivel", "M-03"),
    111: ("b2a8c124d062", "2022-07-swivel", "M-04"),
    112: ("b2a8c124d062", "2022-07-swivel", "M-06"),
    181: ("7100a75d6ebd", "2022-07-juicebox", "M-15"),
    186: ("7100a75d6ebd", "2022-07-juicebox", "M-06"),
    208: ("0315ba9d8121", "2024-07-benddao", "H-08"),
    215: ("e0d2d83ea351", "2023-01-popcorn", "H-08"),
    220: ("e0d2d83ea351", "2023-01-popcorn", "H-11"),
    231: ("e169bbefc6e2", "2022-04-jpegd", "H-03"),
    298: ("af8e4af75dc2", "2022-10-inverse", "M-18"),
    312: ("0315ba9d8121", "2024-07-benddao", "M-15"),
    327: ("af8e4af75dc2", "2022-10-inverse", "M-02"),
}


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
        "# Exact-Title Labels V4 Report",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Leaderboard baseline: `402.6`",
        "- Strategy: second exact-title label batch; freeze allocation, severity, and descriptions.",
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
