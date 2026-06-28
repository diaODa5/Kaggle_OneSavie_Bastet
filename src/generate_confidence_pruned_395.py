from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_3825_no_padding_targeted_v2.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_confidence_pruned_395.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "confidence_pruned_395_report.md"
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"

TARGETED_V2_PROPERTIES = frozenset(range(389, 401))

DROP_DECISIONS = {
    140: (
        "Pure template description with no contract, function, or root-cause detail in a 26-finding repo; "
        "the same repo already has seven other High/Accounting Error/Incorrect Formula predictions."
    ),
    144: (
        "Pure template description and a weaker duplicate of property 142 for the same repo, severity, tag, "
        "and subtag; property 142 at least includes JPEG withdrawal context."
    ),
    293: (
        "Pure template description with no repository-specific evidence in a 15-finding repo."
    ),
    295: (
        "Pure template description with no repository-specific evidence in a 15-finding repo."
    ),
    321: (
        "Pure template description with no repository-specific evidence; the repo already contains another "
        "Medium/Accounting Error/Incorrect Formula finding with concrete VaultProxy context."
    ),
}


def build_confidence_pruned_submission(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = ["Property", "repo_path", "severity", "tag", "subtag", "description"]
    if list(base.columns) != required_columns:
        raise ValueError(f"Unexpected baseline columns: {list(base.columns)}")
    if len(base) != 400:
        raise ValueError(f"Expected a 400-row baseline, got {len(base)}")
    if set(DROP_DECISIONS) & TARGETED_V2_PROPERTIES:
        raise AssertionError("Drop decisions must not include targeted V2 properties")

    available = set(base["Property"].astype(int))
    missing = sorted(set(DROP_DECISIONS) - available)
    if missing:
        raise ValueError(f"Baseline is missing configured drop properties: {missing}")

    dropped = base[base["Property"].isin(DROP_DECISIONS)].copy()
    kept = base[~base["Property"].isin(DROP_DECISIONS)].copy()
    padding = pd.DataFrame(
        [
            {
                "Property": 0,
                "repo_path": "empty",
                "severity": "empty",
                "tag": "empty",
                "subtag": "empty",
                "description": "empty",
            }
            for _ in DROP_DECISIONS
        ],
        columns=required_columns,
    )
    result = pd.concat([kept, padding], ignore_index=True)
    result["Property"] = range(1, 401)

    if len(result) != 400:
        raise AssertionError(f"Expected 400 output rows, got {len(result)}")
    return result, dropped


def write_report(dropped: pd.DataFrame) -> None:
    lines = [
        "# Confidence-Pruned 395 Report",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Strategy: preserve the known 382.5 baseline except for five explicitly audited low-confidence rows.",
        "- Effective findings: `395`",
        "- Padding rows: `5`",
        "- Protected targeted V2 properties: `389..400`",
        "",
        "## Removed Findings",
        "",
    ]
    for _, row in dropped.sort_values("Property").iterrows():
        prop = int(row["Property"])
        lines.extend(
            [
                f"### Property {prop}",
                f"- Repo: `{row['repo_path']}`",
                f"- Labels: `{row['severity']} | {row['tag']} | {row['subtag']}`",
                f"- Reason: {DROP_DECISIONS[prop]}",
                f"- Description: {row['description']}",
                "",
            ]
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    base = pd.read_csv(BASE_PATH)
    result, dropped = build_confidence_pruned_submission(base)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    shutil.copy2(OUTPUT_PATH, ROOT_SUBMISSION_PATH)
    write_report(dropped)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Copied candidate to {ROOT_SUBMISSION_PATH}")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
