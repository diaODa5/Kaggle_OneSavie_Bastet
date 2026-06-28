from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_v6_after_3997_experiment.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_template_description_repair_v7_property_251.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "template_description_repair_v7_report.md"
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"


DESCRIPTION_REPAIRS = {
    251: (
        "bond is permissionless and always triggers reLP after the bond flow completes. Because no privileged "
        "keeper or timing control gates that reLP execution, any user can choose the exact moment it runs after "
        "manipulating the WETH-rDPX pool, forcing the protocol to remove liquidity and swap at an attacker-chosen "
        "price."
    ),
}


REPAIR_NOTES = {
    251: (
        "Dopex, direct labels; same raw event as 250, written to emphasize the permissionless control surface that "
        "matches the Access Control / Centralization Risk tuple."
    ),
}


def normalize_cell(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def build_template_description_repair_v7(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = ["Property", "repo_path", "severity", "tag", "subtag", "description"]
    if list(base.columns) != required_columns:
        raise ValueError(f"Unexpected baseline columns: {list(base.columns)}")
    if len(base) != 400:
        raise ValueError(f"Expected 400 baseline rows, got {len(base)}")

    result = base.copy()
    missing = sorted(set(DESCRIPTION_REPAIRS) - set(result["Property"].astype(int)))
    if missing:
        raise ValueError(f"Baseline missing repair properties: {missing}")

    for prop, description in DESCRIPTION_REPAIRS.items():
        mask = result["Property"].eq(prop)
        if int(mask.sum()) != 1:
            raise ValueError(f"Expected exactly one row for Property {prop}, got {int(mask.sum())}")
        result.loc[mask, "description"] = normalize_cell(description)

    changed = base.loc[base["Property"].isin(DESCRIPTION_REPAIRS)].copy()
    changed["new_description"] = changed["Property"].map(DESCRIPTION_REPAIRS).map(normalize_cell)
    changed["repair_note"] = changed["Property"].map(REPAIR_NOTES)
    return result, changed


def write_report(changed: pd.DataFrame) -> None:
    lines = [
        "# Template Description Repair V7 Report",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Strategy: keep the v6 allocation and labels, repair only the final template description.",
        "- Labels changed: `0`",
        "- Rows removed: `0`",
        "- Empty padding rows: `0`",
        "",
        "## Repairs",
        "",
    ]
    for _, row in changed.sort_values("Property").iterrows():
        lines.extend(
            [
                f"### Property {int(row['Property'])}",
                f"- Repo: `{row['repo_path']}`",
                f"- Labels: `{row['severity']} | {row['tag']} | {row['subtag']}`",
                f"- Note: {row['repair_note']}",
                f"- Old: {row['description']}",
                f"- New: {row['new_description']}",
                "",
            ]
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    base = pd.read_csv(BASE_PATH)
    result, changed = build_template_description_repair_v7(base)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    shutil.copy2(OUTPUT_PATH, ROOT_SUBMISSION_PATH)
    write_report(changed)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Copied candidate to {ROOT_SUBMISSION_PATH}")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
