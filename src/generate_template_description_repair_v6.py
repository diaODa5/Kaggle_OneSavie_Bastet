from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_3997_template_description_repair_v5.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_template_description_repair_v6_conservative_final.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "template_description_repair_v6_report.md"
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"


DESCRIPTION_REPAIRS = {
    58: (
        "Deviation.isWithinDeviationThreshold can divide by the input value a without checking whether it is zero. "
        "Because this internal helper is used by public-facing logic, a zero input can trigger a division-by-zero "
        "revert and block deviation checks that should otherwise complete safely."
    ),
    257: (
        "When a user chooses to lock rewards, the flow reverts if AuraLocker is shut down. The user is then forced "
        "onto the unlocked withdrawal path and pays the 20 percent penalty even though the inability to lock is "
        "caused by the admin-controlled locker state rather than the user's choice."
    ),
    294: (
        "The minimum token amount used for slippage protection is calculated with an incorrect minTokenAAmount "
        "formula. Because the wrong value is passed into the trade path, the transaction's slippage bound can be "
        "too loose or invalid and no longer protects the user as intended."
    ),
    303: (
        "The owner can disable preventSmartContracts, after which arbitrary smart contracts may interact with the "
        "protocol. The optional allowlist-based mitigation does not prevent same-block flash-loan style activity, "
        "so a contract can still exploit underlying flash-loan-sensitive paths once the guard is disabled."
    ),
}


REPAIR_NOTES = {
    58: "Volt M-03; direct severity, nearest label text, clear division-by-zero root cause in Deviation.sol.",
    257: "Aura; rule-text labels but clear locker-shutdown penalty path.",
    294: "Dopex; short raw source but clear incorrect minTokenAAmount slippage formula.",
    303: "Gro M-04; rule-text labels but clear optional flash-loan mitigation issue.",
}


def normalize_cell(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def build_template_description_repair_v6(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "# Template Description Repair V6 Report",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Strategy: keep the 399.7 baseline allocation and labels, repair four clear remaining template descriptions.",
        "- Labels changed: `0`",
        "- Rows removed: `0`",
        "- Empty padding rows: `0`",
        "- Left unchanged intentionally: `251`, because it duplicates the repaired `250` source description but has a different label tuple.",
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
    result, changed = build_template_description_repair_v6(base)

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
