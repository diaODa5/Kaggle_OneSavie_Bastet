from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_40075_template_description_repair_v7.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_description_quality_repair_v9_conservative.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "description_quality_repair_v9_report.md"
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"


DESCRIPTION_REPAIRS = {
    49: (
        "Giant pools can be drained due to a weak vault authenticity check. batchDepositETHForStaking validates "
        "only the supplied vault's liquid staking manager before sending funds, so a malicious vault can pass the "
        "check and withdraw ETH staked in GiantSavETHVaultPool or GiantMevAndFeesPool."
    ),
    57: (
        "Setting a new buffer cap does not reduce the current buffer to the cap. RateLimited.setBufferCap updates "
        "the buffer before storing the new cap and never clamps the updated value, so the buffer can remain larger "
        "than bufferCap and violate the rate limit invariant."
    ),
    91: (
        "Buoy3Pool uses Chainlink latestAnswer, a deprecated API that can return zero when no valid answer is "
        "available instead of reverting. That behavior can feed an invalid oracle price into the pool and corrupt "
        "price-dependent accounting."
    ),
    177: (
        "JBERC20PaymentTerminal calls ERC20 transfer and transferFrom directly without checking the returned "
        "boolean. Tokens that return false rather than revert can silently fail, while the terminal continues "
        "execution as if the payment transfer succeeded."
    ),
}


REPAIR_NOTES = {
    49: "Conservative exact-public style expansion; same repo and labels.",
    57: "Conservative recovery-source expansion; same repo and labels.",
    91: "Conservative recovery-source expansion; same repo and labels.",
    177: "Conservative recovery-source expansion; same repo and labels.",
}


def normalize_cell(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def build_description_quality_repair_v9(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "# Description Quality Repair V9 Report",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Strategy: recover from failed V8 by testing only four conservative short-title expansions.",
        "- Labels changed: `0`",
        "- Rows removed: `0`",
        "- Empty padding rows: `0`",
        "- Excluded from V8: `61`, `67`, `71`, `123`, `226` due to higher semantic-drift risk.",
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
    result, changed = build_description_quality_repair_v9(base)

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
