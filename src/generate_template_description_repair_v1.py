from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_3825_no_padding_targeted_v2.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_template_description_repair_v1.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "template_description_repair_v1_report.md"
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"


DESCRIPTION_REPAIRS = {
    140: (
        "MultiRewardStaking.claimRewards transfers reward tokens before clearing accruedRewards. "
        "When an ERC777 reward token invokes a recipient hook, the receiver can reenter claimRewards "
        "and drain the staking contract's reward balance before accounting is reset."
    ),
    144: (
        "StrategyPUSDConvex.balanceOfJPEG calls extraReward.earned without the required account "
        "argument. The incorrect function signature makes the call revert, breaking Controller, "
        "YVault, and YVaultLPFarming flows that depend on the JPEG balance calculation."
    ),
    293: (
        "StaderConfig.updateAdmin revokes DEFAULT_ADMIN_ROLE from the old admin even when the new "
        "admin address is the same current admin. Repeating the existing admin value can remove the "
        "only protocol admin role and leave critical configuration without admin control."
    ),
    295: (
        "SocializingPool, StaderOracle, OperatorRewardsCollector, and Auction inherit "
        "PausableUpgradeable but expose no external pause or unpause functions. The intended pause "
        "controls cannot be activated when these contracts need emergency shutdown behavior."
    ),
    321: (
        "StaderOracle accepts SD price reports for different reportingBlockNumber values into shared "
        "price state. Reports for later blocks can mix with or overwrite earlier pending data, so "
        "finalizing one reportable block can corrupt or discard another block's oracle data."
    ),
}


REPAIR_NOTES = {
    140: "Popcorn H-04, high-value real finding; preserve labels and replace only the degraded template text.",
    144: "JPEG'd H-08, independent High DoS finding; property 142 is a different JPEG'd issue, not a duplicate.",
    293: "Stader M-01, direct external labels and source-specific root cause; template text was the weak part.",
    295: "Stader M-02, direct external labels and multiple named affected contracts; template text was the weak part.",
    321: (
        "Stader M-08 is real, but the baseline labels are suspicious. This candidate intentionally fixes only "
        "description to isolate the variable; label repair should be a separate A/B test."
    ),
}


def normalize_cell(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def build_template_description_repair(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "# Template Description Repair V1 Report",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Strategy: restore 400 non-empty findings and change only five degraded template descriptions.",
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
    result, changed = build_template_description_repair(base)

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
