from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_3825_no_padding_targeted_v2.csv"
SAMPLE_PATH = ROOT / "data" / "raw_kaggle" / "submission_example.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_exact_public_description_ab4.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "exact_public_description_ab4_report.md"


DESCRIPTION_UPDATES = {
    49: (
        "Giant pools can be drained due to a weak vault authenticity check. "
        "batchDepositETHForStaking validates only the supplied vault's liquid staking manager before sending funds, "
        "so an attacker can provide a malicious vault that passes the check and withdraw ETH staked in "
        "GiantSavETHVaultPool or GiantMevAndFeesPool."
    ),
    129: (
        "Chainlink's latestRoundData may return a stale or incorrect result. "
        "The implementation uses the reported answer without validating updatedAt and answeredInRound, allowing an "
        "outdated price to be used for protocol accounting when the feed stops updating."
    ),
    216: (
        "Users can obtain immediate profit by depositing and redeeming in PerpetualAtlanticVaultLP. "
        "deposit calls previewDeposit before updateFunding, so shares are calculated from stale vault accounting and "
        "can be redeemed in the same block for more assets than the user supplied."
    ),
    222: (
        "VaultController.verifyCreatorOrOwner does not work as intended. "
        "The modifier requires msg.sender to be both the vault creator and owner instead of accepting either role, "
        "blocking legitimate creators and owners from accessing the protected vault functions."
    ),
}


def clean(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def main() -> int:
    base = pd.read_csv(BASE_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    if list(base.columns) != list(sample.columns):
        raise ValueError("Baseline columns do not match submission_example.csv")

    output = base.copy(deep=True)
    changes: list[dict[str, object]] = []
    for property_id, description in DESCRIPTION_UPDATES.items():
        indices = output.index[output["Property"].astype(int).eq(property_id)]
        if len(indices) != 1:
            raise ValueError(f"Property {property_id} was not found exactly once")
        index = indices[0]
        old_description = clean(output.at[index, "description"])
        new_description = clean(description)
        output.at[index, "description"] = new_description
        changes.append(
            {
                "Property": property_id,
                "repo_path": output.at[index, "repo_path"],
                "old_description": old_description,
                "new_description": new_description,
            }
        )

    fixed_columns = ["Property", "repo_path", "severity", "tag", "subtag"]
    if not output[fixed_columns].equals(base[fixed_columns]):
        raise ValueError("Exact-description A/B changed rows, repositories, or labels")
    if int((output["description"] != base["description"]).sum()) != len(DESCRIPTION_UPDATES):
        raise ValueError("Unexpected number of description changes")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    shutil.copy2(OUTPUT_PATH, ROOT / "submission.csv")

    lines = [
        "# Exact Public Description A/B 4",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Preserved: 400 rows, repo allocation, severity, tag, subtag.",
        "- Changed: four descriptions whose public final-report titles exactly match the baseline descriptions.",
        "",
        "## Changes",
    ]
    for change in changes:
        lines.extend(
            [
                f"### Property {change['Property']} / {change['repo_path']}",
                f"- Old: {change['old_description']}",
                f"- New: {change['new_description']}",
                "",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print("Copied candidate to submission.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
