from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_387_template_description_repair_v2.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_template_description_repair_v3_medium_confidence.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "template_description_repair_v3_report.md"
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"


DESCRIPTION_REPAIRS = {
    131: (
        "submitExchangeRateData and submitSDPrice depend on a trusted-node quorum. If trustedNodesCount is reduced "
        "while submissions are in progress, the stored submission count is not adjusted to the new threshold, so "
        "consensus can become unreachable and the oracle update remains permanently stuck."
    ),
    132: (
        "ValidatorWithdrawalVault.settleFunds checks only the balance held in the withdrawal vault when applying a "
        "penalty. Rewards held in the NodeELRewardVault are ignored, so an operator can be treated as underfunded "
        "and penalized even when the validator's total rewards are sufficient."
    ),
    246: (
        "When putOptionsRequired is true, the premium paid during bonding is calculated from block.timestamp and "
        "nextFundingPaymentTimestamp. Users can choose to bond close to the funding update, reducing timeToExpiry "
        "and paying a lower premium for the same amount and market price."
    ),
    253: (
        "VaultLP withdrawal checks rely on totalCollateral, but when all WETH collateral has been locked and "
        "converted into RDPX the live WETH collateral balance becomes zero. Share holders then fail the withdrawal "
        "require checks and cannot claim the RDPX compensation owed to them."
    ),
    258: (
        "VaultLP redeem calculates the returned RDPX from the user's share of vault supply instead of the current "
        "RDPX market value. A user can deposit before a price increase and later redeem against the stale share "
        "accounting to receive more RDPX than intended."
    ),
    259: (
        "The pool add function does not reject an lpToken that is already registered. Two pools can point at the "
        "same LP token, and both use lpToken.balanceOf(this) as supply, so the same balance is counted twice and "
        "reward allocation across pools becomes incorrect."
    ),
    261: (
        "The lock extension check compares unlockInWeeks and unlockTime values that are both expressed in seconds "
        "as if their difference were measured in weeks. The condition can pass after a new week begins instead of "
        "after two weeks, allowing users to extend locks earlier than documented."
    ),
    274: (
        "distributeRewards has no access control and can be called before settleFunds. An attacker can front-run "
        "settlement, distribute the operator share first, and leave operatorShare below penaltyAmount, causing the "
        "validator or operator accounting to absorb a loss."
    ),
    276: (
        "Penalty recovery caps the slashable amount at min(operatorSD, poolThreshold.minThreshold) instead of the "
        "actual penalty exposure. When the penalty is larger than that threshold, most of the penalty cannot be "
        "recovered and the intended slashing mechanism is bypassed."
    ),
    281: (
        "The validator fund distribution logic advances the pool index even when the current funds are insufficient "
        "for that pool. A malicious user can send a tiny amount to influence the next allocation step and redirect "
        "the following validator funding round to a preferred pool."
    ),
    291: (
        "Active auctions can still receive addBid calls immediately before the protocol pause takes effect. A MEV "
        "bot can front-run the pause by bidding across multiple auctions, preventing later competitive bids and "
        "locking near-ending auctions into unfavorable outcomes."
    ),
    296: (
        "The auction endBlock is fixed and publicly known, which gives bidders little reason to reveal demand "
        "before the final block. Rational bidders can wait for last-block or MEV inclusion, reducing price "
        "discovery and leading to a lower final auction price."
    ),
}


REPAIR_NOTES = {
    131: "Stader M-05, direct labels; trusted-node quorum can become unreachable after count changes.",
    132: "Stader M-12, direct labels; penalty settlement ignores rewards held outside the withdrawal vault.",
    246: "Dopex M-12, direct labels; user timing changes premium through timeToExpiry.",
    253: "Dopex M-13, direct labels; zero WETH collateral balance blocks VaultLP compensation withdrawals.",
    258: "Dopex M-02, direct labels; redeem uses share accounting instead of live RDPX market value.",
    259: "Aura M-22, direct labels; duplicate lpToken corrupts reward accounting.",
    261: "Aura M-19, direct labels; seconds are compared as weeks in lock extension logic.",
    274: "Stader M-11, direct labels; unrestricted reward distribution can front-run penalty settlement.",
    276: "Stader M-06, direct labels; penalty recovery cap is too low and bypasses slashing.",
    281: "Stader M-09, direct labels; fund distribution advances index despite insufficient funds.",
    291: "Stader M-07, direct labels; pause can be front-run with bids across active auctions.",
    296: "Stader M-13, nearest tag but clear source; fixed endBlock weakens auction price discovery.",
}


def normalize_cell(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def build_template_description_repair_v3(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "# Template Description Repair V3 Report",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Strategy: keep the 387 baseline allocation and labels, repair 12 medium-confidence template descriptions.",
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
    result, changed = build_template_description_repair_v3(base)

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
