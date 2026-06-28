from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_3959_template_description_repair_v4.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_template_description_repair_v5_remaining_clear.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "template_description_repair_v5_report.md"
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"


DESCRIPTION_REPAIRS = {
    79: (
        "The upgradeable contract is deployed without a constructor or initializer that sets its initial state. "
        "Because initialization is missing, required ownership, configuration, or dependency values can remain at "
        "their defaults, leaving the proxy or implementation in an unsafe uninitialized state."
    ),
    256: (
        "claimRewards can transfer CVX from the user whenever depositCvxMaxAmount is positive even when the user "
        "did not choose Options.LockCvx. Because the lock path is skipped, the transferred CVX remains in the "
        "contract with no refund or withdrawal path for the user."
    ),
    265: (
        "Reward calculations round down and do not adjust accounting when the contract lacks enough reward tokens. "
        "safeRewardTransfer can send less than the amount recorded as paid, so users receive fewer rewards than "
        "owed while the accounting still treats the transfer as successful."
    ),
    271: (
        "The reLP flow removes WETH-rDPX liquidity, swaps part of the WETH to RDPX, and then adds liquidity again "
        "using imprecise calculations. Some executions can revert or leave WETH dust behind, which accumulates and "
        "distorts the value of the protocol's LP position."
    ),
    277: (
        "Users who lock AURA and receive vlAURA do not automatically receive voting power unless they explicitly "
        "set a delegate. The system records voting weight only for delegated balances, so users without a delegate "
        "can hold vlAURA but still have no effective governance power."
    ),
    278: (
        "Adding rewards to a past epoch can be front-run by adding a dust reward to a later epoch first. The later "
        "epoch update changes the expected ordering and causes the legitimate past-epoch reward transaction to "
        "hit a require condition and revert."
    ),
    280: (
        "The reward claim path transfers the reward token before updating the user's accounting state. If the "
        "reward token has a transfer hook or is later upgraded to ERC777-like behavior, the recipient can reenter "
        "before state is finalized and claim against stale balances."
    ),
    282: (
        "When all of a user's locks have expired, kick reward calculation uses only the amount from the final lock "
        "multiplied by expired epochs. It should account for the user's total expired locked amount, so the kicker "
        "can receive less reward than the protocol formula intends."
    ),
    285: (
        "The ERC721 transfer path does not update the stored owner field after a token is moved. Owner is set only "
        "during minting and then remains stale, so later logic or external reads can report the wrong owner for "
        "tokens that have already been transferred."
    ),
    286: (
        "sync updates the protocol's internal asset balances, but it is not called on every path that changes those "
        "balances. Internal accounting can fall out of sync with live token balances, causing later calculations "
        "or external calls to rely on stale reserve values."
    ),
    287: (
        "Reward tokens are swapped directly to WETH without checking whether the direct pair has enough liquidity "
        "or whether another route would provide better output. With low liquidity on the direct path, the protocol "
        "can receive far less WETH than expected."
    ),
    289: (
        "Total voting power is derived from all locked AURA, but individual user voting weights are recorded only "
        "for delegated balances. This mismatch makes aggregate and per-user governance power inconsistent, which "
        "can distort quorum or proposal threshold calculations."
    ),
}


REPAIR_NOTES = {
    79: "Direct labels; missing initialization in an upgradeable deployment pattern.",
    256: "Aura, clear raw cause; CVX transfer can occur without the selected lock path.",
    265: "Aura direct labels; reward underpayment is hidden by accounting that marks transfer successful.",
    271: "Dopex direct labels; imprecise Uniswap reLP leaves dust or reverts.",
    277: "Aura, nearest tag but clear governance root cause; vlAURA has no vote weight without delegate.",
    278: "Aura, nearest subtag but clear front-run DoS root cause around past epoch rewards.",
    280: "Aura direct labels; reward token transfer precedes user state update.",
    282: "Aura, nearest tag but clear kick reward parameter error.",
    285: "Dopex direct labels; ERC721 owner state is not updated after transfer.",
    286: "Dopex direct labels; sync is not called on all balance-changing paths.",
    287: "JPEG'd direct labels; direct reward-token swap can suffer from low liquidity.",
    289: "Aura, nearest tag but clear governance accounting mismatch.",
}


def normalize_cell(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def build_template_description_repair_v5(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "# Template Description Repair V5 Report",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Strategy: keep the 395.9 baseline allocation and labels, repair 12 remaining clear template descriptions.",
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
    result, changed = build_template_description_repair_v5(base)

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
