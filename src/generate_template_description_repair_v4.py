from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_392_template_description_repair_v3.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_template_description_repair_v4_medium_confidence.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "template_description_repair_v4_report.md"
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"


DESCRIPTION_REPAIRS = {
    250: (
        "Bonding automatically triggers reLP, and bond is permissionless. An attacker can manipulate the "
        "WETH-rDPX pool price with a flash loan, call bond to force reLP at the manipulated price, and unwind the "
        "trade after the pool normalizes to extract value from the protocol's liquidity operation."
    ),
    254: (
        "The bonding flow gives users a discount when minting dpxETH, but reserve usage is not based on the exact "
        "amount of WETH the user supplied. The protocol always consumes _amount / 2 of reserve WETH for liquidity, "
        "which can overdraw reserves and eventually block later bonding operations."
    ),
    264: (
        "distributeOther lets any caller notifyRewardAmount and reset the reward distribution window. By repeatedly "
        "calling it with a very small amount, an attacker can roll remaining rewards into a fresh seven-day period "
        "and delay normal users from receiving accrued rewards."
    ),
    266: (
        "When minAmount is zero and _ethToDpxEth is true, the slippage calculation uses getDpxEthPrice to derive "
        "the ETH-to-dpxETH minimum output. It should use the ETH price instead, so the computed minOut is wrong and "
        "the intended slippage protection does not reflect the actual swap direction."
    ),
    267: (
        "The add and set pool functions allow _withUpdate to skip massUpdatePools while still changing "
        "totalAllocPoint. Existing pools keep stale reward accounting after allocation weights change, so rewards "
        "are distributed using inconsistent pool proportions."
    ),
    269: (
        "_getReward iterates from a user's epochIndex through tokenEpochs. If a user waits through many epochs "
        "before claiming, the loop can grow beyond the block gas limit, causing reward claims to revert and leaving "
        "the user's accrued rewards permanently inaccessible."
    ),
    270: (
        "If the staking pool's lpToken is also the JPEG reward token, newly minted reward tokens increase "
        "lpToken.balanceOf(this). The contract uses that balance as lpSupply, so reward minting inflates the "
        "denominator and dilutes rewards owed to actual stakers."
    ),
    272: (
        "The pool add path does not verify that _lpToken differs from the CVX reward token. If CVX itself is added "
        "as an LP token, newly minted CVX rewards are counted as pool supply, inflating lpSupply and diluting the "
        "rewards owed to stakers."
    ),
    273: (
        "The liquidity request validation requires each live token balance to equal _request.maxAmountsIn[i]. An "
        "attacker can front-run the transaction by sending one wei of a token to the contract, breaking the exact "
        "equality check and forcing the liquidity provider's transaction to revert."
    ),
    275: (
        "massUpdatePools is public and loops over the full poolInfo array without a bound. As the number of pools "
        "grows, the call can exceed the block gas limit, preventing some or all pools from updating reward "
        "accounting through that path."
    ),
    279: (
        "The sponsorship flow does not require block.timestamp to be at or before endTime. Funds can still be "
        "sponsored after the reward unlock period has already ended, leaving the deposited funds unable to convert "
        "into any effective reward distribution."
    ),
    290: (
        "The redeem path does not check the protocol pause state and lacks a whenNotPaused guard. Even after the "
        "system is paused, users can still call redeem and change balances through a path that should be disabled "
        "during emergency pause handling."
    ),
}


REPAIR_NOTES = {
    250: "Dopex M-03-style reLP manipulation; tag is nearest-mapped but source root cause is specific.",
    254: "Dopex direct labels; reserve WETH consumption is decoupled from user-supplied WETH.",
    264: "Aura direct labels; permissionless notifyRewardAmount can stretch reward distribution.",
    266: "Dopex direct labels; wrong oracle price is used for ETH-to-dpxETH slippage.",
    267: "Aura direct labels; allocation point changes can skip pool reward updates.",
    269: "Aura direct labels; unbounded reward epoch loop can make claims uncallable.",
    270: "JPEG'd direct labels; reward token used as lpToken inflates lpSupply.",
    272: "Aura direct labels; CVX reward token can be added as an LP token.",
    273: "Aura direct labels; exact balance equality can be broken by dust front-run.",
    275: "Aura direct labels; public massUpdatePools can exceed gas as pool count grows.",
    279: "Aura direct labels; missing endTime guard accepts ineffective sponsorship funds.",
    290: "Dopex direct labels; redeem bypasses pause state.",
}


def normalize_cell(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def build_template_description_repair_v4(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "# Template Description Repair V4 Report",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Strategy: keep the 392 baseline allocation and labels, repair 12 remaining template descriptions.",
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
    result, changed = build_template_description_repair_v4(base)

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
