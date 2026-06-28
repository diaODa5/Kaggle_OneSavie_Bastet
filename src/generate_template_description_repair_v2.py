from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_385_template_description_repair_v1.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_template_description_repair_v2_high_confidence.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "template_description_repair_v2_report.md"
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"


DESCRIPTION_REPAIRS = {
    38: (
        "NFTVault lets two users finalize value changes for the same NFT index, and JPEGLock stores only one "
        "lock position per index. A later finalization overwrites the earlier user's JPEG lock schedule, leaving "
        "the first user unable to unlock their JPEG collateral."
    ),
    41: (
        "RdpxV2Core.addToDelegate increases totalWethDelegated, but withdraw does not reduce that value when "
        "delegated WETH is removed. A later sync subtracts stale delegated WETH from reserves, so lowerDepeg can "
        "underflow and fail when the admin tries to defend the peg."
    ),
    141: (
        "The first yVault depositor can mint a tiny initial share and then donate assets directly to the strategy "
        "to inflate share price. Later depositors can receive zero shares for their deposits, allowing the first "
        "depositor to withdraw assets supplied by others."
    ),
    142: (
        "Controller.setStrategy tries to migrate by calling withdraw on the current strategy for the JPEG token. "
        "The PUSDConvex strategy blacklists JPEG withdrawals, so strategy migration reverts and the vault cannot "
        "complete the intended controller update."
    ),
    143: (
        "yVault.deposit caches balanceBefore and then calls token.transferFrom before minting shares. If the "
        "token gives control to the sender, a reentrant deposit can mint shares against the stale balance and "
        "withdraw more backing assets than were deposited."
    ),
    149: (
        "When putOptionsRequired is true, bond purchases pass timeToExpiry directly into option pricing. During "
        "the final 864 seconds before the funding timestamp, timeToExpiry rounds to zero in the pricing library "
        "and every bond operation that purchases puts reverts."
    ),
    150: (
        "RdpxV2Core calculates a 25 percent out-of-the-money put strike and then rounds it with "
        "PerpetualAtlanticVault.roundUp using a hardcoded 1e6 precision. At realistic rDPX prices this rounds "
        "the strike back to the spot price, creating an in-the-money option and excessive premium."
    ),
    151: (
        "PerpetualAtlanticVaultLP.subtractLoss requires the live collateral balance to exactly equal "
        "_totalCollateral minus the loss. Anyone can send a dust amount of collateral to the vault, break that "
        "equality, and make settlement permanently revert without an admin recovery path."
    ),
    152: (
        "Put settlement thresholds are public and LP shares can be redeemed whenever excess collateral exists. "
        "Liquidity providers can exit before predictable settlements, forcing remaining LPs to absorb losses and "
        "draining available collateral so later bonding transactions revert."
    ),
    202: (
        "ReLPContract calculates lpToRemove as if the protocol owns all liquidity in the UniswapV2 pair. When "
        "outside LPs supply liquidity, the computed amount can exceed the AMO's LP token balance, causing reLP "
        "during bonding to revert."
    ),
    203: (
        "RdpxPriceOracle returns rDPX and LP prices in 1e18 precision, while RdpxV2Core, UniV2LiquidityAmo, and "
        "PerpetualAtlanticVault assume 1e8 precision. The mismatch corrupts bond costs, LP valuation, premiums, "
        "and transfer calculations, including possible underflows."
    ),
    204: (
        "UniV3LiquidityAMO.recoverERC721 sends recovered NFTs to RdpxV2Core, but RdpxV2Core only receives ERC721 "
        "tokens and has no function to transfer or approve them out. Any ERC721 recovered this way becomes "
        "permanently locked in the core contract."
    ),
}


REPAIR_NOTES = {
    38: "JPEG'd H-02, direct labels; replaces CJK-derived template with exact lock overwrite root cause.",
    41: "Dopex H-08, high-value state inconsistency; tag was nearest-mapped but subtag and root cause are strong.",
    141: "JPEG'd H-01, direct labels; classic first-depositor share inflation attack.",
    142: "JPEG'd H-07, direct labels; independent strategy migration DoS finding.",
    143: "JPEG'd H-04, direct labels; yVault deposit reentrancy root cause.",
    149: "Dopex H-06; labels include a rule-text subtag, but report root cause is specific and high severity.",
    150: "Dopex H-01, direct labels; strike precision and hardcoded rounding issue.",
    151: "Dopex H-03; subtag nearest-mapped, but settlement DoS root cause is specific.",
    152: "Dopex H-02, direct labels; predictable settlement and LP exit DoS.",
    202: "Dopex H-09; tag nearest-mapped, but incorrect UniswapV2 liquidity ownership assumption is specific.",
    203: "Dopex H-07, direct labels; oracle decimal mismatch root cause.",
    204: "Dopex H-04, direct labels; ERC721 recovery locks NFTs in core contract.",
}


def normalize_cell(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def build_template_description_repair_v2(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "# Template Description Repair V2 Report",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Strategy: keep the 385 baseline allocation and labels, repair 12 high-confidence template descriptions.",
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
    result, changed = build_template_description_repair_v2(base)

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
