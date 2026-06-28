from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_40075_template_description_repair_v7.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_description_quality_repair_v8_short_titles.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "description_quality_repair_v8_report.md"
ROOT_SUBMISSION_PATH = ROOT / "submission.csv"


DESCRIPTION_REPAIRS = {
    49: (
        "GiantSavETHVaultPool and GiantMevAndFeesPool validate only that a supplied vault points to an accepted "
        "liquid staking manager before sending ETH during batchDepositETHForStaking. A malicious vault can pass "
        "that weak authenticity check and then withdraw ETH that was deposited by Giant pool users."
    ),
    57: (
        "RateLimited.setBufferCap updates the current buffer before lowering the cap, but never clamps the updated "
        "buffer to the new cap. After reducing the cap, the stored buffer can remain above the maximum value and "
        "break the invariant that buffer should never exceed bufferCap."
    ),
    61: (
        "Loans can be created with arbitrary ERC20 tokenContract values, including fee-on-transfer tokens. The "
        "protocol accounts for the requested transfer amount rather than the amount actually received, so transfer "
        "fees can make the recorded balance exceed real funds and drain other users' tokens."
    ),
    67: (
        "When adding liquidity, minLpTokenAmount protects only the number of LP tokens minted and not the pool price "
        "or reserve ratio. The first liquidity provider can set an arbitrary initial price, causing later providers "
        "to deposit at a bad ratio and lose value in one of the pair tokens."
    ),
    71: (
        "Market buy and sell functions calculate share prices from the bonding curve but do not accept user-supplied "
        "minimum output or maximum input bounds. If the curve price changes before execution, users can receive "
        "worse terms than expected with no slippage protection."
    ),
    91: (
        "Buoy3Pool uses Chainlink latestAnswer even though that API is deprecated and can return zero instead of "
        "reverting when no valid answer is available. A zero or stale oracle answer can feed an incorrect price into "
        "the pool rather than stopping the transaction."
    ),
    123: (
        "ForgottenRunesWarriorsMinter.teamSummon lets the owner mint an unrestricted number of NFTs. If the owner "
        "key is compromised or misused during launch, nearly all NFTs can be minted by the owner path, creating a "
        "centralization risk for the collection supply."
    ),
    177: (
        "JBERC20PaymentTerminal calls ERC20 transfer and transferFrom directly and does not check their boolean "
        "return values. Tokens that signal failure by returning false instead of reverting can silently fail while "
        "the terminal continues as if the payment transfer succeeded."
    ),
    226: (
        "The protocol lifecycle state machine can leave deposited ETH permanently frozen. Users who deposited ETH "
        "for staking may be unable to receive funds, rewards, or rotate to another token, making the protocol "
        "insolvent because it cannot pay users back through the expected flow."
    ),
}


REPAIR_NOTES = {
    49: "Same repo and labels in recovery_desc/exact_public; expanded weak vault authenticity root cause.",
    57: "Same repo and labels in recovery_desc; expanded RateLimited cap invariant issue.",
    61: "Same repo and labels in recovery_desc_medium; expanded fee-on-transfer accounting issue.",
    67: "Same repo and labels in recovery_desc_medium; expanded liquidity-ratio loss issue.",
    71: "Same repo and labels in recovery_desc_medium; expanded missing slippage bounds issue.",
    91: "Same repo and labels in recovery_desc; expanded deprecated latestAnswer oracle issue.",
    123: "Same repo and labels in recovery_desc_medium; expanded unrestricted owner minting risk.",
    177: "Same repo and labels in recovery_desc_medium; expanded unchecked ERC20 return value issue.",
    226: "Same repo and labels in recovery_desc_medium; expanded lifecycle insolvency issue.",
}


def normalize_cell(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def build_description_quality_repair_v8(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "# Description Quality Repair V8 Report",
        "",
        f"- Baseline: `{BASE_PATH}`",
        f"- Output: `{OUTPUT_PATH}`",
        "- Strategy: keep the 400.75 baseline allocation and labels, expand nine title-like descriptions.",
        "- Selection rule: same Property, repo, and labels had longer source-backed descriptions in earlier recovery candidates.",
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
    result, changed = build_description_quality_repair_v8(base)

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
