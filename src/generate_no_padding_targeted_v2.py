from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_3818_super_aggressive.csv"
SAMPLE_PATH = ROOT / "data" / "raw_kaggle" / "submission_example.csv"
TRAIN_PATH = ROOT / "data" / "raw_kaggle" / "train.csv"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "outputs" / "reports"


TARGETED_V2_FINDINGS = [
    {
        "repo_path": "103f39b0f29b",
        "severity": "Medium",
        "tag": "Access Control",
        "subtag": "Centralization Risk",
        "description": (
            "RubiconFeeController lets a privileged fee controller set both base and pair-specific fees that are "
            "applied to reactor outputs. If fee bounds are weak, the controller can set excessive fees and capture "
            "most of the swap value from users."
        ),
    },
    {
        "repo_path": "103f39b0f29b",
        "severity": "High",
        "tag": "MEV",
        "subtag": "Front Run",
        "description": (
            "Dutch and exclusive order execution exposes time-decaying prices and filler selection to mempool "
            "timing. A filler can front-run near the decay boundary or exploit stale exclusivity assumptions to "
            "capture value that should remain with the maker."
        ),
    },
    {
        "repo_path": "1167ec3a176e",
        "severity": "Medium",
        "tag": "Accounting Error",
        "subtag": "State Update Inconsistency",
        "description": (
            "Arrakis module changes move liquidity between vault modules while accounting for fees and reserves. "
            "If reserve state is not synchronized before and after setModule, share pricing can diverge from the "
            "actual liquidity held by the vault."
        ),
    },
    {
        "repo_path": "51c6dc5fd57f",
        "severity": "High",
        "tag": "Input Validation",
        "subtag": "Incorrect Parameter",
        "description": (
            "MinterGateway derives minting capacity from collateral, validators, and registrar configuration. "
            "Incorrectly bounded collateral or debt parameters can let a minter inflate approved mint capacity "
            "relative to the backing assets."
        ),
    },
    {
        "repo_path": "51c6dc5fd57f",
        "severity": "Medium",
        "tag": "Governance",
        "subtag": "Bypass Mechanism",
        "description": (
            "Epoch-based voting and delegation can be bypassed if delegation updates are accepted around epoch "
            "boundaries without enforcing the intended snapshot time. This can let voting power move after the "
            "proposal state should already be fixed."
        ),
    },
    {
        "repo_path": "51c6dc5fd57f",
        "severity": "Medium",
        "tag": "EIP712",
        "subtag": "Invalid Validation,Nonce",
        "description": (
            "EIP712 signature helpers must reject expired signatures and consume nonces on every accepted action. "
            "If validation and nonce updates are not coupled, a signature can be reused to repeat delegation or "
            "authorization effects."
        ),
    },
    {
        "repo_path": "9ddd6b83c27e",
        "severity": "High",
        "tag": "Accounting Error",
        "subtag": "Incorrect Formula",
        "description": (
            "DnGmxJuniorVaultManager converts between GLP, USDC debt, and hedge exposure using several price and "
            "precision constants. An incorrect formula in these conversions can overstate vault equity and allow "
            "withdrawals or rebalances that harm remaining users."
        ),
    },
    {
        "repo_path": "9ddd6b83c27e",
        "severity": "Medium",
        "tag": "DoS",
        "subtag": "Bad Condition",
        "description": (
            "BatchingManager processes deposits and withdrawals by round. If round state is not advanced correctly "
            "after partial GLP staking or failed settlement, later users can be stuck in an old round and unable to "
            "complete their vault operation."
        ),
    },
    {
        "repo_path": "9ddd6b83c27e",
        "severity": "Medium",
        "tag": "Slippage",
        "subtag": "Missing minOut / maxAmount",
        "description": (
            "WithdrawPeriphery uses fixed slippage thresholds for conversions involving GLP and underlying assets. "
            "Without per-transaction minOut bounds, a withdrawal can execute through unfavorable GMX pricing and "
            "return less value than the user expected."
        ),
    },
    {
        "repo_path": "e7921851ec01",
        "severity": "High",
        "tag": "Oracle",
        "subtag": "Stale Value",
        "description": (
            "D3Oracle can be configured with Chainlink and sequencer feeds for token valuation. If stale answers or "
            "sequencer downtime are not rejected before liquidation checks, pools can be liquidated or protected "
            "using outdated prices."
        ),
    },
    {
        "repo_path": "e7921851ec01",
        "severity": "High",
        "tag": "Accounting Error",
        "subtag": "Incorrect Parameter",
        "description": (
            "D3VaultLiquidation accepts debtToCover and collateralAmount during public liquidation. If those "
            "parameters are not recomputed from current debt and collateral, a liquidator can claim too much "
            "collateral for too little repayment."
        ),
    },
    {
        "repo_path": "e7921851ec01",
        "severity": "Medium",
        "tag": "DoS, Liquidation",
        "subtag": "Missing Functionality",
        "description": (
            "Pool removal requires liquidation and pending repayment steps before finishPoolRemove can complete. "
            "If deprecated or pending-remove pools are not handled in every liquidation path, repayment can become "
            "stuck and block final pool removal."
        ),
    },
]


PROTECTED_REPOS = {row["repo_path"] for row in TARGETED_V2_FINDINGS}


def normalize(value: str) -> str:
    return " ".join(str(value).split())


def validate_labels(train: pd.DataFrame) -> None:
    for col in ["severity", "tag", "subtag"]:
        legal = {normalize(v) for v in train[col].dropna().astype(str)}
        actual = {normalize(row[col]) for row in TARGETED_V2_FINDINGS}
        bad = sorted(actual - legal)
        if bad:
            raise ValueError(f"Illegal {col} labels: {bad}")


def choose_drop_indices(base: pd.DataFrame, n_drop: int) -> list[int]:
    drop_indices: list[int] = []
    while len(drop_indices) < n_drop:
        active = base.drop(index=drop_indices, errors="ignore")
        counts = active["repo_path"].value_counts()
        candidates = [
            repo
            for repo, count in counts.items()
            if repo not in PROTECTED_REPOS and count > 3
        ]
        if not candidates:
            candidates = [repo for repo, count in counts.items() if repo not in PROTECTED_REPOS and count > 1]
        repo = max(candidates, key=lambda value: counts[value])
        repo_rows = active.index[active["repo_path"] == repo].tolist()
        drop_indices.append(repo_rows[-1])
    return drop_indices


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(BASE_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    train = pd.read_csv(TRAIN_PATH)
    columns = list(sample.columns)
    validate_labels(train)

    drop_indices = choose_drop_indices(base, len(TARGETED_V2_FINDINGS))
    kept = base.drop(index=drop_indices).copy()
    additions = pd.DataFrame(TARGETED_V2_FINDINGS)
    additions.insert(0, "Property", range(1, len(additions) + 1))
    out = pd.concat([kept[columns], additions[columns]], ignore_index=True)
    out["Property"] = range(1, 401)
    for col in ["repo_path", "severity", "tag", "subtag", "description"]:
        out[col] = out[col].astype(str).map(lambda text: " ".join(text.replace("\r", " ").replace("\n", " ").split()))

    if len(out) != 400:
        raise AssertionError(f"Expected 400 rows, got {len(out)}")
    if (out["repo_path"].str.lower() == "empty").any():
        raise AssertionError("No-padding v2 unexpectedly contains empty rows")

    output_path = OUT_DIR / "submission_no_padding_targeted_v2.csv"
    out.to_csv(output_path, index=False)
    shutil.copy2(output_path, ROOT / "submission.csv")

    dropped = base.loc[drop_indices]
    lines = [
        "# No-Padding Targeted V2 Report",
        "",
        f"- Base: `{BASE_PATH}`",
        f"- Output: `{output_path}`",
        "- Strategy: keep 400 non-empty findings, replace rows from overrepresented repos with targeted complex-repo findings.",
        f"- Added findings: `{len(TARGETED_V2_FINDINGS)}`",
        f"- Repos covered: `{out['repo_path'].nunique()}`",
        "",
        "## Added Repo Counts",
    ]
    for repo, count in additions["repo_path"].value_counts().items():
        lines.append(f"- `{repo}`: {count}")
    lines.extend(["", "## Dropped Repo Counts"])
    for repo, count in dropped["repo_path"].value_counts().items():
        lines.append(f"- `{repo}`: {count}")
    (REPORT_DIR / "no_padding_targeted_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {output_path}")
    print("Copied to submission.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
