from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_3825_no_padding_targeted_v2.csv"
SAMPLE_PATH = ROOT / "data" / "raw_kaggle" / "submission_example.csv"
TRAIN_PATH = ROOT / "data" / "raw_kaggle" / "train.csv"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "outputs" / "reports"


TARGETED_V3_FINDINGS = [
    {
        "repo_path": "592eed5791df",
        "severity": "High",
        "tag": "Accounting Error",
        "subtag": "State Update Inconsistency",
        "description": (
            "StakingRewardsV2 can leave newly notified rewards stuck when notifyRewardAmount is called before the "
            "current reward period has finished. The reward-rate update does not account for leftover rewards in a "
            "way that guarantees the full amount becomes claimable by stakers."
        ),
    },
    {
        "repo_path": "592eed5791df",
        "severity": "Medium",
        "tag": "DoS",
        "subtag": "Bad Condition",
        "description": (
            "Reward notification timing can make subsequent staking reward updates revert or distribute zero-value "
            "increments when the remaining period and token decimals interact badly. This can block a valid reward "
            "top-up and delay distribution to users."
        ),
    },
    {
        "repo_path": "9470d2cf198f",
        "severity": "Medium",
        "tag": "Input Validation",
        "subtag": "Missing Upper/Lower Bound Check",
        "description": (
            "Launch participation updates do not enforce consistent lower and upper bounds after currency amount "
            "changes. A user can move a participation into an invalid state relative to balance, allocation, or sale "
            "limits and cause later settlement to behave incorrectly."
        ),
    },
    {
        "repo_path": "9470d2cf198f",
        "severity": "Medium",
        "tag": "Accounting Error",
        "subtag": "Incorrect Formula",
        "description": (
            "Rova participation accounting mixes token-denominated and currency-denominated values during updates. "
            "Using the wrong unit in refund or allocation formulas can overstate user tokens or charge users more "
            "than the intended participation amount."
        ),
    },
    {
        "repo_path": "73f6a793d916",
        "severity": "Medium",
        "tag": "Governance",
        "subtag": "Bypass Mechanism",
        "description": (
            "Vesting escrow delegation relies on clone arguments and adaptor resolution during voting operations. "
            "If the delegatee or adaptor context can be manipulated, a vesting account can bypass the intended "
            "voting-escrow restrictions."
        ),
    },
    {
        "repo_path": "73f6a793d916",
        "severity": "High",
        "tag": "Access Control",
        "subtag": "Asset Theft",
        "description": (
            "Forged immutable arguments in a vesting escrow clone can redirect privileged adaptor calls through an "
            "attacker-controlled address. That can let an attacker execute unauthorized logic against escrowed "
            "voting or token state."
        ),
    },
    {
        "repo_path": "1167ec3a176e",
        "severity": "Medium",
        "tag": "Oracle",
        "subtag": "Price Manipulation / Arbitrage opportunity",
        "description": (
            "Arrakis HOT module deposits depend on pool price ranges and liquidity state. If price bounds are not "
            "passed through consistently, an attacker can manipulate the pool around a deposit and arbitrage the "
            "vault's minted share value."
        ),
    },
    {
        "repo_path": "1167ec3a176e",
        "severity": "Medium",
        "tag": "DoS",
        "subtag": "Missing Functionality",
        "description": (
            "Module migration lacks a robust recovery path when a selected module cannot return or account for all "
            "liquidity. A failed migration can leave liquidity stranded in the module and prevent normal deposits "
            "or withdrawals."
        ),
    },
]


PROTECTED_REPOS = {row["repo_path"] for row in TARGETED_V3_FINDINGS}


def normalize(value: str) -> str:
    return " ".join(str(value).split())


def validate_labels(train: pd.DataFrame) -> None:
    for col in ["severity", "tag", "subtag"]:
        legal = {normalize(v) for v in train[col].dropna().astype(str)}
        actual = {normalize(row[col]) for row in TARGETED_V3_FINDINGS}
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

    drop_indices = choose_drop_indices(base, len(TARGETED_V3_FINDINGS))
    kept = base.drop(index=drop_indices).copy()
    additions = pd.DataFrame(TARGETED_V3_FINDINGS)
    additions.insert(0, "Property", range(1, len(additions) + 1))
    out = pd.concat([kept[columns], additions[columns]], ignore_index=True)
    out["Property"] = range(1, 401)
    for col in ["repo_path", "severity", "tag", "subtag", "description"]:
        out[col] = out[col].astype(str).map(lambda text: " ".join(text.replace("\r", " ").replace("\n", " ").split()))

    if len(out) != 400:
        raise AssertionError(f"Expected 400 rows, got {len(out)}")
    if (out["repo_path"].str.lower() == "empty").any():
        raise AssertionError("No-padding v3 unexpectedly contains empty rows")

    output_path = OUT_DIR / "submission_no_padding_targeted_v3.csv"
    out.to_csv(output_path, index=False)
    shutil.copy2(output_path, ROOT / "submission.csv")

    dropped = base.loc[drop_indices]
    lines = [
        "# No-Padding Targeted V3 Report",
        "",
        f"- Base: `{BASE_PATH}`",
        f"- Output: `{output_path}`",
        "- Strategy: incremental targeted replacement after v2 improved the public score.",
        f"- Added findings: `{len(TARGETED_V3_FINDINGS)}`",
        f"- Repos covered: `{out['repo_path'].nunique()}`",
        "",
        "## Added Repo Counts",
    ]
    for repo, count in additions["repo_path"].value_counts().items():
        lines.append(f"- `{repo}`: {count}")
    lines.extend(["", "## Dropped Repo Counts"])
    for repo, count in dropped["repo_path"].value_counts().items():
        lines.append(f"- `{repo}`: {count}")
    (REPORT_DIR / "no_padding_targeted_v3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {output_path}")
    print("Copied to submission.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
