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


MICRO_FINDINGS = [
    {
        "repo_path": "592eed5791df",
        "severity": "High",
        "tag": "Accounting Error",
        "subtag": "State Update Inconsistency",
        "description": (
            "Calling notifyRewardAmount before the previous staking period has ended leaves part of the newly "
            "notified rewards outside the claimable reward-rate schedule. Those rewards remain stuck in "
            "StakingRewardsV2 instead of being distributed to stakers."
        ),
    },
    {
        "repo_path": "9470d2cf198f",
        "severity": "Medium",
        "tag": "Input Validation",
        "subtag": "Missing Upper/Lower Bound Check",
        "description": (
            "Launch.updateParticipation does not re-check all participation bounds after a user changes the "
            "currency amount. A participant can update into an amount that violates balance or sale limits and "
            "break later settlement or refund accounting."
        ),
    },
    {
        "repo_path": "73f6a793d916",
        "severity": "High",
        "tag": "Access Control",
        "subtag": "Asset Theft",
        "description": (
            "Forgeable immutable clone arguments let an attacker control the factory or voting adaptor resolved by "
            "VestingEscrow. The escrow can then delegatecall into attacker-controlled logic and expose escrowed "
            "voting or token state to unauthorized execution."
        ),
    },
    {
        "repo_path": "51c6dc5fd57f",
        "severity": "Medium",
        "tag": "Arithmetic",
        "subtag": "Precision Loss",
        "description": (
            "M^0 continuous-indexing math converts principal and present value through fixed-point division. "
            "Repeated rounding when balances or rate deltas are small can under-credit users and leave value in "
            "the token accounting."
        ),
    },
]


PROTECTED_REPOS = {row["repo_path"] for row in MICRO_FINDINGS}


def normalize(value: str) -> str:
    return " ".join(str(value).split())


def validate_labels(train: pd.DataFrame) -> None:
    for col in ["severity", "tag", "subtag"]:
        legal = {normalize(v) for v in train[col].dropna().astype(str)}
        actual = {normalize(row[col]) for row in MICRO_FINDINGS}
        bad = sorted(actual - legal)
        if bad:
            raise ValueError(f"Illegal {col} labels: {bad}")


def choose_drop_indices(base: pd.DataFrame, n_drop: int) -> list[int]:
    drop_indices: list[int] = []
    while len(drop_indices) < n_drop:
        active = base.drop(index=drop_indices, errors="ignore")
        counts = active["repo_path"].value_counts()
        candidates = [repo for repo, count in counts.items() if repo not in PROTECTED_REPOS and count > 12]
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

    drop_indices = choose_drop_indices(base, len(MICRO_FINDINGS))
    kept = base.drop(index=drop_indices).copy()
    additions = pd.DataFrame(MICRO_FINDINGS)
    additions.insert(0, "Property", range(1, len(additions) + 1))
    out = pd.concat([kept[columns], additions[columns]], ignore_index=True)
    out["Property"] = range(1, 401)
    for col in ["repo_path", "severity", "tag", "subtag", "description"]:
        out[col] = out[col].astype(str).map(lambda text: " ".join(text.replace("\r", " ").replace("\n", " ").split()))

    if len(out) != 400:
        raise AssertionError(f"Expected 400 rows, got {len(out)}")
    if (out["repo_path"].str.lower() == "empty").any():
        raise AssertionError("Micro AB unexpectedly contains empty rows")
    if out.duplicated(subset=["repo_path", "severity", "tag", "subtag", "description"]).any():
        raise AssertionError("Micro AB contains duplicate finding rows")

    output_path = OUT_DIR / "submission_micro_ab_v1.csv"
    out.to_csv(output_path, index=False)
    shutil.copy2(output_path, ROOT / "submission.csv")

    dropped = base.loc[drop_indices]
    lines = [
        "# Micro AB V1 Report",
        "",
        f"- Base: `{BASE_PATH}`",
        f"- Output: `{output_path}`",
        "- Strategy: 4-row no-padding replacement from known-best baseline.",
        "",
        "## Added",
    ]
    for row in MICRO_FINDINGS:
        lines.append(f"- `{row['repo_path']}` | {row['severity']} | {row['tag']} | {row['subtag']}")
    lines.extend(["", "## Dropped"])
    for repo, count in dropped["repo_path"].value_counts().items():
        lines.append(f"- `{repo}`: {count}")
    (REPORT_DIR / "micro_ab_v1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {output_path}")
    print("Copied to submission.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
