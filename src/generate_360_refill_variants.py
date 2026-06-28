from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SKELETON_360 = ROOT / "outputs" / "submission_precision_padding_360.csv"
BASE_3825 = ROOT / "outputs" / "submission_backup_3825_no_padding_targeted_v2.csv"
DESC_PRECISION = ROOT / "outputs" / "submission_description_precision_v1.csv"
SAMPLE_PATH = ROOT / "data" / "raw_kaggle" / "submission_example.csv"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "outputs" / "reports"


EXTRA_BOLD_ROWS = [
    {
        "repo_path": "103f39b0f29b",
        "severity": "Medium",
        "tag": "Access Control",
        "subtag": "Centralization Risk",
        "description": (
            "RubiconFeeController can configure base and pair-specific fees applied to reactor outputs. Without a "
            "strict upper bound, a privileged fee controller can set excessive fees and capture most of a user's "
            "swap proceeds."
        ),
    },
    {
        "repo_path": "51c6dc5fd57f",
        "severity": "Medium",
        "tag": "Arithmetic",
        "subtag": "Precision Loss",
        "description": (
            "M^0 continuous-indexing math converts principal and present value using fixed-point division. Repeated "
            "rounding around small balances or rate changes can under-credit users and leave value trapped in the "
            "token accounting."
        ),
    },
    {
        "repo_path": "9ddd6b83c27e",
        "severity": "Medium",
        "tag": "ERC4626",
        "subtag": "Rounding Error",
        "description": (
            "Rage Trade vault share conversion can round differently between preview and actual deposit or "
            "withdrawal paths. In edge cases, a user can mint too many shares or receive fewer assets than the "
            "ERC4626 accounting implies."
        ),
    },
    {
        "repo_path": "e7921851ec01",
        "severity": "Medium",
        "tag": "Slippage",
        "subtag": "Invalid Slippage Control / Missing slippage check",
        "description": (
            "D3MMLiquidationRouter swaps collateral during liquidation settlement. Missing minimum output "
            "protection lets the liquidation path execute at an unfavorable price and reduces recovered collateral "
            "for the vault."
        ),
    },
]


def clean(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def row_key(row: pd.Series | dict) -> tuple[str, str, str, str, str]:
    return (
        clean(row["repo_path"]),
        clean(row["severity"]),
        clean(row["tag"]),
        clean(row["subtag"]),
        clean(row["description"]),
    )


def non_empty(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["repo_path"].astype(str).str.lower() != "empty"].copy()


def make_submission(kept: pd.DataFrame, fill_rows: list[dict], columns: list[str]) -> pd.DataFrame:
    additions = pd.DataFrame(fill_rows)
    if "Property" not in additions.columns:
        additions.insert(0, "Property", range(1, len(additions) + 1))
    out = pd.concat([kept[columns], additions[columns]], ignore_index=True)
    out = out[columns].copy()
    out["Property"] = range(1, 401)
    for col in ["repo_path", "severity", "tag", "subtag", "description"]:
        out[col] = out[col].map(clean)
    if len(out) != 400:
        raise AssertionError(f"Expected 400 rows, got {len(out)}")
    if (out["repo_path"].str.lower() == "empty").any():
        raise AssertionError("Refilled submission still contains empty rows")
    if out.duplicated(subset=["repo_path", "severity", "tag", "subtag", "description"]).any():
        raise AssertionError("Refilled submission contains duplicate finding rows")
    return out


def candidate_rows_from(reference: pd.DataFrame, kept: pd.DataFrame) -> list[dict]:
    existing = {row_key(row) for _, row in kept.iterrows()}
    rows = []
    for _, row in non_empty(reference).iterrows():
        key = row_key(row)
        if key not in existing:
            rows.append(row.to_dict())
            existing.add(key)
    return rows


def build_variants() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    skeleton = pd.read_csv(SKELETON_360)
    reference = pd.read_csv(BASE_3825)
    desc_reference = pd.read_csv(DESC_PRECISION) if DESC_PRECISION.exists() else reference
    sample = pd.read_csv(SAMPLE_PATH)
    columns = list(sample.columns)

    kept = non_empty(skeleton)
    if len(kept) != 360:
        raise AssertionError(f"Expected 360 kept rows, got {len(kept)}")

    conservative_pool = candidate_rows_from(reference, kept)
    conservative_fill = conservative_pool[:40]
    conservative = make_submission(kept, conservative_fill, columns)
    conservative_path = OUT_DIR / "submission_360_refilled_from_3825.csv"
    conservative.to_csv(conservative_path, index=False)

    # Bold version: use the description-precision reference where available, then replace four restored
    # rows from overrepresented repos with targeted extras for the repos that helped in v2.
    bold_pool = candidate_rows_from(desc_reference, kept)
    bold_fill = bold_pool[:40]
    if len(bold_fill) < 40:
        raise AssertionError("Not enough rows to fill bold variant")
    protected = {row["repo_path"] for row in EXTRA_BOLD_ROWS}
    extra_keys = {row_key(row) for row in EXTRA_BOLD_ROWS}
    existing_after_fill = {row_key(row) for _, row in kept.iterrows()} | {row_key(row) for row in bold_fill}
    extras = [row for row in EXTRA_BOLD_ROWS if row_key(row) not in existing_after_fill and row_key(row) not in extra_keys]
    # The condition above intentionally deduplicates against kept/fill; use all explicitly unique extras.
    extras = [row for row in EXTRA_BOLD_ROWS if row_key(row) not in existing_after_fill]
    replace_count = min(len(extras), 4)
    if replace_count:
        fill_df = pd.DataFrame(bold_fill)
        drop_positions = []
        counts = pd.concat([kept[["repo_path"]], fill_df[["repo_path"]]], ignore_index=True)["repo_path"].value_counts()
        for pos in reversed(range(len(bold_fill))):
            repo = bold_fill[pos]["repo_path"]
            if repo not in protected and counts.get(repo, 0) > 12:
                drop_positions.append(pos)
                counts[repo] -= 1
            if len(drop_positions) == replace_count:
                break
        for pos in sorted(drop_positions, reverse=True):
            bold_fill.pop(pos)
        bold_fill.extend(extras[:replace_count])
    bold = make_submission(kept, bold_fill, columns)
    bold_path = OUT_DIR / "submission_360_refilled_bold.csv"
    bold.to_csv(bold_path, index=False)
    shutil.copy2(bold_path, ROOT / "submission.csv")

    report_lines = [
        "# 360 Refill Variants",
        "",
        f"- Skeleton: `{SKELETON_360}`",
        f"- Reference: `{BASE_3825}`",
        f"- Conservative output: `{conservative_path}`",
        f"- Bold output: `{bold_path}`",
        "- Current root submission: bold output",
        "",
        "## Variant Summary",
    ]
    for name, df in [("conservative", conservative), ("bold", bold)]:
        report_lines.extend(
            [
                f"### {name}",
                f"- Shape: `{tuple(df.shape)}`",
                f"- Non-empty rows: `{int((df['repo_path'].str.lower() != 'empty').sum())}`",
                f"- Repos covered: `{df['repo_path'].nunique()}`",
                "- Top counts:",
            ]
        )
        for repo, count in df["repo_path"].value_counts().head(10).items():
            report_lines.append(f"  - `{repo}`: {count}")
    (REPORT_DIR / "360_refill_variants_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return conservative_path, bold_path


def main() -> int:
    conservative_path, bold_path = build_variants()
    print(f"Wrote {conservative_path}")
    print(f"Wrote {bold_path}")
    print("Copied bold variant to submission.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
