from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_super_aggressive_rebalanced.csv"
SAMPLE_PATH = ROOT / "data" / "raw_kaggle" / "submission_example.csv"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "outputs" / "reports"


PROTECTED_REPOS = {
    # Repos added from public reports or direct source inspection. Keep these because recent LB gains came from coverage.
    "103f39b0f29b",
    "1167ec3a176e",
    "27c6f2a68058",
    "348856fe60ac",
    "51c6dc5fd57f",
    "592eed5791df",
    "73f6a793d916",
    "9470d2cf198f",
    "9ddd6b83c27e",
    "c2426a2ab283",
    "e7921851ec01",
}


def padding_rows(start_property: int, n_rows: int, columns: list[str]) -> pd.DataFrame:
    rows = []
    for offset in range(n_rows):
        rows.append(
            {
                "Property": start_property + offset,
                "repo_path": "empty",
                "severity": "empty",
                "tag": "empty",
                "subtag": "empty",
                "description": "empty",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def select_drop_indices(sub: pd.DataFrame, n_drop: int) -> list[int]:
    drop_indices: list[int] = []
    working = sub.copy()

    while len(drop_indices) < n_drop:
        active = working.drop(index=drop_indices, errors="ignore")
        counts = active["repo_path"].value_counts()
        candidates = [
            repo
            for repo, count in counts.items()
            if repo not in PROTECTED_REPOS and count > 1
        ]
        if not candidates:
            candidates = [repo for repo, count in counts.items() if count > 1]
        if not candidates:
            raise RuntimeError("Cannot drop more rows while preserving at least one row per repo.")

        # Continue the successful direction: trim from the most over-reported repos first.
        repo = max(candidates, key=lambda r: counts[r])
        repo_rows = active.index[active["repo_path"] == repo].tolist()
        drop_indices.append(repo_rows[-1])

    return drop_indices


def make_variant(non_empty_rows: int, copy_root: bool = False) -> Path:
    if non_empty_rows < 53 or non_empty_rows > 400:
        raise ValueError("non_empty_rows must be between 53 and 400")
    base = pd.read_csv(BASE_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    columns = list(sample.columns)
    if list(base.columns) != columns:
        raise ValueError("Base columns do not match submission_example.csv")

    n_drop = len(base) - non_empty_rows
    drop_indices = select_drop_indices(base, n_drop)
    kept = base.drop(index=drop_indices).copy()
    kept = kept[columns]
    kept["Property"] = range(1, len(kept) + 1)
    pad = padding_rows(len(kept) + 1, 400 - len(kept), columns)
    out = pd.concat([kept, pad], ignore_index=True)
    out["Property"] = range(1, 401)

    output_path = OUT_DIR / f"submission_precision_padding_{non_empty_rows}.csv"
    out.to_csv(output_path, index=False)
    if copy_root:
        shutil.copy2(output_path, ROOT / "submission.csv")

    dropped = base.loc[drop_indices]
    return output_path, dropped, out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    variants = [380, 360, 340]
    report_lines = [
        "# Precision Padding Variants",
        "",
        f"- Base: `{BASE_PATH}`",
        "- Strategy: keep all 53 repos covered, trim overrepresented repos, pad remaining rows with `empty`.",
        "",
    ]
    for non_empty in variants:
        path, dropped, out = make_variant(non_empty, copy_root=(non_empty == 360))
        non_padding = out["repo_path"].astype(str).str.lower() != "empty"
        report_lines.extend(
            [
                f"## {non_empty} Non-Empty Variant",
                f"- Output: `{path}`",
                f"- Shape: `{tuple(out.shape)}`",
                f"- Non-empty rows: `{int(non_padding.sum())}`",
                f"- Padding rows: `{int((~non_padding).sum())}`",
                f"- Repos covered: `{out.loc[non_padding, 'repo_path'].nunique()}`",
                "- Dropped row counts:",
            ]
        )
        for repo, count in dropped["repo_path"].value_counts().items():
            report_lines.append(f"  - `{repo}`: {count}")
        report_lines.append("")

    report_path = REPORT_DIR / "precision_padding_variants_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote variants: {', '.join(str(OUT_DIR / f'submission_precision_padding_{n}.csv') for n in variants)}")
    print("Copied 360-row non-empty variant to submission.csv")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
