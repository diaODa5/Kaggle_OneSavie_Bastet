from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_3825_no_padding_targeted_v2.csv"
TRAIN_PATH = ROOT / "data" / "raw_kaggle" / "train.csv"
SAMPLE_PATH = ROOT / "data" / "raw_kaggle" / "submission_example.csv"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "outputs" / "reports"


LABEL_UPDATES = {
    372: {
        "tag": "Chainlink, Oracle",
        "subtag": "Missing Return Check, Stale Value",
    },
    383: {
        "tag": "Arithmetic,ERC4626",
        "subtag": "Rounding Error",
    },
    385: {
        "tag": "Input Validation, Liquidation",
        "subtag": "Incorrect Parameter",
    },
    398: {
        "tag": "Chainlink, Oracle",
        "subtag": "Missing Return Check, Stale Value",
    },
}


def normalize(value: str) -> str:
    return " ".join(str(value).split())


def validate_labels(sub: pd.DataFrame, train: pd.DataFrame) -> None:
    for col in ["severity", "tag", "subtag"]:
        legal = {normalize(v) for v in train[col].dropna().astype(str)}
        actual = {normalize(v) for v in sub[col].dropna().astype(str)}
        bad = sorted(actual - legal)
        if bad:
            raise ValueError(f"Illegal {col} labels: {bad}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sub = pd.read_csv(BASE_PATH)
    train = pd.read_csv(TRAIN_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    if list(sub.columns) != list(sample.columns):
        raise ValueError("Base columns do not match submission example")

    report_lines = [
        "# Label Precision V1 Report",
        "",
        f"- Base: `{BASE_PATH}`",
        "- Strategy: preserve rows and descriptions, change only selected tag/subtag labels.",
        "",
        "## Label Updates",
    ]
    for prop, updates in LABEL_UPDATES.items():
        idx = sub.index[sub["Property"].astype(int) == prop]
        if len(idx) != 1:
            raise ValueError(f"Property {prop} not found exactly once")
        i = idx[0]
        old_tag = sub.at[i, "tag"]
        old_subtag = sub.at[i, "subtag"]
        for col, value in updates.items():
            sub.at[i, col] = value
        report_lines.append(
            f"- `{prop}` `{sub.at[i, 'repo_path']}`: tag `{old_tag}` -> `{sub.at[i, 'tag']}`, "
            f"subtag `{old_subtag}` -> `{sub.at[i, 'subtag']}`"
        )

    validate_labels(sub, train)
    output_path = OUT_DIR / "submission_label_precision_v1.csv"
    sub.to_csv(output_path, index=False)
    shutil.copy2(output_path, ROOT / "submission.csv")

    report_lines.insert(3, f"- Output: `{output_path}`")
    (REPORT_DIR / "label_precision_v1_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print("Copied to submission.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
