import argparse
from pathlib import Path

import pandas as pd

try:
    from config import (
        ROOT_SUBMISSION_PATH,
        SUBMISSION_EXAMPLE_PATH,
        TEST_CSV_PATH,
        TRAIN_CSV_PATH,
        VALIDATION_REPORT_PATH,
        ensure_project_dirs,
    )
    from utils import load_json, normalize_text, read_csv_required
except ImportError:
    from src.config import (
        ROOT_SUBMISSION_PATH,
        SUBMISSION_EXAMPLE_PATH,
        TEST_CSV_PATH,
        TRAIN_CSV_PATH,
        VALIDATION_REPORT_PATH,
        ensure_project_dirs,
    )
    from src.utils import normalize_text, read_csv_required


def validate_submission_frames(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    test: pd.DataFrame,
    train: pd.DataFrame,
    target_info: dict | None = None,
    expected_rows: int | None = 400,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    expected_columns = list(sample.columns)

    if list(submission.columns) != expected_columns:
        errors.append(f"Column order/name mismatch. expected={expected_columns} actual={list(submission.columns)}")
    if submission.columns.astype(str).str.startswith("Unnamed:").any():
        errors.append("Submission contains an unwanted Unnamed/index column.")
    if expected_rows is not None and len(submission) != expected_rows:
        errors.append(f"Submission must have exactly {expected_rows} data rows. actual={len(submission)}")
    if len(submission) < len(test):
        errors.append(f"Submission row count {len(submission)} is smaller than test repo count {len(test)}.")
    if len(submission) == len(test):
        warnings.append("Submission has exactly one finding per test repo; likely low recall for a finding-level task.")
    if len(test) >= 20 and len(submission) < 150:
        warnings.append("Submission has fewer than 150 findings; likely low recall.")
    if len(submission) > 600:
        warnings.append("Submission has more than 600 findings; this may be too aggressive.")
    if submission.isna().any().any():
        errors.append("Submission contains NaN values.")

    if "repo_path" in submission.columns:
        test_repos = set(test["repo_path"].astype(str))
        repo_values = submission["repo_path"].astype(str)
        padding_mask = repo_values.map(lambda value: normalize_text(value).lower() == "empty")
        actual_repos = set(repo_values[~padding_mask])
        unknown = sorted(actual_repos - test_repos)
        if unknown:
            errors.append(f"Submission contains repo_path values not present in test.csv: {unknown[:20]}")
        if padding_mask.any():
            target_cols = [col for col in ["severity", "tag", "subtag", "description"] if col in submission.columns]
            bad_padding = submission.loc[padding_mask, target_cols].apply(
                lambda row: any(normalize_text(value).lower() != "empty" for value in row),
                axis=1,
            )
            if bad_padding.any():
                errors.append("Rows with repo_path='empty' must use 'empty' for severity/tag/subtag/description.")
    else:
        errors.append("Submission missing repo_path column.")

    if "Property" in submission.columns:
        prop = submission["Property"]
        if prop.duplicated().any():
            errors.append("Property contains duplicate values.")
        expected = [str(i) for i in range(1, len(submission) + 1)]
        if prop.astype(str).tolist() != expected:
            errors.append("Property must be exactly sequential 1..N.")
    else:
        errors.append("Submission missing Property column.")

    for col in ["severity", "tag", "subtag"]:
        if col not in submission.columns:
            errors.append(f"Submission missing {col} column.")
            continue
        legal = {normalize_text(v) for v in train[col].dropna().unique()}
        if "repo_path" in submission.columns:
            non_padding = submission["repo_path"].astype(str).map(lambda value: normalize_text(value).lower() != "empty")
            actual = {normalize_text(v) for v in submission.loc[non_padding, col].dropna().unique()}
        else:
            actual = {normalize_text(v) for v in submission[col].dropna().unique()}
        illegal = sorted(actual - legal)
        if illegal:
            errors.append(f"Column `{col}` has labels not seen in train.csv: {illegal[:20]}")

    for col in submission.columns:
        blank = submission[col].map(lambda value: normalize_text(value) == "")
        ellipsis = submission[col].astype(str).map(lambda value: value.strip() == "...")
        if blank.any():
            errors.append(f"Column `{col}` contains blank values.")
        if ellipsis.any():
            errors.append(f"Column `{col}` contains literal `...`.")

    if "description" in submission.columns:
        if "repo_path" in submission.columns:
            non_padding = submission["repo_path"].astype(str).map(lambda value: normalize_text(value).lower() != "empty")
        else:
            non_padding = pd.Series([True] * len(submission), index=submission.index)
        desc = submission.loc[non_padding, "description"].map(normalize_text)
        if desc.map(lambda value: value in {"", "...", "empty"}).any():
            errors.append("description contains empty/example placeholder values.")
        if desc.map(len).lt(25).any():
            errors.append("description contains values shorter than 25 characters.")
        if submission.loc[non_padding, "description"].astype(str).str.contains(r"[\r\n]", regex=True).any():
            errors.append("description contains newline characters.")
        duplicate_rate = float(desc.duplicated().mean()) if len(desc) else 1.0
        if desc.nunique() <= 1:
            errors.append("description values are all identical.")
        elif duplicate_rate > 0.75:
            warnings.append(f"description duplicate rate is high: {duplicate_rate:.3f}.")
    else:
        duplicate_rate = 1.0
        errors.append("Submission missing description column.")

    return {
        "shape": list(submission.shape),
        "expected_columns": expected_columns,
        "columns": list(submission.columns),
        "test_repo_count": int(len(test)),
        "expected_rows": expected_rows,
        "description_duplicate_rate": duplicate_rate,
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def validate_submission(path=ROOT_SUBMISSION_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"submission.csv not found: {path}")
    submission = pd.read_csv(path)
    sample = read_csv_required(SUBMISSION_EXAMPLE_PATH)
    test = read_csv_required(TEST_CSV_PATH)
    train = read_csv_required(TRAIN_CSV_PATH)
    report = validate_submission_frames(submission, sample, test, train, None, expected_rows=400)
    report["path"] = str(path)
    return report


def write_report(report: dict) -> None:
    lines = [
        "# Validation Report",
        "",
        f"- Submission path: `{report.get('path', '')}`",
        f"- Shape: `{report['shape']}`",
        f"- Test repo count: `{report['test_repo_count']}`",
        f"- Expected row count: `{report['expected_rows']}`",
        f"- Columns: `{report['columns']}`",
        f"- Expected columns: `{report['expected_columns']}`",
        f"- Description duplicate rate: `{report['description_duplicate_rate']:.6f}`",
        f"- Passed: `{report['passed']}`",
        "",
        "## Errors",
    ]
    lines.extend([f"- {err}" for err in report["errors"]] if report["errors"] else ["No validation errors."])
    lines.append("")
    lines.append("## Warnings")
    lines.extend([f"- {warn}" for warn in report["warnings"]] if report["warnings"] else ["No validation warnings."])
    VALIDATION_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Kaggle submission CSV.")
    parser.add_argument("--path", type=Path, default=ROOT_SUBMISSION_PATH)
    args = parser.parse_args(argv)
    ensure_project_dirs()
    print("Validating variable-length finding-level submission.csv...")
    report = validate_submission(args.path)
    write_report(report)
    if not report["passed"]:
        print(f"Validation failed. See {VALIDATION_REPORT_PATH}")
        for err in report["errors"]:
            print(f"- {err}")
        raise SystemExit(1)
    print(f"Validation passed: {args.path}")
    if report["warnings"]:
        print("Warnings:")
        for warn in report["warnings"]:
            print(f"- {warn}")
    print(f"Report: {VALIDATION_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
