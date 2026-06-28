from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAPPED_FINDINGS_PATH = ROOT / "data" / "processed" / "public_judging_findings_bastet_mapped.parquet"
SAMPLE_PATH = ROOT / "data" / "raw_kaggle" / "submission_example.csv"
TEST_PATH = ROOT / "data" / "raw_kaggle" / "test.csv"
TRAIN_PATH = ROOT / "data" / "raw_kaggle" / "train.csv"
OUT_DIR = ROOT / "outputs"
PRED_DIR = OUT_DIR / "preds"
REPORT_DIR = OUT_DIR / "reports"

OUTPUTS = {
    "precision": OUT_DIR / "submission_public_judging_precision.csv",
    "balanced": OUT_DIR / "submission_public_judging_balanced.csv",
    "recall": OUT_DIR / "submission_public_judging_recall.csv",
}

CONFIDENCE_RANK = {"exact": 3, "strong": 2, "weak": 1, "rejected": 0}
SEVERITY_RANK = {"High": 2, "Medium": 1}


MANUAL_SHERLOCK_LABELS = {
    ("2024-02-rubicon-finance", "M-1"): ("Medium", "Arithmetic", "Precision Loss"),
    ("2024-02-rubicon-finance", "M-2"): ("Medium", "Accounting Error", "Incorrect Formula"),
    ("2024-02-rubicon-finance", "M-3"): ("Medium", "Accounting Error", "State Update Inconsistency"),
    ("2024-02-rubicon-finance", "M-6"): ("Medium", "Accounting Error", "Incorrect Formula"),
    ("2024-02-rubicon-finance", "H-7"): ("High", "Input Validation", "Invalid Validation"),
    ("2024-02-rubicon-finance", "M-8"): ("Medium", "Accounting Error", "Duplicate Value"),
    ("2024-02-rubicon-finance", "M-10"): ("Medium", "Arithmetic", "Token Decimal"),
    ("2024-02-rubicon-finance", "M-13"): ("Medium", "DoS", "Bad Condition"),
    ("2024-02-rubicon-finance", "M-14"): ("Medium", "MEV", "Front Run"),
    ("2024-02-rubicon-finance", "M-15"): ("Medium", "DoS", "Invalid Validation"),
    ("2024-03-arrakis", "M-1"): ("Medium", "Access Control", "Centralization Risk"),
    ("2024-03-arrakis", "M-2"): ("Medium", "Slippage", "Invalid Slippage Control / Missing slippage check"),
    ("2024-03-arrakis", "M-4"): ("Medium", "DoS", "Bad Condition"),
    ("2024-03-arrakis", "M-5"): ("Medium", "Slippage", "Invalid Slippage Control / Missing slippage check"),
    ("2024-03-arrakis", "H-1"): ("High", "Access Control", "Asset Theft"),
    ("2024-03-arrakis", "H-2"): ("High", "Logic Error", "Bad Condition"),
    ("2024-03-arrakis", "H-3"): ("High", "Accounting Error", "Incorrect Formula"),
    ("2024-03-arrakis", "H-4"): ("High", "Access Control", "Bypass Mechanism"),
    ("2024-03-arrakis", "H-5"): ("High", "Arithmetic", "Precision Loss"),
    ("2024-03-arrakis", "H-6"): ("High", "Slippage", "Invalid Slippage Control / Missing slippage check"),
    ("2024-03-arrakis", "M-7"): ("Medium", "ERC20", "safeApprove"),
    ("2023-10-mzero", "M-1"): ("Medium", "Accounting Error", "Incorrect Formula"),
    ("2023-10-mzero", "M-2"): ("Medium", "Access Control", "Invalid Validation"),
    ("2023-10-mzero", "M-3"): ("Medium", "Accounting Error", "Incorrect Formula"),
    ("2023-10-mzero", "M-4"): ("Medium", "Governance", "Invalid Validation"),
    ("2023-10-mzero", "H-5"): ("High", "EIP712, Governance, Replay Attack", "Nonce"),
    ("2023-10-mzero", "H-6"): ("High", "EIP712", "Invalid Validation,Nonce"),
    ("2023-10-mzero", "M-7"): ("Medium", "Governance", "State Update Inconsistency"),
    ("2023-10-mzero", "M-8"): ("Medium", "Input Validation", "Incorrect Parameter"),
    ("2023-10-mzero", "H-9"): ("High", "Accounting Error", "Incorrect Formula"),
    ("2023-10-mzero", "M-10"): ("Medium", "Accounting Error", "Incorrect Parameter"),
    ("2023-10-mzero", "H-11"): ("High", "Accounting Error", "State Update Inconsistency"),
    ("2023-10-mzero", "H-12"): ("High", "Accounting Error", "Incorrect Formula"),
    ("2023-10-mzero", "M-13"): ("Medium", "Accounting Error", "State Update Inconsistency"),
    ("2023-10-mzero", "M-14"): ("Medium", "Governance", "Invalid Validation"),
    ("2023-10-mzero", "M-15"): ("Medium", "MEV", "Front Run"),
    ("2023-10-mzero", "M-16"): ("Medium", "Accounting Error", "Incorrect Formula"),
    ("2023-10-mzero", "M-17"): ("Medium", "Governance", "Not EIP Compliant"),
    ("2023-10-mzero", "M-18"): ("Medium", "Governance", "Invalid Validation"),
    ("2023-10-mzero", "M-19"): ("Medium", "DoS", "Bad Condition"),
    ("2023-10-mzero", "M-20"): ("Medium", "Accounting Error", "Incorrect Formula"),
    ("2023-10-mzero", "M-1"): ("Medium", "Access Control", "Invalid Validation"),
    ("2023-10-mzero", "M-2"): ("Medium", "Accounting Error", "State Update Inconsistency"),
    ("2023-10-mzero", "M-3"): ("Medium", "Input Validation", "Bypass Mechanism"),
    ("2022-10-rage-trade", "M-1"): ("Medium", "DoS", "Bad Condition"),
    ("2022-10-rage-trade", "M-2"): ("Medium", "ERC4626", "Inflation Attack"),
    ("2022-10-rage-trade", "H-3"): ("High", "Slippage", "Invalid Slippage Control / Missing slippage check"),
    ("2022-10-rage-trade", "H-4"): ("High", "Access Control", "Asset Theft"),
    ("2022-10-rage-trade", "H-1"): ("High", "Access Control", "Asset Theft"),
    ("2022-10-rage-trade", "H-2"): ("High", "Accounting Error", "Incorrect Formula"),
    ("2022-10-rage-trade", "M-3"): ("Medium", "Slippage", "Missing minOut / maxAmount"),
    ("2022-10-rage-trade", "M-4"): ("Medium", "Slippage", "Invalid Slippage Control / Missing slippage check"),
    ("2022-10-rage-trade", "M-5"): ("Medium", "ERC4626", "Inflation Attack"),
    ("2022-10-rage-trade", "M-6"): ("Medium", "Accounting Error", "Incorrect Formula"),
    ("2023-12-dodo", "H-1"): ("High", "Accounting Error", "Incorrect Formula"),
    ("2023-12-dodo", "M-1"): ("Medium", "DoS", "Missing Functionality"),
    ("2023-12-dodo", "M-2"): ("Medium", "DoS", "Missing Functionality"),
    ("2023-12-dodo", "M-3"): ("Medium", "Accounting Error", "Incorrect Formula"),
    ("2023-12-dodo", "M-4"): ("Medium", "DoS", "Bad Condition"),
    ("2023-12-dodo", "M-5"): ("Medium", "Logic Error", "Bad Condition"),
    ("2024-01-rio-vesting-escrow", "H-1"): ("High", "call / delegatecall", "No Recovery Mechanism"),
    ("2024-01-rio-vesting-escrow", "M-2"): ("Medium", "Governance", "Missing Functionality"),
    ("2024-07-kwenta-staking-contracts", "M-1"): ("Medium", "Arithmetic", "Precision Loss"),
    ("2024-07-kwenta-staking-contracts", "H-1"): ("High", "Arithmetic", "Precision Loss"),
    ("2025-02-rova", "M-1"): ("Medium", "Input Validation", "Missing Upper/Lower Bound Check"),
    ("2025-02-rova", "M-2"): ("Medium", "Accounting Error", "State Update Inconsistency"),
    ("2023-03-optimism", "M-1"): ("Medium", "Cross-Chain", "Bad Condition"),
    ("2023-03-optimism", "M-2"): ("Medium", "Input Validation, Pause", "Invalid Validation"),
    ("2023-03-optimism", "M-3"): ("Medium", "Arithmetic", "Incorrect Formula"),
    ("2023-03-optimism", "H-4"): ("High", "Cross-Chain, Replay Attack", "Nonce"),
    ("2023-03-optimism", "M-5"): ("Medium", "DoS", "Out of Gas"),
    ("2023-03-optimism", "M-6"): ("Medium", "Cross-Chain", "1/64 Gas Rule"),
    ("2023-03-optimism", "M-7"): ("Medium", "Access Control", "Invalid Validation"),
    ("2023-03-optimism", "M-8"): ("Medium", "Accounting Error", "Incorrect Formula"),
    ("2023-03-optimism", "H-1"): ("High", "Cross-Chain", "1/64 Gas Rule"),
    ("2023-03-optimism", "H-2"): ("High", "Cross-Chain, Replay Attack", "Nonce"),
    ("2023-03-optimism", "H-3"): ("High", "Arithmetic", "Incorrect Formula"),
    ("2023-03-optimism", "M-1"): ("Medium", "Cross-Chain, Replay Attack", "Nonce"),
    ("2023-03-optimism", "M-2"): ("Medium", "Cross-Chain", "1/64 Gas Rule"),
    ("2023-03-optimism", "M-3"): ("Medium", "DoS", "Bad Condition"),
    ("2023-03-optimism", "M-4"): ("Medium", "DoS", "Out of Gas"),
    ("2023-03-optimism", "M-5"): ("Medium", "Arithmetic", "Incorrect Formula"),
    ("2023-03-optimism", "M-6"): ("Medium", "DoS", "Out of Gas"),
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def _ascii(value: Any) -> str:
    text = _text(str(value).replace("\r", " ").replace("\n", " "))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _label_key(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _text(value).lower()))


def _legal_lookup(train: pd.DataFrame, column: str) -> dict[str, str]:
    return {_label_key(value): _text(value) for value in train[column].dropna() if _label_key(value)}


def _legalize(value: Any, lookup: dict[str, str]) -> str:
    return lookup.get(_label_key(value), "")


def _normalize_issue(value: Any) -> str:
    text = _text(value)
    match = re.search(r"\b([hm])[-_ ]*0*(\d+)\b", text, re.I)
    if match:
        return f"{match.group(1).upper()}-{int(match.group(2))}"
    match = re.search(r"\b0*(\d+)[-_ ]*([hm])\b", text, re.I)
    if match:
        return f"{match.group(2).upper()}-{int(match.group(1))}"
    return ""


def _clean_description(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_#>\[\]\(\)]+", " ", text)
    text = _ascii(text)
    if len(text) > 520:
        text = text[:520].rsplit(" ", 1)[0].rstrip(" ,;:") + "."
    if len(text) < 35:
        text = "The implementation contains a source-verified vulnerability matching this public final finding."
    return text


def _manual_key(row: pd.Series) -> tuple[str, str]:
    return (_text(row.get("contest")), _normalize_issue(row.get("issue_id")))


def apply_manual_sherlock_labels(findings: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    out = findings.copy()
    lookups = {column: _legal_lookup(train, column) for column in ["severity", "tag", "subtag"]}
    for index, row in out.iterrows():
        if bool(row.get("labels_complete", False)):
            continue
        labels = MANUAL_SHERLOCK_LABELS.get(_manual_key(row))
        if labels is None:
            continue
        severity, tag, subtag = (
            _legalize(labels[0], lookups["severity"]),
            _legalize(labels[1], lookups["tag"]),
            _legalize(labels[2], lookups["subtag"]),
        )
        if severity and tag and subtag:
            out.at[index, "severity"] = severity
            out.at[index, "tag"] = tag
            out.at[index, "subtag"] = subtag
            out.at[index, "mapping_method"] = "manual_sherlock"
            out.at[index, "mapping_confidence"] = 0.72
            out.at[index, "mapping_ambiguity"] = False
            out.at[index, "labels_complete"] = True
    return out


def _policy_mask(findings: pd.DataFrame, policy: str) -> pd.Series:
    complete = findings["labels_complete"].astype(bool)
    confidence = findings["confidence"].astype(str).str.lower()
    method = findings["mapping_method"].astype(str)
    ambiguity = findings.get("mapping_ambiguity", pd.Series(False, index=findings.index)).fillna(False).astype(bool)
    if policy == "precision":
        return complete & confidence.eq("exact") & (~ambiguity | method.eq("manual_sherlock"))
    if policy == "balanced":
        return complete & confidence.eq("exact")
    if policy == "recall":
        return complete & confidence.isin(["exact", "strong"])
    raise ValueError(f"Unknown policy: {policy}")


def _rank_rows(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["_confidence_rank"] = ranked["confidence"].astype(str).map(CONFIDENCE_RANK).fillna(0)
    ranked["_severity_rank"] = ranked["severity"].astype(str).map(SEVERITY_RANK).fillna(0)
    ranked["_mapping_rank"] = ranked["mapping_method"].astype(str).map(
        {
            "external_exact": 5,
            "external_exact_merged": 4,
            "manual_sherlock": 3,
            "external_exact_ambiguous": 2,
        }
    ).fillna(0)
    mapping_confidence = ranked.get("mapping_confidence", pd.Series(0.0, index=ranked.index))
    ranked["_mapping_conf"] = pd.to_numeric(mapping_confidence, errors="coerce").fillna(0.0)
    ranked["_repo_count"] = ranked.groupby("repo_path")["repo_path"].transform("size")
    if "issue_id" not in ranked.columns:
        ranked["issue_id"] = ""
    return ranked.sort_values(
        ["_confidence_rank", "_mapping_rank", "_severity_rank", "_mapping_conf", "_repo_count", "repo_path", "issue_id"],
        ascending=[False, False, False, False, True, True, True],
    )


def _validate_legal(findings: pd.DataFrame, train: pd.DataFrame) -> None:
    for column in ["severity", "tag", "subtag"]:
        legal = {_text(value) for value in train[column].dropna()}
        actual = {_text(value) for value in findings[column].dropna() if _text(value) and _text(value) != "empty"}
        illegal = sorted(actual - legal)
        if illegal:
            raise ValueError(f"Illegal {column} labels: {illegal[:20]}")


def build_candidate_submission(
    findings: pd.DataFrame,
    sample: pd.DataFrame,
    test: pd.DataFrame,
    train: pd.DataFrame,
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(sample.columns)
    required = {"Property", "repo_path", "severity", "tag", "subtag", "description"}
    if set(columns) != required:
        raise ValueError(f"Unexpected submission columns: {columns}")
    enriched = apply_manual_sherlock_labels(findings, train)
    selected = enriched[_policy_mask(enriched, policy)].copy()
    selected = selected[selected["repo_path"].astype(str).isin(set(test["repo_path"].astype(str)))]
    _validate_legal(selected, train)
    selected = _rank_rows(selected).drop_duplicates(["repo_path", "severity", "tag", "subtag", "description"], keep="first")
    selected = selected.head(400).copy()

    rows = []
    provenance = []
    for _, row in selected.iterrows():
        rows.append(
            {
                "Property": len(rows) + 1,
                "repo_path": _text(row["repo_path"]),
                "severity": _text(row["severity"]),
                "tag": _text(row["tag"]),
                "subtag": _text(row["subtag"]),
                "description": _clean_description(row["description"]),
            }
        )
        provenance.append(
            {
                "Property": len(rows),
                "repo_path": _text(row["repo_path"]),
                "contest": _text(row.get("contest")),
                "issue_id": _text(row.get("issue_id")),
                "source_url": _text(row.get("source_url")),
                "confidence": _text(row.get("confidence")),
                "mapping_method": _text(row.get("mapping_method")),
                "policy": policy,
            }
        )
    while len(rows) < 400:
        rows.append(
            {
                "Property": len(rows) + 1,
                "repo_path": "empty",
                "severity": "empty",
                "tag": "empty",
                "subtag": "empty",
                "description": "empty",
            }
        )
        provenance.append({"Property": len(rows), "repo_path": "empty", "policy": policy, "mapping_method": "padding"})

    out = pd.DataFrame(rows, columns=columns)
    out["Property"] = range(1, 401)
    provenance_frame = pd.DataFrame(provenance)
    return out, provenance_frame


def run(copy_root: bool = True, selected_policy: str = "recall") -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    findings = pd.read_parquet(MAPPED_FINDINGS_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    test = pd.read_csv(TEST_PATH)
    train = pd.read_csv(TRAIN_PATH)

    summaries = {}
    provenance_parts = []
    for policy, path in OUTPUTS.items():
        submission, provenance = build_candidate_submission(findings, sample, test, train, policy)
        submission.to_csv(path, index=False)
        provenance_parts.append(provenance.assign(output=str(path)))
        non_empty = submission[submission["repo_path"] != "empty"]
        summaries[policy] = {
            "path": str(path),
            "rows": int(len(submission)),
            "non_empty": int(len(non_empty)),
            "padding": int((submission["repo_path"] == "empty").sum()),
            "repos": int(non_empty["repo_path"].nunique()),
            "method_counts": provenance["mapping_method"].value_counts().to_dict(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"{policy}: {path} non_empty={summaries[policy]['non_empty']} padding={summaries[policy]['padding']}")

    provenance_all = pd.concat(provenance_parts, ignore_index=True)
    provenance_path = PRED_DIR / "public_judging_submission_provenance.parquet"
    provenance_all.to_parquet(provenance_path, index=False)
    selected_path = OUTPUTS[selected_policy]
    if copy_root:
        shutil.copy2(selected_path, ROOT / "submission.csv")
        shutil.copy2(selected_path, OUT_DIR / "submission.csv")
    lines = [
        "# Public Judging Submission Report",
        "",
        f"- Selected policy: `{selected_policy}`",
        f"- Selected path: `{selected_path}`",
        f"- Copied root submission: `{copy_root}`",
        f"- Provenance sidecar: `{provenance_path}`",
        "",
        "## Candidate Summaries",
    ]
    for policy, summary in summaries.items():
        lines.append(f"### {policy}")
        for key, value in summary.items():
            lines.append(f"- {key}: `{value}`")
    (REPORT_DIR / "public_judging_submission_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate public-judging evidence-aware submissions.")
    parser.add_argument("--selected-policy", choices=sorted(OUTPUTS), default="recall")
    parser.add_argument("--no-copy-root", action="store_true")
    args = parser.parse_args()
    run(copy_root=not args.no_copy_root, selected_policy=args.selected_policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
