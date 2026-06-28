import argparse
import ast
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "public_judging_findings.parquet"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "public_judging_manifest.csv"
DEFAULT_ZIP_PATH = PROJECT_ROOT / "data" / "raw_kaggle" / "test.zip"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "test_public_finding_matches.parquet"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "public_finding_verification_report.md"

PARSER_COLUMNS = [
    "platform",
    "contest",
    "issue_id",
    "title",
    "severity",
    "description",
    "referenced_files",
    "referenced_functions",
    "source_path",
    "source_url",
]
EVIDENCE_COLUMNS = [
    "repo_path",
    "confidence",
    "file_match_kind",
    "matched_files",
    "matched_functions",
    "matched_identifiers",
    "matched_snippets",
    "evidence_summary",
    "rejection_reason",
]
OUTPUT_COLUMNS = ["repo_path", *PARSER_COLUMNS, *EVIDENCE_COLUMNS[1:]]

EXCLUDED_SEGMENTS = {
    "__macosx",
    "artifacts",
    "build",
    "cache",
    "dependencies",
    "deps",
    "generated",
    "lib",
    "libs",
    "mock",
    "mocks",
    "node_modules",
    "out",
    "test",
    "tests",
    "vendor",
}
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
FENCE_RE = re.compile(r"```(?:solidity|sol)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
SOLIDITY_NOISE = {
    "address",
    "bool",
    "bytes",
    "calldata",
    "contract",
    "external",
    "false",
    "function",
    "internal",
    "memory",
    "private",
    "public",
    "returns",
    "solidity",
    "storage",
    "string",
    "struct",
    "true",
    "uint256",
}


@dataclass(frozen=True)
class SourceFile:
    repo_path: str
    path: str
    archive_path: str
    text: str
    normalized_text: str
    identifiers: frozenset[str]


SourceIndex = dict[str, dict[str, SourceFile]]


def _path_parts(value: str) -> tuple[str, ...]:
    return tuple(part.lower() for part in PurePosixPath(value.replace("\\", "/")).parts)


def is_excluded_path(value: str) -> bool:
    return bool(set(_path_parts(value)) & EXCLUDED_SEGMENTS)


def build_source_index(zip_path: Path | str) -> SourceIndex:
    index: SourceIndex = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            archive_path = info.filename.replace("\\", "/")
            parts = PurePosixPath(archive_path).parts
            if (
                len(parts) < 4
                or parts[0] != "test"
                or not archive_path.lower().endswith(".sol")
            ):
                continue
            repo_path = parts[1]
            relative_path = PurePosixPath(*parts[2:]).as_posix()
            if is_excluded_path(relative_path):
                continue
            text = zf.read(info).decode("utf-8", errors="replace")
            index.setdefault(repo_path, {})[relative_path] = SourceFile(
                repo_path=repo_path,
                path=relative_path,
                archive_path=archive_path,
                text=text,
                normalized_text=_normalize_source_text(text),
                identifiers=frozenset(IDENTIFIER_RE.findall(text)),
            )
    return index


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if hasattr(value, "tolist") and not isinstance(value, str):
        return _as_list(value.tolist())
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        for loader in (json.loads, ast.literal_eval):
            try:
                return _as_list(loader(text))
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
    return [text]


def _normalize_reference_path(value: str) -> str:
    text = value.strip().strip("`'\"()[]{}.,:;")
    text = text.replace("\\", "/").split("#", 1)[0].split("?", 1)[0]
    is_url = "://" in text or text.lower().startswith(("github.com/", "www.github.com/"))
    parsed = urlparse(text if "://" in text else f"https://{text}") if is_url else None
    if parsed and parsed.netloc:
        parts = list(PurePosixPath(parsed.path).parts)
        if "blob" in parts:
            blob_index = parts.index("blob")
            parts = parts[blob_index + 2 :]
        elif len(parts) >= 3 and parsed.netloc.lower() == "github.com":
            parts = parts[2:]
        text = PurePosixPath(*parts).as_posix()
    text = re.sub(r"^(?:\./)+", "", text)
    return text.lstrip("/")


def _normalize_source_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _extract_snippets(description: str) -> list[str]:
    candidates: list[str] = []
    for block in FENCE_RE.findall(description):
        candidates.extend(block.splitlines())
    candidates.extend(INLINE_CODE_RE.findall(description))
    snippets = []
    for candidate in candidates:
        clean = candidate.strip()
        if (
            len(clean) >= 16
            and not clean.lower().startswith(("http://", "https://"))
            and not clean.lower().endswith(".sol")
            and clean not in snippets
        ):
            snippets.append(clean)
    return snippets


def _identifiers(text: str) -> list[str]:
    return sorted(
        {
            identifier
            for identifier in IDENTIFIER_RE.findall(text)
            if identifier.lower() not in SOLIDITY_NOISE
        }
    )


def _code_identifiers(description: str, functions: list[str]) -> list[str]:
    code = list(functions)
    code.extend(INLINE_CODE_RE.findall(description))
    code.extend(FENCE_RE.findall(description))
    return _identifiers(" ".join(code))


def _find_referenced_files(
    references: list[str],
    repo_sources: dict[str, SourceFile],
) -> tuple[list[str], str, list[str], list[str]]:
    exact: set[str] = set()
    basename: set[str] = set()
    excluded: list[str] = []
    normalized: list[str] = []
    lower_to_path = {path.lower(): path for path in repo_sources}
    by_basename: dict[str, list[str]] = {}
    for path in repo_sources:
        by_basename.setdefault(PurePosixPath(path).name.lower(), []).append(path)

    for reference in references:
        path = _normalize_reference_path(reference)
        if not path.lower().endswith(".sol"):
            continue
        normalized.append(path)
        if is_excluded_path(path):
            excluded.append(path)
            continue
        direct = lower_to_path.get(path.lower())
        if direct:
            exact.add(direct)
            continue
        matches = by_basename.get(PurePosixPath(path).name.lower(), [])
        if len(matches) == 1:
            basename.add(matches[0])

    if exact:
        return sorted(exact | basename), "exact", excluded, normalized
    if basename:
        return sorted(basename), "basename", excluded, normalized
    return [], "none", excluded, normalized


def verify_finding(
    finding: dict[str, Any] | pd.Series,
    repo_path: str,
    source_index: SourceIndex,
) -> dict[str, Any]:
    row = dict(finding)
    repo_sources = source_index.get(str(repo_path), {})
    references = _as_list(row.get("referenced_files"))
    functions = sorted(set(_as_list(row.get("referenced_functions"))))
    matched_files, file_match_kind, excluded, normalized_refs = _find_referenced_files(
        references,
        repo_sources,
    )
    valid_refs = [path for path in normalized_refs if path not in excluded]

    if not repo_sources:
        rejection_reason = "repository_not_indexed"
    elif normalized_refs and not valid_refs:
        rejection_reason = "excluded_referenced_path"
    elif valid_refs and not matched_files:
        rejection_reason = "referenced_file_not_found"
    else:
        rejection_reason = ""

    search_paths = matched_files or sorted(repo_sources)
    matched_functions: set[str] = set()
    function_files: set[str] = set()
    for function in functions:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function):
            continue
        for path in search_paths:
            if function in repo_sources[path].identifiers:
                matched_functions.add(function)
                function_files.add(path)

    snippets = _extract_snippets(str(row.get("description", "") or ""))
    matched_snippets: list[str] = []
    snippet_files: set[str] = set()
    for snippet in snippets:
        normalized_snippet = _normalize_source_text(snippet)
        for path in search_paths:
            if normalized_snippet in repo_sources[path].normalized_text:
                matched_snippets.append(snippet)
                snippet_files.add(path)
                break

    candidate_identifiers = _code_identifiers(
        str(row.get("description", "") or ""),
        functions,
    )
    matched_identifiers: set[str] = set()
    identifier_files: set[str] = set()
    for identifier in candidate_identifiers:
        for path in search_paths:
            if identifier in repo_sources[path].identifiers:
                matched_identifiers.add(identifier)
                identifier_files.add(path)

    all_matched_files = sorted(
        set(matched_files) | function_files | snippet_files | identifier_files
    )
    corroborated = bool(matched_functions or matched_snippets)
    if rejection_reason:
        confidence = "rejected"
    elif file_match_kind == "exact" and corroborated:
        confidence = "exact"
    elif file_match_kind == "exact":
        confidence = "strong"
    elif file_match_kind == "basename" and corroborated:
        confidence = "strong"
    elif file_match_kind == "basename":
        confidence = "weak"
    elif matched_snippets:
        confidence = "strong"
    elif matched_functions or matched_identifiers:
        confidence = "weak"
    else:
        confidence = "rejected"
        rejection_reason = "no_source_evidence"

    evidence_bits = []
    if file_match_kind != "none":
        evidence_bits.append(f"{file_match_kind} file: {', '.join(matched_files)}")
    if matched_functions:
        evidence_bits.append(f"functions: {', '.join(sorted(matched_functions))}")
    if matched_snippets:
        evidence_bits.append(f"snippets: {len(matched_snippets)}")
    if matched_identifiers:
        evidence_bits.append(f"identifiers: {', '.join(sorted(matched_identifiers))}")

    return {
        "repo_path": str(repo_path),
        "confidence": confidence,
        "file_match_kind": file_match_kind,
        "matched_files": all_matched_files,
        "matched_functions": sorted(matched_functions),
        "matched_identifiers": sorted(matched_identifiers),
        "matched_snippets": matched_snippets,
        "evidence_summary": "; ".join(evidence_bits),
        "rejection_reason": rejection_reason,
    }


def _manifest_mapping(manifest: pd.DataFrame) -> dict[tuple[str, str], str]:
    if manifest.empty:
        return {}
    rows = manifest.copy()
    if "resolution_status" in rows:
        rows = rows[rows["resolution_status"].astype(str).str.lower() == "resolved"]
    mapping = {}
    for _, row in rows.iterrows():
        key = (
            str(row.get("platform", "") or "").strip().lower(),
            str(row.get("contest_slug", "") or "").strip().lower(),
        )
        repo_path = str(row.get("repo_path", "") or "").strip()
        if all(key) and repo_path:
            mapping[key] = repo_path
    return mapping


def verify_findings(
    findings: pd.DataFrame,
    source_index: SourceIndex,
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    missing = [column for column in PARSER_COLUMNS if column not in findings.columns]
    if missing:
        raise ValueError(f"Parsed findings missing required columns: {', '.join(missing)}")
    mapping = _manifest_mapping(manifest if manifest is not None else pd.DataFrame())
    records = []
    for _, finding_row in findings.iterrows():
        direct_repo = str(finding_row.get("repo_path", "") or "").strip()
        key = (
            str(finding_row.get("platform", "") or "").strip().lower(),
            str(finding_row.get("contest", "") or "").strip().lower(),
        )
        repo_path = direct_repo or mapping.get(key, "")
        evidence = (
            verify_finding(finding_row, repo_path, source_index)
            if repo_path
            else {
                "repo_path": "",
                "confidence": "rejected",
                "file_match_kind": "none",
                "matched_files": [],
                "matched_functions": [],
                "matched_identifiers": [],
                "matched_snippets": [],
                "evidence_summary": "",
                "rejection_reason": "contest_repo_mapping_missing",
            }
        )
        record = {"repo_path": evidence["repo_path"]}
        record.update({column: finding_row[column] for column in PARSER_COLUMNS})
        record.update({column: evidence[column] for column in EVIDENCE_COLUMNS[1:]})
        records.append(record)
    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def write_report(
    matches: pd.DataFrame,
    report_path: Path,
    notes: list[str] | None = None,
) -> None:
    notes = notes or []
    status = "degraded" if notes else "complete"
    counts = (
        matches["confidence"].value_counts().reindex(
            ["exact", "strong", "weak", "rejected"],
            fill_value=0,
        )
        if not matches.empty
        else pd.Series({"exact": 0, "strong": 0, "weak": 0, "rejected": 0})
    )
    accepted = matches[matches["confidence"].isin(["exact", "strong", "weak"])]
    lines = [
        "# Public Finding Verification Report",
        "",
        f"- Status: `{status}`",
        f"- Parsed findings evaluated: `{len(matches)}`",
        f"- Accepted with source evidence: `{len(accepted)}`",
        f"- Exact: `{int(counts['exact'])}`",
        f"- Strong: `{int(counts['strong'])}`",
        f"- Weak: `{int(counts['weak'])}`",
        f"- Rejected: `{int(counts['rejected'])}`",
        "- Exclusions: dependencies, tests, mocks, generated output, and `__MACOSX`",
        "",
    ]
    if notes:
        lines.extend(["## Degraded Inputs", ""])
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    lines.extend(
        [
            "## Confidence Policy",
            "",
            "- `exact`: exact project path plus matching function or source snippet.",
            "- `strong`: exact path alone, basename plus corroboration, or a source snippet.",
            "- `weak`: basename-only or identifier/function-only source overlap.",
            "- `rejected`: missing mapping/source, excluded path, missing referenced file, or no evidence.",
            "",
            "## Evidence Coverage",
            "",
            "| Confidence | Findings |",
            "| --- | ---: |",
        ]
    )
    for confidence in ["exact", "strong", "weak", "rejected"]:
        lines.append(f"| {confidence} | {int(counts[confidence])} |")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    findings_path: Path | str = DEFAULT_FINDINGS_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    zip_path: Path | str = DEFAULT_ZIP_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    report_path: Path | str = DEFAULT_REPORT_PATH,
) -> pd.DataFrame:
    findings_path = Path(findings_path)
    manifest_path = Path(manifest_path)
    zip_path = Path(zip_path)
    output_path = Path(output_path)
    report_path = Path(report_path)
    notes = []

    if not findings_path.exists():
        notes.append(f"Missing parsed findings: `{findings_path}`")
    if not manifest_path.exists():
        notes.append(f"Missing contest manifest: `{manifest_path}`")
    if not zip_path.exists():
        notes.append(f"Missing test source ZIP: `{zip_path}`")

    matches = _empty_output()
    if findings_path.exists() and zip_path.exists():
        findings = pd.read_parquet(findings_path)
        manifest = (
            pd.read_csv(manifest_path, dtype=str).fillna("")
            if manifest_path.exists()
            else pd.DataFrame()
        )
        if not manifest_path.exists() and "repo_path" not in findings.columns:
            notes.append("Verification skipped because findings have no direct repo mapping.")
        else:
            try:
                matches = verify_findings(findings, build_source_index(zip_path), manifest)
            except ValueError as exc:
                notes.append(str(exc))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(output_path, index=False)
    write_report(matches, report_path, notes)
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify parsed public findings against Solidity sources in test.zip."
    )
    parser.add_argument("--findings", type=Path, default=DEFAULT_FINDINGS_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--test-zip", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    matches = run(
        findings_path=args.findings,
        manifest_path=args.manifest,
        zip_path=args.test_zip,
        output_path=args.output,
        report_path=args.report,
    )
    print(f"Verified findings: {len(matches)}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
