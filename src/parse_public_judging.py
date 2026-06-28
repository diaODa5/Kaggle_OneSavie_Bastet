import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "external" / "public_judging"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "public_judging_findings.parquet"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "public_judging_parse_report.md"

OUTPUT_COLUMNS = [
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
    "raw_tag",
    "raw_subtag",
]

REJECT_TERMS = {
    "qa",
    "gas",
    "informational",
    "info",
    "low",
    "invalid",
    "duplicate",
    "withdrawn",
}
FINAL_TERMS = {
    "accepted",
    "confirmed",
    "final",
    "validated",
}
FILE_RE = re.compile(
    r"(?<![\w.-])((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:sol|vy|rs|go|move|ts|js|py|java|cpp|c|h))\b"
)
FUNCTION_RE = re.compile(
    r"\b(?:[A-Za-z_]\w*(?:::|\.))([A-Za-z_]\w*)\b|\b([A-Za-z_]\w*)\s*\("
)
URL_RE = re.compile(r"https?://[^\s)>]+")
CONTEST_RE = re.compile(r"(.+)-(judging|findings)(?:-[A-Za-z0-9._-]+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class Document:
    contest_dir: str
    relative_path: str
    source_path: str
    text: str

    @property
    def name(self) -> str:
        return PurePosixPath(self.relative_path).name


def portable_path(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _canonical_contest_dir(value: str) -> str | None:
    match = CONTEST_RE.fullmatch(value)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2).lower()}"


def _contest_parts(parts: Iterable[str]) -> tuple[str, str] | None:
    for part in parts:
        canonical = _canonical_contest_dir(part)
        if canonical:
            return part, canonical
    return None


def _discover_directory_documents(root: Path) -> list[Document]:
    documents: list[Document] = []
    contest_dirs = []
    if root.is_dir() and _canonical_contest_dir(root.name):
        contest_dirs.append(root)
    contest_dirs.extend(
        path
        for path in root.rglob("*")
        if path.is_dir() and _canonical_contest_dir(path.name)
    )
    seen = set()
    for contest_dir in sorted(contest_dirs):
        resolved = contest_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        for path in sorted(contest_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".json"}:
                continue
            relative = path.relative_to(contest_dir).as_posix()
            documents.append(
                Document(
                    contest_dir=_canonical_contest_dir(contest_dir.name) or contest_dir.name,
                    relative_path=relative,
                    source_path=portable_path(path),
                    text=_read_text(path),
                )
            )
    return documents


def _discover_zip_documents(root: Path) -> list[Document]:
    documents: list[Document] = []
    zip_paths = [root] if root.is_file() and root.suffix.lower() == ".zip" else sorted(root.rglob("*.zip"))
    for archive in zip_paths:
        try:
            with zipfile.ZipFile(archive) as zf:
                for info in sorted(zf.infolist(), key=lambda item: item.filename):
                    member = PurePosixPath(info.filename)
                    contest_info = _contest_parts(member.parts)
                    if info.is_dir() or not contest_info or member.suffix.lower() not in {".md", ".json"}:
                        continue
                    contest_root, contest_dir = contest_info
                    contest_index = member.parts.index(contest_root)
                    relative = PurePosixPath(*member.parts[contest_index + 1 :]).as_posix()
                    text = zf.read(info).decode("utf-8", errors="replace")
                    documents.append(
                        Document(
                            contest_dir=contest_dir,
                            relative_path=relative,
                            source_path=f"{portable_path(archive)}!{info.filename}",
                            text=text,
                        )
                    )
        except (OSError, zipfile.BadZipFile):
            continue
    return documents


def discover_documents(root: Path) -> list[Document]:
    if not root.exists():
        return []
    documents = _discover_zip_documents(root)
    if root.is_dir():
        documents.extend(_discover_directory_documents(root))
    return documents


def _platform_and_contest(contest_dir: str) -> tuple[str, str]:
    match = CONTEST_RE.fullmatch(contest_dir)
    if not match:
        return "", contest_dir
    platform = "sherlock" if match.group(2).lower() == "judging" else "code4rena"
    return platform, match.group(1)


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if not match:
        return {}, text
    metadata = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower()] = value.strip().strip("\"'")
    return metadata, text[match.end() :]


def _severity(value: Any) -> str:
    text = str(value or "").strip().lower()
    if re.search(r"(?:^|\b)(?:h|high|1\s*\(\s*high risk\s*\))(?:\b|$)", text):
        return "High"
    if re.search(r"(?:^|\b)(?:m|med|medium|2\s*\(\s*med risk\s*\))(?:\b|$)", text):
        return "Medium"
    return ""


def _labels(record: dict[str, Any]) -> list[str]:
    values = record.get("labels", [])
    if isinstance(values, str):
        return [values]
    if not isinstance(values, list):
        return []
    return [str(item.get("name", "")) if isinstance(item, dict) else str(item) for item in values]


def _is_rejected(values: Iterable[Any]) -> bool:
    tokens = set()
    for value in values:
        tokens.update(re.findall(r"[a-z]+", str(value or "").lower()))
    return bool(tokens & REJECT_TERMS)


def _is_final(values: Iterable[Any]) -> bool:
    tokens = set()
    for value in values:
        tokens.update(re.findall(r"[a-z]+", str(value or "").lower()))
    return bool(tokens & FINAL_TERMS)


def _clean_title(title: str) -> str:
    title = re.sub(r"^\s*\[+[HM]-\d+\]+\s*", "", title, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", title).strip(" -:#")


def _clean_description(text: str) -> str:
    text = re.sub(r"^\s*#+\s+.*$", "", text, count=1, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _raw_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        values = [
            str(item.get("name", "")) if isinstance(item, dict) else str(item)
            for item in value
        ]
        return ", ".join(item.strip() for item in values if item.strip())
    if isinstance(value, dict):
        return str(value.get("name", "")).strip()
    return str(value).strip()


def _raw_categories(values: dict[str, Any]) -> tuple[str, str]:
    raw_tag = _raw_label(values.get("tag")) or _raw_label(values.get("category"))
    raw_subtag = _raw_label(values.get("subtag")) or _raw_label(values.get("tags"))
    return raw_tag, raw_subtag


def _references(text: str) -> tuple[list[str], list[str]]:
    files = sorted(set(FILE_RE.findall(text)))
    functions = []
    for match in FUNCTION_RE.finditer(text):
        name = match.group(1) or match.group(2)
        if name and name not in {"if", "for", "while", "require", "assert", "return"}:
            functions.append(name)
    return files, sorted(set(functions))


def _record(
    document: Document,
    issue_id: Any,
    title: str,
    severity: str,
    description: str,
    source_url: str = "",
    source_suffix: str = "",
    raw_tag: str = "",
    raw_subtag: str = "",
) -> dict[str, Any]:
    platform, contest = _platform_and_contest(document.contest_dir)
    if not source_url:
        owner = "sherlock-audit" if platform == "sherlock" else "code-423n4"
        suffix = "judging" if platform == "sherlock" else "findings"
        source_url = f"https://github.com/{owner}/{contest}-{suffix}"
    files, functions = _references(f"{title}\n{description}")
    return {
        "platform": platform,
        "contest": contest,
        "issue_id": str(issue_id or ""),
        "title": _clean_title(title),
        "severity": severity,
        "description": _clean_description(description),
        "referenced_files": files,
        "referenced_functions": functions,
        "source_path": document.source_path + source_suffix,
        "source_url": source_url,
        "raw_tag": raw_tag,
        "raw_subtag": raw_subtag,
    }


def _parse_sherlock_readme(document: Document) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"(?im)^#\s+Issue\s+([HML]-\d+)\s*:\s*(.+?)\s*$"
    )
    matches = list(pattern.finditer(document.text))
    records = []
    for index, match in enumerate(matches):
        issue_id = match.group(1).upper()
        severity = _severity(issue_id)
        if not severity:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document.text)
        body = document.text[match.end() : end].strip()
        url_match = URL_RE.search(body)
        records.append(
            _record(
                document,
                issue_id,
                match.group(2),
                severity,
                body,
                url_match.group(0) if url_match else "",
                f"#{issue_id}",
            )
        )
    return records


def _parse_sherlock_accepted_file(document: Document, issue_id: str) -> dict[str, Any] | None:
    metadata, body = _front_matter(document.text)
    severity = _severity(metadata.get("severity") or issue_id)
    if not severity or _is_rejected([metadata.get("status"), metadata.get("severity")]):
        return None
    heading = re.search(r"(?m)^#\s+(.+?)\s*$", body)
    title = metadata.get("title") or (heading.group(1) if heading else PurePosixPath(document.name).stem)
    url_match = URL_RE.search(body)
    url = metadata.get("url", "") or (url_match.group(0) if url_match else "")
    raw_tag, raw_subtag = _raw_categories(metadata)
    return _record(
        document,
        issue_id,
        title,
        severity,
        body,
        url,
        raw_tag=raw_tag,
        raw_subtag=raw_subtag,
    )


def _parse_sherlock(documents: list[Document]) -> list[dict[str, Any]]:
    accepted = []
    for document in documents:
        path = PurePosixPath(document.relative_path)
        if len(path.parts) < 2 or not re.fullmatch(r"\d+-[HM]", path.parent.name, re.IGNORECASE):
            continue
        if not re.search(r"-(?:best|report)\.md$", path.name, re.IGNORECASE):
            continue
        record = _parse_sherlock_accepted_file(document, path.parent.name.upper())
        if record:
            accepted.append(record)
    if accepted:
        return accepted

    readmes = [
        doc
        for doc in documents
        if doc.relative_path.lower() == "readme.md" and re.search(r"(?im)^#\s+Issue\s+[HML]-\d+\s*:", doc.text)
    ]
    if readmes:
        records = []
        for document in readmes:
            records.extend(_parse_sherlock_readme(document))
        return records

    records = []
    for document in documents:
        if document.name.lower() == "readme.md" or not document.name.lower().endswith(".md"):
            continue
        metadata, _ = _front_matter(document.text)
        if str(metadata.get("status", "")).lower() not in {"accepted", "valid", "validated", "confirmed"}:
            continue
        issue_id = metadata.get("id") or PurePosixPath(document.name).stem
        record = _parse_sherlock_accepted_file(document, str(issue_id))
        if record:
            records.append(record)
    return records


def _parse_code4rena_report(document: Document) -> list[dict[str, Any]]:
    heading_re = re.compile(
        r"(?im)^#{1,3}\s+"
        r"(?:\[\[([HM]-\d+)\]\s*(.+?)\]\((https?://[^)]+)\)"
        r"|\[([HM]-\d+)\]\s*(.+?)"
        r"|([HM]-\d+)\s*:\s*(.+?))\s*$"
    )
    matches = list(heading_re.finditer(document.text))
    records = []
    for index, match in enumerate(matches):
        issue_id = next(value for value in (match.group(1), match.group(4), match.group(6)) if value)
        title = next(value for value in (match.group(2), match.group(5), match.group(7)) if value)
        url = match.group(3) or ""
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document.text)
        body = document.text[match.end() : end].strip()
        records.append(
            _record(document, issue_id.upper(), title, _severity(issue_id), body, url, f"#{issue_id.upper()}")
        )
    return records


def _json_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    for key in ("issues", "findings", "data", "results"):
        if key in value:
            return _json_records(value[key])
    if any(key in value for key in ("title", "body", "severity", "number")):
        return [value]
    records = []
    for child in value.values():
        records.extend(_json_records(child))
    return records


def _parse_code4rena_json(document: Document) -> list[dict[str, Any]]:
    try:
        data = json.loads(document.text)
    except json.JSONDecodeError:
        return []
    findings = []
    for item in _json_records(data):
        labels = _labels(item)
        state = str(item.get("state") or "").strip().lower()
        status = str(item.get("status") or "").strip().lower()
        if state in {"open", "pending"} or status in {"open", "pending", "submitted"}:
            continue
        status_values = labels + [
            status,
            state,
            item.get("verdict"),
            item.get("risk"),
            item.get("severity"),
        ]
        if _is_rejected(status_values) or not _is_final(status_values):
            continue
        severity = _severity(item.get("severity") or item.get("risk") or " ".join(labels) or item.get("title"))
        if not severity:
            continue
        issue_id = item.get("number") or item.get("id") or item.get("issue_id")
        title = str(item.get("title") or item.get("name") or "")
        body = str(item.get("body") or item.get("description") or item.get("content") or "")
        source_url = str(item.get("html_url") or item.get("url") or "")
        raw_tag, raw_subtag = _raw_categories(item)
        findings.append(
            _record(
                document,
                issue_id,
                title,
                severity,
                body,
                source_url,
                f"#{issue_id}",
                raw_tag,
                raw_subtag,
            )
        )
    return findings


def _parse_code4rena(documents: list[Document]) -> list[dict[str, Any]]:
    reports = [
        document
        for document in documents
        if document.name.lower() == "report.md"
    ]
    report_records = []
    for document in reports:
        report_records.extend(_parse_code4rena_report(document))
    if report_records:
        return report_records

    records = []
    for document in documents:
        if document.name.lower().endswith(".json"):
            records.extend(_parse_code4rena_json(document))
    return records


def parse_public_judging(root: Path | str = DEFAULT_INPUT_DIR) -> pd.DataFrame:
    documents = discover_documents(Path(root))
    grouped: dict[str, list[Document]] = {}
    for document in documents:
        grouped.setdefault(document.contest_dir, []).append(document)

    records = []
    for contest_dir in sorted(grouped):
        platform, _ = _platform_and_contest(contest_dir)
        contest_documents = grouped[contest_dir]
        if platform == "sherlock":
            records.extend(_parse_sherlock(contest_documents))
        elif platform == "code4rena":
            records.extend(_parse_code4rena(contest_documents))

    valid = [
        record
        for record in records
        if record["severity"] in {"High", "Medium"}
        and record["title"]
        and record["description"]
    ]
    if not valid:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame = pd.DataFrame(valid, columns=OUTPUT_COLUMNS)
    frame = frame.drop_duplicates(
        subset=["platform", "contest", "issue_id", "title", "severity"],
        keep="first",
    )
    return frame.reset_index(drop=True)


def write_report(findings: pd.DataFrame, input_dir: Path, report_path: Path) -> None:
    counts = (
        findings.groupby(["platform", "contest", "severity"], dropna=False)
        .size()
        .reset_index(name="count")
        if not findings.empty
        else pd.DataFrame(columns=["platform", "contest", "severity", "count"])
    )
    lines = [
        "# Public Judging Parse Report",
        "",
        f"- Input directory: `{portable_path(input_dir)}`",
        f"- Parsed findings: `{len(findings)}`",
        f"- Contests with findings: `{findings['contest'].nunique() if not findings.empty else 0}`",
        f"- High findings: `{int((findings['severity'] == 'High').sum()) if not findings.empty else 0}`",
        f"- Medium findings: `{int((findings['severity'] == 'Medium').sum()) if not findings.empty else 0}`",
        "",
        "## Per-Contest Counts",
        "",
        "| Platform | Contest | Severity | Count |",
        "| --- | --- | --- | ---: |",
    ]
    for row in counts.itertuples(index=False):
        lines.append(f"| {row.platform} | {row.contest} | {row.severity} | {row.count} |")
    if counts.empty:
        lines.append("| - | - | - | 0 |")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    input_dir: Path | str = DEFAULT_INPUT_DIR,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    report_path: Path | str = DEFAULT_REPORT_PATH,
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    report_path = Path(report_path)
    findings = parse_public_judging(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    findings.to_parquet(output_path, index=False)
    write_report(findings, input_dir, report_path)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse cached Sherlock and Code4rena findings.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    findings = run(args.input_dir, args.output, args.report)
    print(f"Parsed findings: {len(findings)}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
