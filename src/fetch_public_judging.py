import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "public_judging_manifest.csv"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "external" / "public_judging"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "public_judging_fetch_report.md"

METADATA_SUFFIXES = {".md", ".markdown", ".json", ".txt"}
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_MEMBER_BYTES = 20 * 1024 * 1024
MAX_EXTRACTED_BYTES = 200 * 1024 * 1024
USER_AGENT = "OneSavie-Bastet-public-judging-fetcher/1.0"
URL_COLUMNS = (
    "candidate_public_url",
    "public_judging_url",
    "judging_url",
    "findings_url",
    "public_repo_url",
    "public_url",
)


class UnsafeArchiveError(ValueError):
    pass


def _require_positive_timeout(timeout) -> None:
    if timeout <= 0:
        raise ValueError("timeout must be positive")


def _positive_timeout(value: str) -> int:
    timeout = int(value)
    if timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return timeout


def normalize_github_repo_url(url: str) -> tuple[str, str] | None:
    value = str(url or "").strip()
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme.lower() != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
    ):
        return None
    http_match = re.match(
        r"https://github\.com/([^/]+)/([^/#?]+?)(?:\.git)?(?:[/#?].*)?$",
        value,
        flags=re.I,
    )
    if http_match:
        return http_match.group(1), re.sub(r"\.git$", "", http_match.group(2), flags=re.I)
    return None


def archive_candidates(repository_url: str) -> list[str]:
    value = str(repository_url or "").strip()
    if not value:
        return []
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme.lower() != "https"
        or parsed.netloc.lower() not in {"github.com", "codeload.github.com"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
    ):
        return []
    if re.match(r"https://codeload\.github\.com/[^/]+/[^/]+/zip/", value, flags=re.I):
        return [value]
    if re.match(r"https://github\.com/[^/]+/[^/]+/archive/", value, flags=re.I):
        return [value]
    repo = normalize_github_repo_url(value)
    if repo is None:
        return [value] if value.lower().endswith(".zip") else []
    owner, name = repo
    return [
        f"https://codeload.github.com/{owner}/{name}/zip/refs/heads/main",
        f"https://codeload.github.com/{owner}/{name}/zip/refs/heads/master",
        f"https://github.com/{owner}/{name}/archive/refs/heads/main.zip",
        f"https://github.com/{owner}/{name}/archive/refs/heads/master.zip",
    ]


def manifest_url(row: dict) -> str:
    for column in URL_COLUMNS:
        value = str(row.get(column, "") or "").strip()
        if value:
            return value
    return ""


def contest_key(row: dict) -> str:
    raw = (
        row.get("contest_slug")
        or row.get("judging_slug")
        or row.get("origin_slug")
        or row.get("repo_path")
        or "contest"
    )
    key = re.sub(r"[^A-Za-z0-9._-]+", "-", str(raw).strip()).strip(".-").lower()
    source_hash = hashlib.sha256(_source_identity(row).encode("utf-8")).hexdigest()[:12]
    return f"{key or 'contest'}-{source_hash}"


def _source_identity(row: dict) -> str:
    return (
        manifest_url(row)
        or str(row.get("repo_path", "") or "").strip()
        or str(row.get("contest_slug", "") or "").strip()
        or str(row.get("judging_slug", "") or "").strip()
        or str(row.get("origin_slug", "") or "").strip()
        or "contest"
    )


def _safe_member_path(name: str) -> PurePosixPath:
    normalized = str(name).replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise UnsafeArchiveError(f"unsafe ZIP member path: {name}")
    return path


def _validate_archive(zf: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    selected = []
    seen = set()
    total_bytes = 0
    for info in zf.infolist():
        path = _safe_member_path(info.filename)
        unix_mode = info.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise UnsafeArchiveError(f"symbolic link ZIP member is not allowed: {info.filename}")
        if info.is_dir():
            continue
        normalized = path.as_posix()
        if normalized in seen:
            raise UnsafeArchiveError(f"duplicate ZIP member path: {info.filename}")
        seen.add(normalized)
        if info.file_size > MAX_MEMBER_BYTES:
            raise UnsafeArchiveError(f"ZIP member exceeds size limit: {info.filename}")
        if path.suffix.lower() not in METADATA_SUFFIXES:
            continue
        total_bytes += info.file_size
        if total_bytes > MAX_EXTRACTED_BYTES:
            raise UnsafeArchiveError("selected ZIP metadata exceeds extraction size limit")
        selected.append((info, path))
    return selected


def extract_metadata(archive_path: Path, destination: Path) -> dict:
    archive_path = Path(archive_path)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"extraction destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive_path) as zf:
            selected = _validate_archive(zf)
            if not selected:
                raise UnsafeArchiveError("ZIP contains no extractable metadata files")
            for info, relative_path in selected:
                target = temp_dir.joinpath(*relative_path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            total_files = sum(1 for info in zf.infolist() if not info.is_dir())
        temp_dir.replace(destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {
        "extracted_files": len(selected),
        "skipped_files": total_files - len(selected),
        "extracted_bytes": sum(info.file_size for info, _ in selected),
    }


def _default_opener(url: str, timeout: int):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/zip, application/octet-stream;q=0.9, */*;q=0.1",
        },
    )
    return urllib.request.urlopen(request, timeout=timeout)


def _response_content_length(response) -> int | None:
    headers = getattr(response, "headers", None)
    value = headers.get("Content-Length") if headers is not None else None
    if value is None:
        getheader = getattr(response, "getheader", None)
        value = getheader("Content-Length") if getheader is not None else None
    if value is None:
        return None
    length = int(value)
    if length < 0:
        raise ValueError("Content-Length must not be negative")
    return length


def _read_response(response, deadline: float | None = None) -> bytes:
    try:
        content_length = _response_content_length(response)
        if content_length is not None and content_length > MAX_ARCHIVE_BYTES:
            raise ValueError(f"download exceeds {MAX_ARCHIVE_BYTES}-byte size limit")
        chunks = []
        total = 0
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("download deadline exceeded")
            chunk = response.read(min(1024 * 1024, MAX_ARCHIVE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError(f"download exceeds {MAX_ARCHIVE_BYTES}-byte size limit")
        payload = b"".join(chunks)
        if content_length is not None and len(payload) != content_length:
            raise ValueError(
                f"Content-Length mismatch: expected {content_length} bytes, received {len(payload)}"
            )
        return payload
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            close()


def _write_temp_archive(payload: bytes, archive_dir: Path, key: str) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{key}.", suffix=".zip", dir=archive_dir)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def _archive_metadata_path(archive_path: Path) -> Path:
    return Path(archive_path).with_suffix(".json")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_archive_metadata(
    metadata_path: Path,
    source_url: str,
    sha256: str,
    archive_url: str,
) -> None:
    metadata_path.write_text(
        json.dumps(
            {"url": source_url, "sha256": sha256, "archive_url": archive_url},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _unlink_if_same(path: Path, expected: Path) -> None:
    try:
        if path.exists() and os.path.samefile(path, expected):
            path.unlink()
    except FileNotFoundError:
        pass


def _publish_archive_bundle(
    temp_archive: Path,
    archive_path: Path,
    source_url: str,
    sha256: str,
    archive_url: str,
) -> None:
    metadata_path = _archive_metadata_path(archive_path)
    fd, metadata_name = tempfile.mkstemp(
        prefix=f".{archive_path.stem}.",
        suffix=".json",
        dir=archive_path.parent,
    )
    os.close(fd)
    temp_metadata = Path(metadata_name)
    archive_published = False
    metadata_published = False
    try:
        _write_archive_metadata(temp_metadata, source_url, sha256, archive_url)
        os.link(temp_archive, archive_path)
        archive_published = True
        os.link(temp_metadata, metadata_path)
        metadata_published = True
    except Exception:
        if metadata_published:
            _unlink_if_same(metadata_path, temp_metadata)
        if archive_published:
            _unlink_if_same(archive_path, temp_archive)
        raise
    finally:
        temp_metadata.unlink(missing_ok=True)


def _validate_cached_archive(
    archive_path: Path,
    extracted_path: Path,
    source_url: str,
) -> dict:
    metadata_path = _archive_metadata_path(archive_path)
    if not metadata_path.is_file():
        raise ValueError(f"cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("url") != source_url:
        raise ValueError("cache source URL does not match manifest")
    actual_sha256 = _sha256_file(archive_path)
    if metadata.get("sha256") != actual_sha256:
        raise ValueError("cache SHA256 does not match archive")
    with zipfile.ZipFile(archive_path) as zf:
        selected = _validate_archive(zf)
        if not selected:
            raise UnsafeArchiveError("ZIP contains no extractable metadata files")
    if extracted_path.exists():
        if not any(path.is_file() for path in extracted_path.rglob("*")):
            raise ValueError("cached extraction is empty")
        return {}
    return extract_metadata(archive_path, extracted_path)


def _base_result(row: dict, key: str) -> dict:
    return {
        "key": key,
        "repo_path": str(row.get("repo_path", "") or ""),
        "contest_slug": str(row.get("contest_slug", "") or key),
        "platform": str(row.get("platform", "") or ""),
        "status": "",
        "source": "",
        "url": manifest_url(row),
        "archive_path": "",
        "extracted_path": "",
        "attempts": [],
    }


def fetch_contest(row: dict, cache_dir: Path, opener=None, timeout: int = 30) -> dict:
    _require_positive_timeout(timeout)
    cache_dir = Path(cache_dir)
    opener = opener or _default_opener
    key = contest_key(row)
    result = _base_result(row, key)
    archive_path = cache_dir / "archives" / f"{key}.zip"
    extracted_path = cache_dir / "extracted" / key
    dropin_path = cache_dir / "dropins" / f"{key}.zip"

    if archive_path.exists():
        try:
            summary = _validate_cached_archive(archive_path, extracted_path, manifest_url(row))
            result.update(summary)
            result.update(
                status="cached",
                source="archive_cache",
                archive_path=str(archive_path),
                extracted_path=str(extracted_path),
            )
        except Exception as exc:
            result.update(status="failed", source="archive_cache", error=f"{type(exc).__name__}: {exc}")
        return result

    if dropin_path.exists():
        temp_archive = None
        extraction_committed = False
        try:
            temp_archive = _write_temp_archive(dropin_path.read_bytes(), archive_path.parent, key)
            summary = extract_metadata(temp_archive, extracted_path)
            extraction_committed = True
            _publish_archive_bundle(
                temp_archive,
                archive_path,
                manifest_url(row),
                _sha256_file(temp_archive),
                str(dropin_path),
            )
            result.update(
                status="fetched",
                source="local_dropin",
                archive_path=str(archive_path),
                extracted_path=str(extracted_path),
                **summary,
            )
            return result
        except Exception as exc:
            if extraction_committed:
                shutil.rmtree(extracted_path, ignore_errors=True)
            result["attempts"].append(
                {"url": str(dropin_path), "error": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            if temp_archive is not None:
                temp_archive.unlink(missing_ok=True)

    source_url = manifest_url(row)
    candidates = archive_candidates(source_url)
    if not source_url or not candidates:
        result.update(status="unavailable", error="no supported judging/findings repository URL")
        return result

    for url in candidates:
        temp_archive = None
        extraction_committed = False
        try:
            deadline = time.monotonic() + timeout
            response = opener(url, timeout)
            payload = _read_response(response, deadline=deadline)
            temp_archive = _write_temp_archive(payload, archive_path.parent, key)
            if not zipfile.is_zipfile(temp_archive):
                raise zipfile.BadZipFile("downloaded response is not a ZIP archive")
            summary = extract_metadata(temp_archive, extracted_path)
            extraction_committed = True
            _publish_archive_bundle(
                temp_archive,
                archive_path,
                source_url,
                _sha256_bytes(payload),
                url,
            )
            result["attempts"].append({"url": url, "error": ""})
            result.update(
                status="fetched",
                source="network",
                url=url,
                archive_path=str(archive_path),
                extracted_path=str(extracted_path),
                **summary,
            )
            return result
        except Exception as exc:
            if extraction_committed:
                shutil.rmtree(extracted_path, ignore_errors=True)
            result["attempts"].append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if temp_archive is not None:
                temp_archive.unlink(missing_ok=True)

    result.update(status="failed", error="all archive candidates failed")
    return result


def read_manifest(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as manifest_file:
        return list(csv.DictReader(manifest_file))


def _markdown_cell(value) -> str:
    return str(value or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def write_report(results: list[dict], report_path: Path, manifest_path: Path, note: str = "") -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(result.get("status", "unknown") for result in results)
    lines = [
        "# Public Judging Fetch Report",
        "",
        f"- manifest: `{manifest_path}`",
        f"- contests: `{len(results)}`",
        f"- fetched: `{counts['fetched']}`",
        f"- cached: `{counts['cached']}`",
        f"- unavailable: `{counts['unavailable']}`",
        f"- failed: `{counts['failed']}`",
    ]
    if note:
        lines.extend(["", f"- status: `unavailable`", f"- detail: {_markdown_cell(note)}"])
    lines.extend(
        [
            "",
            "## Contests",
            "",
            "| contest | platform | status | source | URL / error |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for result in results:
        detail = result.get("error", "") or result.get("url", "")
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(result.get("contest_slug") or result.get("key")),
                    _markdown_cell(result.get("platform")),
                    _markdown_cell(result.get("status")),
                    _markdown_cell(result.get("source")),
                    _markdown_cell(detail),
                ]
            )
            + " |"
        )
    failures = [
        (result, attempt)
        for result in results
        for attempt in result.get("attempts", [])
        if attempt.get("error")
    ]
    if failures:
        lines.extend(
            [
                "",
                "## Failed Attempts",
                "",
                "| contest | URL | error |",
                "| --- | --- | --- |",
            ]
        )
        for result, attempt in failures:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(result.get("contest_slug") or result.get("key")),
                        _markdown_cell(attempt.get("url")),
                        _markdown_cell(attempt.get("error")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Local Drop-ins",
            "",
            "Place a ZIP at `data/external/public_judging/dropins/<contest-key>.zip` and rerun.",
            "A successful archive is cached immutably under `archives/`; selected metadata is under `extracted/`.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fetch(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    opener=None,
    timeout: int = 30,
) -> list[dict]:
    _require_positive_timeout(timeout)
    manifest_path = Path(manifest_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "archives").mkdir(exist_ok=True)
    (cache_dir / "dropins").mkdir(exist_ok=True)
    (cache_dir / "extracted").mkdir(exist_ok=True)
    if not manifest_path.exists():
        note = f"manifest not found: {manifest_path}"
        write_report([], report_path, manifest_path, note=note)
        print(note)
        print(f"Report: {report_path}")
        return []

    rows = read_manifest(manifest_path)
    results = []
    seen_sources = set()
    for row in rows:
        source = _source_identity(row)
        if source in seen_sources:
            result = _base_result(row, contest_key(row))
            result.update(status="failed", error=f"duplicate source in manifest: {source}")
        else:
            seen_sources.add(source)
            result = fetch_contest(row, cache_dir, opener=opener, timeout=timeout)
        results.append(result)
    write_report(results, report_path, manifest_path)
    counts = Counter(result["status"] for result in results)
    print(
        "Fetch summary: "
        + ", ".join(f"{status}={counts[status]}" for status in ("fetched", "cached", "unavailable", "failed"))
    )
    print(f"Report: {report_path}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and cache public judging repository archives.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--timeout", type=_positive_timeout, default=30)
    args = parser.parse_args()
    run_fetch(args.manifest, args.cache_dir, args.report, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
