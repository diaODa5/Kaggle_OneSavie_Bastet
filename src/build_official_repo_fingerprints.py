import hashlib
import json
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import pandas as pd

try:
    from config import (
        OFFICIAL_REPO_FINGERPRINT_REPORT_PATH,
        OFFICIAL_REPO_FINGERPRINTS_PATH,
        OFFICIAL_TEST_ZIP_PATH,
        OFFICIAL_TRAIN_ZIP_PATH,
        PROJECT_ROOT,
        TEST_CSV_PATH,
        TRAIN_CSV_PATH,
        ensure_project_dirs,
    )
    from build_external_fingerprints import (
        CHUNK_SIZE,
        GIT_CANDIDATE_FILES,
        GIT_NAME_PREFIXES,
        METADATA_FILENAMES,
        METADATA_SUFFIXES,
        SOURCE_SUFFIXES,
        has_generated_part,
    )
except ImportError:
    from src.config import (
        OFFICIAL_REPO_FINGERPRINT_REPORT_PATH,
        OFFICIAL_REPO_FINGERPRINTS_PATH,
        OFFICIAL_TEST_ZIP_PATH,
        OFFICIAL_TRAIN_ZIP_PATH,
        PROJECT_ROOT,
        TEST_CSV_PATH,
        TRAIN_CSV_PATH,
        ensure_project_dirs,
    )
    from src.build_external_fingerprints import (
        CHUNK_SIZE,
        GIT_CANDIDATE_FILES,
        GIT_NAME_PREFIXES,
        METADATA_FILENAMES,
        METADATA_SUFFIXES,
        SOURCE_SUFFIXES,
        has_generated_part,
    )


@dataclass
class RepoGroup:
    split: str
    repo_hash: str
    infos: list[zipfile.ZipInfo] = field(default_factory=list)


def normalize_zip_name(name: str) -> str:
    return name.replace("\\", "/").lstrip("/")


def repo_hash_from_name(name: str, split: str) -> str | None:
    parts = normalize_zip_name(name).split("/")
    if len(parts) >= 3 and parts[0] == split and parts[1]:
        return parts[1]
    return None


def relative_to_repo(name: str) -> str:
    parts = normalize_zip_name(name).split("/")
    return "/".join(parts[2:])


def display_path(path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def update_text(digest, text: str) -> None:
    digest.update(text.encode("utf-8"))


def path_list_hash(infos: list[zipfile.ZipInfo]) -> str:
    lines = [relative_to_repo(info.filename) for info in sorted(infos, key=lambda x: normalize_zip_name(x.filename))]
    return sha256_text("\n".join(lines) + ("\n" if lines else ""))


def manifest_hash(infos: list[zipfile.ZipInfo]) -> str:
    lines = [
        f"{relative_to_repo(info.filename)}\t{info.file_size}\t{info.CRC:08x}"
        for info in sorted(infos, key=lambda x: normalize_zip_name(x.filename))
    ]
    return sha256_text("\n".join(lines) + ("\n" if lines else ""))


def is_solidity(info: zipfile.ZipInfo) -> bool:
    return PurePosixPath(normalize_zip_name(info.filename)).suffix.lower() == ".sol"


def is_source_like(info: zipfile.ZipInfo) -> bool:
    rel = relative_to_repo(info.filename)
    if has_generated_part(rel):
        return False
    return PurePosixPath(rel).suffix.lower() in SOURCE_SUFFIXES


def is_metadata(info: zipfile.ZipInfo) -> bool:
    rel = relative_to_repo(info.filename)
    if has_generated_part(rel):
        return False
    path = PurePosixPath(rel)
    name = path.name.lower()
    if name in METADATA_FILENAMES:
        return True
    return "config" in name and path.suffix.lower() in METADATA_SUFFIXES | {".js", ".ts"}


def is_git_candidate(info: zipfile.ZipInfo) -> bool:
    rel = relative_to_repo(info.filename).lower()
    return rel in GIT_CANDIDATE_FILES or rel.startswith(GIT_NAME_PREFIXES)


def stream_content_hash(zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo]) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    bytes_read = 0
    files_read = 0
    for info in sorted(infos, key=lambda x: normalize_zip_name(x.filename)):
        update_text(digest, f"path:{relative_to_repo(info.filename)}\nsize:{info.file_size}\ncrc:{info.CRC:08x}\n")
        with zf.open(info) as src:
            while True:
                chunk = src.read(CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                bytes_read += len(chunk)
        update_text(digest, "\n--end-file--\n")
        files_read += 1
    return digest.hexdigest(), bytes_read, files_read


def cheap_file_sha256(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[str, int]:
    digest = hashlib.sha256()
    bytes_read = 0
    with zf.open(info) as src:
        while True:
            chunk = src.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
    return digest.hexdigest(), bytes_read


def add_row(rows: list[dict], split: str, repo_hash: str, name: str, value, source: str, bytes_read=0, file_count=0) -> None:
    rows.append(
        {
            "split": split,
            "repo_path": repo_hash,
            "fingerprint_name": name,
            "fingerprint_value": "" if value is None else str(value),
            "source": source,
            "bytes_read": int(bytes_read),
            "file_count": int(file_count),
        }
    )


def group_zip(zf: zipfile.ZipFile, split: str) -> dict[str, RepoGroup]:
    groups: dict[str, RepoGroup] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = normalize_zip_name(info.filename)
        if name.startswith("__MACOSX/"):
            continue
        repo_hash = repo_hash_from_name(name, split)
        if not repo_hash:
            continue
        groups.setdefault(repo_hash, RepoGroup(split=split, repo_hash=repo_hash)).infos.append(info)
    return groups


def build_rows_for_zip(zip_path, split: str) -> tuple[list[dict], dict[str, RepoGroup]]:
    rows: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        groups = group_zip(zf, split)
        print(f"{split}: grouped {len(groups)} repos from {display_path(zip_path)}")
        for idx, repo_hash in enumerate(sorted(groups), start=1):
            infos = groups[repo_hash].infos
            solidity_infos = [info for info in infos if is_solidity(info)]
            source_infos = [info for info in infos if is_source_like(info)]
            metadata_infos = [info for info in infos if is_metadata(info)]
            git_infos = [info for info in infos if is_git_candidate(info)]
            if idx == 1 or idx % 20 == 0:
                print(f"{split}: fingerprinting {idx}/{len(groups)} {repo_hash}")
            add_row(rows, split, repo_hash, "repo_hash_raw", repo_hash, "hash", 0, 1)
            add_row(rows, split, repo_hash, "repo_path_list_sha256", path_list_hash(infos), "path_list", 0, len(infos))
            add_row(rows, split, repo_hash, "repo_manifest_sha256", manifest_hash(infos), "manifest", 0, len(infos))
            add_row(rows, split, repo_hash, "solidity_path_list_sha256", path_list_hash(solidity_infos), "solidity_paths", 0, len(solidity_infos))
            value, bytes_read, file_count = stream_content_hash(zf, solidity_infos)
            add_row(rows, split, repo_hash, "solidity_content_aggregate_sha256", value, "solidity_content", bytes_read, file_count)
            add_row(rows, split, repo_hash, "source_like_path_list_sha256", path_list_hash(source_infos), "source_like_paths", 0, len(source_infos))
            value, bytes_read, file_count = stream_content_hash(zf, source_infos)
            add_row(rows, split, repo_hash, "source_like_content_aggregate_sha256", value, "source_like_content", bytes_read, file_count)
            value, bytes_read, file_count = stream_content_hash(zf, metadata_infos)
            add_row(rows, split, repo_hash, "package_config_metadata_aggregate_sha256", value, "package_config_metadata", bytes_read, file_count)
            git_name_infos = [info for info in git_infos if relative_to_repo(info.filename).lower().startswith(GIT_NAME_PREFIXES)]
            add_row(rows, split, repo_hash, "git_refs_logs_name_list_sha256", path_list_hash(git_name_infos), "git_names", 0, len(git_name_infos))
            for info in sorted(git_infos, key=lambda x: normalize_zip_name(x.filename)):
                rel = relative_to_repo(info.filename).lower()
                if rel in GIT_CANDIDATE_FILES:
                    value, bytes_read = cheap_file_sha256(zf, info)
                    name = "git_" + rel.removeprefix(".git/").replace("/", "_").replace("-", "_") + "_sha256"
                    add_row(rows, split, repo_hash, name, value, "git_metadata", bytes_read, 1)
            suffix_counts = Counter(PurePosixPath(relative_to_repo(info.filename)).suffix.lower() or "<none>" for info in infos)
            stats = {
                "repo_file_count": len(infos),
                "repo_bytes": sum(info.file_size for info in infos),
                "solidity_file_count": len(solidity_infos),
                "source_like_file_count": len(source_infos),
                "metadata_file_count": len(metadata_infos),
                "git_candidate_file_count": len(git_infos),
                "top_suffix_counts": dict(suffix_counts.most_common(20)),
            }
            add_row(rows, split, repo_hash, "aggregate_stats_json", json.dumps(stats, sort_keys=True), "aggregate_stats", 0, len(infos))
            for stat_name, stat_value in stats.items():
                if stat_name != "top_suffix_counts":
                    add_row(rows, split, repo_hash, stat_name, stat_value, "aggregate_stats", 0, len(infos))
    return rows, groups


def main() -> int:
    ensure_project_dirs()
    for path in [OFFICIAL_TRAIN_ZIP_PATH, OFFICIAL_TEST_ZIP_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing official repo zip: {path}")
    all_rows = []
    group_summary = {}
    for split, path in [("train", OFFICIAL_TRAIN_ZIP_PATH), ("test", OFFICIAL_TEST_ZIP_PATH)]:
        rows, groups = build_rows_for_zip(path, split)
        all_rows.extend(rows)
        group_summary[split] = groups
    df = pd.DataFrame(all_rows)
    OFFICIAL_REPO_FINGERPRINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OFFICIAL_REPO_FINGERPRINTS_PATH, index=False)

    train_csv = pd.read_csv(TRAIN_CSV_PATH)
    test_csv = pd.read_csv(TEST_CSV_PATH)
    train_dirs = set(group_summary["train"].keys())
    test_dirs = set(group_summary["test"].keys())
    train_expected = set(train_csv["repo_path"].astype(str))
    test_expected = set(test_csv["repo_path"].astype(str))
    lines = [
        "# Official Repo Fingerprint Report",
        "",
        f"- train zip: `{display_path(OFFICIAL_TRAIN_ZIP_PATH)}`",
        f"- test zip: `{display_path(OFFICIAL_TEST_ZIP_PATH)}`",
        f"- fingerprint rows: `{len(df)}`",
        f"- train repo dirs: `{len(train_dirs)}`",
        f"- test repo dirs: `{len(test_dirs)}`",
        f"- train dirs match train.csv: `{train_dirs == train_expected}`",
        f"- test dirs match test.csv: `{test_dirs == test_expected}`",
        f"- missing train dirs: `{sorted(train_expected - train_dirs)}`",
        f"- missing test dirs: `{sorted(test_expected - test_dirs)}`",
        f"- output: `{display_path(OFFICIAL_REPO_FINGERPRINTS_PATH)}`",
    ]
    OFFICIAL_REPO_FINGERPRINT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OFFICIAL_REPO_FINGERPRINTS_PATH} rows={len(df)}")
    print(f"Report: {OFFICIAL_REPO_FINGERPRINT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
