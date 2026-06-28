import json
import re
import zipfile
from collections import Counter
from pathlib import PurePosixPath

import pandas as pd

try:
    from config import (
        OFFICIAL_EXTERNAL_REPO_MATCHES_PATH,
        OFFICIAL_TEST_ZIP_PATH,
        TEST_CSV_PATH,
        UNMATCHED_TEST_REPO_IDENTITY_PATH,
        UNMATCHED_TEST_REPO_IDENTITY_REPORT_PATH,
        ensure_project_dirs,
    )
    from utils import normalize_text
except ImportError:
    from src.config import (
        OFFICIAL_EXTERNAL_REPO_MATCHES_PATH,
        OFFICIAL_TEST_ZIP_PATH,
        TEST_CSV_PATH,
        UNMATCHED_TEST_REPO_IDENTITY_PATH,
        UNMATCHED_TEST_REPO_IDENTITY_REPORT_PATH,
        ensure_project_dirs,
    )
    from src.utils import normalize_text


TEXT_SUFFIXES = {".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".js", ".ts", ".sol"}
METADATA_NAMES = {
    "package.json",
    "foundry.toml",
    "hardhat.config.js",
    "hardhat.config.ts",
    "brownie-config.yaml",
    "remappings.txt",
    "scope.txt",
    "contest.json",
}


def repo_hash_from_name(name: str) -> str | None:
    parts = name.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[0] == "test":
        return parts[1]
    return None


def rel_path(name: str) -> str:
    return "/".join(name.replace("\\", "/").split("/")[2:])


def read_text(zf: zipfile.ZipFile, info: zipfile.ZipInfo, max_bytes: int = 300_000) -> str:
    if info.file_size > max_bytes:
        return ""
    try:
        return zf.read(info).decode("utf-8", errors="replace")
    except Exception:
        return ""


def origin_from_git_config(text: str) -> tuple[str, str]:
    match = re.search(r"url\s*=\s*(\S+)", text, flags=re.I)
    if not match:
        return "", ""
    url = match.group(1).strip()
    slug_match = re.search(r"(?:github\.com[:/])([^/\s]+/[^/\s]+?)(?:\.git)?$", url, flags=re.I)
    if not slug_match:
        return url, ""
    owner_repo = slug_match.group(1)
    slug = owner_repo.split("/", 1)[1]
    return url, slug


def extract_package_names(text: str) -> list[str]:
    names = []
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("name"), str):
            names.append(data["name"])
    except Exception:
        pass
    return names


def extract_identifiers(text: str) -> dict[str, list[str]]:
    contracts = re.findall(r"\b(?:contract|interface|library)\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    functions = re.findall(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    imports = re.findall(r"\bimport\s+(?:[^;]*?\s+from\s+)?[\"']([^\"']+)[\"']", text)
    return {
        "contracts": sorted(set(contracts))[:80],
        "functions": sorted(set(functions))[:120],
        "imports": sorted(set(imports))[:80],
    }


def analyze_repo(zf: zipfile.ZipFile, repo_hash: str, infos: list[zipfile.ZipInfo]) -> dict:
    rels = [rel_path(info.filename) for info in infos if not info.is_dir()]
    suffix_counts = Counter(PurePosixPath(path).suffix.lower() or "<none>" for path in rels)
    metadata_snippets = {}
    package_names = []
    all_contracts: list[str] = []
    all_functions: list[str] = []
    all_imports: list[str] = []
    origin_url = ""
    origin_slug = ""
    readme_titles = []
    report_like_files = []

    for info in infos:
        rel = rel_path(info.filename)
        name = PurePosixPath(rel).name
        lower = rel.lower()
        if lower == ".git/config":
            text = read_text(zf, info)
            origin_url, origin_slug = origin_from_git_config(text)
            metadata_snippets[".git/config"] = normalize_text(text[:500])
            continue
        if name.lower() in METADATA_NAMES or name.lower().startswith("readme"):
            text = read_text(zf, info)
            if text:
                metadata_snippets[rel] = normalize_text(text[:1200])
                package_names.extend(extract_package_names(text))
                if name.lower().startswith("readme"):
                    for line in text.splitlines():
                        clean = normalize_text(re.sub(r"^[#*\-\s]+", "", line))
                        if clean and len(clean) > 4:
                            readme_titles.append(clean[:160])
                            break
        if any(token in lower for token in ["4naly3er", "bot-report", "slither", "report"]):
            report_like_files.append(rel)
        if PurePosixPath(rel).suffix.lower() == ".sol":
            text = read_text(zf, info, max_bytes=700_000)
            ids = extract_identifiers(text)
            all_contracts.extend(ids["contracts"])
            all_functions.extend(ids["functions"])
            all_imports.extend(ids["imports"])

    path_tokens = sorted(
        {
            token.lower()
            for path in rels
            for token in re.split(r"[^A-Za-z0-9]+", path)
            if len(token) >= 4 and token.lower() not in {"contracts", "contract", "test", "tests", "src", "lib", "node", "modules"}
        }
    )
    identity_text = " ".join(
        [
            origin_slug,
            origin_url,
            " ".join(package_names),
            " ".join(readme_titles[:5]),
            " ".join(sorted(set(all_contracts))[:80]),
            " ".join(path_tokens[:200]),
        ]
    )
    return {
        "repo_path": repo_hash,
        "file_count": len(rels),
        "byte_count": int(sum(info.file_size for info in infos)),
        "solidity_file_count": int(suffix_counts.get(".sol", 0)),
        "source_like_file_count": int(sum(suffix_counts.get(s, 0) for s in [".sol", ".ts", ".js", ".go", ".rs", ".py"])),
        "origin_url": origin_url,
        "origin_slug": origin_slug,
        "package_names_json": json.dumps(sorted(set(package_names)), ensure_ascii=False),
        "readme_titles_json": json.dumps(readme_titles[:8], ensure_ascii=False),
        "contracts_json": json.dumps(sorted(set(all_contracts))[:120], ensure_ascii=False),
        "functions_json": json.dumps(sorted(set(all_functions))[:160], ensure_ascii=False),
        "imports_json": json.dumps(sorted(set(all_imports))[:120], ensure_ascii=False),
        "report_like_files_json": json.dumps(sorted(set(report_like_files))[:80], ensure_ascii=False),
        "path_tokens_json": json.dumps(path_tokens[:250], ensure_ascii=False),
        "top_suffix_counts_json": json.dumps(dict(suffix_counts.most_common(20)), ensure_ascii=False),
        "identity_text": normalize_text(identity_text),
    }


def main() -> int:
    ensure_project_dirs()
    if not OFFICIAL_TEST_ZIP_PATH.exists():
        raise FileNotFoundError(f"Missing test zip: {OFFICIAL_TEST_ZIP_PATH}")
    test = pd.read_csv(TEST_CSV_PATH)
    matches = pd.read_parquet(OFFICIAL_EXTERNAL_REPO_MATCHES_PATH)
    very_high = set(matches[(matches["split"] == "test") & (matches["confidence"] == "very_high")]["repo_path"].astype(str))
    target_repos = sorted(set(test["repo_path"].astype(str)) - very_high)

    with zipfile.ZipFile(OFFICIAL_TEST_ZIP_PATH) as zf:
        groups: dict[str, list[zipfile.ZipInfo]] = {repo: [] for repo in target_repos}
        for info in zf.infolist():
            if info.is_dir():
                continue
            repo = repo_hash_from_name(info.filename)
            if repo in groups:
                groups[repo].append(info)
        rows = [analyze_repo(zf, repo, groups[repo]) for repo in target_repos]

    out = pd.DataFrame(rows)
    UNMATCHED_TEST_REPO_IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(UNMATCHED_TEST_REPO_IDENTITY_PATH, index=False)

    summary_cols = ["repo_path", "file_count", "solidity_file_count", "origin_slug", "origin_url"]
    summary_lines = ["| " + " | ".join(summary_cols) + " |", "| " + " | ".join(["---"] * len(summary_cols)) + " |"]
    for _, row in out[summary_cols].iterrows():
        summary_lines.append("| " + " | ".join(normalize_text(row[col]) for col in summary_cols) + " |")

    lines = [
        "# Unmatched Test Repo Identity Report",
        "",
        f"- non-very-high test repos: `{len(target_repos)}`",
        f"- output: `{UNMATCHED_TEST_REPO_IDENTITY_PATH}`",
        "",
        "## Summary",
        "",
        *summary_lines,
        "",
        "## Details",
    ]
    for _, row in out.iterrows():
        lines.extend(
            [
                "",
                f"### {row['repo_path']}",
                f"- origin_url: `{row['origin_url']}`",
                f"- origin_slug: `{row['origin_slug']}`",
                f"- files: `{row['file_count']}`, solidity: `{row['solidity_file_count']}`",
                f"- packages: `{row['package_names_json']}`",
                f"- readme_titles: `{row['readme_titles_json']}`",
                f"- report_like_files: `{row['report_like_files_json']}`",
                f"- contracts sample: `{row['contracts_json'][:800]}`",
            ]
        )
    UNMATCHED_TEST_REPO_IDENTITY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analyzed unmatched repos: {len(out)}")
    print(f"Saved: {UNMATCHED_TEST_REPO_IDENTITY_PATH}")
    print(f"Report: {UNMATCHED_TEST_REPO_IDENTITY_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
