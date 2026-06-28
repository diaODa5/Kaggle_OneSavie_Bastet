import re
import zlib
import zipfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath

import pandas as pd

try:
    from config import (
        EXTERNAL_BASTET_MAPPED_FINDINGS_PATH,
        EXTERNAL_BASTET_V0_DIR,
        OFFICIAL_EXTERNAL_REPO_MATCHES_PATH,
        OFFICIAL_TEST_ZIP_PATH,
        TEST_CSV_PATH,
        UNMATCHED_TEST_REPO_FILE_OVERLAP_PATH,
        UNMATCHED_TEST_REPO_FILE_OVERLAP_REPORT_PATH,
        ensure_project_dirs,
    )
    from utils import normalize_text
except ImportError:
    from src.config import (
        EXTERNAL_BASTET_MAPPED_FINDINGS_PATH,
        EXTERNAL_BASTET_V0_DIR,
        OFFICIAL_EXTERNAL_REPO_MATCHES_PATH,
        OFFICIAL_TEST_ZIP_PATH,
        TEST_CSV_PATH,
        UNMATCHED_TEST_REPO_FILE_OVERLAP_PATH,
        UNMATCHED_TEST_REPO_FILE_OVERLAP_REPORT_PATH,
        ensure_project_dirs,
    )
    from src.utils import normalize_text


SOURCE_SUFFIXES = {
    ".sol",
    ".vy",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".py",
    ".rs",
    ".go",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".sh",
    ".md",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".csv",
}
DEPENDENCY_PARTS = {
    "lib",
    "libs",
    "node_modules",
    "dependencies",
    "deps",
    "vendor",
    "third_party",
    "forge-std",
    "openzeppelin-contracts",
    "openzeppelin",
    "solmate",
    "ds-test",
}
GENERATED_PARTS = {"out", "cache", "artifacts", "build", "dist", "coverage", "broadcast", ".git", "__pycache__"}


def repo_hash_from_zip_name(name: str) -> str | None:
    parts = name.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[0] == "test":
        return parts[1]
    return None


def rel_from_zip_name(name: str) -> str:
    return "/".join(name.replace("\\", "/").split("/")[2:])


def is_source_path(path: str) -> bool:
    parts = [p.lower() for p in path.replace("\\", "/").split("/") if p]
    if any(part in GENERATED_PARTS for part in parts):
        return False
    return PurePosixPath(path).suffix.lower() in SOURCE_SUFFIXES


def is_dependency_path(path: str) -> bool:
    parts = [p.lower() for p in path.replace("\\", "/").split("/") if p]
    return any(part in DEPENDENCY_PARTS for part in parts)


def path_variants(path: str) -> set[str]:
    path = path.replace("\\", "/").strip("/")
    parts = [p for p in path.split("/") if p]
    variants = {path.lower()}
    if len(parts) > 1:
        variants.add("/".join(parts[1:]).lower())
    if len(parts) > 2:
        variants.add("/".join(parts[2:]).lower())
    # Common official zips include a project wrapper such as optimism/ or token/.
    for i, part in enumerate(parts):
        if part.lower() in {"contracts", "src", "packages", "pkg", "op-geth", "optimism"}:
            variants.add("/".join(parts[i:]).lower())
            break
    return {v for v in variants if v}


def official_signatures(target_repos: set[str]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {repo: [] for repo in target_repos}
    with zipfile.ZipFile(OFFICIAL_TEST_ZIP_PATH) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            repo = repo_hash_from_zip_name(info.filename)
            if repo not in result:
                continue
            rel = rel_from_zip_name(info.filename)
            if not is_source_path(rel):
                continue
            sig = {
                "repo_path": repo,
                "path": rel,
                "size": int(info.file_size),
                "crc": f"{info.CRC:08x}",
                "is_solidity": PurePosixPath(rel).suffix.lower() == ".sol",
                "is_dependency": is_dependency_path(rel),
            }
            result[repo].append(sig)
    return result


def file_crc(path: Path) -> str:
    crc = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            crc = zlib.crc32(chunk, crc)
    return f"{crc & 0xFFFFFFFF:08x}"


def external_signatures() -> dict[str, list[dict]]:
    repos_dir = EXTERNAL_BASTET_V0_DIR / "repos"
    if not repos_dir.exists():
        raise FileNotFoundError(f"External repos directory is missing: {repos_dir}")
    result: dict[str, list[dict]] = {}
    for repo_dir in sorted(p for p in repos_dir.iterdir() if p.is_dir()):
        repo_path = f"repos/{repo_dir.name}"
        rows = []
        for path in repo_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(repo_dir).as_posix()
            if not is_source_path(rel):
                continue
            try:
                size = path.stat().st_size
                crc = file_crc(path)
            except OSError:
                continue
            rows.append(
                {
                    "external_repo_path": repo_path,
                    "path": rel,
                    "size": int(size),
                    "crc": crc,
                    "is_solidity": PurePosixPath(rel).suffix.lower() == ".sol",
                    "is_dependency": is_dependency_path(rel),
                }
            )
        result[repo_path] = rows
    return result


def build_external_index(ext_sigs: dict[str, list[dict]]) -> tuple[dict[tuple[str, int, str], list[dict]], dict[tuple[int, str], list[dict]]]:
    path_index: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    content_index: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for repo, rows in ext_sigs.items():
        for row in rows:
            for variant in path_variants(row["path"]):
                path_index[(variant, row["size"], row["crc"])].append(row)
            content_index[(row["size"], row["crc"])].append(row)
    return path_index, content_index


def path_similarity(a: str, b: str) -> float:
    av = path_variants(a)
    bv = path_variants(b)
    best = 0.0
    for x in av:
        for y in bv:
            if x == y:
                return 1.0
            best = max(best, SequenceMatcher(None, x, y).ratio())
    return best


def score_repo(official_rows: list[dict], matched_rows: list[dict]) -> dict:
    total_project = sum(1 for row in official_rows if not row["is_dependency"])
    total_sol_project = sum(1 for row in official_rows if row["is_solidity"] and not row["is_dependency"])
    project_matches = [row for row in matched_rows if not row["official_is_dependency"] and not row["external_is_dependency"]]
    solidity_project_matches = [row for row in project_matches if row["official_is_solidity"] and row["external_is_solidity"]]
    dep_matches = [row for row in matched_rows if row["official_is_dependency"] or row["external_is_dependency"]]
    project_ratio = len(project_matches) / max(total_project, 1)
    solidity_ratio = len(solidity_project_matches) / max(total_sol_project, 1)
    dep_penalty = min(len(dep_matches), 30) * 0.01
    score = len(project_matches) * 1.5 + len(solidity_project_matches) * 2.0 + project_ratio * 30 + solidity_ratio * 30 - dep_penalty
    return {
        "score": float(score),
        "project_match_count": int(len(project_matches)),
        "solidity_project_match_count": int(len(solidity_project_matches)),
        "dependency_match_count": int(len(dep_matches)),
        "project_file_count": int(total_project),
        "solidity_project_file_count": int(total_sol_project),
        "project_match_ratio": float(project_ratio),
        "solidity_project_match_ratio": float(solidity_ratio),
    }


def match_one_repo(
    repo: str,
    rows: list[dict],
    path_index: dict[tuple[str, int, str], list[dict]],
    content_index: dict[tuple[int, str], list[dict]],
) -> list[dict]:
    by_ext: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        seen = set()
        for variant in path_variants(row["path"]):
            for hit in path_index.get((variant, row["size"], row["crc"]), []):
                key = (hit["external_repo_path"], hit["path"])
                if key in seen:
                    continue
                seen.add(key)
                by_ext[hit["external_repo_path"]].append(
                    {
                        "official_path": row["path"],
                        "external_path": hit["path"],
                        "official_is_solidity": bool(row["is_solidity"]),
                        "external_is_solidity": bool(hit["is_solidity"]),
                        "official_is_dependency": bool(row["is_dependency"]),
                        "external_is_dependency": bool(hit["is_dependency"]),
                        "path_similarity": 1.0,
                    }
                )
        # Content-identical files can move under a different wrapper prefix.
        # Only count content-only matches when they are not pure dependencies
        # or when paths still look related.
        for hit in content_index.get((row["size"], row["crc"]), []):
            key = (hit["external_repo_path"], hit["path"])
            if key in seen:
                continue
            sim = path_similarity(row["path"], hit["path"])
            both_dependency = row["is_dependency"] and hit["is_dependency"]
            if both_dependency and sim < 0.80:
                continue
            if sim < 0.25 and (row["is_dependency"] or hit["is_dependency"]):
                continue
            seen.add(key)
            by_ext[hit["external_repo_path"]].append(
                {
                    "official_path": row["path"],
                    "external_path": hit["path"],
                    "official_is_solidity": bool(row["is_solidity"]),
                    "external_is_solidity": bool(hit["is_solidity"]),
                    "official_is_dependency": bool(row["is_dependency"]),
                    "external_is_dependency": bool(hit["is_dependency"]),
                    "path_similarity": float(sim),
                }
            )
    candidates = []
    for ext_repo, matched in by_ext.items():
        scored = score_repo(rows, matched)
        scored.update(
            {
                "repo_path": repo,
                "mapped_external_repo_path": ext_repo,
                "sample_matches": "; ".join(
                    f"{m['official_path']} -> {m['external_path']}" for m in matched[:8]
                ),
            }
        )
        if scored["project_match_count"] >= 30 and scored["project_match_ratio"] >= 0.70:
            confidence = "very_high"
        elif scored["project_match_count"] >= 20 and scored["project_match_ratio"] >= 0.40:
            confidence = "high"
        elif scored["project_match_count"] >= 8 and scored["project_match_ratio"] >= 0.20:
            confidence = "medium"
        elif scored["project_match_count"] >= 3:
            confidence = "low"
        else:
            confidence = "none"
        scored["confidence"] = confidence
        candidates.append(scored)
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def main() -> int:
    ensure_project_dirs()
    test = pd.read_csv(TEST_CSV_PATH)
    matches = pd.read_parquet(OFFICIAL_EXTERNAL_REPO_MATCHES_PATH)
    very_high = set(matches[(matches["split"] == "test") & (matches["confidence"] == "very_high")]["repo_path"].astype(str))
    target_repos = sorted(set(test["repo_path"].astype(str)) - very_high)
    official = official_signatures(set(target_repos))
    external = external_signatures()
    path_index, content_index = build_external_index(external)
    findings = pd.read_parquet(EXTERNAL_BASTET_MAPPED_FINDINGS_PATH)
    findings_counts = findings.groupby("repo_path").size().to_dict()

    rows = []
    for repo in target_repos:
        candidates = match_one_repo(repo, official.get(repo, []), path_index, content_index)
        if candidates:
            for rank, cand in enumerate(candidates[:5], start=1):
                cand["rank"] = rank
                cand["matched_findings_count"] = int(findings_counts.get(cand["mapped_external_repo_path"], 0))
                rows.append(cand)
        else:
            rows.append(
                {
                    "repo_path": repo,
                    "mapped_external_repo_path": "",
                    "rank": 1,
                    "confidence": "none",
                    "score": 0.0,
                    "project_match_count": 0,
                    "solidity_project_match_count": 0,
                    "dependency_match_count": 0,
                    "project_file_count": len([r for r in official.get(repo, []) if not r["is_dependency"]]),
                    "solidity_project_file_count": len([r for r in official.get(repo, []) if r["is_solidity"] and not r["is_dependency"]]),
                    "project_match_ratio": 0.0,
                    "solidity_project_match_ratio": 0.0,
                    "matched_findings_count": 0,
                    "sample_matches": "",
                }
            )

    out = pd.DataFrame(rows).sort_values(["repo_path", "rank"])
    UNMATCHED_TEST_REPO_FILE_OVERLAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(UNMATCHED_TEST_REPO_FILE_OVERLAP_PATH, index=False)

    top = out[out["rank"] == 1].copy()
    lines = [
        "# Unmatched Test Repo File Overlap Match Report",
        "",
        f"- target repos: `{len(target_repos)}`",
        f"- external repos scanned: `{len(external)}`",
        f"- output: `{UNMATCHED_TEST_REPO_FILE_OVERLAP_PATH}`",
        f"- top confidence counts: `{top['confidence'].value_counts().to_dict()}`",
        "",
        "## Top Matches",
        "",
        "| repo_path | mapped_external_repo_path | confidence | findings | project_matches | project_ratio | solidity_matches | solidity_ratio | score |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in top.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    normalize_text(row["repo_path"]),
                    normalize_text(row["mapped_external_repo_path"]),
                    normalize_text(row["confidence"]),
                    str(int(row["matched_findings_count"])),
                    str(int(row["project_match_count"])),
                    f"{float(row['project_match_ratio']):.3f}",
                    str(int(row["solidity_project_match_count"])),
                    f"{float(row['solidity_project_match_ratio']):.3f}",
                    f"{float(row['score']):.2f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Evidence Samples"])
    for _, row in top.iterrows():
        lines.append(f"- `{row['repo_path']}` -> `{row['mapped_external_repo_path']}` ({row['confidence']}): {normalize_text(row['sample_matches'])}")
    UNMATCHED_TEST_REPO_FILE_OVERLAP_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Top confidence counts: {top['confidence'].value_counts().to_dict()}")
    print(f"Saved: {UNMATCHED_TEST_REPO_FILE_OVERLAP_PATH}")
    print(f"Report: {UNMATCHED_TEST_REPO_FILE_OVERLAP_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
