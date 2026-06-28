import re
import zipfile
from pathlib import Path

import pandas as pd

try:
    from config import (
        OFFICIAL_EXTERNAL_REPO_MATCHES_PATH,
        OFFICIAL_TEST_ZIP_PATH,
        PROCESSED_DIR,
        REPORTS_DIR,
        TEST_CSV_PATH,
        UNMATCHED_TEST_REPO_IDENTITY_PATH,
        ensure_project_dirs,
    )
except ImportError:
    from src.config import (
        OFFICIAL_EXTERNAL_REPO_MATCHES_PATH,
        OFFICIAL_TEST_ZIP_PATH,
        PROCESSED_DIR,
        REPORTS_DIR,
        TEST_CSV_PATH,
        UNMATCHED_TEST_REPO_IDENTITY_PATH,
        ensure_project_dirs,
    )


PUBLIC_JUDGING_MANIFEST_PATH = PROCESSED_DIR / "public_judging_manifest.csv"
PUBLIC_JUDGING_MANIFEST_REPORT_PATH = REPORTS_DIR / "public_judging_manifest_report.md"

VERIFIED_SHERLOCK_SLUGS = {
    "2022-10-rage-trade",
    "2023-03-optimism",
    "2023-10-mzero",
    "2023-12-dodo",
    "2024-01-rio-vesting-escrow",
    "2024-02-rubicon-finance",
    "2024-03-arrakis",
    "2024-07-kwenta-staking-contracts",
    "2025-02-rova",
}
VERIFIED_CODE4RENA_SLUGS = {
    "2021-09-sushimiso",
    "2022-01-xdefi",
    "2022-04-xtribe",
    "2022-07-juicebox",
    "2024-02-thruster",
    "2024-03-coinbase",
}
DEPENDENCY_ORIGINS = {
    ("foundry-rs", "forge-std"),
}
SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")

MANIFEST_COLUMNS = [
    "repo_path",
    "contest_slug",
    "platform",
    "source_repo_url",
    "public_judging_url",
    "public_archive_verified",
    "resolution_status",
    "provenance",
    "origin_url",
    "origin_slug",
    "matched_external_repo_path",
    "match_confidence",
    "match_score",
    "matched_fingerprints",
    "test_git_head",
    "test_commit",
    "test_commit_provenance",
]


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def parse_github_repo(url: str) -> tuple[str, str]:
    value = clean(url).removesuffix("/")
    match = re.search(
        r"(?:github\.com[:/])(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+?)(?:\.git)?$",
        value,
        flags=re.I,
    )
    if not match:
        return "", ""
    return match.group("owner"), match.group("repo")


def verified_public_archive(platform: str, slug: str) -> bool:
    if platform == "sherlock":
        return slug in VERIFIED_SHERLOCK_SLUGS
    if platform == "code4rena":
        return slug in VERIFIED_CODE4RENA_SLUGS
    return False


def derive_public_judging_identity(
    origin_url: str = "",
    origin_slug: str = "",
    official_match_slug: str = "",
) -> dict[str, object]:
    owner, url_slug = parse_github_repo(origin_url)
    slug = clean(origin_slug) or url_slug
    owner_lower = owner.lower()

    if (owner_lower, slug.lower()) in DEPENDENCY_ORIGINS:
        return {
            "contest_slug": "",
            "platform": "",
            "source_repo_url": "",
            "public_judging_url": "",
            "public_archive_verified": False,
            "resolution_status": "unresolved",
            "provenance": "ignored_dependency_origin",
        }

    if owner_lower == "sherlock-audit" and slug:
        platform = "sherlock"
        source_repo_url = f"https://github.com/sherlock-audit/{slug}"
        public_judging_url = f"{source_repo_url}-judging"
        provenance = "test_git_origin"
    elif owner_lower == "code-423n4" and slug:
        platform = "code4rena"
        source_repo_url = f"https://github.com/code-423n4/{slug}"
        public_judging_url = f"{source_repo_url}-findings"
        provenance = "test_git_origin"
    elif clean(official_match_slug):
        slug = clean(official_match_slug).removeprefix("repos/")
        platform = "code4rena"
        source_repo_url = f"https://github.com/code-423n4/{slug}"
        public_judging_url = f"{source_repo_url}-findings"
        provenance = "official_fingerprint_match"
    else:
        return {
            "contest_slug": "",
            "platform": "",
            "source_repo_url": "",
            "public_judging_url": "",
            "public_archive_verified": False,
            "resolution_status": "unresolved",
            "provenance": "missing_identity",
        }

    return {
        "contest_slug": slug,
        "platform": platform,
        "source_repo_url": source_repo_url,
        "public_judging_url": public_judging_url,
        "public_archive_verified": verified_public_archive(platform, slug),
        "resolution_status": "resolved",
        "provenance": provenance,
    }


def read_test_commit_metadata(zip_path: Path) -> dict[str, dict[str, str]]:
    heads: dict[str, str] = {}
    loose_refs: dict[tuple[str, str], str] = {}
    packed_refs: dict[str, str] = {}

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = info.filename.replace("\\", "/").split("/")
            if len(parts) < 4 or parts[0] != "test":
                continue
            repo_path = parts[1]
            relative = "/".join(parts[2:])
            if relative == ".git/HEAD":
                heads[repo_path] = zf.read(info).decode("utf-8", errors="replace").strip()
            elif relative.startswith(".git/refs/"):
                loose_refs[(repo_path, relative)] = zf.read(info).decode("utf-8", errors="replace").strip()
            elif relative == ".git/packed-refs":
                packed_refs[repo_path] = zf.read(info).decode("utf-8", errors="replace")

    metadata: dict[str, dict[str, str]] = {}
    for repo_path, head in heads.items():
        commit = ""
        provenance = ""
        if SHA1_RE.fullmatch(head):
            commit = head
            provenance = ".git/HEAD"
        elif head.startswith("ref: "):
            ref_name = head.removeprefix("ref: ").strip()
            ref_path = f".git/{ref_name}"
            loose_commit = loose_refs.get((repo_path, ref_path), "")
            if SHA1_RE.fullmatch(loose_commit):
                commit = loose_commit
                provenance = ref_path
            else:
                for line in packed_refs.get(repo_path, "").splitlines():
                    if not line or line.startswith(("#", "^")):
                        continue
                    packed_commit, packed_name = line.split(" ", 1)
                    if packed_name == ref_name and SHA1_RE.fullmatch(packed_commit):
                        commit = packed_commit
                        provenance = ".git/packed-refs"
                        break
        metadata[repo_path] = {
            "test_git_head": head,
            "test_commit": commit,
            "test_commit_provenance": provenance,
        }
    return metadata


def build_manifest(
    test_repos: pd.DataFrame,
    identities: pd.DataFrame,
    official_matches: pd.DataFrame,
    commits: dict[str, dict[str, str]],
) -> pd.DataFrame:
    identity_by_repo = {
        clean(row["repo_path"]): row
        for _, row in identities.iterrows()
        if clean(row.get("repo_path", ""))
    }
    test_matches = official_matches.copy()
    if "split" in test_matches:
        test_matches = test_matches[test_matches["split"].astype(str) == "test"]
    match_by_repo = {
        clean(row["repo_path"]): row
        for _, row in test_matches.iterrows()
        if clean(row.get("repo_path", ""))
    }

    rows = []
    for repo_path in sorted(set(test_repos["repo_path"].astype(str))):
        identity_row = identity_by_repo.get(repo_path, {})
        match_row = match_by_repo.get(repo_path, {})
        origin_url = clean(identity_row.get("origin_url", ""))
        origin_slug = clean(identity_row.get("origin_slug", ""))
        matched_path = clean(match_row.get("matched_external_repo_path", ""))
        resolved = derive_public_judging_identity(
            origin_url=origin_url,
            origin_slug=origin_slug,
            official_match_slug=matched_path,
        )
        commit_metadata = commits.get(
            repo_path,
            {
                "test_git_head": "",
                "test_commit": "",
                "test_commit_provenance": "",
            },
        )
        rows.append(
            {
                "repo_path": repo_path,
                **resolved,
                "origin_url": origin_url,
                "origin_slug": origin_slug,
                "matched_external_repo_path": matched_path,
                "match_confidence": clean(match_row.get("confidence", "")),
                "match_score": match_row.get("score", ""),
                "matched_fingerprints": clean(match_row.get("matched_fingerprints", "")),
                **commit_metadata,
            }
        )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def write_report(manifest: pd.DataFrame, path: Path) -> None:
    resolved = manifest[manifest["resolution_status"] == "resolved"]
    verified = manifest[manifest["public_archive_verified"]]
    unresolved = manifest[manifest["resolution_status"] != "resolved"]
    lines = [
        "# Public Judging Manifest Report",
        "",
        f"- test repositories: `{len(manifest)}`",
        f"- resolved candidates: `{len(resolved)}`",
        f"- verified public archives: `{len(verified)}`",
        f"- unresolved repositories: `{len(unresolved)}`",
        f"- repositories with exact test commit: `{int(manifest['test_commit'].astype(bool).sum())}`",
        f"- platform counts: `{resolved['platform'].value_counts().to_dict()}`",
        f"- provenance counts: `{manifest['provenance'].value_counts().to_dict()}`",
        "- output: `data/processed/public_judging_manifest.csv`",
        "",
        "## Verified Public Archives",
        "",
        "| repo_path | platform | contest_slug | public_judging_url | test_commit |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in verified.sort_values(["platform", "contest_slug", "repo_path"]).iterrows():
        lines.append(
            f"| {row['repo_path']} | {row['platform']} | {row['contest_slug']} | "
            f"{row['public_judging_url']} | {row['test_commit']} |"
        )
    lines.extend(
        [
            "",
            "## Unresolved Repositories",
            "",
            "| repo_path | origin_url | provenance | test_commit |",
            "| --- | --- | --- | --- |",
        ]
    )
    for _, row in unresolved.sort_values("repo_path").iterrows():
        lines.append(
            f"| {row['repo_path']} | {row['origin_url']} | {row['provenance']} | {row['test_commit']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_project_dirs()
    for input_path in [
        TEST_CSV_PATH,
        UNMATCHED_TEST_REPO_IDENTITY_PATH,
        OFFICIAL_EXTERNAL_REPO_MATCHES_PATH,
        OFFICIAL_TEST_ZIP_PATH,
    ]:
        if not input_path.exists():
            raise FileNotFoundError(f"Missing manifest input: {input_path}")

    manifest = build_manifest(
        test_repos=pd.read_csv(TEST_CSV_PATH),
        identities=pd.read_parquet(UNMATCHED_TEST_REPO_IDENTITY_PATH),
        official_matches=pd.read_parquet(OFFICIAL_EXTERNAL_REPO_MATCHES_PATH),
        commits=read_test_commit_metadata(OFFICIAL_TEST_ZIP_PATH),
    )
    PUBLIC_JUDGING_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(PUBLIC_JUDGING_MANIFEST_PATH, index=False)
    write_report(manifest, PUBLIC_JUDGING_MANIFEST_REPORT_PATH)
    print(f"Wrote manifest: {PUBLIC_JUDGING_MANIFEST_PATH} rows={len(manifest)}")
    print(f"Verified public archives: {int(manifest['public_archive_verified'].sum())}")
    print(f"Unresolved repositories: {int((manifest['resolution_status'] != 'resolved').sum())}")
    print(f"Report: {PUBLIC_JUDGING_MANIFEST_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
