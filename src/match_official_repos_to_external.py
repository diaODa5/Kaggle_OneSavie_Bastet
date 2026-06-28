import json

import pandas as pd

try:
    from config import (
        EXTERNAL_REPO_FINGERPRINTS_PATH,
        HIGH_CONF_TRAIN_HASH_EXTERNAL_PAIRS_PATH,
        OFFICIAL_EXTERNAL_MATCH_REPORT_PATH,
        OFFICIAL_EXTERNAL_REPO_MATCHES_PATH,
        OFFICIAL_REPO_FINGERPRINTS_PATH,
        ensure_project_dirs,
    )
except ImportError:
    from src.config import (
        EXTERNAL_REPO_FINGERPRINTS_PATH,
        HIGH_CONF_TRAIN_HASH_EXTERNAL_PAIRS_PATH,
        OFFICIAL_EXTERNAL_MATCH_REPORT_PATH,
        OFFICIAL_EXTERNAL_REPO_MATCHES_PATH,
        OFFICIAL_REPO_FINGERPRINTS_PATH,
        ensure_project_dirs,
    )


FINGERPRINT_WEIGHTS = {
    "source_like_content_aggregate_sha256": 100.0,
    "solidity_content_aggregate_sha256": 90.0,
    "package_config_metadata_aggregate_sha256": 40.0,
    "source_like_path_list_sha256": 25.0,
    "solidity_path_list_sha256": 20.0,
    "repo_manifest_sha256": 15.0,
    "repo_path_list_sha256": 12.0,
    "git_index_sha256": 10.0,
    "git_config_sha256": 5.0,
}
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def confidence_from_score(score: float, names: set[str]) -> str:
    if "source_like_content_aggregate_sha256" in names or "solidity_content_aggregate_sha256" in names:
        return "very_high"
    if score >= 50:
        return "high"
    if score >= 25:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def main() -> int:
    ensure_project_dirs()
    if not OFFICIAL_REPO_FINGERPRINTS_PATH.exists():
        raise FileNotFoundError(f"Missing official fingerprints: {OFFICIAL_REPO_FINGERPRINTS_PATH}")
    if not EXTERNAL_REPO_FINGERPRINTS_PATH.exists():
        raise FileNotFoundError(f"Missing external fingerprints: {EXTERNAL_REPO_FINGERPRINTS_PATH}")
    official = pd.read_parquet(OFFICIAL_REPO_FINGERPRINTS_PATH)
    external = pd.read_parquet(EXTERNAL_REPO_FINGERPRINTS_PATH)

    official = official[official["fingerprint_name"].isin(FINGERPRINT_WEIGHTS)].copy()
    external = external[external["fingerprint_name"].isin(FINGERPRINT_WEIGHTS)].copy()
    official = official[(official["fingerprint_value"].astype(str) != "") & (official["fingerprint_value"].astype(str) != EMPTY_SHA256)]
    external = external[(external["fingerprint_value"].astype(str) != "") & (external["fingerprint_value"].astype(str) != EMPTY_SHA256)]
    official = official[official["file_count"].astype(float) > 0]
    external = external[external["file_count"].astype(float) > 0]

    joined = official.merge(
        external,
        on=["fingerprint_name", "fingerprint_value"],
        how="inner",
        suffixes=("_official", "_external"),
    )
    if joined.empty:
        matches = pd.DataFrame(columns=["split", "repo_path", "matched_external_repo_path", "score", "confidence", "matched_fingerprints", "match_count"])
    else:
        joined["weight"] = joined["fingerprint_name"].map(FINGERPRINT_WEIGHTS).fillna(1.0)
        agg_rows = []
        for (split, repo_hash, ext_repo), group in joined.groupby(["split", "repo_path", "external_repo_path"]):
            names = sorted(set(group["fingerprint_name"]))
            agg_rows.append(
                {
                    "split": split,
                    "repo_path": repo_hash,
                    "matched_external_repo_path": ext_repo,
                    "score": float(group["weight"].sum()),
                    "confidence": confidence_from_score(float(group["weight"].sum()), set(names)),
                    "matched_fingerprints": ",".join(names),
                    "match_count": int(len(group)),
                }
            )
        all_matches = pd.DataFrame(agg_rows).sort_values(["split", "repo_path", "score"], ascending=[True, True, False])
        matches = all_matches.groupby(["split", "repo_path"], as_index=False).head(1).reset_index(drop=True)

    OFFICIAL_EXTERNAL_REPO_MATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(OFFICIAL_EXTERNAL_REPO_MATCHES_PATH, index=False)

    known_eval = {}
    if HIGH_CONF_TRAIN_HASH_EXTERNAL_PAIRS_PATH.exists() and not matches.empty:
        known = pd.read_csv(HIGH_CONF_TRAIN_HASH_EXTERNAL_PAIRS_PATH)
        train_matches = matches[matches["split"] == "train"]
        merged = known.merge(train_matches, left_on="train_repo_path", right_on="repo_path", how="left", suffixes=("_known", "_match"))
        merged["correct_known_pair"] = merged["top_external_repo_path"] == merged["matched_external_repo_path"]
        cols = [
            "train_repo_path",
            "top_external_repo_path",
            "matched_external_repo_path",
            "confidence_match",
            "matched_fingerprints",
            "correct_known_pair",
        ]
        for col in cols:
            if col not in merged.columns:
                merged[col] = ""
        known_eval = {
            "known_pair_count": int(len(merged)),
            "known_pair_matched_count": int(merged["matched_external_repo_path"].notna().sum()),
            "known_pair_correct_count": int(merged["correct_known_pair"].sum()),
            "known_pair_rows": merged[cols].to_dict(orient="records"),
        }
    summary = {
        "official_match_rows": int(len(matches)),
        "train_match_count": int((matches["split"] == "train").sum()) if not matches.empty else 0,
        "test_match_count": int((matches["split"] == "test").sum()) if not matches.empty else 0,
        "confidence_counts": matches["confidence"].value_counts().to_dict() if not matches.empty else {},
        "test_confidence_counts": matches[matches["split"] == "test"]["confidence"].value_counts().to_dict() if not matches.empty else {},
        "known_eval": known_eval,
    }
    lines = [
        "# Official External Repo Match Report",
        "",
        f"- official match rows: `{summary['official_match_rows']}`",
        f"- train match count: `{summary['train_match_count']}`",
        f"- test match count: `{summary['test_match_count']}`",
        f"- confidence counts: `{summary['confidence_counts']}`",
        f"- test confidence counts: `{summary['test_confidence_counts']}`",
        "",
        "## Known Pair Evaluation",
        "```json",
        json.dumps(known_eval, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Top Test Matches",
        "```text",
        matches[matches["split"] == "test"].head(80).to_string(index=False) if not matches.empty else "No matches.",
        "```",
    ]
    OFFICIAL_EXTERNAL_MATCH_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Train matches: {summary['train_match_count']} Test matches: {summary['test_match_count']}")
    print(f"Test confidence: {summary['test_confidence_counts']}")
    print(f"Report: {OFFICIAL_EXTERNAL_MATCH_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
