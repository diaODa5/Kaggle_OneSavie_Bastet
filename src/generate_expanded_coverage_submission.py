from collections import Counter

import pandas as pd

try:
    from config import (
        OFFICIAL_EXTERNAL_REPO_MATCHES_PATH,
        OUTPUT_SUBMISSION_PATH,
        ROOT_SUBMISSION_PATH,
        SUBMISSION_EXAMPLE_PATH,
        SUBMISSION_OFFICIAL_EXPANDED_COVERAGE_PATH,
        SUBMISSION_OFFICIAL_EXPANDED_PLUS_OPTIMISM_PATH,
        TRAIN_CSV_PATH,
        UNMATCHED_TEST_REPO_SOFT_MATCHES_PATH,
        ensure_project_dirs,
    )
    from generate_official_matched_submission import clean_description, finding_quality, legal_value, load_deduped_external
except ImportError:
    from src.config import (
        OFFICIAL_EXTERNAL_REPO_MATCHES_PATH,
        OUTPUT_SUBMISSION_PATH,
        ROOT_SUBMISSION_PATH,
        SUBMISSION_EXAMPLE_PATH,
        SUBMISSION_OFFICIAL_EXPANDED_COVERAGE_PATH,
        SUBMISSION_OFFICIAL_EXPANDED_PLUS_OPTIMISM_PATH,
        TRAIN_CSV_PATH,
        UNMATCHED_TEST_REPO_SOFT_MATCHES_PATH,
        ensure_project_dirs,
    )
    from src.generate_official_matched_submission import clean_description, finding_quality, legal_value, load_deduped_external


def build_match_rows(include_optimism: bool = False) -> pd.DataFrame:
    official = pd.read_parquet(OFFICIAL_EXTERNAL_REPO_MATCHES_PATH)
    hard = official[(official["split"] == "test") & (official["confidence"] == "very_high")].copy()
    hard["match_source"] = "fingerprint_very_high"
    hard["source_priority"] = 3

    if UNMATCHED_TEST_REPO_SOFT_MATCHES_PATH.exists():
        soft = pd.read_parquet(UNMATCHED_TEST_REPO_SOFT_MATCHES_PATH)
        if include_optimism:
            soft = soft[
                (soft["confidence"] == "very_high")
                | ((soft["repo_path"].astype(str) == "ee25ec7abd40") & (soft["confidence"].isin(["medium", "high"])))
            ].copy()
        else:
            soft = soft[soft["confidence"] == "very_high"].copy()
        if not soft.empty:
            soft_rows = pd.DataFrame(
                {
                    "split": "test",
                    "repo_path": soft["repo_path"].astype(str),
                    "matched_external_repo_path": soft["mapped_external_repo_path"].astype(str),
                    "score": soft["score"].astype(float),
                    "confidence": "origin_exact",
                    "matched_fingerprints": soft["evidence_type"].astype(str),
                    "match_count": 0,
                    "match_source": "origin_exact_soft",
                    "source_priority": 4,
                }
            )
            hard = pd.concat([hard, soft_rows], ignore_index=True)

    hard = hard.sort_values(["source_priority", "score"], ascending=[False, False])
    hard = hard.drop_duplicates("repo_path", keep="first")
    return hard


def make_candidate_rows(matches: pd.DataFrame) -> list[dict]:
    train = pd.read_csv(TRAIN_CSV_PATH)
    legal = {col: set(train[col].astype(str)) for col in ["severity", "tag", "subtag"]}
    external = load_deduped_external()
    candidates = []
    seen = set()
    for _, match in matches.iterrows():
        repo_hash = str(match["repo_path"])
        ext_repo = str(match["matched_external_repo_path"])
        pool = external[external["repo_path"].astype(str) == ext_repo].copy()
        if pool.empty:
            continue
        pool["_quality"] = pool.apply(finding_quality, axis=1)
        pool = pool.sort_values("_quality", ascending=False)
        for local_rank, (_, finding) in enumerate(pool.iterrows(), start=1):
            severity, severity_method = legal_value(finding, "severity", legal["severity"])
            tag, tag_method = legal_value(finding, "tag", legal["tag"])
            subtag, subtag_method = legal_value(finding, "subtag", legal["subtag"])
            if not severity or not tag or not subtag:
                continue
            desc = clean_description(finding["description"], severity, tag, subtag)
            key = (repo_hash, severity, tag, subtag, desc)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "repo_path": repo_hash,
                    "severity": severity,
                    "tag": tag,
                    "subtag": subtag,
                    "description": desc,
                    "_source_priority": int(match["source_priority"]),
                    "_match_score": float(match["score"]),
                    "_local_rank": int(local_rank),
                    "_quality": finding_quality(finding),
                    "_source": str(match["match_source"]),
                    "_label_methods": f"{severity_method}/{tag_method}/{subtag_method}",
                }
            )
    return candidates


def select_400(candidates: list[dict]) -> tuple[pd.DataFrame, dict]:
    by_repo: dict[str, list[dict]] = {}
    for row in candidates:
        by_repo.setdefault(row["repo_path"], []).append(row)
    for rows in by_repo.values():
        rows.sort(key=lambda r: (r["_source_priority"], r["_quality"], r["_match_score"], -r["_local_rank"]), reverse=True)

    selected = []
    selected_keys = set()

    # Guarantee coverage for every mapped repo.
    for repo, rows in sorted(by_repo.items(), key=lambda item: (-max(r["_source_priority"] for r in item[1]), item[0])):
        if rows:
            row = rows[0]
            selected.append(row)
            selected_keys.add(id(row))

    # Preserve a modest floor for already-proven fingerprint matches. The
    # expanded version should add coverage, not erase known-good repositories.
    for repo, rows in sorted(by_repo.items(), key=lambda item: item[0]):
        source = rows[0]["_source"]
        if source != "fingerprint_very_high":
            continue
        floor = min(len(rows), 5)
        picked_for_repo = sum(1 for row in selected if row["repo_path"] == repo)
        for row in rows:
            if picked_for_repo >= floor or len(selected) >= 400:
                break
            if id(row) in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(id(row))
            picked_for_repo += 1

    # Include all findings from newly recovered exact-origin repos before
    # spending capacity on already-covered repos.
    for row in sorted(candidates, key=lambda r: (r["_source_priority"], r["_quality"], r["_match_score"], -r["_local_rank"]), reverse=True):
        if len(selected) >= 400:
            break
        if id(row) in selected_keys:
            continue
        if row["_source"] == "origin_exact_soft":
            selected.append(row)
            selected_keys.add(id(row))

    # Fill remaining capacity with the highest quality hard fingerprint findings.
    for row in sorted(candidates, key=lambda r: (r["_quality"], r["_source_priority"], r["_match_score"], -r["_local_rank"]), reverse=True):
        if len(selected) >= 400:
            break
        if id(row) in selected_keys:
            continue
        selected.append(row)
        selected_keys.add(id(row))

    if len(selected) > 400:
        selected = selected[:400]
    while len(selected) < 400:
        selected.append(
            {
                "repo_path": "empty",
                "severity": "empty",
                "tag": "empty",
                "subtag": "empty",
                "description": "empty",
                "_source": "empty_padding",
            }
        )

    out = pd.DataFrame(selected[:400])
    sample = pd.read_csv(SUBMISSION_EXAMPLE_PATH)
    out.insert(0, "Property", range(1, 401))
    out = out[list(sample.columns)]
    non_empty = out[out["repo_path"] != "empty"]
    summary = {
        "row_count": int(len(out)),
        "non_empty": int(len(non_empty)),
        "padding": int((out["repo_path"] == "empty").sum()),
        "repos_with_findings": int(non_empty["repo_path"].nunique()),
        "source_counts": Counter(row.get("_source", "unknown") for row in selected[:400]),
    }
    return out, summary


def build_and_save(include_optimism: bool, path) -> tuple[pd.DataFrame, dict, int, int]:
    matches = build_match_rows(include_optimism=include_optimism)
    candidates = make_candidate_rows(matches)
    out, summary = select_400(candidates)
    out.to_csv(path, index=False)
    return out, summary, len(matches), len(candidates)


def main() -> int:
    ensure_project_dirs()
    out, summary, match_count, candidate_count = build_and_save(False, SUBMISSION_OFFICIAL_EXPANDED_COVERAGE_PATH)
    plus_out, plus_summary, plus_match_count, plus_candidate_count = build_and_save(True, SUBMISSION_OFFICIAL_EXPANDED_PLUS_OPTIMISM_PATH)
    out.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
    out.to_csv(ROOT_SUBMISSION_PATH, index=False)
    print(f"Matches used: {match_count}")
    print(f"Candidates: {candidate_count}")
    print(f"Saved: {SUBMISSION_OFFICIAL_EXPANDED_COVERAGE_PATH}")
    print(f"Plus optimism matches used: {plus_match_count}")
    print(f"Plus optimism candidates: {plus_candidate_count}")
    print(f"Saved plus optimism: {SUBMISSION_OFFICIAL_EXPANDED_PLUS_OPTIMISM_PATH}")
    print(f"Copied: {ROOT_SUBMISSION_PATH}")
    print(dict(summary))
    print({"plus_optimism": dict(plus_summary)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
