import json
import re
from difflib import SequenceMatcher

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from config import (
        EXTERNAL_BASTET_MAPPED_FINDINGS_PATH,
        EXTERNAL_REPO_FINGERPRINTS_PATH,
        OFFICIAL_EXTERNAL_REPO_MATCHES_PATH,
        UNMATCHED_TEST_REPO_IDENTITY_PATH,
        UNMATCHED_TEST_REPO_SOFT_MATCHES_PATH,
        UNMATCHED_TEST_REPO_SOFT_MATCH_REPORT_PATH,
        ensure_project_dirs,
    )
    from utils import normalize_text
except ImportError:
    from src.config import (
        EXTERNAL_BASTET_MAPPED_FINDINGS_PATH,
        EXTERNAL_REPO_FINGERPRINTS_PATH,
        OFFICIAL_EXTERNAL_REPO_MATCHES_PATH,
        UNMATCHED_TEST_REPO_IDENTITY_PATH,
        UNMATCHED_TEST_REPO_SOFT_MATCHES_PATH,
        UNMATCHED_TEST_REPO_SOFT_MATCH_REPORT_PATH,
        ensure_project_dirs,
    )
    from src.utils import normalize_text


def compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value).lower())


def no_year(slug: str) -> str:
    return re.sub(r"^\d{4}-\d{2}-", "", normalize_text(slug).lower())


def external_slug_table(fps: pd.DataFrame) -> pd.DataFrame:
    piv = fps.pivot_table(
        index="external_repo_path",
        columns="fingerprint_name",
        values="fingerprint_value",
        aggfunc="first",
    ).reset_index()
    for col in ["slug", "slug_lower", "slug_no_year_prefix", "slug_compact"]:
        if col not in piv.columns:
            piv[col] = ""
    piv["slug"] = piv["slug"].map(normalize_text)
    piv["slug_lower"] = piv["slug_lower"].map(lambda x: normalize_text(x).lower())
    piv["slug_no_year_prefix"] = piv["slug_no_year_prefix"].map(lambda x: normalize_text(x).lower())
    piv["slug_compact"] = piv["slug_compact"].map(lambda x: normalize_text(x).lower())
    return piv


def best_text_match(identity: pd.Series, external: pd.DataFrame) -> tuple[str, float, str]:
    texts = [normalize_text(identity.get("identity_text", ""))]
    ext_texts = (
        external["slug"].fillna("")
        + " "
        + external["slug_no_year_prefix"].fillna("")
        + " "
        + external["external_repo_path"].fillna("")
    ).map(normalize_text).tolist()
    if not texts[0] or not ext_texts:
        return "", 0.0, ""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    matrix = vectorizer.fit_transform(texts + ext_texts)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    idx = int(sims.argmax())
    return str(external.iloc[idx]["external_repo_path"]), float(sims[idx]), str(external.iloc[idx]["slug"])


def candidate_for_row(row: pd.Series, external: pd.DataFrame, existing: pd.DataFrame, findings_counts: dict[str, int]) -> dict:
    repo_hash = str(row["repo_path"])
    origin_slug = normalize_text(row.get("origin_slug", ""))
    origin_slug_lower = origin_slug.lower()
    origin_compact = compact(origin_slug)
    origin_no_year = no_year(origin_slug)

    candidates = []
    if origin_slug:
        exact = external[external["slug_lower"] == origin_slug_lower]
        for _, ext in exact.iterrows():
            candidates.append((ext["external_repo_path"], "origin_slug_exact", 1.0, "origin slug exactly matches external repo slug"))
        compact_exact = external[(external["slug_compact"] == origin_compact) & (external["slug_compact"] != "")]
        for _, ext in compact_exact.iterrows():
            candidates.append((ext["external_repo_path"], "origin_slug_compact_exact", 0.96, "compact origin slug matches external slug"))
        no_year_exact = external[(external["slug_no_year_prefix"] == origin_no_year) & (external["slug_no_year_prefix"] != "")]
        for _, ext in no_year_exact.iterrows():
            score = 0.90 if str(ext["slug_lower"]).startswith(origin_slug_lower[:7]) else 0.70
            candidates.append((ext["external_repo_path"], "origin_project_name_match", score, "project name matches after removing date prefix"))

    ex = existing[existing["repo_path"].astype(str) == repo_hash]
    for _, match in ex.iterrows():
        conf = str(match["confidence"])
        score = {"high": 0.86, "low": 0.55, "very_high": 1.0}.get(conf, 0.4)
        candidates.append((match["matched_external_repo_path"], f"existing_{conf}_fingerprint", score, f"existing fingerprint score={match['score']}"))

    text_repo, text_score, text_slug = best_text_match(row, external)
    if text_repo:
        candidates.append((text_repo, "identity_text_tfidf", min(text_score, 0.75), f"best TF-IDF identity match slug={text_slug}, score={text_score:.3f}"))

    if not candidates:
        return {
            "repo_path": repo_hash,
            "origin_slug": origin_slug,
            "mapped_external_repo_path": "",
            "confidence": "none",
            "evidence_type": "none",
            "score": 0.0,
            "matched_findings_count": 0,
            "notes": "no external candidate",
        }

    by_repo: dict[str, dict] = {}
    for repo, evidence_type, score, note in candidates:
        repo = normalize_text(repo)
        cur = by_repo.setdefault(repo, {"score": 0.0, "evidence": [], "notes": []})
        cur["score"] += score
        cur["evidence"].append(evidence_type)
        cur["notes"].append(note)
    best_repo, best = max(by_repo.items(), key=lambda item: item[1]["score"])
    score = float(best["score"])
    evidence = ",".join(best["evidence"])
    if "origin_slug_exact" in best["evidence"] or "origin_slug_compact_exact" in best["evidence"]:
        confidence = "very_high"
    elif score >= 1.05 and "existing_high_fingerprint" in best["evidence"]:
        confidence = "high"
    elif score >= 0.85:
        confidence = "medium"
    elif score >= 0.50:
        confidence = "low"
    else:
        confidence = "none"

    if confidence in {"low", "none"} and "origin_slug" in row and origin_slug and best_repo:
        notes = "; ".join(best["notes"])
    else:
        notes = "; ".join(best["notes"])
    return {
        "repo_path": repo_hash,
        "origin_slug": origin_slug,
        "origin_url": normalize_text(row.get("origin_url", "")),
        "mapped_external_repo_path": best_repo if confidence != "none" else "",
        "confidence": confidence,
        "evidence_type": evidence,
        "score": score,
        "matched_findings_count": int(findings_counts.get(best_repo, 0)) if confidence != "none" else 0,
        "notes": notes,
    }


def main() -> int:
    ensure_project_dirs()
    identity = pd.read_parquet(UNMATCHED_TEST_REPO_IDENTITY_PATH)
    fps = pd.read_parquet(EXTERNAL_REPO_FINGERPRINTS_PATH)
    existing = pd.read_parquet(OFFICIAL_EXTERNAL_REPO_MATCHES_PATH)
    external = external_slug_table(fps)
    findings = pd.read_parquet(EXTERNAL_BASTET_MAPPED_FINDINGS_PATH)
    findings_counts = findings.groupby("repo_path").size().to_dict()

    rows = [candidate_for_row(row, external, existing[existing["split"] == "test"], findings_counts) for _, row in identity.iterrows()]
    out = pd.DataFrame(rows).sort_values(["confidence", "score"], ascending=[True, False])
    UNMATCHED_TEST_REPO_SOFT_MATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(UNMATCHED_TEST_REPO_SOFT_MATCHES_PATH, index=False)

    confidence_counts = out["confidence"].value_counts().to_dict()
    lines = [
        "# Unmatched Test Repo Soft Match Report",
        "",
        f"- input identity rows: `{len(identity)}`",
        f"- external repos: `{len(external)}`",
        f"- output: `{UNMATCHED_TEST_REPO_SOFT_MATCHES_PATH}`",
        f"- confidence counts: `{confidence_counts}`",
        "",
        "## Matches",
        "",
        "| repo_path | origin_slug | mapped_external_repo_path | confidence | findings | score | evidence |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for _, row in out.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    normalize_text(row["repo_path"]),
                    normalize_text(row["origin_slug"]),
                    normalize_text(row["mapped_external_repo_path"]),
                    normalize_text(row["confidence"]),
                    str(int(row["matched_findings_count"])),
                    f"{float(row['score']):.3f}",
                    normalize_text(row["evidence_type"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Notes"])
    for _, row in out.iterrows():
        lines.append(f"- `{row['repo_path']}`: {normalize_text(row['notes'])}")
    UNMATCHED_TEST_REPO_SOFT_MATCH_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Soft matched rows: {len(out)}")
    print(f"Confidence counts: {confidence_counts}")
    print(f"Saved: {UNMATCHED_TEST_REPO_SOFT_MATCHES_PATH}")
    print(f"Report: {UNMATCHED_TEST_REPO_SOFT_MATCH_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
