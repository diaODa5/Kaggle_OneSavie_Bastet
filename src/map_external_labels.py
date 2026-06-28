import json
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from config import (
        EXTERNAL_BASTET_FINDINGS_PATH,
        EXTERNAL_BASTET_MAPPED_FINDINGS_PATH,
        EXTERNAL_LABEL_MAPPING_PATH,
        LABEL_MAPPING_REPORT_PATH,
        TRAIN_CSV_PATH,
        ensure_project_dirs,
    )
    from external_utils import fuzzy_ratio, norm_label
    from utils import normalize_text
except ImportError:
    from src.config import (
        EXTERNAL_BASTET_FINDINGS_PATH,
        EXTERNAL_BASTET_MAPPED_FINDINGS_PATH,
        EXTERNAL_LABEL_MAPPING_PATH,
        LABEL_MAPPING_REPORT_PATH,
        TRAIN_CSV_PATH,
        ensure_project_dirs,
    )
    from src.external_utils import fuzzy_ratio, norm_label
    from src.utils import normalize_text


LABEL_COLUMNS = ["severity", "tag", "subtag"]
NEAREST_SIM_THRESHOLD = 0.55


def compile_legal_lookup(values: set[str]) -> dict[str, str]:
    return {norm_label(v): v for v in values if normalize_text(v)}


def legalize_label(value: str, legal: set[str], lookup: dict[str, str]) -> str | None:
    clean = normalize_text(value)
    if not clean:
        return None
    if clean in legal:
        return clean
    normalized = norm_label(clean)
    if normalized in lookup:
        return lookup[normalized]
    return None


def infer_tag_subtag_from_text(text: str, legal: dict[str, set[str]], lookups: dict[str, dict[str, str]]) -> tuple[str | None, str | None]:
    t = normalize_text(text).lower()
    rules = [
        (r"\b(reentr|nonreentrant|cei|check[- ]effects|external call)\b", "Reentrancy", "Violating CEI / Missing nonReentrant"),
        (r"\b(chainlink|oracle|latestrounddata|updatedat|stale|price feed|price oracle)\b", "Chainlink, Oracle", "Stale Value"),
        (r"\b(slippage|minout|min amount|amountoutmin|minreceived|deadline)\b", "Slippage", "Missing minOut / maxAmount"),
        (r"\b(flash ?loan|executeoperation|flash borrower)\b", "Flashloan", "Invalid Validation"),
        (r"\b(erc4626|vault|converttoassets|converttoshares|previewdeposit|previewwithdraw|inflation attack)\b", "ERC4626", "Inflation Attack"),
        (r"\b(overflow|underflow|precision|rounding|decimal|division before|unsafe downcast|loss of precision)\b", "Arithmetic", "Precision Loss"),
        (r"\b(accounting|incorrect formula|miscalculat|wrong calculat|reward|fee|interest|balance|share price|index|accrual)\b", "Accounting Error", "Incorrect Formula"),
        (r"\b(access control|onlyowner|owner|admin|role|permission|privilege|unauthorized|centralization)\b", "Access Control", "Centralization Risk"),
        (r"\b(initializer|initialize|proxy|upgradeable|storage gap|implementation contract)\b", "Upgradeable", "Missing Initialization"),
        (r"\b(signature|ecrecover|permit|replay|nonce|domain separator|eip712)\b", "EIP712", "Nonce"),
        (r"\b(erc20|token transfer|transferfrom|approve|safeapprove|fee on transfer|missing return)\b", "ERC20", "Missing Return Check"),
        (r"\b(liquidat|bad debt|collateral|health factor|auction)\b", "Liquidation", "Unfair Liquidation"),
        (r"\b(governance|proposal|voting|vote|dao|multisig)\b", "Governance", "Centralization Risk"),
        (r"\b(front[ -]?run|sandwich|mev|arbitrage)\b", "MEV", "Front Run"),
        (r"\b(dos|denial of service|out of gas|gas grief|unbounded loop|cannot withdraw|locked funds|freeze|revert|bricked)\b", "DoS", "Bad Condition"),
        (r"\b(missing check|invalid validation|validate|validation|parameter|wrong parameter|input|bypass|unchecked)\b", "Input Validation", "Invalid Validation"),
    ]
    for pattern, tag, subtag in rules:
        if re.search(pattern, t):
            legal_tag = legalize_label(tag, legal["tag"], lookups["tag"])
            legal_subtag = legalize_label(subtag, legal["subtag"], lookups["subtag"])
            return legal_tag, legal_subtag
    return None, None


def build_direct_mapping(kaggle_values: set[str], external_values: set[str]) -> dict[str, str | None]:
    kaggle_norm = {norm_label(v): v for v in kaggle_values}
    mapping: dict[str, str | None] = {}
    for value in sorted(external_values):
        clean = normalize_text(value)
        if not clean:
            mapping[value] = None
            continue
        n = norm_label(clean)
        if clean in kaggle_values:
            mapping[value] = clean
        elif n in kaggle_norm:
            mapping[value] = kaggle_norm[n]
        else:
            best = None
            best_score = 0.0
            for k_norm, k_value in kaggle_norm.items():
                score = fuzzy_ratio(n, k_norm)
                if score > best_score:
                    best_score = score
                    best = k_value
            mapping[value] = best if best_score >= 0.92 else None
    return mapping


def nearest_train_labels(train: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    train_text = (train["description"].fillna("") + " " + train["tag"].fillna("") + " " + train["subtag"].fillna("")).map(normalize_text)
    ext_text = (external["description"].fillna("") + " " + external["report_text"].fillna("")).map(normalize_text)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, max_features=50000)
    x_train = vectorizer.fit_transform(train_text)
    x_ext = vectorizer.transform(ext_text)
    sims = x_ext @ x_train.T
    nearest = []
    for i in range(sims.shape[0]):
        row = sims.getrow(i)
        if row.nnz == 0:
            nearest.append((0, 0.0))
        else:
            pos = row.data.argmax()
            nearest.append((int(row.indices[pos]), float(row.data[pos])))
    nn = pd.DataFrame(nearest, columns=["nearest_train_index", "nearest_train_similarity"])
    for col in LABEL_COLUMNS:
        nn[f"nearest_{col}"] = [train.iloc[idx][col] for idx, _ in nearest]
    return nn


def main() -> int:
    ensure_project_dirs()
    if not EXTERNAL_BASTET_FINDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing external findings: {EXTERNAL_BASTET_FINDINGS_PATH}")
    train = pd.read_csv(TRAIN_CSV_PATH)
    external = pd.read_parquet(EXTERNAL_BASTET_FINDINGS_PATH)
    nn = nearest_train_labels(train, external)
    mapped = external.copy().reset_index(drop=True)
    for col in LABEL_COLUMNS:
        mapped[f"raw_{col}"] = mapped[col].map(normalize_text) if col in mapped.columns else ""
    mapping_json = {}
    report = {}
    legal_sets = {col: set(train[col].dropna().map(normalize_text)) for col in LABEL_COLUMNS}
    lookups = {col: compile_legal_lookup(legal_sets[col]) for col in LABEL_COLUMNS}
    rule_text = (
        mapped["description"].fillna("").map(normalize_text)
        + " "
        + mapped.get("report_text", pd.Series([""] * len(mapped))).fillna("").map(normalize_text)
    )
    rule_pairs = [infer_tag_subtag_from_text(text, legal_sets, lookups) for text in rule_text]
    for col in LABEL_COLUMNS:
        legal = legal_sets[col]
        values = set(mapped[col].dropna().map(normalize_text)) if col in mapped.columns else set()
        direct = build_direct_mapping(legal, values)
        mapping_json[col] = direct
        out_values = []
        out_methods = []
        method_counts = {"direct": 0, "rule_text": 0, "nearest_description": 0, "global_fallback": 0}
        global_mode = train[col].mode().iloc[0]
        for i, value in enumerate(mapped[col].fillna("").map(normalize_text)):
            direct_value = direct.get(value)
            if direct_value in legal:
                out_values.append(direct_value)
                out_methods.append("direct")
                method_counts["direct"] += 1
            elif col in {"tag", "subtag"} and rule_pairs[i][0 if col == "tag" else 1] in legal:
                out_values.append(rule_pairs[i][0 if col == "tag" else 1])
                out_methods.append("rule_text")
                method_counts["rule_text"] += 1
            else:
                nearest_value = normalize_text(nn.iloc[i][f"nearest_{col}"])
                nearest_similarity = float(nn.iloc[i]["nearest_train_similarity"])
                if nearest_value in legal and nearest_similarity >= NEAREST_SIM_THRESHOLD:
                    out_values.append(nearest_value)
                    out_methods.append("nearest_description")
                    method_counts["nearest_description"] += 1
                else:
                    out_values.append(global_mode)
                    out_methods.append("global_fallback")
                    method_counts["global_fallback"] += 1
        mapped[col] = out_values
        mapped[f"{col}_mapping_method"] = out_methods
        illegal = sorted(set(mapped[col]) - legal)
        if illegal:
            raise ValueError(f"Illegal mapped {col} labels remain: {illegal[:20]}")
        report[col] = {
            "legal_unique": len(legal),
            "external_raw_unique": len(values),
            "method_counts": method_counts,
        }
    mapped = pd.concat([mapped, nn], axis=1)
    EXTERNAL_BASTET_MAPPED_FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_parquet(EXTERNAL_BASTET_MAPPED_FINDINGS_PATH, index=False)
    EXTERNAL_LABEL_MAPPING_PATH.write_text(json.dumps(mapping_json, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Label Mapping Report",
        "",
        f"- external input shape: `{tuple(external.shape)}`",
        f"- mapped output shape: `{tuple(mapped.shape)}`",
        f"- mapped findings saved: `{EXTERNAL_BASTET_MAPPED_FINDINGS_PATH}`",
        f"- mapping saved: `{EXTERNAL_LABEL_MAPPING_PATH}`",
        "",
        "```json",
        json.dumps(report, indent=2, ensure_ascii=False),
        "```",
    ]
    LABEL_MAPPING_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Mapped external shape: {mapped.shape}")
    print(f"Report: {LABEL_MAPPING_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
