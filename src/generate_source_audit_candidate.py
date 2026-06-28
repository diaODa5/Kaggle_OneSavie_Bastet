from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_ZIP = ROOT / "data" / "raw_kaggle" / "test.zip"
TRAIN_CSV = ROOT / "data" / "raw_kaggle" / "train.csv"
TEST_CSV = ROOT / "data" / "raw_kaggle" / "test.csv"
SAMPLE_CSV = ROOT / "data" / "raw_kaggle" / "submission_example.csv"
BASELINE_CSV = ROOT / "outputs" / "submission_backup_3825_no_padding_targeted_v2.csv"
EVIDENCE_PARQUET = ROOT / "data" / "processed" / "source_audit_evidence.parquet"
OUTPUT_CSV = ROOT / "outputs" / "submission_source_audit_v1.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "source_audit_findings.md"
FUSION_REPORT_PATH = ROOT / "outputs" / "reports" / "source_audit_fusion_report.md"


CANDIDATES = [
    {
        "replace_property": 374,
        "repo_path": "103f39b0f29b",
        "grade": "B",
        "severity": "Medium",
        "tag": "Arithmetic",
        "subtag": "Precision Loss",
        "title": "MAX_FEE rounding mismatch reverts valid orders",
        "description": (
            "RubiconFeeController rounds fee outputs up with mulDivUp, while ProtocolFees enforces MAX_FEE using "
            "mulDivDown. At the maximum configured fee, the one-unit rounding mismatch can revert otherwise valid orders."
        ),
        "sources": [
            {
                "path": "gladius-contracts-internal/src/fee-controllers/RubiconFeeController.sol",
                "start": 77,
                "end": 84,
            },
            {
                "path": "gladius-contracts-internal/src/base/ProtocolFees.sol",
                "start": 88,
                "end": 96,
            },
            {
                "path": "gladius-contracts-internal/src/fee-controllers/RubiconFeeController.sol",
                "start": 120,
                "end": 135,
            },
        ],
        "condition": {
            "source_required": [
                ["mulDivUp"],
                ["mulDivDown", "MAX_FEE"],
                ["fee > gladiusReactor.MAX_FEE()", "bFee > gladiusReactor.MAX_FEE()"],
            ],
            "source_forbidden": [[], [], []],
            "file_forbidden": [[], [], []],
        },
    },
    {
        "replace_property": 369,
        "repo_path": "51c6dc5fd57f",
        "grade": "B",
        "severity": "High",
        "tag": "Input Validation, Replay Attack",
        "subtag": "Invalid Validation",
        "title": "Collateral update signatures can be replayed",
        "description": (
            "MinterGateway validator signatures do not include a nonce, and updateCollateral records only the minimum "
            "timestamp among accepted signatures. A minter can combine previously valid signatures with a newer "
            "signature to restore stale collateral and mint or withdraw against assets no longer held."
        ),
        "sources": [
            {"path": "protocol/src/MinterGateway.sol", "start": 958, "end": 978},
            {"path": "protocol/src/MinterGateway.sol", "start": 1045, "end": 1105},
            {"path": "protocol/src/MinterGateway.sol", "start": 868, "end": 875},
        ],
        "condition": {
            "source_required": [
                ["UPDATE_COLLATERAL_TYPEHASH", "timestamp_"],
                ["timestamps_[index_]", "UIntMath.min40", "SignatureChecker.isValidSignature"],
                ["StaleCollateralUpdate", "newTimestamp_ <= lastUpdateTimestamp_"],
            ],
            "source_forbidden": [[], [], []],
            "file_forbidden": [
                ["collateralUpdateNonce", "updateCollateralNonce"],
                [],
                [],
            ],
        },
    },
    {
        "replace_property": 392,
        "repo_path": "51c6dc5fd57f",
        "grade": "B",
        "severity": "High",
        "tag": "Accounting Error",
        "subtag": "Incorrect Parameter",
        "title": "Mint solvency check excludes pending penalties",
        "description": (
            "MinterGateway checks mint solvency using activeOwedMOf, which excludes unavoidable penalties for missed "
            "collateral updates. An undercollateralized minter can therefore mint additional M before pending penalties "
            "are reflected, creating unbacked debt."
        ),
        "sources": [
            {"path": "protocol/src/MinterGateway.sol", "start": 513, "end": 520},
            {"path": "protocol/src/MinterGateway.sol", "start": 584, "end": 588},
            {"path": "protocol/src/MinterGateway.sol", "start": 1018, "end": 1031},
        ],
        "condition": {
            "source_required": [
                ["function activeOwedMOf", "_getPresentAmount(uint112(_rawOwedM[minter_]))"],
                ["_getPenaltyPrincipalForMissedCollateralUpdates", "_getPresentAmount(penaltyPrincipal_)"],
                ["activeOwedMOf(minter_)", "additionalOwedM_", "finalActiveOwedM_"],
            ],
            "source_forbidden": [[], [], ["getPenaltyPrincipalForMissedCollateralUpdates(minter_)"]],
            "file_forbidden": [[], [], []],
        },
    },
    {
        "replace_property": 381,
        "repo_path": "9ddd6b83c27e",
        "grade": "B",
        "severity": "High",
        "tag": "Access Control",
        "subtag": "Asset Theft",
        "title": "Anyone can consume a user's vault-share approval",
        "description": (
            "WithdrawPeriphery lets the caller choose both the share owner and receiver without requiring the owner to "
            "equal msg.sender. Anyone can consume a victim's approval to the periphery and redirect the redeemed assets."
        ),
        "sources": [
            {"path": "dn-gmx-vaults/contracts/periphery/WithdrawPeriphery.sol", "start": 107, "end": 145},
        ],
        "condition": {
            "source_required": [[
                "function withdrawToken",
                "dnGmxJuniorVault.withdraw",
                "function redeemToken",
                "dnGmxJuniorVault.redeem",
            ]],
            "source_forbidden": [["from == msg.sender", "msg.sender == from"]],
            "file_forbidden": [[]],
        },
    },
    {
        "replace_property": 383,
        "repo_path": "9ddd6b83c27e",
        "grade": "B",
        "severity": "Medium",
        "tag": "ERC4626",
        "subtag": "Inflation Attack",
        "title": "Senior vault is vulnerable to first-depositor inflation",
        "description": (
            "DnGmxSeniorVault uses ERC4626 share conversion against donation-inflatable totalAssets without virtual or "
            "dead shares. An early depositor can donate aUSDC to inflate the exchange rate, causing later deposits to "
            "round down and transferring value to the attacker."
        ),
        "sources": [
            {"path": "dn-gmx-vaults/contracts/ERC4626/ERC4626Upgradeable.sol", "start": 192, "end": 196},
            {"path": "dn-gmx-vaults/contracts/vaults/DnGmxSeniorVault.sol", "start": 368, "end": 373},
        ],
        "condition": {
            "source_required": [
                ["assets.mulDivDown(supply, totalAssets())"],
                ["aUsdc.balanceOf(address(this))", "totalUsdcBorrowed()"],
            ],
            "source_forbidden": [[], []],
            "file_forbidden": [["virtualShares", "deadShares"], ["virtualShares", "deadShares"]],
        },
    },
    {
        "replace_property": 399,
        "repo_path": "e7921851ec01",
        "grade": "B",
        "severity": "High",
        "tag": "Accounting Error",
        "subtag": "Incorrect Formula",
        "title": "D3Trading omits swap fees from settlement amounts",
        "description": (
            "D3Trading calculates swapFee but querySellTokens subtracts only mtFee, while queryBuyTokens prices "
            "toAmount plus mtFee without adding swapFee. Traders can avoid the protocol swap fee and distort swap accounting."
        ),
        "sources": [
            {"path": "dodo-v3/contracts/DODOV3MM/D3Pool/D3Trading.sol", "start": 179, "end": 245},
        ],
        "condition": {
            "source_required": [[
                "swapFee = DecimalMath.mulFloor",
                "receiveToAmount - mtFee",
                "toAmountWithFee = toAmount + mtFee",
            ]],
            "source_forbidden": [["receiveToAmount - swapFee", "toAmount + swapFee + mtFee"]],
            "file_forbidden": [[]],
        },
    },
    {
        "replace_property": 396,
        "repo_path": "9ddd6b83c27e",
        "grade": "B",
        "severity": "Medium",
        "tag": "DoS",
        "subtag": "Bad Condition",
        "title": "Public batch execution bypasses keeper pause control",
        "description": (
            "DnGmxBatchingManager restricts pause and unpause operations to keepers, but executeBatchDeposit is public "
            "and unpauses deposits whenever the contract is paused. An attacker can reopen deposits during the GMX "
            "cooldown workflow and disrupt batch settlement."
        ),
        "sources": [
            {
                "path": "dn-gmx-vaults/contracts/vaults/DnGmxBatchingManager.sol",
                "start": 158,
                "end": 166,
            },
            {
                "path": "dn-gmx-vaults/contracts/vaults/DnGmxBatchingManager.sol",
                "start": 239,
                "end": 253,
            },
            {
                "path": "dn-gmx-vaults/contracts/vaults/DnGmxBatchingManager.sol",
                "start": 356,
                "end": 379,
            },
        ],
        "condition": {
            "source_required": [
                ["function pauseDeposit() external onlyKeeper", "function unpauseDeposit() external onlyKeeper"],
                ["function executeBatchDeposit() external", "if (paused()) _unpause()"],
                ["if (vaultBatchingState.roundGlpStaked == 0) return"],
            ],
            "source_forbidden": [[], ["onlyKeeper"], []],
            "file_forbidden": [[], [], []],
        },
    },
    {
        "replace_property": 391,
        "repo_path": "1167ec3a176e",
        "grade": "B",
        "severity": "Medium",
        "tag": "Slippage",
        "subtag": "Invalid Slippage Control / Missing slippage check",
        "title": "Public HOT vault calls omit execution price bounds",
        "description": (
            "ValantisHOTModulePublic deposits with both expected price bounds set to zero, while direct public-vault "
            "mint and burn functions expose no caller-supplied token limits. Deposits and withdrawals can be sandwiched "
            "at manipulated reserve ratios."
        ),
        "sources": [
            {"path": "arrakis-modular/src/ArrakisMetaVaultPublic.sol", "start": 46, "end": 98},
            {"path": "arrakis-modular/src/modules/ValantisHOTModulePublic.sol", "start": 84, "end": 92},
        ],
        "condition": {
            "source_required": [
                ["function mint", "function burn"],
                ["alm.depositLiquidity(amount0, amount1, 0, 0)"],
            ],
            "source_forbidden": [["amount0Min", "amount1Min"], ["amount0Min", "amount1Min"]],
            "file_forbidden": [[], []],
        },
    },
]


def clean_text(value: object) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def strip_solidity_comments(text: str) -> str:
    without_blocks = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//.*?$", " ", without_blocks, flags=re.MULTILINE)


def verify_condition(text: str, condition: dict) -> tuple[bool, dict]:
    text = strip_solidity_comments(text)
    required = list(condition.get("required_all", []))
    forbidden = list(condition.get("forbidden_any", []))
    missing = [token for token in required if token not in text]
    forbidden_present = [token for token in forbidden if token in text]
    return not missing and not forbidden_present, {
        "missing": missing,
        "forbidden_present": forbidden_present,
    }


def verify_source_requirements(
    excerpts: list[str],
    full_texts: list[str],
    condition: dict,
) -> tuple[bool, list[dict]]:
    source_required = condition.get("source_required", [])
    source_forbidden = condition.get("source_forbidden", [])
    file_forbidden = condition.get("file_forbidden", [])
    if not (
        len(excerpts)
        == len(full_texts)
        == len(source_required)
        == len(source_forbidden)
        == len(file_forbidden)
    ):
        raise ValueError("Source verification configuration lengths do not match.")

    results = []
    for excerpt, full_text, required, forbidden, forbidden_file in zip(
        excerpts,
        full_texts,
        source_required,
        source_forbidden,
        file_forbidden,
    ):
        code = strip_solidity_comments(excerpt)
        full_code = strip_solidity_comments(full_text)
        missing = [token for token in required if token not in code]
        forbidden_present = [token for token in forbidden if token in code]
        forbidden_file_present = [token for token in forbidden_file if token in full_code]
        results.append(
            {
                "missing": missing,
                "forbidden_present": forbidden_present,
                "forbidden_file_present": forbidden_file_present,
            }
        )
    passed = all(
        not result["missing"]
        and not result["forbidden_present"]
        and not result["forbidden_file_present"]
        for result in results
    )
    return passed, results


def read_source_excerpt(
    archive: zipfile.ZipFile,
    repo_path: str,
    source: dict,
) -> tuple[str, str]:
    member = f"test/{repo_path}/{source['path']}"
    lines = archive.read(member).decode("utf-8", errors="replace").splitlines()
    start = int(source["start"])
    end = int(source["end"])
    if start < 1 or end < start or end > len(lines):
        raise ValueError(
            f"Invalid source range {start}-{end} for {member} with {len(lines)} lines."
        )
    excerpt = "\n".join(lines[start - 1 : end])
    location = f"{source['path']}:{start}-{end}"
    return location, excerpt


def build_evidence() -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(TEST_ZIP) as archive:
        for candidate in CANDIDATES:
            locations = []
            excerpts = []
            full_texts = []
            for source in candidate["sources"]:
                location, excerpt = read_source_excerpt(archive, candidate["repo_path"], source)
                locations.append(location)
                excerpts.append(excerpt)
                member = f"test/{candidate['repo_path']}/{source['path']}"
                full_texts.append(archive.read(member).decode("utf-8", errors="replace"))
            verified, source_details = verify_source_requirements(
                excerpts,
                full_texts,
                candidate["condition"],
            )
            row = {key: value for key, value in candidate.items() if key not in {"sources", "condition"}}
            row.update(
                {
                    "verified": verified,
                    "source_locations": json.dumps(locations, ensure_ascii=False),
                    "source_excerpt": "\n\n".join(excerpts),
                    "required_tokens": json.dumps(
                        candidate["condition"]["source_required"], ensure_ascii=False
                    ),
                    "missing_tokens": json.dumps(
                        [details["missing"] for details in source_details], ensure_ascii=False
                    ),
                    "forbidden_present": json.dumps(
                        [
                            details["forbidden_present"]
                            + details["forbidden_file_present"]
                            for details in source_details
                        ],
                        ensure_ascii=False,
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def apply_replacements(
    baseline: pd.DataFrame,
    evidence: pd.DataFrame,
    max_changes: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = evidence[
        evidence["verified"].astype(bool) & evidence["grade"].isin(["A", "B"])
    ].copy()
    if len(accepted) > max_changes:
        raise ValueError(f"Evidence contains {len(accepted)} changes; maximum is {max_changes}.")
    if accepted["replace_property"].duplicated().any():
        raise ValueError("Evidence contains duplicate replacement Property values.")

    out = baseline.copy()
    changes = []
    for _, finding in accepted.iterrows():
        prop = int(finding["replace_property"])
        matches = out.index[out["Property"].astype(int) == prop].tolist()
        if len(matches) != 1:
            raise ValueError(f"Replacement Property {prop} was not found exactly once.")
        index = matches[0]
        if str(out.at[index, "repo_path"]) != str(finding["repo_path"]):
            raise ValueError(f"Replacement Property {prop} must stay in the same repository.")

        before = out.loc[index].to_dict()
        for column in ["severity", "tag", "subtag", "description"]:
            out.at[index, column] = clean_text(finding[column])
        changes.append(
            {
                "Property": prop,
                "repo_path": finding["repo_path"],
                "grade": finding["grade"],
                "title": finding.get("title", ""),
                "old_severity": before["severity"],
                "old_tag": before["tag"],
                "old_subtag": before["subtag"],
                "new_severity": finding["severity"],
                "new_tag": finding["tag"],
                "new_subtag": finding["subtag"],
            }
        )

    if out.shape != baseline.shape:
        raise AssertionError("Fusion changed submission shape.")
    if out["Property"].astype(int).tolist() != baseline["Property"].astype(int).tolist():
        raise AssertionError("Fusion changed Property ordering.")
    if out["repo_path"].value_counts().to_dict() != baseline["repo_path"].value_counts().to_dict():
        raise AssertionError("Fusion changed per-repository finding counts.")
    return out, pd.DataFrame(changes)


def validate_labels(evidence: pd.DataFrame, train: pd.DataFrame) -> None:
    for column in ["severity", "tag", "subtag"]:
        legal = {clean_text(value) for value in train[column].dropna()}
        actual = {clean_text(value) for value in evidence[column]}
        illegal = sorted(actual - legal)
        if illegal:
            raise ValueError(f"Illegal {column} labels in evidence: {illegal}")


def validate_final_candidate(
    candidate: pd.DataFrame,
    sample: pd.DataFrame,
    test: pd.DataFrame,
    train: pd.DataFrame,
) -> None:
    errors = []
    if list(candidate.columns) != list(sample.columns):
        errors.append("columns must exactly match submission_example.csv")
    if len(candidate) != 400:
        errors.append(f"candidate must contain exactly 400 rows, got {len(candidate)}")
    if candidate.columns.astype(str).str.startswith("Unnamed:").any():
        errors.append("candidate contains an Unnamed index column")
    if "Property" not in candidate or candidate["Property"].astype(int).tolist() != list(range(1, 401)):
        errors.append("Property must be exactly 1..400")
    if candidate.isna().any().any():
        errors.append("candidate contains NaN")

    test_repos = set(test["repo_path"].astype(str))
    candidate_repos = set(candidate["repo_path"].astype(str)) if "repo_path" in candidate else set()
    if not candidate_repos.issubset(test_repos):
        errors.append("candidate contains repo_path values outside test.csv")
    if test_repos - candidate_repos:
        errors.append("candidate does not cover every test repo_path")

    for column in ["severity", "tag", "subtag"]:
        legal = {clean_text(value) for value in train[column].dropna()}
        actual = {clean_text(value) for value in candidate[column]}
        if actual - legal:
            errors.append(f"candidate contains illegal {column} labels: {sorted(actual - legal)}")

    for column in candidate.columns:
        values = candidate[column].astype(str)
        if values.map(lambda value: clean_text(value) == "").any():
            errors.append(f"candidate contains blank values in {column}")
        if values.str.strip().eq("...").any():
            errors.append(f"candidate contains literal ellipsis in {column}")
    descriptions = candidate["description"].astype(str)
    if descriptions.str.contains(r"[\r\n]", regex=True).any():
        errors.append("candidate descriptions contain newlines")
    if descriptions.map(len).lt(25).any():
        errors.append("candidate contains descriptions shorter than 25 characters")
    if candidate.duplicated(["repo_path", "severity", "tag", "subtag", "description"]).any():
        errors.append("candidate contains duplicate finding rows")
    if errors:
        raise ValueError("; ".join(errors))


def write_reports(evidence: pd.DataFrame, changes: pd.DataFrame) -> None:
    evidence_lines = [
        "# Source Audit Findings",
        "",
        f"- Exact source archive: `{TEST_ZIP}`",
        f"- Verified candidates: `{int(evidence['verified'].sum())}` / `{len(evidence)}`",
        "- Grade B = source-proven vulnerable condition in the exact Kaggle test snapshot.",
        "",
    ]
    for _, row in evidence.iterrows():
        evidence_lines.extend(
            [
                f"## Property {int(row['replace_property'])}: {row['title']}",
                "",
                f"- Repo: `{row['repo_path']}`",
                f"- Grade: `{row['grade']}`",
                f"- Verified: `{row['verified']}`",
                f"- Label: `{row['severity']} / {row['tag']} / {row['subtag']}`",
                f"- Source: `{row['source_locations']}`",
                f"- Description: {row['description']}",
                "",
            ]
        )
    REPORT_PATH.write_text("\n".join(evidence_lines), encoding="utf-8")

    fusion_lines = [
        "# Source Audit Fusion Report",
        "",
        f"- Immutable baseline: `{BASELINE_CSV}`",
        f"- Candidate: `{OUTPUT_CSV}`",
        f"- Changed rows: `{len(changes)}`",
        "- Per-repository finding counts: unchanged.",
        "- Total findings: unchanged at 400.",
        "",
        "## Replacements",
        "",
    ]
    for _, row in changes.iterrows():
        fusion_lines.append(
            f"- Property `{int(row['Property'])}` ({row['repo_path']}): "
            f"`{row['old_severity']} / {row['old_tag']} / {row['old_subtag']}` -> "
            f"`{row['new_severity']} / {row['new_tag']} / {row['new_subtag']}` "
            f"[{row['grade']}: {row['title']}]"
        )
    FUSION_REPORT_PATH.write_text("\n".join(fusion_lines) + "\n", encoding="utf-8")


def main() -> int:
    for path in [EVIDENCE_PARQUET.parent, OUTPUT_CSV.parent, REPORT_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(BASELINE_CSV)
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    sample = pd.read_csv(SAMPLE_CSV)
    evidence = build_evidence()
    validate_labels(evidence, train)

    if not evidence["verified"].all():
        failed = evidence.loc[~evidence["verified"], ["title", "missing_tokens", "forbidden_present"]]
        raise RuntimeError(f"Source verification failed:\n{failed.to_string(index=False)}")

    candidate, changes = apply_replacements(baseline, evidence, max_changes=8)
    if len(changes) < 4:
        raise RuntimeError("Fewer than four source-proven improvements qualified; refusing to create candidate.")
    candidate = candidate[list(sample.columns)]
    for column in candidate.columns:
        if column != "Property":
            candidate[column] = candidate[column].map(clean_text)
    validate_final_candidate(candidate, sample, test, train)

    evidence.to_parquet(EVIDENCE_PARQUET, index=False)
    candidate.to_csv(OUTPUT_CSV, index=False)
    shutil.copy2(OUTPUT_CSV, ROOT / "submission.csv")
    shutil.copy2(OUTPUT_CSV, ROOT / "outputs" / "submission.csv")
    write_reports(evidence, changes)

    print(f"Verified source findings: {len(evidence)}")
    print(f"Changed baseline rows: {len(changes)}")
    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Copied recommended candidate to: {ROOT / 'submission.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
