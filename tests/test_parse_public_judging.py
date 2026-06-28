import json
import tempfile
import unittest
import zipfile
from pathlib import Path


class ParsePublicJudgingTests(unittest.TestCase):
    def test_project_files_use_portable_relative_paths(self):
        from src.parse_public_judging import PROJECT_ROOT, portable_path

        self.assertEqual(
            portable_path(PROJECT_ROOT / "data" / "external" / "public_judging"),
            "data/external/public_judging",
        )

    def test_output_schema_includes_raw_category_labels(self):
        from src.parse_public_judging import OUTPUT_COLUMNS, parse_public_judging

        with tempfile.TemporaryDirectory() as tmp:
            findings = parse_public_judging(Path(tmp))

        self.assertEqual(OUTPUT_COLUMNS[-2:], ["raw_tag", "raw_subtag"])
        self.assertEqual(list(findings.columns), OUTPUT_COLUMNS)

    def test_parses_sherlock_markdown_front_matter(self):
        from src.parse_public_judging import parse_public_judging

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contest = root / "2024-01-example-judging"
            contest.mkdir()
            (contest / "42.md").write_text(
                """---
id: 42
title: Missing authorization in withdraw
severity: High
status: accepted
category: Access Control
tags: Missing Authorization
url: https://github.com/sherlock-audit/2024-01-example-judging/issues/42
---

## Summary

`Vault.withdraw` in `src/Vault.sol` does not verify the caller.

## Vulnerability Detail

Anyone can call `withdraw()` and drain the vault.
""",
                encoding="utf-8",
            )

            findings = parse_public_judging(root)

        self.assertEqual(len(findings), 1)
        row = findings.iloc[0]
        self.assertEqual(row["platform"], "sherlock")
        self.assertEqual(row["contest"], "2024-01-example")
        self.assertEqual(row["issue_id"], "42")
        self.assertEqual(row["title"], "Missing authorization in withdraw")
        self.assertEqual(row["severity"], "High")
        self.assertIn("Anyone can call", row["description"])
        self.assertEqual(row["referenced_files"], ["src/Vault.sol"])
        self.assertIn("withdraw", row["referenced_functions"])
        self.assertTrue(row["source_path"].endswith("42.md"))
        self.assertEqual(row["raw_tag"], "Access Control")
        self.assertEqual(row["raw_subtag"], "Missing Authorization")
        self.assertEqual(
            row["source_url"],
            "https://github.com/sherlock-audit/2024-01-example-judging/issues/42",
        )

    def test_parses_code4rena_json_issue(self):
        from src.parse_public_judging import parse_public_judging

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contest = root / "2024-02-example-findings"
            contest.mkdir()
            issue = {
                "number": 17,
                "title": "[M-03] Incorrect fee accounting",
                "body": (
                    "## Vulnerability details\n"
                    "`FeeManager.collectFees()` in `contracts/FeeManager.sol` "
                    "uses the stale total and overcharges users."
                ),
                "html_url": "https://github.com/code-423n4/2024-02-example-findings/issues/17",
                "labels": [{"name": "2 (Med Risk)"}, {"name": "validated"}],
                "state": "closed",
                "tag": "Arithmetic",
                "subtag": "Incorrect Accounting",
            }
            (contest / "issues.json").write_text(json.dumps([issue]), encoding="utf-8")

            findings = parse_public_judging(root)

        self.assertEqual(len(findings), 1)
        row = findings.iloc[0]
        self.assertEqual(row["platform"], "code4rena")
        self.assertEqual(row["contest"], "2024-02-example")
        self.assertEqual(row["issue_id"], "17")
        self.assertEqual(row["title"], "Incorrect fee accounting")
        self.assertEqual(row["severity"], "Medium")
        self.assertEqual(row["referenced_files"], ["contracts/FeeManager.sol"])
        self.assertIn("collectFees", row["referenced_functions"])
        self.assertTrue(row["source_path"].endswith("issues.json#17"))
        self.assertEqual(row["raw_tag"], "Arithmetic")
        self.assertEqual(row["raw_subtag"], "Incorrect Accounting")
        self.assertEqual(row["source_url"], issue["html_url"])

    def test_sherlock_prefers_best_and_report_group_files_over_raw_issues(self):
        from src.parse_public_judging import parse_public_judging

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            modern = root / "rage-trade-judging"
            (modern / "001-M").mkdir(parents=True)
            (modern / "12.md").write_text(
                "# Raw duplicate submission\n\nSeverity: Medium\n\nDo not parse me.",
                encoding="utf-8",
            )
            (modern / "001-M" / "7-best.md").write_text(
                """---
title: Accepted modern finding
severity: Medium
---
The bug is in `src/Modern.sol` within `settle()`.
""",
                encoding="utf-8",
            )
            (modern / "002-H").mkdir()
            (modern / "002-H" / "9-report.md").write_text(
                """# Accepted legacy finding

Severity: High

Source: https://github.com/sherlock-audit/rage-trade-judging/issues/9

`contracts/Legacy.sol::execute` permits unauthorized execution.
""",
                encoding="utf-8",
            )

            findings = parse_public_judging(root)

        self.assertEqual(set(findings["title"]), {"Accepted modern finding", "Accepted legacy finding"})
        self.assertEqual(set(findings["issue_id"]), {"001-M", "002-H"})
        self.assertFalse(findings["description"].str.contains("Do not parse me").any())
        legacy = findings.loc[findings["issue_id"] == "002-H"].iloc[0]
        self.assertEqual(
            legacy["source_url"],
            "https://github.com/sherlock-audit/rage-trade-judging/issues/9",
        )

    def test_sherlock_prefers_consolidated_rova_readme_issue_sections(self):
        from src.parse_public_judging import parse_public_judging

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contest = root / "2025-02-rova-judging"
            contest.mkdir()
            (contest / "README.md").write_text(
                """# Contest results

# Issue M-1: Tokens can be locked permanently

Source: https://github.com/sherlock-audit/2025-02-rova-judging/issues/55

Calling `lock()` in `src/Locker.sol` before initialization permanently traps funds.

# Issue H-2: Arbitrary execution drains assets

`Executor.execute()` in `src/Executor.sol` lacks authorization.

# Issue L-3: Minor event mismatch

This low issue must not be retained.
""",
                encoding="utf-8",
            )
            (contest / "55.md").write_text(
                "# Raw submission copy\n\nSeverity: Medium\n\nDuplicate raw body.",
                encoding="utf-8",
            )

            findings = parse_public_judging(root)

        self.assertEqual(list(findings["issue_id"]), ["M-1", "H-2"])
        self.assertEqual(list(findings["severity"]), ["Medium", "High"])
        self.assertEqual(
            findings.iloc[0]["source_url"],
            "https://github.com/sherlock-audit/2025-02-rova-judging/issues/55",
        )
        self.assertFalse(findings["description"].str.contains("Duplicate raw body").any())

    def test_sherlock_group_accepted_files_take_precedence_over_consolidated_readme(self):
        from src.parse_public_judging import parse_public_judging

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contest = root / "priority-judging"
            (contest / "001-M").mkdir(parents=True)
            (contest / "001-M" / "7-best.md").write_text(
                """---
title: Group accepted finding
severity: Medium
---
The accepted group body references `src/Accepted.sol`.
""",
                encoding="utf-8",
            )
            (contest / "README.md").write_text(
                """# Contest results

# Issue H-1: Conflicting README finding

The consolidated body references `src/Readme.sol`.
""",
                encoding="utf-8",
            )

            findings = parse_public_judging(root)

        self.assertEqual(list(findings["issue_id"]), ["001-M"])
        self.assertEqual(list(findings["title"]), ["Group accepted finding"])
        self.assertFalse(findings["description"].str.contains("consolidated body").any())

    def test_parses_code4rena_linked_report_headings_and_rejects_non_hm_sections(self):
        from src.parse_public_judging import parse_public_judging

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contest = root / "2024-07-traitforge-findings"
            contest.mkdir()
            (contest / "report.md").write_text(
                """# Findings

## [[H-01] Wrong minting logic](https://github.com/code-423n4/2024-07-traitforge-findings/issues/231)

`TraitForgeNft::mintWithBudget` in `contracts/TraitForgeNft.sol` uses the wrong counter.

## [M-02] Incorrect fee calculation

The `calculateFee()` function in `contracts/Fee.sol` rounds against users.

## [QA-01] Missing documentation

Informational only.
""",
                encoding="utf-8",
            )

            findings = parse_public_judging(root)

        self.assertEqual(list(findings["issue_id"]), ["H-01", "M-02"])
        self.assertEqual(list(findings["severity"]), ["High", "Medium"])
        self.assertEqual(findings.iloc[0]["title"], "Wrong minting logic")
        self.assertEqual(findings.iloc[0]["raw_tag"], "")
        self.assertEqual(findings.iloc[0]["raw_subtag"], "")
        self.assertEqual(
            findings.iloc[0]["source_url"],
            "https://github.com/code-423n4/2024-07-traitforge-findings/issues/231",
        )
        self.assertEqual(
            findings.iloc[1]["source_url"],
            "https://github.com/code-423n4/2024-07-traitforge-findings",
        )

    def test_rejects_non_final_or_non_high_medium_records(self):
        from src.parse_public_judging import parse_public_judging

        rejected = [
            ("qa", "QA"),
            ("gas", "Gas"),
            ("informational", "Informational"),
            ("low", "Low"),
            ("invalid", "Invalid"),
            ("duplicate", "Duplicate"),
            ("withdrawn", "Withdrawn"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contest = root / "filter-findings"
            contest.mkdir()
            issues = [
                {
                    "number": index,
                    "title": f"Rejected {name}",
                    "body": "Body",
                    "labels": [{"name": label}],
                    "state": "closed",
                }
                for index, (name, label) in enumerate(rejected, start=1)
            ]
            issues.append(
                {
                    "number": 100,
                    "title": "Accepted",
                    "body": "Accepted body",
                    "labels": [{"name": "High"}, {"name": "validated"}],
                    "state": "closed",
                }
            )
            issues.append(
                {
                    "number": 101,
                    "title": "Still pending",
                    "body": "Pending body",
                    "labels": [{"name": "High"}],
                    "state": "open",
                }
            )
            issues.append(
                {
                    "number": 102,
                    "title": "Closed but not adjudicated",
                    "body": "Closed body",
                    "labels": [{"name": "High"}],
                    "state": "closed",
                }
            )
            (contest / "issues.json").write_text(json.dumps(issues), encoding="utf-8")

            findings = parse_public_judging(root)

        self.assertEqual(list(findings["title"]), ["Accepted"])

    def test_code4rena_json_accepts_only_explicit_final_statuses(self):
        from src.parse_public_judging import parse_public_judging

        allowed = ["final", "validated", "accepted", "confirmed"]
        disallowed = ["valid", "adjudicated", "approved", "finalized", "upheld"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contest = root / "status-findings"
            contest.mkdir()
            issues = [
                {
                    "number": index,
                    "title": f"Status {status}",
                    "body": "Finding body",
                    "severity": "High",
                    "status": status,
                    "state": "closed",
                }
                for index, status in enumerate(allowed + disallowed, start=1)
            ]
            (contest / "issues.json").write_text(json.dumps(issues), encoding="utf-8")

            findings = parse_public_judging(root)

        self.assertEqual(
            list(findings["title"]),
            [f"Status {status}" for status in allowed],
        )

    def test_scans_supported_zip_archives_without_extraction(self):
        from src.parse_public_judging import parse_public_judging

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "cached.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr(
                    "2024-03-zipped-findings-main/report.md",
                    """## [M-01] Zipped accounting issue

`account()` in `src/Account.sol` loses precision.
""",
                )

            findings = parse_public_judging(root)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.iloc[0]["contest"], "2024-03-zipped")
        self.assertIn("cached.zip!", findings.iloc[0]["source_path"])

    def test_run_writes_parquet_and_per_contest_report(self):
        import pandas as pd

        from src.parse_public_judging import run

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "cache"
            contest = input_dir / "2024-04-output-findings"
            contest.mkdir(parents=True)
            (contest / "report.md").write_text(
                """## [H-01] Output finding

`run()` in `src/Output.sol` is unsafe.
""",
                encoding="utf-8",
            )
            output_path = tmp_path / "findings.parquet"
            report_path = tmp_path / "parse_report.md"

            findings = run(input_dir, output_path, report_path)
            saved = pd.read_parquet(output_path)
            report = report_path.read_text(encoding="utf-8")

        self.assertEqual(len(findings), 1)
        self.assertEqual(saved.loc[0, "title"], "Output finding")
        self.assertIn("Parsed findings: `1`", report)
        self.assertIn("| code4rena | 2024-04-output | High | 1 |", report)


if __name__ == "__main__":
    unittest.main()
