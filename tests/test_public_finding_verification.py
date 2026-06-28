import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd


def write_test_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "test/repo-a/src/Vault.sol",
            """
pragma solidity ^0.8.20;

contract Vault {
    mapping(address => uint256) public balances;

    function withdraw(uint256 amount) external {
        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
    }

    function settleAccount(address account) external {
        balances[account] = 0;
    }
}
""",
        )
        zf.writestr(
            "test/repo-a/contracts/FeeManager.sol",
            """
contract FeeManager {
    function collectFees(uint256 total) external returns (uint256) {
        return total / 100;
    }
}
""",
        )
        zf.writestr(
            "test/repo-a/lib/forge-std/src/Test.sol",
            "contract Test { function assertEq(uint256, uint256) internal {} }",
        )
        zf.writestr(
            "test/repo-a/test/Vault.t.sol",
            "contract VaultTest { function testWithdraw() external {} }",
        )
        zf.writestr(
            "test/repo-a/mocks/MockToken.sol",
            "contract MockToken {}",
        )
        zf.writestr(
            "test/repo-a/generated/Bindings.sol",
            "contract Bindings {}",
        )
        zf.writestr(
            "__MACOSX/test/repo-a/src/._Vault.sol",
            "ignored",
        )


def finding(**overrides):
    row = {
        "platform": "sherlock",
        "contest": "demo-contest",
        "issue_id": "M-1",
        "title": "Withdrawal accounting is incorrect",
        "severity": "Medium",
        "description": "The withdrawal path can lose funds.",
        "referenced_files": [],
        "referenced_functions": [],
        "source_path": "demo-judging/1.md",
        "source_url": "https://example.test/1",
    }
    row.update(overrides)
    return row


class PublicFindingVerificationTests(unittest.TestCase):
    def test_zip_index_keeps_only_project_solidity_sources(self):
        from src.verify_public_findings_against_test import build_source_index

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "test.zip"
            write_test_zip(zip_path)

            index = build_source_index(zip_path)

        self.assertEqual(
            set(index["repo-a"]),
            {"src/Vault.sol", "contracts/FeeManager.sol"},
        )
        self.assertIn("function withdraw", index["repo-a"]["src/Vault.sol"].text)

    def test_exact_path_and_function_match_is_exact(self):
        from src.verify_public_findings_against_test import (
            build_source_index,
            verify_finding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "test.zip"
            write_test_zip(zip_path)
            index = build_source_index(zip_path)

            result = verify_finding(
                finding(
                    referenced_files=["src/Vault.sol"],
                    referenced_functions=["withdraw"],
                ),
                "repo-a",
                index,
            )

        self.assertEqual(result["confidence"], "exact")
        self.assertEqual(result["matched_files"], ["src/Vault.sol"])
        self.assertEqual(result["matched_functions"], ["withdraw"])
        self.assertEqual(result["file_match_kind"], "exact")

    def test_basename_and_function_match_is_strong(self):
        from src.verify_public_findings_against_test import (
            build_source_index,
            verify_finding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "test.zip"
            write_test_zip(zip_path)
            index = build_source_index(zip_path)

            result = verify_finding(
                finding(
                    referenced_files=["packages/protocol/Vault.sol"],
                    referenced_functions=["withdraw"],
                ),
                "repo-a",
                index,
            )

        self.assertEqual(result["confidence"], "strong")
        self.assertEqual(result["matched_files"], ["src/Vault.sol"])
        self.assertEqual(result["file_match_kind"], "basename")

    def test_identifier_overlap_without_file_reference_is_weak(self):
        from src.verify_public_findings_against_test import (
            build_source_index,
            verify_finding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "test.zip"
            write_test_zip(zip_path)
            index = build_source_index(zip_path)

            result = verify_finding(
                finding(
                    title="settleAccount clears balances too early",
                    description="Calling settleAccount for an account erases balances.",
                    referenced_functions=["settleAccount"],
                ),
                "repo-a",
                index,
            )

        self.assertEqual(result["confidence"], "weak")
        self.assertEqual(result["matched_functions"], ["settleAccount"])
        self.assertIn("src/Vault.sol", result["matched_files"])

    def test_dependency_reference_is_rejected_even_if_archive_contains_it(self):
        from src.verify_public_findings_against_test import (
            build_source_index,
            verify_finding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "test.zip"
            write_test_zip(zip_path)
            index = build_source_index(zip_path)

            result = verify_finding(
                finding(
                    referenced_files=["lib/forge-std/src/Test.sol"],
                    referenced_functions=["assertEq"],
                ),
                "repo-a",
                index,
            )

        self.assertEqual(result["confidence"], "rejected")
        self.assertEqual(result["rejection_reason"], "excluded_referenced_path")
        self.assertEqual(result["matched_files"], [])

    def test_explicit_missing_file_is_rejected_despite_identifier_elsewhere(self):
        from src.verify_public_findings_against_test import (
            build_source_index,
            verify_finding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "test.zip"
            write_test_zip(zip_path)
            index = build_source_index(zip_path)

            result = verify_finding(
                finding(
                    referenced_files=["src/MissingVault.sol"],
                    referenced_functions=["withdraw"],
                ),
                "repo-a",
                index,
            )

        self.assertEqual(result["confidence"], "rejected")
        self.assertEqual(result["rejection_reason"], "referenced_file_not_found")

    def test_source_snippet_match_is_strong(self):
        from src.verify_public_findings_against_test import (
            build_source_index,
            verify_finding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "test.zip"
            write_test_zip(zip_path)
            index = build_source_index(zip_path)

            result = verify_finding(
                finding(
                    description=(
                        "The following source line permits unsafe accounting:\n"
                        "```solidity\n"
                        "balances[msg.sender] -= amount;\n"
                        "```"
                    )
                ),
                "repo-a",
                index,
            )

        self.assertEqual(result["confidence"], "strong")
        self.assertEqual(result["matched_snippets"], ["balances[msg.sender] -= amount;"])
        self.assertEqual(result["matched_files"], ["src/Vault.sol"])

    def test_plain_prose_words_do_not_count_as_identifier_evidence(self):
        from src.verify_public_findings_against_test import (
            build_source_index,
            verify_finding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "test.zip"
            write_test_zip(zip_path)
            index = build_source_index(zip_path)

            result = verify_finding(
                finding(
                    title="Balances are not described clearly",
                    description="The report discusses balances and account behavior in prose.",
                ),
                "repo-a",
                index,
            )

        self.assertEqual(result["confidence"], "rejected")
        self.assertEqual(result["matched_identifiers"], [])

    def test_inline_code_identifier_records_its_source_file(self):
        from src.verify_public_findings_against_test import (
            build_source_index,
            verify_finding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "test.zip"
            write_test_zip(zip_path)
            index = build_source_index(zip_path)

            result = verify_finding(
                finding(description="The `balances` state can become inconsistent."),
                "repo-a",
                index,
            )

        self.assertEqual(result["confidence"], "weak")
        self.assertEqual(result["matched_identifiers"], ["balances"])
        self.assertEqual(result["matched_files"], ["src/Vault.sol"])

    def test_dataframe_verification_maps_contest_to_repo_and_preserves_schema(self):
        from src.verify_public_findings_against_test import (
            OUTPUT_COLUMNS,
            build_source_index,
            verify_findings,
        )

        findings = pd.DataFrame(
            [
                finding(
                    referenced_files=["contracts/FeeManager.sol"],
                    referenced_functions=["collectFees"],
                )
            ]
        )
        manifest = pd.DataFrame(
            [
                {
                    "repo_path": "repo-a",
                    "platform": "sherlock",
                    "contest_slug": "demo-contest",
                    "resolution_status": "resolved",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "test.zip"
            write_test_zip(zip_path)
            index = build_source_index(zip_path)

            verified = verify_findings(findings, index, manifest)

        self.assertEqual(list(verified.columns), OUTPUT_COLUMNS)
        self.assertEqual(verified.loc[0, "repo_path"], "repo-a")
        self.assertEqual(verified.loc[0, "confidence"], "exact")
        self.assertEqual(verified.loc[0, "title"], findings.loc[0, "title"])

    def test_run_writes_empty_artifacts_with_clear_upstream_degradation(self):
        from src.verify_public_findings_against_test import OUTPUT_COLUMNS, run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "test.zip"
            write_test_zip(zip_path)
            output_path = root / "matches.parquet"
            report_path = root / "report.md"

            result = run(
                findings_path=root / "missing-findings.parquet",
                manifest_path=root / "missing-manifest.csv",
                zip_path=zip_path,
                output_path=output_path,
                report_path=report_path,
            )

            saved = pd.read_parquet(output_path)
            report = report_path.read_text(encoding="utf-8")

        self.assertEqual(list(result.columns), OUTPUT_COLUMNS)
        self.assertTrue(saved.empty)
        self.assertIn("Status: `degraded`", report)
        self.assertIn("Missing parsed findings", report)
        self.assertIn("Missing contest manifest", report)


if __name__ == "__main__":
    unittest.main()
